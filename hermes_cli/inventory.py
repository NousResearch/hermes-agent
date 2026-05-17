"""Provider/model inventory context — shared substrate for the dashboard
``/api/model/options``, the TUI ``model.options``/``model.save_key``
JSON-RPC handlers, and the interactive picker.

Before this module the three call-sites each duplicated:

1. The 17-LOC config-slice that pulls ``model.{default,name,provider,base_url}``,
   ``providers:``, and ``custom_providers:`` out of ``load_config()``;
2. The call into ``list_authenticated_providers`` with the resulting kwargs;
3. (TUI only) a 45-LOC post-pass that merges authenticated rows with
   unconfigured ``CANONICAL_PROVIDERS`` rows and emits ``authenticated``/
   ``auth_type``/``key_env``/``warning`` hints for the picker UI.

Consolidating those three steps into one entry point eliminates two bugs
the duplicates were hiding:

- The dashboard read ``cfg.get("custom_providers")`` directly, missing the
  v12+ keyed ``providers:`` form (which the TUI handled via
  ``get_compatible_custom_providers``).
- The TUI's canonical-merge keyed on ``is_user_defined`` to decide
  ordering. Section 3 of ``list_authenticated_providers`` sets
  ``is_user_defined=True`` even for canonical slugs that appear in the
  ``providers:`` config dict, which silently demoted them to the tail of
  the picker. ``_reorder_canonical`` keys on slug membership instead.

Substrate facts (verified May 2026):
- ``list_authenticated_providers`` already populates each row's
  ``models`` from the curated catalog (same source as the picker). Do
  NOT call ``provider_model_ids()`` per row to "freshen" — that bypasses
  curation and pulls in non-agentic models (Nous /models returns ~400
  IDs including TTS, embeddings, rerankers, image/video generators).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional


# ─── Public types ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class ConfigContext:
    """Snapshot of the model + provider config every inventory caller
    needs. Built once via ``load_picker_context()``; the TUI overlays
    live agent state via ``with_overrides()`` before passing through.
    """

    current_provider: str
    current_model: str
    current_base_url: str
    user_providers: dict
    custom_providers: list

    def with_overrides(
        self,
        *,
        current_provider: Optional[str] = None,
        current_model: Optional[str] = None,
        current_base_url: Optional[str] = None,
    ) -> "ConfigContext":
        """Return a copy with truthy overrides applied.

        Truthy-only because the TUI reads agent attributes that may be
        empty strings before an agent is spawned — empties must NOT
        clobber the disk-config values.
        """
        kw: dict = {}
        if current_provider:
            kw["current_provider"] = current_provider
        if current_model:
            kw["current_model"] = current_model
        if current_base_url:
            kw["current_base_url"] = current_base_url
        return replace(self, **kw) if kw else self


def load_picker_context() -> ConfigContext:
    """Load the disk-config snapshot every consumer needs.

    Replaces the inline 17-LOC config-slice that ``web_server.py`` and
    ``tui_gateway/server.py`` (×2 sites) used to do.
    """
    from hermes_cli.config import get_compatible_custom_providers, load_config

    cfg = load_config()
    model_cfg = cfg.get("model", {})
    if isinstance(model_cfg, dict):
        current_model = model_cfg.get("default", model_cfg.get("name", "")) or ""
        current_provider = model_cfg.get("provider", "") or ""
        current_base_url = model_cfg.get("base_url", "") or ""
    else:
        # config.model can be a bare string in older configs.
        current_model = str(model_cfg) if model_cfg else ""
        current_provider = ""
        current_base_url = ""
    raw = cfg.get("providers")
    return ConfigContext(
        current_provider=current_provider,
        current_model=current_model,
        current_base_url=current_base_url,
        user_providers=raw if isinstance(raw, dict) else {},
        custom_providers=get_compatible_custom_providers(cfg),
    )


# ─── Public: payload builder ────────────────────────────────────────────


def build_models_payload(
    ctx: ConfigContext,
    *,
    include_unconfigured: bool = False,
    picker_hints: bool = False,
    canonical_order: bool = False,
    max_models: int = 50,
) -> dict:
    """Build the ``{providers, model, provider}`` shape every consumer
    needs from a single substrate call.

    Flags:
    - ``include_unconfigured``: append ``CANONICAL_PROVIDERS`` rows that
      ``list_authenticated_providers`` didn't emit (TUI uses this to show
      the full provider universe in the picker).
    - ``picker_hints``: add ``authenticated``/``auth_type``/``key_env``/
      ``warning`` per row (TUI ``ModelPickerDialog`` shape).
    - ``canonical_order``: reorder canonical-slug rows to
      ``CANONICAL_PROVIDERS`` declaration order; truly-custom rows go
      last (TUI display order).
    """
    from hermes_cli.model_switch import list_authenticated_providers

    rows = list_authenticated_providers(
        current_provider=ctx.current_provider,
        current_base_url=ctx.current_base_url,
        current_model=ctx.current_model,
        user_providers=ctx.user_providers,
        custom_providers=ctx.custom_providers,
        max_models=max_models,
    )

    if include_unconfigured:
        rows = list(rows) + _append_unconfigured_rows(rows, ctx)
    if picker_hints:
        _apply_picker_hints(rows)
    if canonical_order:
        rows = _reorder_canonical(rows)

    return {
        "providers": rows,
        "model": ctx.current_model,
        "provider": ctx.current_provider,
    }


# ─── Internal: row post-processing ──────────────────────────────────────


def _append_unconfigured_rows(rows: list[dict], ctx: ConfigContext) -> list[dict]:
    """Build skeleton rows for canonical providers missing from ``rows``."""
    from hermes_cli.models import CANONICAL_PROVIDERS, _PROVIDER_LABELS

    seen = {r["slug"].lower() for r in rows}
    cur = (ctx.current_provider or "").lower()
    extras: list[dict] = []
    for entry in CANONICAL_PROVIDERS:
        if entry.slug.lower() in seen:
            continue
        extras.append(
            {
                "slug": entry.slug,
                "name": _PROVIDER_LABELS.get(entry.slug, entry.label),
                "is_current": entry.slug.lower() == cur,
                "is_user_defined": False,
                "models": [],
                "total_models": 0,
                "source": "canonical",
            }
        )
    return extras


def _apply_picker_hints(rows: list[dict]) -> None:
    """Add ``authenticated``/``auth_type``/``key_env``/``warning`` per row.

    Mutates ``rows`` in-place. Rows already from
    ``list_authenticated_providers`` are marked ``authenticated=True``;
    the unconfigured skeleton rows from ``_append_unconfigured_rows`` get
    the picker's setup-hint shape.
    """
    from hermes_cli.auth import PROVIDER_REGISTRY

    for row in rows:
        if "authenticated" in row:
            continue
        # Distinguish authenticated rows (returned by
        # list_authenticated_providers) from skeleton rows (from
        # _append_unconfigured_rows). The skeleton rows have empty
        # `models` AND source="canonical"; authenticated rows have
        # populated `models` OR a non-canonical source.
        is_skeleton = row.get("source") == "canonical" and not row.get("models")
        row["authenticated"] = not is_skeleton
        if not is_skeleton or row.get("is_user_defined"):
            continue
        cfg = PROVIDER_REGISTRY.get(row["slug"])
        auth_type = cfg.auth_type if cfg else "api_key"
        key_env = (
            cfg.api_key_env_vars[0]
            if (cfg and cfg.api_key_env_vars)
            else ""
        )
        row["auth_type"] = auth_type
        row["key_env"] = key_env
        row["warning"] = (
            f"paste {key_env} to activate"
            if auth_type == "api_key" and key_env
            else f"run `hermes model` to configure ({auth_type})"
        )


def _reorder_canonical(rows: list[dict]) -> list[dict]:
    """Canonical slugs in ``CANONICAL_PROVIDERS`` declaration order;
    truly-custom rows last.

    Keys on slug membership, NOT ``is_user_defined`` — section 3 of
    ``list_authenticated_providers`` sets ``is_user_defined=True`` on
    rows from the ``providers:`` config dict even when the slug is
    canonical. Keying on the flag would silently demote canonical
    providers configured via the new keyed schema.
    """
    from hermes_cli.models import CANONICAL_PROVIDERS

    order = {e.slug: i for i, e in enumerate(CANONICAL_PROVIDERS)}
    canon = sorted(
        (r for r in rows if r["slug"] in order),
        key=lambda r: order[r["slug"]],
    )
    extras = [r for r in rows if r["slug"] not in order]
    return canon + extras
