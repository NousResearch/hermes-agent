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
    pricing: bool = False,
    capabilities: bool = False,
    force_fresh_nous_tier: bool = False,
    refresh: bool = False,
    max_models: int | None = None,
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
    - ``pricing``: enrich each row with formatted per-model pricing and,
      for Nous, ``free_tier``/``unavailable_models`` so the GUI picker can
      show $/Mtok columns and gate paid models on free accounts —
      mirroring the ``hermes model`` CLI picker. Adds network calls
      (pricing fetch + Nous tier check); only set for interactive pickers.
    - ``capabilities``: add a per-row ``capabilities`` map
      ``{model: {fast, reasoning}}`` so pickers can gate the model-options
      controls (fast toggle / reasoning) to what each model actually
      supports, instead of offering knobs the backend would reject.
    - ``force_fresh_nous_tier``: bypass the short Nous free-tier cache when
      selecting Portal-recommended Nous models and applying tier gating. Keep
      this false for UI picker opens; explicit auth/model flows can opt in
      when they need freshly-purchased credits to show up immediately.
    - ``refresh``: bust the per-provider model-id disk cache so every row
      re-fetches its live catalog. Set only for an explicit user-triggered
      "refresh models" action; normal picker opens leave it false to stay
      snappy on the 1h cache.
    """
    from hermes_cli.model_switch import list_authenticated_providers

    rows = list_authenticated_providers(
        current_provider=ctx.current_provider,
        current_base_url=ctx.current_base_url,
        current_model=ctx.current_model,
        user_providers=ctx.user_providers,
        custom_providers=ctx.custom_providers,
        force_fresh_nous_tier=force_fresh_nous_tier,
        max_models=max_models,
        refresh=refresh,
    )

    # --- Deduplicate: remove models from aggregators that overlap with
    # user-defined providers.  When a local proxy (e.g. litellm-proxy)
    # serves a model whose name also appears in an aggregator's curated
    # catalog, the picker would show the model under both providers.
    # Selecting it from the aggregator row sets model.provider to the
    # aggregator (e.g. openrouter) instead of the user's proxy — silently
    # breaking the call.  Filtering at the payload level keeps the
    # aggregator rows honest: they only show models the user can't get
    # from a more-specific provider.  (#45954)
    try:
        from hermes_cli.providers import is_routing_aggregator as _is_routing_aggregator
    except Exception:
        _is_routing_aggregator = None  # type: ignore[assignment]

    if _is_routing_aggregator is not None:
        user_models: set[str] = set()
        for row in rows:
            if row.get("is_user_defined"):
                user_models.update(m.lower() for m in (row.get("models") or []))
        if user_models:
            for row in rows:
                # A user's own configured provider is never an "aggregator
                # duplicate" of itself: user_models is built from these very
                # rows, and is_routing_aggregator() reports True for every
                # custom:* slug.  Without this guard the dedup strips a
                # user-defined custom provider's entire model list (all of it
                # lives in user_models), emptying its picker row.
                if row.get("is_user_defined"):
                    continue
                slug = row.get("slug", "")
                # Only strip overlaps from TRUE routing aggregators (OpenRouter,
                # custom:* proxies). Flat-namespace resellers (opencode-go /
                # opencode-zen) serve every listed model as a first-party model,
                # so their rows must keep models that a user's proxy happens to
                # share a name with — otherwise a subscription provider's own
                # catalog (minimax-m3, glm-5, deepseek-v4-flash, ...) is silently
                # gutted in the picker. (#47077)
                if not _is_routing_aggregator(slug):
                    continue
                original = row.get("models") or []
                filtered = [m for m in original if m.lower() not in user_models]
                if len(filtered) < len(original):
                    row["models"] = filtered
                    row["total_models"] = len(filtered)

    if include_unconfigured:
        rows = list(rows) + _append_unconfigured_rows(rows, ctx)
    if picker_hints:
        _apply_picker_hints(rows)
    if canonical_order:
        rows = _reorder_canonical(rows)
    if pricing:
        _apply_pricing(rows, force_fresh_nous_tier=force_fresh_nous_tier)
    if capabilities:
        _apply_capabilities(rows)

    return {
        "providers": rows,
        "model": ctx.current_model,
        "provider": ctx.current_provider,
    }


def _apply_capabilities(rows: list[dict]) -> None:
    """Attach a ``{model: {fast, reasoning}}`` map to each provider row.

    `fast` mirrors ``model_supports_fast_mode`` (the same gate the runtime
    enforces). `reasoning` comes from the models.dev catalog when known and
    defaults to True otherwise — the effort dial is broadly accepted and a
    no-op on models that ignore it, whereas hiding it from a capable-but-
    uncatalogued model is the worse failure.
    """
    from hermes_cli.models import model_supports_fast_mode

    try:
        from agent.models_dev import get_model_capabilities
    except Exception:
        get_model_capabilities = None  # type: ignore[assignment]

    for row in rows:
        slug = row.get("slug") or ""
        caps: dict[str, dict[str, bool]] = {}

        for model in row.get("models") or []:
            reasoning = True
            if get_model_capabilities is not None and slug:
                try:
                    meta = get_model_capabilities(slug, model)
                    if meta is not None:
                        reasoning = bool(meta.supports_reasoning)
                except Exception:
                    reasoning = True

            caps[model] = {
                "fast": bool(model_supports_fast_mode(model)),
                "reasoning": reasoning,
            }

        row["capabilities"] = caps


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


def _apply_pricing(
    rows: list[dict],
    *,
    force_fresh_nous_tier: bool = False,
) -> None:
    """Enrich each provider row with per-model pricing + Nous tier gating.

    Mutates ``rows`` in-place. For every row whose provider supports live
    pricing (openrouter / nous / novita) adds::

        row["pricing"] = {model_id: {"input": "$3.00", "output": "$15.00",
                                     "cache": "$0.30" | None, "free": bool}}

    For Nous additionally adds::

        row["free_tier"] = bool            # current account is free-tier
        row["unavailable_models"] = [...]  # paid models a free user can't pick

    Prices are pre-formatted via ``_format_price_per_mtok`` so the GUI just
    renders strings — identical formatting to the CLI picker. All failures
    are swallowed (best-effort): a row simply gets no ``pricing`` key.
    """
    from hermes_cli.models import (
        _format_price_per_mtok,
        check_nous_free_tier,
        get_pricing_for_provider,
        partition_nous_models_by_tier,
    )

    # Resolve Nous free-tier once (cached in models.py for the TTL window).
    nous_free_tier: Optional[bool] = None

    for row in rows:
        slug = str(row.get("slug", "")).lower()
        models = row.get("models") or []
        if not models:
            continue
        try:
            raw_pricing = get_pricing_for_provider(slug) or {}
        except Exception:
            raw_pricing = {}
        if not raw_pricing:
            continue

        formatted: dict[str, dict] = {}
        for mid in models:
            p = raw_pricing.get(mid)
            if not p:
                continue
            inp_raw = p.get("prompt", "")
            out_raw = p.get("completion", "")
            cache_raw = p.get("input_cache_read", "")
            inp = _format_price_per_mtok(inp_raw) if inp_raw != "" else ""
            out = _format_price_per_mtok(out_raw) if out_raw != "" else ""
            cache = _format_price_per_mtok(cache_raw) if cache_raw else None
            # A model is "free" when both input and output cost nothing.
            is_free = inp == "free" and (out == "free" or out == "")
            formatted[mid] = {
                "input": inp,
                "output": out,
                "cache": cache,
                "free": is_free,
            }

        if formatted:
            row["pricing"] = formatted

        if slug == "nous":
            try:
                if nous_free_tier is None:
                    nous_free_tier = check_nous_free_tier(
                        force_fresh=force_fresh_nous_tier
                    )
                row["free_tier"] = bool(nous_free_tier)
                if nous_free_tier:
                    _selectable, unavailable = partition_nous_models_by_tier(
                        list(models), raw_pricing, free_tier=True
                    )
                    row["unavailable_models"] = unavailable
                else:
                    row["unavailable_models"] = []
            except Exception:
                # Tier detection failed — fail open (no gating) so the user
                # is never blocked from picking a model.
                row["free_tier"] = False
                row["unavailable_models"] = []
