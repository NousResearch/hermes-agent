"""Provider/model inventory — single source of truth for the picker,
dashboard ``/api/model/options``, TUI ``model.options``, and the new
``hermes models``/``hermes providers`` CLI surfaces.

Issue #23359 documents the four pre-PR surfaces that each duplicated
config-slice + post-processing inline. This module owns that traversal so
consumers reduce to ``build_payload(kind, ctx=load_picker_context())``.

Substrate facts (verified May 2026):

- ``CANONICAL_PROVIDERS`` (35) is the universe — the picker iterates it.
  ``PROVIDER_REGISTRY`` (33) and ``_PROVIDER_MODELS`` (32) each miss
  different providers; iterating either drops real providers.
- ``profile.fetch_models()`` returns the unfiltered raw catalog
  (OpenRouter raw=367 vs picker-filtered=34). Always go through
  ``provider_model_ids()``.
- ``_PROVIDER_MODELS["openrouter"]`` is ``None``. Use the
  ``OPENROUTER_MODELS`` constant directly for the offline path.
- ``list_authenticated_providers`` section 3 emits rows from the
  ``providers:`` config dict with ``is_user_defined=True`` even when the
  slug is canonical. Reorder must key on slug∈CANONICAL, not the flag.
- ``provider_model_ids`` has no auth precheck — it'll attempt HTTP for
  every slug. ``_skip_live_for`` gates that.
- ``get_compatible_custom_providers(cfg)`` is the canonical merged view
  of legacy ``custom_providers`` list + v12+ keyed ``providers`` dict.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from typing import Optional

SCHEMA_VERSION = 1

# Curated allowlist: the only ``oauth_external`` provider with a real
# ``/models`` REST endpoint. Most use proprietary URI schemes
# (``cloudcode-pa://google``) where probing falls through to curated and
# masks failure. See substrate notes in references/.
_OAUTH_EXTERNAL_HAS_MODELS = frozenset({"openai-codex"})


# ─── Public types ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ConfigContext:
    """Snapshot of model + provider config used by every inventory caller."""

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
        """Return a copy with truthy overrides applied (TUI session state)."""
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

    Replaces the inline 17-LOC config-slice that web_server.py,
    tui_gateway/server.py, and the CLI handlers each used to do.
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


# ─── Internal: per-provider HTTP gate ───────────────────────────────────


def _skip_live_for(slug: str) -> bool:
    """Return True if calling ``provider_model_ids(slug)`` will likely fail
    or fall through to curated catalogs (masking failure as success).

    Gates the live HTTP path so ``--all`` doesn't make ~20 spurious calls.
    """
    from hermes_cli.auth import PROVIDER_REGISTRY
    from providers import get_provider_profile

    profile = get_provider_profile(slug)
    cfg = PROVIDER_REGISTRY.get(slug)
    if profile is None:
        # Legacy provider (lmstudio, tencent-tokenhub) — registry-only.
        return cfg is None
    if profile.auth_type in ("aws_sdk", "external_process"):
        return True
    if profile.auth_type == "oauth_external":
        return slug not in _OAUTH_EXTERNAL_HAS_MODELS
    if profile.auth_type == "oauth_device_code":
        try:
            from hermes_cli.auth import resolve_nous_runtime_credentials

            return not resolve_nous_runtime_credentials()
        except Exception:
            return True
    if profile.auth_type == "copilot":
        return False
    # api_key — skip if no env var
    return not _is_configured_env_only(slug)


def _is_configured_env_only(slug: str) -> bool:
    """Env-var-only auth check for the ``--offline`` path.

    Examines ``profile.env_vars`` ∪ ``PROVIDER_REGISTRY[slug].api_key_env_vars``
    ∪ ``HERMES_OVERLAYS[slug].extra_env_vars``. No ``auth.json`` /
    credential pool / Claude-Code creds — deterministic from environment.
    """
    from hermes_cli.auth import PROVIDER_REGISTRY
    from providers import get_provider_profile

    candidates: list[str] = []
    profile = get_provider_profile(slug)
    if profile and profile.env_vars:
        candidates.extend(profile.env_vars)
    cfg = PROVIDER_REGISTRY.get(slug)
    if cfg and cfg.api_key_env_vars:
        candidates.extend(cfg.api_key_env_vars)
    try:
        from hermes_cli.providers import HERMES_OVERLAYS

        overlay = HERMES_OVERLAYS.get(slug)
        if overlay and overlay.extra_env_vars:
            candidates.extend(overlay.extra_env_vars)
    except Exception:
        pass
    return any(os.environ.get(v) for v in candidates)


# ─── Internal: model resolution ─────────────────────────────────────────


def _models_for(slug: str, *, live: bool) -> tuple[list[str], str]:
    """Return ``(model_ids, source)`` for a canonical provider slug.

    ``source`` ∈ {"catalog","curated","fallback","empty"}. Static path
    NEVER calls ``model_ids()`` / ``get_curated_nous_model_ids()`` /
    ``fetch_openrouter_models()`` — all three remote-fetch.
    """
    from hermes_cli.models import OPENROUTER_MODELS, _PROVIDER_MODELS
    from providers import get_provider_profile

    if live and not _skip_live_for(slug):
        try:
            from hermes_cli.models import provider_model_ids

            ids = provider_model_ids(slug)
            if ids:
                return list(ids), "catalog"
        except Exception:
            pass
    if slug == "openrouter":
        ids = [m for m, _ in OPENROUTER_MODELS]
        if ids:
            return ids, "curated"
    curated = list(_PROVIDER_MODELS.get(slug, []) or [])
    if curated:
        return curated, "curated"
    profile = get_provider_profile(slug)
    if profile and profile.fallback_models:
        return list(profile.fallback_models), "fallback"
    return [], "empty"


# ─── Internal: row enrichment ───────────────────────────────────────────


def _enrich_row(row: dict) -> dict:
    """Attach ``api_mode``/``auth_type``/``base_url``/``env_vars``/
    ``env_vars_present`` from profile or registry. Mutates and returns.
    Env-var values are NEVER recorded — only presence booleans.
    """
    from hermes_cli.auth import PROVIDER_REGISTRY
    from providers import get_provider_profile

    slug = row["slug"]
    profile = get_provider_profile(slug)
    cfg = PROVIDER_REGISTRY.get(slug)
    if profile:
        row.setdefault("api_mode", profile.api_mode)
        row.setdefault("auth_type", profile.auth_type)
        row.setdefault("base_url", profile.base_url)
        env_vars = list(profile.env_vars)
    elif cfg:
        # Legacy (lmstudio, tencent-tokenhub) — both OpenAI-compatible.
        row.setdefault("api_mode", "chat_completions")
        row.setdefault("auth_type", cfg.auth_type)
        row.setdefault("base_url", cfg.inference_base_url)
        env_vars = list(cfg.api_key_env_vars)
    else:
        row.setdefault("api_mode", "unknown")
        row.setdefault("auth_type", "unknown")
        row.setdefault("base_url", "")
        env_vars = []
    try:
        from hermes_cli.providers import HERMES_OVERLAYS

        overlay = HERMES_OVERLAYS.get(slug)
        if overlay and overlay.extra_env_vars:
            for ev in overlay.extra_env_vars:
                if ev not in env_vars:
                    env_vars.append(ev)
    except Exception:
        pass
    row["env_vars"] = env_vars
    row["env_vars_present"] = [bool(os.environ.get(v)) for v in env_vars]
    return row


# ─── Internal: enumeration ──────────────────────────────────────────────


def _enumerate(
    ctx: ConfigContext, *, live: bool, include_unconfigured: bool, max_models: int
) -> list[dict]:
    """Build the row list. ``live=False`` synthesizes from in-repo data
    only (no HTTP, env-only auth derivation); ``live=True`` delegates to
    ``list_authenticated_providers`` (the canonical picker code path).
    """
    from hermes_cli.models import CANONICAL_PROVIDERS, _PROVIDER_LABELS

    if live:
        from hermes_cli.model_switch import list_authenticated_providers

        rows: list[dict] = []
        for r in list_authenticated_providers(
            current_provider=ctx.current_provider,
            current_base_url=ctx.current_base_url,
            current_model=ctx.current_model,
            user_providers=ctx.user_providers,
            custom_providers=ctx.custom_providers,
            max_models=max_models,
        ):
            r = dict(r)
            r["auth_state"] = "configured"
            _enrich_row(r)
            rows.append(r)
        if include_unconfigured:
            seen = {r["slug"].lower() for r in rows}
            cur = (ctx.current_provider or "").lower()
            for entry in CANONICAL_PROVIDERS:
                if entry.slug.lower() in seen:
                    continue
                row = {
                    "slug": entry.slug,
                    "name": _PROVIDER_LABELS.get(entry.slug, entry.label),
                    "is_current": entry.slug.lower() == cur,
                    "is_user_defined": False,
                    "models": [],
                    "total_models": 0,
                    "source": "canonical",
                    "auth_state": "unconfigured",
                }
                _enrich_row(row)
                rows.append(row)
        # Live rows already have models; tag source for renderers.
        for row in rows:
            row.setdefault(
                "model_source",
                "user-config" if row.get("is_user_defined") else "catalog",
            )
        return rows

    # Static path.
    rows = []
    cur = (ctx.current_provider or "").lower()
    for entry in CANONICAL_PROVIDERS:
        configured = _is_configured_env_only(entry.slug)
        if not configured and not include_unconfigured:
            continue
        ids, source = _models_for(entry.slug, live=False)
        row = {
            "slug": entry.slug,
            "name": _PROVIDER_LABELS.get(entry.slug, entry.label),
            "is_current": entry.slug.lower() == cur,
            "is_user_defined": False,
            "models": ids,
            "total_models": len(ids),
            "source": "canonical",
            "model_source": source,
            "auth_state": "configured" if configured else "unconfigured",
        }
        _enrich_row(row)
        rows.append(row)
    cur_url = (ctx.current_base_url or "").rstrip("/").lower()
    for cp in ctx.custom_providers:
        if not isinstance(cp, dict):
            continue
        name = str(cp.get("name") or "").strip()
        if not name:
            continue
        api_key = str(cp.get("api_key") or "").strip()
        configured = bool(cp.get("base_url"))
        if api_key:
            if api_key.startswith("${") and api_key.endswith("}"):
                configured = bool(os.environ.get(api_key[2:-1]))
            else:
                configured = True
        if not configured and not include_unconfigured:
            continue
        cp_url = str(cp.get("base_url") or "").strip().rstrip("/").lower()
        models = [
            m if isinstance(m, str) else (m.get("id") or m.get("name") or "")
            for m in (cp.get("models") or [])
            if (isinstance(m, str) and m) or (isinstance(m, dict) and (m.get("id") or m.get("name")))
        ]
        rows.append(
            {
                "slug": f"custom:{name}",
                "name": name,
                "is_current": bool(cp_url) and cp_url == cur_url,
                "is_user_defined": True,
                "models": models,
                "total_models": len(models),
                "source": "user-config",
                "model_source": "user-config",
                "auth_state": "configured" if configured else "unconfigured",
                "api_mode": "chat_completions",
                "auth_type": "api_key",
                "base_url": str(cp.get("base_url") or ""),
                "env_vars": [],
                "env_vars_present": [],
            }
        )
    return rows


# ─── Internal: picker-shape post-processing ─────────────────────────────


def _apply_picker_hints(rows: list[dict]) -> None:
    """Add ``authenticated``/``key_env``/``warning`` per row (TUI consumer)."""
    from hermes_cli.auth import PROVIDER_REGISTRY

    for row in rows:
        if "authenticated" in row:
            continue
        configured = row.get("auth_state") == "configured"
        row["authenticated"] = configured
        if configured or row.get("is_user_defined"):
            continue
        cfg = PROVIDER_REGISTRY.get(row["slug"])
        auth_type = cfg.auth_type if cfg else row.get("auth_type", "api_key")
        key_env = (
            cfg.api_key_env_vars[0]
            if (cfg and cfg.api_key_env_vars)
            else ""
        )
        row.setdefault("auth_type", auth_type)
        row["key_env"] = key_env
        row["warning"] = (
            f"paste {key_env} to activate"
            if auth_type == "api_key" and key_env
            else f"run `hermes model` to configure ({auth_type})"
        )


def _reorder_canonical(rows: list[dict]) -> list[dict]:
    """Canonical slugs in CANONICAL_PROVIDERS order; truly-custom rows last.

    Keys on slug membership, NOT ``is_user_defined`` — section 3 of
    ``list_authenticated_providers`` sets ``is_user_defined=True`` for
    rows from ``providers:`` config dict even when the slug is canonical.
    """
    from hermes_cli.models import CANONICAL_PROVIDERS

    order = {e.slug: i for i, e in enumerate(CANONICAL_PROVIDERS)}
    canon = sorted((r for r in rows if r["slug"] in order), key=lambda r: order[r["slug"]])
    extras = [r for r in rows if r["slug"] not in order]
    return canon + extras


# ─── Internal: provider-arg resolution ──────────────────────────────────


def _resolve_provider(slug: str, ctx: ConfigContext) -> Optional[dict]:
    """Resolve a user-supplied ``--provider NAME`` argument.

    Returns ``{"slug": canonical_slug, "kind": "canonical"|"custom",
    "name": display_name}`` or ``None`` if unknown.
    """
    from hermes_cli.models import CANONICAL_PROVIDERS, _PROVIDER_LABELS, normalize_provider
    from providers import get_provider_profile

    if not slug:
        return None
    if slug.startswith("custom:"):
        bare = slug[7:].strip()
        for cp in ctx.custom_providers:
            if isinstance(cp, dict) and str(cp.get("name") or "").strip().lower() == bare.lower():
                return {"slug": str(cp["name"]).strip(), "kind": "custom", "name": str(cp["name"]).strip()}
        return None
    profile = get_provider_profile(slug)
    if profile is not None:
        return {
            "slug": profile.name,
            "kind": "canonical",
            "name": _PROVIDER_LABELS.get(profile.name, profile.name),
        }
    canon = normalize_provider(slug)
    canonical_set = {p.slug: p for p in CANONICAL_PROVIDERS}
    if canon in canonical_set:
        p = canonical_set[canon]
        return {"slug": canon, "kind": "canonical", "name": _PROVIDER_LABELS.get(canon, p.label)}
    for cp in ctx.custom_providers:
        if isinstance(cp, dict) and str(cp.get("name") or "").strip().lower() == slug.lower():
            return {"slug": str(cp["name"]).strip(), "kind": "custom", "name": str(cp["name"]).strip()}
    return None


# ─── Public: payload builder ────────────────────────────────────────────


def build_payload(
    kind: str,
    *,
    ctx: Optional[ConfigContext] = None,
    provider: Optional[str] = None,
    include_unconfigured: bool = False,
    live: bool = True,
    offline: bool = False,
    picker_hints: bool = False,
    canonical_order: bool = False,
    max_models: int = 50,
) -> dict:
    """Single payload builder for all 3 surfaces.

    ``kind`` ∈ {"models", "providers", "status"}:
      - models: per-row models list; supports scoped ``provider=`` lookup.
      - providers: same row shape; ignores models field for the renderer.
      - status: row shape minus the models payload (auth snapshot only).

    ``offline=True`` forces ``live=False`` and bypasses
    ``list_authenticated_providers`` (which has unconditional HTTP).
    ``picker_hints``/``canonical_order`` are TUI shape extras.
    """
    if kind not in ("models", "providers", "status"):
        raise ValueError(f"unknown kind: {kind!r}")
    if offline:
        live = False
    if ctx is None:
        ctx = load_picker_context()

    if kind == "models" and provider:
        # Scoped lookup — always returns one row, never filtered by creds.
        r = _resolve_provider(provider, ctx)
        if r is None:
            raise ValueError(f"unknown provider: {provider!r}")
        if r["kind"] == "canonical":
            ids, source = _models_for(r["slug"], live=live)
            row = {
                "slug": r["slug"],
                "name": r["name"],
                "is_current": r["slug"].lower() == (ctx.current_provider or "").lower(),
                "is_user_defined": False,
                "models": ids,
                "total_models": len(ids),
                "source": "canonical",
                "model_source": source,
                "auth_state": "configured" if _is_configured_env_only(r["slug"]) else "unconfigured",
            }
            _enrich_row(row)
        else:
            cp_entry = next(
                (cp for cp in ctx.custom_providers if isinstance(cp, dict)
                 and str(cp.get("name") or "").strip() == r["slug"]),
                None,
            )
            if cp_entry is None:
                raise ValueError(f"custom provider not found: {r['slug']!r}")
            models = [m for m in (cp_entry.get("models") or []) if isinstance(m, str)]
            row = {
                "slug": f"custom:{r['slug']}",
                "name": r["name"],
                "is_current": False,
                "is_user_defined": True,
                "models": models,
                "total_models": len(models),
                "source": "user-config",
                "model_source": "user-config",
                "auth_state": "configured",
                "api_mode": "chat_completions",
                "auth_type": "api_key",
                "base_url": str(cp_entry.get("base_url") or ""),
                "env_vars": [],
                "env_vars_present": [],
            }
        rows = [row]
    else:
        rows = _enumerate(
            ctx,
            live=live,
            include_unconfigured=include_unconfigured or kind == "status",
            max_models=max_models,
        )

    if kind == "status":
        # Drop the models payload; status is an auth snapshot.
        for row in rows:
            row.pop("models", None)
            row.pop("total_models", None)
            row.pop("model_source", None)

    if picker_hints:
        _apply_picker_hints(rows)
    if canonical_order:
        rows = _reorder_canonical(rows)

    return {
        "schema_version": SCHEMA_VERSION,
        "current": {"provider": ctx.current_provider, "model": ctx.current_model},
        "providers": rows,
    }


# ─── Public: rendering ──────────────────────────────────────────────────


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths) and len(cell) > widths[i]:
                widths[i] = len(cell)
    lines = ["  ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) for cells in [headers, *rows]]
    return "\n".join(lines)


def render_text(payload: dict, kind: str) -> str:
    """Plain-text rendering for the default (non-``--json``) CLI mode."""
    rows = payload.get("providers", [])
    if not rows:
        return f"(no {kind})"
    if kind == "providers":
        body = [
            [r["slug"], str(r.get("auth_state", "unknown")),
             str(r.get("api_mode", "")), str(r.get("base_url", ""))]
            for r in rows
        ]
        configured = sum(1 for r in rows if r.get("auth_state") == "configured")
        return _format_table(
            ["SLUG", "AUTH", "API_MODE", "BASE_URL"], body
        ) + f"\n\n{configured} of {len(rows)} providers configured"
    if kind == "status":
        body = [[r["slug"], str(r.get("auth_state", "unknown"))] for r in rows]
        return _format_table(["PROVIDER", "AUTH"], body) + f"\n\n{len(rows)} providers"
    # models
    cur_p = payload.get("current", {}).get("provider", "")
    cur_m = payload.get("current", {}).get("model", "")
    body = []
    for r in rows:
        for m in r.get("models", []) or []:
            mark = " *current*" if r["slug"] == cur_p and m == cur_m else ""
            body.append([r["slug"], f"{m}{mark}", str(r.get("model_source", "?"))])
    if not body:
        return "(no models)"
    total = sum(len(r.get("models", []) or []) for r in rows)
    return _format_table(
        ["PROVIDER", "MODEL", "SOURCE"], body
    ) + f"\n\n{total} models across {len(rows)} provider(s)"


def dump_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)
