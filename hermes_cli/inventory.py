"""Provider/model inventory — shared substrate for the dashboard ``/api/model/options``, the TUI
``model.options``/``model.save_key`` RPC handlers, and the interactive picker."""

from __future__ import annotations

from contextvars import copy_context
from dataclasses import dataclass, replace
from threading import Lock, Thread, current_thread
from typing import Any, Optional


_pricing_prewarm_lock = Lock()
_pricing_prewarm_threads: dict[tuple[str, tuple[tuple[str, str], ...]], Thread] = {}


# ─── Public types ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class ConfigContext:
    """Disk-config snapshot (``load_picker_context()``); the TUI overlays live agent state via
    ``with_overrides()``."""

    current_provider: str
    current_model: str
    current_base_url: str
    user_providers: dict
    custom_providers: list
    excluded_providers: list = None

    def with_overrides(
        self, *, current_provider: Optional[str] = None, current_model: Optional[str] = None,
        current_base_url: Optional[str] = None,
    ) -> "ConfigContext":
        """Copy with TRUTHY overrides applied: the TUI reads agent attributes that may be empty strings
        before an agent is spawned — empties must not clobber the disk-config values."""
        overrides = (("current_provider", current_provider), ("current_model", current_model),
                     ("current_base_url", current_base_url))
        kw = {k: v for k, v in overrides if v}
        return replace(self, **kw) if kw else self


def load_picker_context() -> ConfigContext:
    """Load the disk-config snapshot every consumer needs.

    Replaces the inline 17-LOC config-slice that ``web_server.py`` and
    ``tui_gateway/server.py`` (×2 sites) used to do.
    """
    from hermes_cli.config import (
        coerce_provider_id,
        get_compatible_custom_providers,
        load_config,
        stringify_provider_map,
    )

    cfg = load_config()
    model_cfg = cfg.get("model", {})
    if isinstance(model_cfg, dict):
        # PyYAML parses unquoted scalars as int (`provider: 2070`). Keep these
        # as strings so picker/options paths never call `.strip()` on an int.
        current_model = str(model_cfg.get("default", model_cfg.get("name", "")) or "")
        current_provider = coerce_provider_id(model_cfg.get("provider", ""))
        current_base_url = str(model_cfg.get("base_url", "") or "")
    else:
        # config.model can be a bare string in older configs.
        current_model = str(model_cfg) if model_cfg else ""
        current_provider = ""
        current_base_url = ""
    excluded = cfg.get("model_catalog", {}).get("excluded_providers") or []
    return ConfigContext(
        current_provider=current_provider,
        current_model=current_model,
        current_base_url=current_base_url,
        user_providers=stringify_provider_map(cfg.get("providers")),
        custom_providers=get_compatible_custom_providers(cfg),
        excluded_providers=excluded if isinstance(excluded, list) else [],
    )


def _slug(row: dict) -> str:
    return str(row.get("slug") or "").strip().lower()


def _without_slug(rows: list[dict], slug: str) -> list[dict]:
    return [r for r in rows if _slug(r) != slug]


# ─── Public: payload builder ────────────────────────────────────────────


def build_models_payload(
    ctx: ConfigContext,
    *,
    explicit_only: bool = False,
    include_unconfigured: bool = False,
    picker_hints: bool = False,
    canonical_order: bool = False,
    pricing: bool = False,
    pricing_cache_only: bool = False,
    capabilities: bool = False,
    featured: bool = False,
    force_fresh_nous_tier: bool = False,
    refresh: bool = False,
    probe_custom_providers: bool = True,
    probe_current_custom_provider: bool = False,
    for_picker: bool = False,
    max_models: int | None = None,
) -> dict:
    """Build the ``{providers, model, provider}`` shape every consumer
    needs from a single substrate call.

    Flags:
    - ``explicit_only``: keep only providers the user explicitly configured
      (current provider, providers from config, or providers backed by
      provider-specific env vars). This hides ambient / auto-seeded
      credentials from desktop chat pickers.
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
    - ``pricing_cache_only``: when pricing is enabled, use only values already
      resident in process caches. Normal picker opens use this while a
      background worker warms cold pricing endpoints.
    - ``capabilities``: add a per-row ``capabilities`` map
      ``{model: {fast, reasoning}}`` so pickers can gate the model-options
      controls (fast toggle / reasoning) to what each model actually
      supports, instead of offering knobs the backend would reject.
    - ``featured``: add a per-row ``featured_models`` list — the newest few
      models per lab (by models.dev release_date, ranked within the row's own
      models; see ``_FEATURED_PER_LAB``) for aggregator providers that serve
      dozens of models across many labs. Pickers default their visible set to
      these; the rest of ``models`` stays reachable via search / show-all. Empty
      for single-lab providers (callers fall back to top-N). Derived live from
      models.dev — no allowlist.
    - ``force_fresh_nous_tier``: bypass the short Nous free-tier cache when
      selecting Portal-recommended Nous models and applying tier gating. Keep
      this false for UI picker opens; explicit auth/model flows can opt in
      when they need freshly-purchased credits to show up immediately.
    - ``refresh``: bust the per-provider model-id disk cache so every row
      re-fetches its live catalog. Set only for an explicit user-triggered
      "refresh models" action; normal picker opens leave it false to stay
      snappy on the 1h cache.
    - ``probe_custom_providers``: allow saved custom/provider endpoints to
      run live ``/models`` discovery while building the payload. GUI picker
      opens should leave this false unless the user explicitly refreshes; the
      row can still render its configured model immediately, and slow/offline
      local endpoints no longer block the dialog.
    - ``probe_current_custom_provider``: when ``probe_custom_providers`` is
      false, still live-probe the current custom endpoint. This keeps normal
      GUI/TUI picker opens fast while making the active custom provider's model
      list match the classic CLI picker.
    - ``for_picker``: interactive-picker visibility. Keeps providers whose
      credential pool exists but is entirely rate-limited (exhausted) in the
      list. Rate limits are per-model, so a different model under the same
      provider may still work; hiding the provider strands the user. Set for
      any surface a human is choosing from, not for programmatic resolution.
    """
    from hermes_cli.model_switch import list_authenticated_providers

    rows = list_authenticated_providers(
        current_provider=ctx.current_provider, current_base_url=ctx.current_base_url,
        current_model=ctx.current_model, user_providers=ctx.user_providers,
        custom_providers=ctx.custom_providers, force_fresh_nous_tier=force_fresh_nous_tier,
        max_models=max_models, refresh=refresh, probe_custom_providers=probe_custom_providers,
        probe_current_custom_provider=probe_current_custom_provider, for_picker=for_picker,
        excluded_providers=ctx.excluded_providers or [],
    )

    # Managed local runtime: staged GGUFs are selectable like any provider's
    # models. list_authenticated_providers can't know about them (no
    # credential, no custom_providers entry — the credential is
    # reachability), so inject the row here where every picker surface
    # inherits it. Present whenever models are staged; picking one routes
    # through the llamacpp alias -> managed/detected server resolution.
    local_row = _local_runtime_row(ctx)
    if local_row is not None:
        rows = [r for r in rows if str(r.get("slug", "")).lower() != "llamacpp"]
        rows.append(local_row)
        # A live session on the managed server reports provider "custom"
        # (the resolution seam's generic label for a raw base_url), which
        # would otherwise materialize a duplicate "Custom endpoint" row
        # carrying the same staged models and stealing the checkmark. The
        # Local row owns the managed server's identity — drop custom rows
        # that point at the managed endpoint.
        if local_row.get("is_current"):
            def _is_managed_custom(row: dict) -> bool:
                if str(row.get("slug", "")).lower() != "custom":
                    return False
                models = {str(m) for m in (row.get("models") or [])}
                return bool(models) and models <= set(local_row["models"])

            rows = [r for r in rows if not _is_managed_custom(r)]

    moa_row = _moa_provider_row(ctx.current_provider)
    if moa_row is not None:
        rows = [moa_row] + _without_slug(rows, "moa")

    if explicit_only:
        rows = _filter_explicit_provider_rows(rows, ctx)
        # Desktop chat pickers request the explicit subset without the full
        # unconfigured provider universe. If the configured current provider
        # has lost its credential, list_authenticated_providers() omits it;
        # keep that one row visible so the UI can show the saved selection and
        # a re-auth affordance instead of appearing to jump to another provider.
        # Exception: a "custom" current whose endpoint is the managed local
        # server is already represented (with the checkmark) by the Local row
        # — the skeleton would resurrect the duplicate the dedup above removed.
        _local_owns_current = bool(local_row and local_row.get("is_current")
                                   and (ctx.current_provider or "").lower() == "custom")
        if not _local_owns_current:
            rows = list(rows) + _append_unconfigured_rows(
                rows, ctx, current_only=True
            )

    # A local proxy serving a model also in an aggregator's catalog would show under both, and picking
    # the aggregator row silently breaks the call — aggregators only list models no specific provider has.
    _strip_aggregator_overlaps(rows)

    if include_unconfigured:
        rows = list(rows) + _without_slug(_append_unconfigured_rows(rows, ctx), "moa")
    if picker_hints:
        _apply_picker_hints(rows)
    if canonical_order:
        rows = _reorder_canonical(rows)
    if pricing:
        _apply_pricing(
            rows,
            force_fresh_nous_tier=force_fresh_nous_tier,
            cached_only=pricing_cache_only,
        )
    if capabilities:
        _apply_capabilities(rows)
    if featured:
        _apply_featured(rows)
    _apply_custom_aliases(rows)

    return {"providers": rows, "model": ctx.current_model, "provider": ctx.current_provider}


def _strip_aggregator_overlaps(rows: list[dict]) -> None:
    """Drop models from TRUE routing aggregators (OpenRouter, custom:* proxies) that a user-defined
    provider also serves. The is_user_defined guard matters: is_routing_aggregator() is True for every
    custom:* slug, so without it the dedup would empty a user's own custom row. Flat-namespace
    resellers (opencode-go/zen) serve every model first-party and keep shared names."""
    try:
        from hermes_cli.providers import is_routing_aggregator
    except Exception:
        return

    user_models: set[str] = set()
    for row in rows:
        if row.get("is_user_defined"):
            user_models.update(m.lower() for m in (row.get("models") or []))
    if not user_models:
        return
    for row in rows:
        if row.get("is_user_defined") or not is_routing_aggregator(row.get("slug", "")):
            continue
        # Only strip overlaps from TRUE routing aggregators (OpenRouter, custom:* proxies). Flat-namespace
        # resellers (opencode-go / opencode-zen) serve every listed model as a first-party model, so their
        # rows must keep models that a user's proxy happens to share a name with — otherwise a subscription
        # provider's own catalog (minimax-m3, glm-5, deepseek-v4-flash, ...) is silently gutted in the
        # picker. (#47077)
        original = row.get("models") or []
        filtered = [m for m in original if m.lower() not in user_models]
        if len(filtered) < len(original):
            row["models"] = filtered
            row["total_models"] = len(filtered)


def build_model_options_payload(
    ctx: ConfigContext, *, explicit_only: bool = False, include_unconfigured: bool = False,
    refresh: bool = False,
) -> dict:
    """Shared API-server/dashboard/TUI payload. Normal open probes only the current custom provider so
    offline saved endpoints don't block the picker; explicit refresh probes all and busts the cache."""
    refresh = bool(refresh)
    payload = build_models_payload(
        ctx,
        explicit_only=bool(explicit_only),
        include_unconfigured=bool(include_unconfigured),
        picker_hints=True,
        canonical_order=True,
        pricing=True,
        pricing_cache_only=not refresh,
        capabilities=True,
        featured=True,
        refresh=refresh,
        probe_custom_providers=refresh,
        probe_current_custom_provider=not refresh,
    )
    if not refresh:
        _prewarm_pricing_async(
            payload["providers"],
            current_provider=ctx.current_provider,
            current_base_url=ctx.current_base_url,
        )
    return payload


# ─── Public: auxiliary-task pickers ─────────────────────────────────────


def build_aux_picker_rows(
    *, current_provider: str = "", current_model: str = "", current_base_url: str = "",
    max_models: int | None = None,
) -> list[dict]:
    """Provider rows for any auxiliary-task picker (vision, compression, …). Honours
    ``excluded_providers``; exhausted-pool providers stay visible (``for_picker``); only the active
    custom endpoint is probed. ``moa`` is excluded: auxiliary_client unwraps it to its aggregator
    anyway, so offering it would be a choice silently rewritten.

    Aux pickers kept re-deriving their own kwargs and each one silently dropped a different slice of the
    user's configuration. Two independent contributor PRs landed against the same two call sites for exactly
    this: 52642 (user ``providers:`` / ``custom_providers:`` entries never appeared) and #66624 (providers
    with an exhausted credential pool were hidden). Both were per-site kwarg patches, so the next aux picker
    would have reintroduced the same gap. Routing through one function makes the correct behaviour the
    default that a new caller cannot forget:
    """
    ctx = load_picker_context().with_overrides(
        current_provider=current_provider, current_model=current_model, current_base_url=current_base_url,
    )
    rows = build_models_payload(
        ctx, for_picker=True, probe_custom_providers=False, probe_current_custom_provider=True,
        max_models=max_models,
    )["providers"]
    return _without_slug(rows, "moa")


def format_aux_picker_entries(
    rows: list[dict], *, current_provider: str = "", current_base_url: str = "",
) -> list[tuple[str, str, list[str]]]:
    """Render aux-picker rows as ``(slug, label, models)``. A raw ``base_url`` custom endpoint is
    "current" only through that URL, never a slug — so with ``current_base_url`` set no row is marked."""
    entries: list[tuple[str, str, list[str]]] = []
    current_slug = str(current_provider or "").strip().lower()
    has_base_url = bool(str(current_base_url or "").strip())
    for row in rows:
        slug = str(row.get("slug") or "")
        name = row.get("name") or slug
        total = row.get("total_models") or len(row.get("models") or [])
        model_hint = f" — {total} models" if total else ""
        marker = "  ← current" if slug.lower() == current_slug and current_slug and not has_base_url else ""
        entries.append((slug, f"{name}{model_hint}{marker}", list(row.get("models") or [])))
    return entries


def _reasoning_catalog_reader(slug: str):
    """Per-model reasoning-capability reader for aggregators that publish one.

    Cache-only — building the picker payload must never block on HTTP. A cold
    cache warms in the background so the next open is accurate; until then the
    model reports no restriction and the UI offers the full scale.
    """
    try:
        from hermes_cli.models import (
            nous_model_reasoning_capabilities,
            openrouter_model_reasoning_capabilities,
            warm_nous_reasoning_caps_async,
            warm_openrouter_reasoning_caps_async,
        )
    except Exception:
        return None

    if slug == "nous":
        warm_nous_reasoning_caps_async()
        return nous_model_reasoning_capabilities
    if slug == "openrouter":
        warm_openrouter_reasoning_caps_async()
        return openrouter_model_reasoning_capabilities
    return None


def _apply_capabilities(rows: list[dict]) -> None:
    """Attach a ``{model: {fast, reasoning, ...}}`` map to each provider row.

    `fast` mirrors ``model_supports_fast_mode`` (the same gate the runtime
    enforces). `reasoning` comes from the models.dev catalog when known and
    defaults to True otherwise — the effort dial is broadly accepted and a
    no-op on models that ignore it, whereas hiding it from a capable-but-
    uncatalogued model is the worse failure.

    Aggregators that publish per-model reasoning detail add
    `can_disable_reasoning`, False on reasoning-mandatory routes whose upstream
    answers a disable with HTTP 400. Omitted when the catalog doesn't say,
    which the UI reads as "no restriction known". Such a catalog also overrides
    `reasoning` itself when it reports a route that takes no reasoning
    parameter — a definitive negative from the provider actually serving the
    model outranks the models.dev inference.

    The catalog's `supported_efforts` list is deliberately NOT forwarded: it
    under-reports. The Portal accepts and honors levels a route doesn't
    advertise (``z-ai/glm-5.3`` publishes ``max, high, low`` yet serves
    ``minimal`` at its lowest thinking), so filtering the picker by that list
    would hide levels that demonstrably work.
    """
    from hermes_cli.models import model_supports_fast_mode

    try:
        from agent.models_dev import get_model_capabilities
    except Exception:
        get_model_capabilities = None  # type: ignore[assignment]

    for row in rows:
        slug = row.get("slug") or ""
        caps: dict[str, dict[str, Any]] = {}
        read_reasoning_catalog = _reasoning_catalog_reader(slug.lower())

        for model in row.get("models") or []:
            reasoning = True
            if get_model_capabilities is not None and slug:
                try:
                    meta = get_model_capabilities(slug, model)
                    if meta is not None:
                        reasoning = bool(meta.supports_reasoning)
                except Exception:
                    reasoning = True

            entry: dict[str, Any] = {
                "fast": bool(model_supports_fast_mode(model)),
                "reasoning": reasoning,
            }

            if reasoning and read_reasoning_catalog is not None:
                try:
                    detail = read_reasoning_catalog(model)
                except Exception:
                    detail = None
                if detail and not detail.get("supports_reasoning"):
                    # For a route it serves, the aggregator's own catalog beats
                    # models.dev: no reasoning parameter means no reasoning
                    # controls, so there is no disable to describe either.
                    entry["reasoning"] = False
                elif detail:
                    entry["can_disable_reasoning"] = not detail.get("mandatory")

            caps[model] = entry

        row["capabilities"] = caps


# Newest N models per lab an aggregator row features by default (older tail behind search/show-all);
# 5 keeps a lab's headliners without letting a prolific vendor flood the view.
_FEATURED_PER_LAB = 5


def _apply_featured(rows: list[dict]) -> None:
    """Attach a ``featured_models`` shortlist to each aggregator row: newest ``_FEATURED_PER_LAB`` per
    vendor by models.dev ``release_date`` (ranked within the row, never vs. today, so it is stable);
    ties keep curated order. Non-aggregators get an empty list and keep top-N behaviour."""
    try:
        from agent.models_dev import get_model_info
    except Exception:
        get_model_info = None  # type: ignore[assignment]

    for row in rows:
        slug = str(row.get("slug") or "").strip().lower()
        models = row.get("models") or []

        by_lab: dict[str, list[tuple[int, str, str]]] = {}  # only multi-lab aggregators get a shortlist
        for pos, model in enumerate(models):
            lab = model.split("/", 1)[0] if "/" in model else ""
            if not lab:  # no vendor prefix → single-namespace provider, not an aggregator
                by_lab = {}
                break
            date = ""
            if get_model_info is not None:
                info = get_model_info(slug, model) or get_model_info("openrouter", model)
                date = getattr(info, "release_date", "") if info else ""
            by_lab.setdefault(lab, []).append((pos, date, model))

        if len(by_lab) < 2:
            row["featured_models"] = []
            continue

        featured: list[str] = []
        for entries in by_lab.values():
            # Newest release_date first; earlier list position breaks ties (sole key when undated).
            ranked = sorted(entries, key=lambda e: (e[1], -e[0]), reverse=True)
            featured.extend(model for _pos, _date, model in ranked[:_FEATURED_PER_LAB])
        order = {m: i for i, m in enumerate(models)}  # keep the row's model order for stable rendering
        row["featured_models"] = sorted(featured, key=lambda m: order[m])


def _apply_custom_aliases(rows: list[dict]) -> None:
    """Attach the accepted identity set to each user-defined row: ``model.options`` reports the canonical
    ``custom:<key>`` while rows carry the bare key as ``slug``, so GUI exact-match never finds the row.

    GUI pickers compare the two to decide which row is active; exact equality never matches for custom
    providers (#87035). Exposing ``aliases`` — every current and legacy spelling from
    :func:`hermes_cli.providers.custom_provider_aliases` — lets the frontend do a membership check instead.
    """
    from hermes_cli.providers import custom_provider_aliases

    for row in rows:
        if not row.get("is_user_defined"):
            continue
        try:
            row["aliases"] = sorted(
                custom_provider_aliases(str(row.get("name", "")), str(row.get("slug", ""))))
        except Exception:
            continue


# ─── Internal: row post-processing ──────────────────────────────────────


def _provider_auth_hint(slug: str) -> tuple[str, str]:
    """``(auth_type, key_env)`` for a canonical provider (``("api_key", "")`` when unregistered)."""
    from hermes_cli.auth import PROVIDER_REGISTRY

    cfg = PROVIDER_REGISTRY.get(slug)
    auth_type = cfg.auth_type if cfg else "api_key"
    key_env = cfg.api_key_env_vars[0] if (cfg and cfg.api_key_env_vars) else ""
    return auth_type, key_env


def _row(slug: str, name: str, is_current: bool, **extra: Any) -> dict:
    return {"slug": slug, "name": name, "is_current": is_current, "is_user_defined": False, **extra}


def _canonical_row(entry, cur: str, **extra: Any) -> dict:
    from hermes_cli.models import _PROVIDER_LABELS

    return _row(entry.slug, _PROVIDER_LABELS.get(entry.slug, entry.label), entry.slug.lower() == cur, **extra)


def _append_unconfigured_rows(
    rows: list[dict], ctx: ConfigContext, *, current_only: bool = False,
) -> list[dict]:
    """Empty setup skeletons for canonical providers missing from ``rows`` — except the *current* one:
    if config.yaml still points at it but credentials are gone, keep a row carrying the saved model so
    GUI pickers don't silently snap to another provider."""
    from hermes_cli.models import CANONICAL_PROVIDERS, _model_requires_account_discovery

    seen = {r["slug"].lower() for r in rows}
    cur = (ctx.current_provider or "").lower()
    cur_model = str(ctx.current_model or "").strip()
    extras: list[dict] = []
    for entry in CANONICAL_PROVIDERS:
        if entry.slug.lower() in seen:
            continue
        if current_only and entry.slug.lower() != cur:
            continue
        if entry.slug.lower() == cur:
            saved_model = "" if _model_requires_account_discovery(entry.slug, cur_model) else cur_model
            auth_type, key_env = _provider_auth_hint(entry.slug)
            tail = (
                "Astra requires successful account-scoped model discovery."
                if cur_model and not saved_model else "Showing the saved model only."
            )
            warning = (
                f"Configured provider missing usable credentials; paste {key_env} to reactivate. {tail}"
                if auth_type == "api_key" and key_env
                else f"Configured provider is not authenticated; run `hermes model` to reactivate. {tail}"
            )
            extras.append(_canonical_row(
                entry, cur, models=[saved_model] if saved_model else [], total_models=1 if saved_model else 0,
                source="configured-current", authenticated=False, auth_type=auth_type, key_env=key_env,
                warning=warning,
            ))
            continue
        extras.append(_canonical_row(entry, cur, models=[], total_models=0, source="canonical"))
    return extras


def _anthropic_oauth_credentials_present() -> bool:
    """True when the user explicitly authenticated Anthropic via OAuth.

    Two deliberate flows leave no trace in active_provider /
    model.provider / API-key env vars: Hermes' own Anthropic device flow
    (token in auth.json) and a Claude Code login (~/.claude/.credentials.json).
    ``list_authenticated_providers`` already accepts both readers as real
    credentials when discovering rows; this mirrors that acceptance so the
    desktop explicit-only filter does not silently drop a provider the user
    deliberately signed into. Unlike ambient CLI tokens (gh -> copilot),
    an OAuth access token only exists after an interactive login.
    """
    try:
        from agent.anthropic_adapter import (
            read_claude_code_credentials,
            read_hermes_oauth_credentials,
        )

        hermes_creds = read_hermes_oauth_credentials() or {}
        if hermes_creds.get("accessToken"):
            return True
        cc_creds = read_claude_code_credentials() or {}
        if cc_creds.get("accessToken"):
            return True
    except Exception:
        return False
    # Pool-only OAuth entries (auth.json credential_pool.anthropic) are the
    # canonical location for wired tokens and equally deliberate — the
    # discovery side accepts them via pool.has_credentials(), so the filter
    # must too or those rows are built and then silently dropped. Read-only
    # dict access (no load_pool) so a picker open never mutates auth.json.
    try:
        from agent.credential_pool import AUTH_TYPE_OAUTH
        from hermes_cli.auth import read_credential_pool

        for entry in read_credential_pool("anthropic"):
            if (
                isinstance(entry, dict)
                and entry.get("auth_type") == AUTH_TYPE_OAUTH
                and str(entry.get("access_token") or "").strip()
            ):
                return True
    except Exception:
        pass
    return False


def _filter_explicit_provider_rows(rows: list[dict], ctx: ConfigContext) -> list[dict]:
    """Keep only rows backed by explicit user configuration.

        readers = (read_hermes_oauth_credentials, read_claude_code_credentials)
        if any((read() or {}).get("accessToken") for read in readers):
            return True
    except Exception:
        return False
    # Pool-only OAuth entries (auth.json credential_pool.anthropic) are equally deliberate — discovery
    # accepts them via pool.has_credentials(), so the filter must too or those rows are built then
    # silently dropped. Read-only (no load_pool) so a picker open never mutates auth.json.
    try:
        from agent.credential_pool import AUTH_TYPE_OAUTH
        from hermes_cli.auth import read_credential_pool

        for entry in read_credential_pool("anthropic"):
            if (isinstance(entry, dict) and entry.get("auth_type") == AUTH_TYPE_OAUTH
                    and str(entry.get("access_token") or "").strip()):
                return True
    except Exception:
        pass
    return False


def _filter_explicit_provider_rows(rows: list[dict], ctx: ConfigContext) -> list[dict]:
    """Keep only rows backed by explicit user configuration — ``list_authenticated_providers`` also
    discovers ambient credentials (e.g. GitHub CLI -> Copilot) Desktop chat pickers must not show."""
    from hermes_cli.auth import is_provider_explicitly_configured

    current_slug = str(ctx.current_provider or "").strip().lower()
    kept: list[dict] = []
    for row in rows:
        slug = str(row.get("slug", "")).strip().lower()
        if not slug:
            continue
        if row.get("is_user_defined"):
            kept.append(row)
            continue
        if current_slug and slug == current_slug:
            kept.append(row)
            continue
        if row.get("source") == "local-runtime":
            # Managed local models are explicit configuration by existence:
            # the user downloaded gigabytes into the machine-scoped models
            # dir. There is deliberately no config credential to find
            # (credential is reachability), so without this clause the row
            # only survives on the profile where Use was last clicked —
            # every other profile loses local models from its picker.
            kept.append(row)
            continue
        if slug == "moa":
            # MoA is a virtual routing mode, not an independently configured
            # provider. Hide it from explicit-only pickers unless it is the
            # current provider (handled above) or the user explicitly wrote an
            # enabled MoA preset into config.yaml. Use raw config so the
            # DEFAULT_CONFIG preset does not make every desktop picker show MoA.
            if _raw_config_has_enabled_moa_preset():
                kept.append(row)
            continue
        if _provider_is_keyless(slug):
            # Keyless providers (opencode-free) require no configuration at
            # all — there is nothing to "explicitly configure", and hiding
            # them would defeat their purpose (zero-setup discoverability).
            kept.append(row)
            continue
        if slug == "anthropic" and _anthropic_oauth_credentials_present():
            # Anthropic OAuth logins (Hermes device flow / Claude Code) are
            # deliberate sign-ins that leave no trace in active_provider,
            # model.provider, or API-key env vars. The strict gate below
            # would drop the row even though list_authenticated_providers
            # just accepted those same credentials when building it.
            kept.append(row)
            continue
        if _external_process_signed_in(slug):
            # External-process providers (copilot-acp) authenticate through
            # their own CLI (`copilot login`), which — like the Anthropic
            # OAuth case above — leaves no trace in active_provider,
            # model.provider, or env vars. Verified CLI credentials are a
            # deliberate sign-in; without this the desktop picker drops the
            # row the picker-discovery side just accepted.
            kept.append(row)
            continue
        if is_provider_explicitly_configured(slug):
            kept.append(row)
    return kept


def _external_process_signed_in(slug: str) -> bool:
    """True when an external-process provider has verified CLI credentials."""
    try:
        from hermes_cli.auth import (
            PROVIDER_REGISTRY,
            get_external_process_provider_status,
        )
        pconfig = PROVIDER_REGISTRY.get(slug)
        if not pconfig or pconfig.auth_type != "external_process":
            return False
        return bool(get_external_process_provider_status(slug).get("auth_verified"))
    except Exception:
        return False


def _provider_is_keyless(slug: str) -> bool:
    """True when the provider's Hermes overlay declares it keyless."""
    try:
        from hermes_cli.providers import HERMES_OVERLAYS
        overlay = HERMES_OVERLAYS.get(slug)
        return bool(overlay is not None and getattr(overlay, "keyless", False))
    except Exception:
        return False


def _raw_config_has_enabled_moa_preset() -> bool:
    """True when the user's RAW config enables MoA: ``load_config()`` merges the DEFAULT_CONFIG preset for
    everyone, which is not a user choice; visible once one enabled preset (or legacy flat config) is saved."""
    try:
        from hermes_cli.config import read_raw_config

        raw = read_raw_config()
    except Exception:
        return False

    moa = raw.get("moa") if isinstance(raw, dict) else None
    if not isinstance(moa, dict):
        return False

    presets = moa.get("presets")
    if isinstance(presets, dict):
        return any(
            not isinstance(preset, dict) or preset.get("enabled", True)
            for name, preset in presets.items() if str(name or "").strip()
        )

    legacy_keys = {"reference_models", "aggregator", "reference_temperature", "aggregator_temperature",
                   "max_tokens", "reference_max_tokens", "fanout"}
    return any(key in moa for key in legacy_keys) and bool(moa.get("enabled", True))


def _apply_picker_hints(rows: list[dict]) -> None:
    """Add ``authenticated``/``auth_type``/``key_env``/``warning`` per row."""
    for row in rows:
        if "authenticated" in row:
            continue
        # Skeleton rows (_append_unconfigured_rows) have empty `models` AND source="canonical".
        is_skeleton = row.get("source") == "canonical" and not row.get("models")
        row["authenticated"] = not is_skeleton
        if not is_skeleton or row.get("is_user_defined"):
            continue
        auth_type, key_env = _provider_auth_hint(row["slug"])
        row["auth_type"] = auth_type
        row["key_env"] = key_env
        row["warning"] = (f"paste {key_env} to activate" if auth_type == "api_key" and key_env
                          else f"run `hermes model` to configure ({auth_type})")


def _reorder_canonical(rows: list[dict]) -> list[dict]:
    """Canonical slugs in ``CANONICAL_PROVIDERS`` order, truly-custom rows last. Keys on slug membership,
    NOT ``is_user_defined`` — ``providers:`` config rows carry that flag even for canonical slugs."""
    from hermes_cli.models import CANONICAL_PROVIDERS

    order = {e.slug: i for i, e in enumerate(CANONICAL_PROVIDERS)}
    canon = sorted((r for r in rows if r["slug"] in order), key=lambda r: order[r["slug"]])
    extras = [r for r in rows if r["slug"] not in order]
    return canon + extras


def _apply_pricing(
    rows: list[dict],
    *,
    force_fresh_nous_tier: bool = False,
    cached_only: bool = False,
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
        compute_sale_discount,
        get_cached_nous_free_tier,
        get_pricing_for_provider,
    )
    from hermes_cli.models import (
        check_nous_free_tier,
        get_cached_nous_free_tier,
        partition_nous_models_by_tier,
    )

    nous_free_tier: Optional[bool] = None  # resolved once (cached in models.py for the TTL window)

    for row in rows:
        slug = str(row.get("slug", "")).lower()
        models = row.get("models") or []
        if not models:
            continue
        if row.get("free_tier_row"):
            # The free tier's one model has no Portal pricing and no entitlement to read: pricing
            # it would lock the only row a free-tier install can select.
            continue
        try:
            pricing_kwargs = {"cached_only": True} if cached_only else {}
            raw_pricing = get_pricing_for_provider(slug, **pricing_kwargs) or {}
        except Exception:
            raw_pricing = {}
        cached_nous_tier: Optional[bool] = None
        if slug == "nous" and cached_only:
            cached_nous_tier = get_cached_nous_free_tier()
            if cached_nous_tier is None:
                # Entitlement is not yet known. Keep the response nonblocking,
                # but fail closed until this profile's prewarm has populated
                # both caches; otherwise a free account can briefly select
                # paid models on its first picker open.
                row["free_tier_pending"] = True
                row["unavailable_models"] = list(models)
                # Every model renders locked until the prewarm lands; say why
                # on the existing per-provider warning surface instead of
                # leaving the user staring at a greyed-out list.
                if not row.get("warning"):
                    row["warning"] = (
                        "Checking Nous plan entitlement… models unlock on the "
                        "next picker open or refresh."
                    )
                continue
        if not raw_pricing:
            if slug == "nous":
                row["free_tier"] = bool(cached_nous_tier)
                row["pricing_pending"] = True
                row["unavailable_models"] = (
                    list(models) if cached_nous_tier else []
                )
            continue

        formatted: dict[str, dict] = {}
        for mid in models:
            p = raw_pricing.get(mid)
            if not p:
                continue
            inp_raw, out_raw = p.get("prompt", ""), p.get("completion", "")
            cache_raw = p.get("input_cache_read", "")
            inp = _format_price_per_mtok(inp_raw) if inp_raw != "" else ""
            out = _format_price_per_mtok(out_raw) if out_raw != "" else ""
            entry: dict = {
                "input": inp, "output": out,
                "cache": _format_price_per_mtok(cache_raw) if cache_raw else None,
                "free": inp == "free" and out in ("free", ""),  # both input and output cost nothing
            }
            # Sale chrome is Nous Portal-only. Other providers (OpenRouter,
            # Novita, …) never get discount_percent / was_* even if a nested
            # pricing.original somehow appeared in their catalog. Free / $0
            # models get flat -100% chrome (was_* only when the gateway
            # served an original).
            if slug == "nous":
                sale = compute_sale_discount(
                    inp_raw, out_raw, p.get("original")
                )
                if sale is not None:
                    discount_percent, was_prompt_raw, was_out_raw = sale
                    entry["discount_percent"] = discount_percent
                    for key, was_raw in (("was_input", was_prompt_raw), ("was_output", was_out_raw)):
                        if was_raw != "":
                            entry[key] = _format_price_per_mtok(was_raw)
            formatted[mid] = entry

        if formatted:
            row["pricing"] = formatted

        if slug == "nous":
            try:
                if nous_free_tier is None:
                    if cached_only:
                        nous_free_tier = cached_nous_tier
                    else:
                        nous_free_tier = check_nous_free_tier(
                            force_fresh=force_fresh_nous_tier
                        )
                row["free_tier"] = bool(nous_free_tier)
                row["unavailable_models"] = (
                    partition_nous_models_by_tier(list(models), raw_pricing, free_tier=True)[1]
                    if nous_free_tier else [])
            except Exception:  # tier detection failed — fail open (no gating)
                row["free_tier"] = False
                row["unavailable_models"] = []


def _local_runtime_row(ctx: "ConfigContext") -> dict | None:
    """Build the ``llamacpp`` provider row from staged local models.

    Present whenever GGUFs are staged in the managed models directory —
    downloaded models must be selectable even before the server is running
    (selection starts it via the runtime_provider seam / activate flow).
    Returns ``None`` when nothing is staged.
    """
    try:
        from hermes_cli.local_runtime.bootstrap import staged_model_ids

        staged = staged_model_ids()
        if not staged:
            return None
        current = (ctx.current_provider or "").strip().lower() in (
            "llamacpp", "llama.cpp", "llama-cpp")
        if not current:
            # A LIVE session on the managed server reports provider "custom"
            # (the resolution seam's label) with the managed base_url. Match
            # on the endpoint so the picker still marks this row current —
            # otherwise the session the user is chatting in shows no
            # selection.
            try:
                from hermes_cli.local_runtime.endpoint import _state_endpoint

                managed = _state_endpoint()
                current = bool(
                    managed
                    and (ctx.current_base_url or "").strip().rstrip("/")
                    == managed["base_url"].rstrip("/"))
            except Exception:
                current = False
        return {
            "slug": "llamacpp",
            # Bare "Local" everywhere user-facing: the engine name is an
            # implementation detail (the pane brands this "Local models").
            "name": "Local",
            "is_current": current,
            "is_user_defined": False,
            "models": staged,
            "total_models": len(staged),
            "source": "local-runtime",
            "authenticated": True,       # the credential is reachability
            "auth_type": "local",
            "warning": None,
        }
    except Exception:
        return None


def _prewarm_pricing_async(
    rows: list[dict],
    *,
    current_provider: str = "",
    current_base_url: str = "",
) -> Optional[Thread]:
    """Warm picker pricing caches without delaying the current payload."""
    from hermes_constants import hermes_home_key
    from hermes_cli.models import pricing_cache_scope

    profile_key = hermes_home_key()
    endpoint_scope = tuple(
        sorted(
            (
                slug,
                pricing_cache_scope(
                    slug,
                    current_provider=current_provider,
                    current_base_url=current_base_url,
                ),
            )
            for slug in {
                str(row.get("slug") or "").lower()
                for row in rows
                if row.get("slug")
            }
        )
    )
    prewarm_key = (profile_key, endpoint_scope)

    with _pricing_prewarm_lock:
        current = _pricing_prewarm_threads.get(prewarm_key)
        if current is not None and current.is_alive():
            return current

        # The worker mutates only private copies while the pricing helpers
        # populate their shared process caches.
        worker_rows = [
            {**row, "models": list(row.get("models") or [])}
            for row in rows
        ]

        def _worker() -> None:
            try:
                _apply_pricing(worker_rows)
            finally:
                with _pricing_prewarm_lock:
                    if _pricing_prewarm_threads.get(prewarm_key) is current_thread():
                        _pricing_prewarm_threads.pop(prewarm_key, None)

        worker_context = copy_context()
        thread = Thread(
            target=worker_context.run,
            args=(_worker,),
            name="hermes-picker-pricing-prewarm",
            daemon=True,
        )
        _pricing_prewarm_threads[prewarm_key] = thread
        thread.start()
        return thread


def _moa_provider_row(current_provider: str = "") -> dict | None:
    """Build the virtual ``moa`` provider row for model pickers.

        staged = staged_model_ids()
        if not staged:
            return None
        current = (ctx.current_provider or "").strip().lower() in ("llamacpp", "llama.cpp", "llama-cpp")
        if not current:
            # A LIVE session on the managed server reports provider "custom" with the managed base_url;
            # match on the endpoint so the session being chatted in still shows a selection.
            try:
                from hermes_cli.local_runtime.endpoint import _state_endpoint

                managed = _state_endpoint()
                current = bool(managed and (ctx.current_base_url or "").strip().rstrip("/")
                               == managed["base_url"].rstrip("/"))
            except Exception:
                current = False
        # Bare "Local" user-facing (engine name is an implementation detail); authenticated = reachability.
        return _row("llamacpp", "Local", current, models=staged, total_models=len(staged),
                    source="local-runtime", authenticated=True, auth_type="local", warning=None)
    except Exception:
        return None


def _prewarm_pricing_async(
    rows: list[dict], *, current_provider: str = "", current_base_url: str = "",
) -> Optional[Thread]:
    """Warm picker pricing caches without delaying the current payload (one worker per
    profile + endpoint scope; a live worker is reused)."""
    from hermes_constants import hermes_home_key
    from hermes_cli.models_pricing import pricing_cache_scope

    slugs = {str(row.get("slug") or "").lower() for row in rows if row.get("slug")}
    endpoint_scope = tuple(sorted(
        (slug, pricing_cache_scope(slug, current_provider=current_provider, current_base_url=current_base_url))
        for slug in slugs))
    prewarm_key = (hermes_home_key(), endpoint_scope)

    with _pricing_prewarm_lock:
        current = _pricing_prewarm_threads.get(prewarm_key)
        if current is not None and current.is_alive():
            return current
        # The worker mutates only private copies; the pricing helpers populate shared process caches.
        worker_rows = [{**row, "models": list(row.get("models") or [])} for row in rows]

        def _worker() -> None:
            try:
                _apply_pricing(worker_rows)
            finally:
                with _pricing_prewarm_lock:
                    if _pricing_prewarm_threads.get(prewarm_key) is current_thread():
                        _pricing_prewarm_threads.pop(prewarm_key, None)

        thread = Thread(target=copy_context().run, args=(_worker,),
                        name="hermes-picker-pricing-prewarm", daemon=True)
        _pricing_prewarm_threads[prewarm_key] = thread
        thread.start()
        return thread


def _moa_provider_row(current_provider: str = "") -> dict | None:
    """The virtual ``moa`` row shared by the CLI inventory and gateway picker; ``None`` without presets."""
    try:
        from hermes_cli.config import load_config
        from hermes_cli.moa_config import normalize_moa_config

        cfg = normalize_moa_config(load_config().get("moa") or {})
        models = list(cfg.get("presets", {}).keys())
        if not models:
            return None
        return _row(
            "moa", "Mixture of Agents", (current_provider or "").lower() == "moa", models=models,
            total_models=len(models), source="virtual", authenticated=True, auth_type="virtual",
            warning="Aggregator acts as the selected model; references provide analysis before each call.")
    except Exception:
        return None
