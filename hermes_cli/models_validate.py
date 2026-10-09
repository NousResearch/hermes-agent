"""Validate a requested ``/model`` value against the active provider's catalog.

Split out of ``hermes_cli.models``. Catalog fetchers defined in ``hermes_cli.models`` are looked up
there at call time (``_m.<name>``) so ``patch("hermes_cli.models.<name>")`` mocks keep intercepting;
local-server probes are looked up on ``hermes_cli.models_local`` (``_ml.<name>``) the same way.

Every provider branch returns a verdict dict (see :func:`_verdict`) or ``None`` for "not decided
here — keep walking the ladder". The ladder ORDER is behavior (see ``_LADDER``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Any, Callable, Optional

from utils import base_url_host_matches
from hermes_constants import openrouter_variant_base


# ── Verdicts ─────────────────────────────────────────────────────────────

def _verdict(accepted: bool, persist: bool, recognized: bool, message: Optional[str]) -> dict[str, Any]:
    return {"accepted": accepted, "persist": persist, "recognized": recognized, "message": message}


def _accept() -> dict[str, Any]:
    return _verdict(True, True, True, None)


def _accept_with_note(message: str) -> dict[str, Any]:
    return _verdict(True, True, True, message)


def _reject(message: str) -> dict[str, Any]:
    return _verdict(False, False, False, message)


def _soft_accept(message: Optional[str]) -> dict[str, Any]:
    """Accept + persist an unrecognized name, with a warning."""
    return _verdict(True, True, False, message)


# ── Catalog matching ─────────────────────────────────────────────────────

@dataclass
class _Match:
    exact: bool = False
    suggestion_text: str = ""

    def verdict(self, req: "_Request") -> Optional[dict[str, Any]]:
        """Accept on exact membership, else None so the branch composes its own message."""
        return _accept() if self.exact else None


def _match_in_catalog(
    query: str,
    candidates,
    *,
    case_insensitive: bool = False,
    suggest_query: Optional[str] = None,
    suggest_cutoff: float = 0.5,
    suggest_label: str = "Similar models",
) -> _Match:
    """Shared ladder: exact membership → suggestion text. Never rewrites the id: a requested model
    that is merely CLOSE to a catalog entry is the user's selection (a newer release the listing
    lacks, a dated snapshot, a qualifier) and goes to the wire verbatim — fuzzy "auto-correction"
    swapped `deepseek-v4.1-flash` for `deepseek-v4-flash`, `gemini-3.8-flash` for `gemini-3.6-flash`
    and `model:nitro` for `model` under the user's own label. The vendor's 400 names the valid ids.
    ``case_insensitive`` matches lower-cased ids and maps results back to the catalog's spelling
    (MiniMax ships mixed-case ids). ``suggest_query`` overrides the string the suggestion search
    uses (some branches search on the raw request, not the lookup form)."""
    pool = list(candidates)
    display = None
    if suggest_query is None:
        suggest_query = query
    if case_insensitive:
        display = {c.lower(): c for c in pool}
        pool = list(display)
        query, suggest_query = query.lower(), suggest_query.lower()

    def _show(cid: str) -> str:
        return display[cid] if display is not None else cid

    if query in set(pool):
        return _Match(exact=True)
    suggestions = get_close_matches(suggest_query, pool, n=3, cutoff=suggest_cutoff)
    if not suggestions:
        return _Match()
    return _Match(suggestion_text=f"\n  {suggest_label}: " + ", ".join(f"`{_show(s)}`" for s in suggestions))


# ── Request context ──────────────────────────────────────────────────────

@dataclass
class _Request:
    requested: str
    lookup: str                 # id used for catalog membership (copilot-normalized / preset base)
    provider: Optional[str]     # raw caller value (Ollama checks look at this, not ``normalized``)
    normalized: str
    api_key: Optional[str]
    base_url: Optional[str]
    api_mode: Optional[str]
    headers: Optional[dict[str, str]]


# ── Provider branches (None = not decided here) ─────────────────────────

def _validate_moa(req: _Request) -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config
        from hermes_cli.moa_config import normalize_moa_config

        cfg = normalize_moa_config(load_config().get("moa") or {})
        if req.requested in cfg["presets"]:
            return _accept()
        return _reject(f"MoA preset `{req.requested}` was not found. Run `hermes moa list`.")
    except Exception as exc:
        return _reject(f"Could not read MoA presets: {exc}")


def _reject_whitespace(req: _Request) -> Optional[dict[str, Any]]:
    # Cloud catalogs never contain spaces. Self-hosted servers and a user-configured
    # base_url do (VLLM "My Custom Model", a local router's "Go reasoning"). The
    # step stays here — later branches must not see a cloud id the catalog cannot serve.
    if any(ch.isspace() for ch in req.requested) and not _whitespace_allowed(req):
        return _reject("Model names cannot contain spaces.")
    return None


_SELF_HOSTED_PROVIDERS = frozenset({
    "lmstudio", "ollama", "local", "vllm", "llamacpp", "llama.cpp", "llama-cpp",
})


def _provider_token(provider: Optional[str]) -> str:
    return str(provider or "").strip().lower()


def _is_self_hosted_provider(provider: Optional[str]) -> bool:
    """Local servers and the custom endpoint bucket, including aliases that normalize to it."""
    raw = _provider_token(provider)
    if not raw:
        return False
    if raw == "custom" or raw.startswith("custom:"):
        return True
    from hermes_cli import models as _m

    normalized = _m.normalize_provider(raw)
    if normalized == "custom" or normalized.startswith("custom:"):
        return True
    return raw in _SELF_HOSTED_PROVIDERS or normalized in _SELF_HOSTED_PROVIDERS


def _non_public_host(host: str) -> bool:
    """Loopback, LAN, and single-label hosts are a user's endpoint, never a vendor catalog."""
    host = (host or "").lower().rstrip(".")
    if not host or host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"} or host.endswith(".localhost"):
        return bool(host)
    if host.endswith((".local", ".lan", ".internal", ".home", ".localdomain")):
        return True
    if "." not in host:
        return True
    parts = host.split(".")
    if len(parts) == 4 and all(part.isdigit() for part in parts):
        octets = [int(part) for part in parts]
        if octets[0] in {10, 127} or (octets[0] == 192 and octets[1] == 168):
            return True
        if octets[0] == 172 and 16 <= octets[1] <= 31:
            return True
    return False


def _stock_host(provider: str) -> str:
    """Vendor host for *provider*, offline. Empty when this install cannot name one.

    ``get_provider`` can answer with a different id (``openai`` → the OpenRouter
    overlay when models.dev is cold). A foreign row is not this provider's stock
    endpoint — using it would exempt the real cloud host.
    """
    from hermes_cli.providers import get_provider
    from utils import base_url_hostname

    token = _provider_token(provider)
    if not token:
        return ""
    try:
        pdef = get_provider(token, allow_network=False)
    except Exception:
        return ""
    if pdef is None or _provider_token(pdef.id) != token:
        return ""
    return base_url_hostname(pdef.base_url or "")


def provider_allows_model_whitespace(provider: Optional[str], base_url: Optional[str] = None) -> bool:
    """True when a spaced id is a real selection, not a cloud-catalog typo.

    Self-hosted providers always qualify. A base_url qualifies when the user
    configured it: a non-public host, or a public host that is not this
    provider's own stock endpoint. A public host with no stock to compare
    against stays rejected, so a cloud URL cannot slip through a cold catalog.
    """
    if _is_self_hosted_provider(provider):
        return True
    url = str(base_url or "").strip()
    if not url:
        return False
    from utils import base_url_hostname

    host = base_url_hostname(url)
    if not host:
        return False
    if _non_public_host(host):
        return True
    raw = _provider_token(provider)
    from hermes_cli import models as _m

    normalized = _m.normalize_provider(raw) if raw else ""
    stock = _stock_host(normalized or raw)
    return bool(stock) and host != stock


def _whitespace_allowed(req: _Request) -> bool:
    # The openrouter→custom rewrite lives on ``normalized``; aliases live on the raw value.
    if _is_self_hosted_provider(req.normalized) or _is_self_hosted_provider(req.provider):
        return True
    return provider_allows_model_whitespace(req.provider or req.normalized, req.base_url)


def offered_model_ids(models, provider: Optional[str], base_url: Optional[str] = None) -> list:
    """Ids a picker may show. Drops whitespace the validator will refuse; keeps the rest."""
    ids = list(models or [])
    if provider_allows_model_whitespace(provider, base_url):
        return ids
    return [model_id for model_id in ids if not (isinstance(model_id, str) and any(ch.isspace() for ch in model_id))]


def drop_unofferable_model_ids(rows: list) -> None:
    """In-place: picker rows must not offer an id ``validate_requested_model`` will refuse for whitespace."""
    for row in rows:
        if not isinstance(row, dict):
            continue
        provider = row.get("slug")
        base_url = row.get("api_url") or row.get("base_url")
        models = row.get("models")
        if isinstance(models, list):
            filtered = offered_model_ids(models, provider, base_url)
            removed = len(models) - len(filtered)
            if removed:
                row["models"] = filtered
                total = row.get("total_models")
                if isinstance(total, int):
                    row["total_models"] = max(0, total - removed)
        featured = row.get("featured_models")
        if isinstance(featured, list):
            row["featured_models"] = offered_model_ids(featured, provider, base_url)


def _parse_openrouter_preset(req: _Request) -> Optional[dict[str, Any]]:
    """OpenRouter presets are account-scoped, so ``@preset/<slug>`` never appears in the public
    /v1/models listing. A bare preset is accepted unverified; ``<model>@preset/<slug>`` validates
    the base model; the full id (suffix included) goes to the wire. OpenRouter validates the slug
    at request time."""
    marker = "@preset/"
    if marker not in req.requested:
        return None
    if req.requested.count(marker) != 1:
        preset_slug, preset_base = "", req.requested
    else:
        preset_base, preset_slug = req.requested.split(marker, 1)
    if re.fullmatch(r"[A-Za-z0-9._~-]+", preset_slug) is None:
        return _reject("OpenRouter preset slugs must be non-empty URL-safe identifiers using only "
                       "letters, digits, '.', '_', '~', or '-'.")
    if not preset_base:
        return _soft_accept(None)
    req.lookup = preset_base
    return None


def _validate_lmstudio(req: _Request) -> dict[str, Any]:
    from hermes_cli import models_local as _ml
    from hermes_cli.auth import AuthError

    # probe_lmstudio_models distinguishes None (unreachable / malformed) from [] (reachable,
    # nothing chat-capable loaded); fetch_lmstudio_models collapses both to [].
    try:
        models = _ml.probe_lmstudio_models(api_key=req.api_key, base_url=req.base_url)
    except AuthError as exc:
        return _reject(f"{exc} Set `LM_API_KEY` (or update it) to match the server's bearer token.")
    if models is None:
        return _reject(f"Could not reach LM Studio's `/api/v1/models` to validate `{req.requested}`.")
    if not models:
        return _reject("LM Studio is reachable but no chat-capable models are loaded. "
                       f"Load `{req.requested}` in LM Studio (Developer tab → Load Model) and try again.")
    if req.lookup in set(models):
        return _accept()
    return _reject(f"Model `{req.requested}` was not found in LM Studio's model listing.")


def _ollama_probe_headers(req: _Request) -> dict[str, str]:
    """Headers for the Ollama native probe. Configured ``providers.ollama.extra_headers`` apply only
    when the probed endpoint is the configured one (never leak them to a different host). Caller
    headers win; a caller ``api_key`` becomes the Authorization header unless the caller sent one."""
    from hermes_cli import models as _m
    from hermes_cli import models_local as _ml
    from hermes_cli.models_local import _configured_ollama_base_url, _drop_authorization

    configured_base = _configured_ollama_base_url()
    configured_allowed = not configured_base or _ml._same_ollama_native_root(req.base_url or "", configured_base)
    configured = _m._get_ollama_native_headers(req.base_url, api_key=req.api_key) if configured_allowed else {}
    if req.headers is None:
        return configured
    out = dict(configured)
    _drop_authorization(out)
    out.update(req.headers)
    if req.api_key and not any(key.lower() == "authorization" for key in req.headers):
        _drop_authorization(out)
        out["Authorization"] = f"Bearer {req.api_key}"
    return out


def _validate_ollama_native(req: _Request) -> Optional[dict[str, Any]]:
    """Runs for EVERY provider: the native ``/api/tags`` catalog is used whenever the endpoint
    looks like a local Ollama server. Also resolves ``base_url`` for the raw ``ollama`` provider,
    which later branches (custom) rely on."""
    from hermes_cli import models as _m
    from hermes_cli import models_local as _ml

    if str(req.provider or "").strip().lower() == "ollama" and not req.base_url:
        req.base_url = _m._get_ollama_base_url()
    headers = _ollama_probe_headers(req)
    if not _ml.should_use_ollama_native_catalog(req.provider, req.base_url, headers=headers):
        return None
    models = _ml.probe_ollama_local_models(req.base_url, headers=headers)
    if models is None:
        # A failed native probe is not authoritative; fall back to the OpenAI-compatible catalog.
        models = _m.probe_api_models(
            req.api_key, _ml._normalize_openai_base_url(req.base_url), request_headers=headers,
        ).get("models")
    if models is None:
        return _soft_accept(
            f"Note: could not reach this Ollama endpoint's `/api/tags` model listing to validate `{req.requested}`. "
            "Hermes will save the model name, but local Ollama model discovery could not verify it."
        )
    match = _match_in_catalog(req.lookup, models, suggest_label="Similar local Ollama models")
    if match.exact:
        return _accept()
    empty_hint = " No models are currently listed by `/api/tags`." if not models else ""
    return _soft_accept(
        f"Note: `{req.requested}` was not found in this Ollama endpoint's `/api/tags` model listing."
        f"{empty_hint} It may still work if the server supports hidden or aliased models."
        f"{match.suggestion_text}"
    )


def _validate_custom(req: _Request) -> dict[str, Any]:
    from hermes_cli import models as _m

    # Probe with the auth shape the api_mode expects.
    anthropic_style = req.api_mode == "anthropic_messages"
    probe_kwargs = {"api_mode": req.api_mode} if anthropic_style else {}
    probe = _m.probe_api_models(req.api_key, req.base_url, request_headers=req.headers, **probe_kwargs)
    api_models = probe.get("models")
    if api_models is not None:
        match = _match_in_catalog(req.lookup, api_models, suggest_query=req.requested)
        verdict = match.verdict(req)
        if verdict is not None:
            return verdict
        message = (
            f"Note: `{req.requested}` was not found in this custom endpoint's model listing "
            f"({probe.get('probed_url')}). It may still work if the server supports hidden or aliased models."
            f"{match.suggestion_text}"
        )
        if probe.get("used_fallback"):
            message += (f"\n  Endpoint verification succeeded after trying `{probe.get('resolved_base_url')}`. "
                        "Consider saving that as your base URL.")
        return _soft_accept(message)

    # Many OpenAI-compatible and Anthropic-compatible proxies (DashScope coding plan, Cline,
    # MiniMax) never implement GET /models; /chat/completions works fine. Rejecting the switch
    # here bricked `/model` for them (#12220), so both chat modes persist the name unverified.
    accepted = req.api_mode in ("chat_completions", "anthropic_messages")
    message = f"Note: could not reach this custom endpoint's model listing at `{probe.get('probed_url')}`. "
    if accepted:
        message += (f"`{req.requested}` was accepted without verification — if this endpoint does not "
                    "serve it, inference will fail; check the provider's model catalog or the model name.")
    else:
        message += f"`{req.requested}` was not saved; the endpoint should expose `/models` for verification."
    if probe.get("suggested_base_url"):
        message += f"\n  If this server expects `/v1`, try base URL: `{probe.get('suggested_base_url')}`"
    return _verdict(accepted, True, False, message)


def _static_catalog(normalized: str) -> list[str]:
    from hermes_cli import models as _m

    try:
        return _m.provider_model_ids(normalized)
    except Exception:
        return []


_STATIC_FAMILY_PREFIXES = {
    # Plausibility gate (#45006): the soft-accept (#16172 / #19729) exists for entitlement-gated *hidden*
    # slugs the curated listing hasn't caught up with — but those are always the provider's own family
    # (openai-codex -> gpt-*; xai-oauth -> grok-*). Accepting an unrelated typed name (e.g. `qwen3.5-4b`,
    # `llama-3.1-8b`) here turns what should be an actionable "did you mean --provider <x>?" error into a
    # confusing success that 400s on the next turn. Only soft- accept names that share the provider's family
    # prefix; reject the rest with guidance to pin the right provider.
    "openai-codex": ("gpt-", "codex-", "o1", "o3", "o4"),
    "xai-oauth": ("grok-",),
}
_STATIC_LABELS = {"openai-codex": "OpenAI Codex", "xai-oauth": "xAI Grok OAuth (SuperGrok / Premium+)"}


def _family_head(model_id: str) -> str:
    """Vendor family token of a model id: ``gpt-5.5`` → ``gpt``, ``claude-opus-5`` → ``claude``."""
    return re.split(r"[-./:]", model_id.strip().lower(), maxsplit=1)[0]


def static_model_provider_conflict(model_name: str, provider: Optional[str], *, limit: int = 5) -> Optional[dict[str, Any]]:
    """Offline model×provider coherence from the curated catalogs only (no network: this runs on
    ``session.create``). ``None`` = coherent or undecidable — custom / aggregator / catalog-less
    providers, names in the provider's own family (a newer ``gpt-*`` the curated list lacks) and
    names no vendor lists (hidden or preview slugs) stay permissive. A conflict is a name outside
    the provider's family that another native vendor's catalog lists — or any foreign-family name
    on the OAuth catalogs with a strict family gate (``_STATIC_FAMILY_PREFIXES``) (#96817)."""
    from hermes_cli import models as _m

    requested = (model_name or "").strip()
    normalized = _m.normalize_provider(provider)
    catalog = list(_m._PROVIDER_MODELS.get(normalized, ()))
    if not requested or not catalog or normalized == "moa" or normalized in _m._AGGREGATOR_PROVIDERS:
        return None
    if _m._model_in_provider_catalog(requested.lower(), _m._provider_keys(normalized)):
        return None
    if _family_head(requested) in {_family_head(m) for m in catalog}:
        return None
    strict = normalized in _STATIC_FAMILY_PREFIXES
    if not strict and next(_m._static_catalog_matches(requested, normalized), None) is None:
        return None
    suggestions = get_close_matches(requested, catalog, n=limit, cutoff=0.4) or catalog[:limit]
    label = _m._PROVIDER_LABELS.get(normalized, normalized)
    return {
        "model": requested, "provider": normalized, "suggestions": suggestions,
        "message": (f"Model `{requested}` is not served by provider `{normalized}` ({label}). "
                    f"Closest {label} models: " + ", ".join(f"`{s}`" for s in suggestions) + "."),
    }


def _validate_static_catalog(req: _Request) -> Optional[dict[str, Any]]:
    """openai-codex / xai-oauth: no /v1/models probing — validate against the curated catalog.
    Returns None (fall through) when the catalog is empty."""
    catalog = _static_catalog(req.normalized)
    if req.normalized == "openai-codex":
        from agent.model_metadata import CODEX_CONTEXT_VARIANT_SUFFIX, is_codex_context_variant

        # Ineligible ``-900k`` aliases must be rejected BEFORE the hidden-slug soft-accept:
        # the suffix is a Hermes picker convention, so an unknown `*-900k` can never be a real
        # hidden provider slug — soft-accepting one silently runs at 272K on a different model.
        if req.lookup.strip().lower().endswith(CODEX_CONTEXT_VARIANT_SUFFIX) and req.lookup not in set(catalog):
            if is_codex_context_variant(req.lookup):
                # Valid variant a stale catalog hasn't synthesized yet.
                return _accept()
            base_guess = req.lookup[: -len(CODEX_CONTEXT_VARIANT_SUFFIX)]
            return _reject(
                f"`{req.requested}` is not a valid large-context variant — `{base_guess}` enforces the "
                "standard 272K window on Codex, so no `-900k` option exists for it. Pick the base model, "
                "or a verified variant from the `/model` picker (e.g. `gpt-5.6-sol-900k`)."
            )
    if not catalog:
        return None
    match = _match_in_catalog(req.lookup, catalog)
    verdict = match.verdict(req)
    if verdict is not None:
        return verdict
    label = _STATIC_LABELS[req.normalized]
    # Plausibility gate: the soft-accept exists for entitlement-gated *hidden* slugs the curated
    # listing hasn't caught up with — always the provider's own family (gpt-* / grok-*). An
    # unrelated name (`qwen3.5-4b`) would turn an actionable "did you mean --provider <x>?" into
    # a confusing success that 400s on the next turn, so reject it with guidance instead.
    prefixes = _STATIC_FAMILY_PREFIXES.get(req.normalized, ())
    lower = req.lookup.strip().lower()
    if prefixes and not any(lower.startswith(p) for p in prefixes):
        return _reject(
            f"`{req.requested}` doesn't look like a {label} model and isn't in its listing, so it was not "
            "accepted. If it belongs to another configured provider, switch with `--provider <slug>` "
            f"(or select it from the `/model` picker).{match.suggestion_text}"
        )
    return _soft_accept(
        f"Note: `{req.requested}` was not found in the {label} model listing. "
        "It may still work if your account has access to a newer or hidden model ID."
        f"{match.suggestion_text}"
    )


def _validate_minimax(req: _Request) -> Optional[dict[str, Any]]:
    """MiniMax has no /models endpoint — static catalog, case-insensitive (ids like MiniMax-M2.7).
    Returns None when the catalog is empty."""
    catalog = _static_catalog(req.normalized)
    if not catalog:
        return None
    match = _match_in_catalog(req.lookup, catalog, case_insensitive=True)
    return match.verdict(req) or _soft_accept(
        f"Note: `{req.requested}` was not found in the MiniMax catalog."
        f"{match.suggestion_text}"
        "\n  MiniMax does not expose a /models endpoint, so Hermes cannot verify the model name."
        "\n  The model may still work if it exists on the server."
    )


def _validate_anthropic(req: _Request) -> Optional[dict[str, Any]]:
    """Native Anthropic: /v1/models needs x-api-key (or OAuth Bearer) + anthropic-version, so the
    generic Bearer probe 401s — use the native fetcher. None (fall through) when no token is
    resolvable or the network failed."""
    from hermes_cli import models as _m

    models = _m._fetch_anthropic_models(base_url=req.base_url or None, api_key=req.api_key or None)
    if models is None:
        return None
    match = _match_in_catalog(req.lookup, models, suggest_query=req.requested)
    # Accept anyway — Anthropic gates newer/preview models (snapshot IDs, early access) behind
    # accounts even though they aren't listed on /v1/models.
    return match.verdict(req) or _soft_accept(
        f"Note: `{req.requested}` was not found in Anthropic's /v1/models listing. "
        f"It may still work if you have early-access or snapshot IDs."
        f"{match.suggestion_text}"
    )


def _validate_anthropic_messages(req: _Request) -> dict[str, Any]:
    """Anthropic Messages transport: probe /v1/models and soft-accept either way, but say which
    happened — a proxy that never implemented the listing is a different situation from a reachable
    listing that simply doesn't name the slug (vendors alias ids: ``kimi-k3`` is served as ``k3``)."""
    from hermes_cli import models as _m

    models = _m.fetch_api_models(req.api_key, req.base_url, api_mode=req.api_mode)
    if models is None:
        return _soft_accept(
            f"Note: could not verify `{req.requested}` against this endpoint's model listing.  Many "
            "Anthropic-compatible proxies do not implement GET /v1/models.  The model name has been accepted "
            "without verification."
        )
    # Vendor alias pairs sit below the default 0.5 similarity cutoff (kimi-k3 vs k3 ≈ 0.44).
    match = _match_in_catalog(req.lookup, models, case_insensitive=True, suggest_query=req.requested,
                              suggest_cutoff=0.4)
    return match.verdict(req) or _soft_accept(
        f"Note: `{req.requested}` is not named in this endpoint's model listing (it may still serve it "
        f"under an alias).{match.suggestion_text}"
        "\n  The model name has been accepted without verification."
    )


def _nous_portal_recommended_names() -> set[str]:
    """Lower-cased ids from the Portal's live recommended-models feed (empty on any failure)."""
    from hermes_cli import models as _m

    try:
        payload = _m.fetch_nous_recommended_models(_m._resolve_nous_portal_url())
        return {
            name.lower()
            for tier in ("freeRecommendedModels", "paidRecommendedModels")
            for entry in (payload.get(tier) or [])
            if (name := _m._extract_model_name(entry))
        }
    except Exception:
        return set()


def _validate_managed_local(req: _Request) -> Optional[dict[str, Any]]:
    """The managed llama.cpp runtime: the staged library on disk is the source of truth, not the
    live listing. The router's model list is spawn-only (a GGUF landed after its start is
    invisible to GET /models until a bounce), so validating a freshly downloaded model against
    the live listing rejects the very file the user just staged — the Local Models "Use" flow
    and the composer picker could never succeed for a non-catalog model. A staged id accepts
    (case-insensitive: typing matches the file name, the router registers the preset id);
    anything else falls through to the live listing, which stays authoritative for ids that
    were never downloaded here."""
    from hermes_cli.local_runtime.bootstrap import staged_model_ids

    staged = {sid.lower() for sid in staged_model_ids()}
    if req.lookup.strip().lower() in staged:
        return _accept_with_note(
            f"Note: `{req.requested}` was not found in the live /v1/models listing "
            "but is downloaded in the managed local-models library — accepted."
        )
    return None


def _profile_catalog(normalized: str, base_url: Optional[str] = None) -> tuple[list[str], bool]:
    """``(catalog, authoritative)`` for a profile whose catalog is not the generic
    ``{base_url}/models`` listing — it overrides ``fetch_models`` or points ``models_url``
    elsewhere — so that listing is not authoritative for it (a relay may 200 with a different
    product catalog, #101705). The catalog is *authoritative* (a miss is a reject, never an
    acceptance by the generic listing, #116667) when the profile serves it from an endpoint of
    its own and it is available; a bare ``fetch_models`` override may just re-shape the generic
    listing, and an unavailable/empty catalog keeps the generic listing as the validator.
    ``([], False)`` for profiles without a catalog of their own."""
    from providers import get_provider_profile
    from providers.base import ProviderProfile

    profile = get_provider_profile(normalized)
    if profile is None:
        return [], False
    # A non-default runtime base_url is a user-configured relay/proxy; its own /models listing
    # decides validation, not the provider profile's canonical catalog endpoint.
    if base_url and (base_url.rstrip("/") != (profile.base_url or "").rstrip("/")):
        return [], False
    generic = (profile.base_url or "").rstrip("/") + "/models"
    own_endpoint = bool(profile.models_url) and profile.models_url.rstrip("/") != generic
    if not own_endpoint and type(profile).fetch_models is ProviderProfile.fetch_models:
        return [], False
    catalog = _static_catalog(normalized)
    return catalog, own_endpoint and bool(catalog)


def _validate_live_listing(req: _Request) -> Optional[dict[str, Any]]:
    """Generic live /v1/models probe. Returns None when the API was unreachable (the caller then
    tries Bedrock discovery / the curated catalog). A profile that owns its catalog is validated
    against that catalog (``provider_model_ids`` — the picker's list) before the generic listing."""
    from hermes_cli import models as _m

    catalog, authoritative = _profile_catalog(req.normalized, req.base_url)
    if catalog:
        match = _match_in_catalog(req.lookup, catalog, suggest_query=req.requested)
        if match.exact:
            return _accept()
        if authoritative:
            # The catalog endpoint the profile declares decides: a miss there is a reject, never
            # an acceptance by the generic listing, which for such relays lists a different
            # product line.
            return _reject(
                f"Model `{req.requested}` was not found in this provider's catalog.{match.suggestion_text}")
    api_models = _m.fetch_api_models(req.api_key, req.base_url)
    if api_models is None:
        return None
    if req.normalized == "gemini":
        # Gemini's OpenAI-compat listing prefixes ids with "models/"; curated list and user input
        # use the bare id, so strip before comparing.
        api_models = [m[len("models/"):] if isinstance(m, str) and m.startswith("models/") else m for m in api_models]
    match = _match_in_catalog(req.lookup, api_models)
    if match.exact:
        return _accept()
    # OpenRouter routing variants (":nitro", ":floor", ...) are request-time modifiers, not
    # catalog entries — validate the BASE but keep the suffixed id.
    variant_base = openrouter_variant_base(req.lookup) if req.normalized == "openrouter" else None
    if variant_base is not None and variant_base in set(api_models):
        return _accept()
    # Listed but not found: the account may reach models absent from the public listing
    # (e.g. Z.AI Pro/Max plans use glm-5 on coding endpoints) — warn but allow where plausible.
    # Curated-catalog soft-accept: providers omit valid models from live listings (stale cache,
    # partial rollout, gated previews). EXCEPTION: official OpenAI hosts (canonical + data-
    # residency regional) — their listing is access-scoped and authoritative, so an absent model
    # is one this key CANNOT serve; a soft-accept would 400 at first use. Custom OpenAI-compatible
    # proxies keep the fallback.
    listing_authoritative = False
    if req.normalized in ("openai", "openai-api"):
        from hermes_cli.providers import is_official_openai_host

        listing_authoritative = is_official_openai_host(req.base_url)
    if not listing_authoritative and _m._model_in_provider_catalog(
        (variant_base or req.lookup).lower(), _m._provider_keys(req.normalized)
    ):
        return _accept_with_note(f"Note: `{req.requested}` was not found in the live /v1/models listing "
                                 "but exists in the curated catalog — accepted.")
    # Nous: the Portal's recommended-models feed can list a model before the curated list or the
    # docs-hosted manifest catches up; `hermes chat` already accepts those at model-list build
    # time, so mirror that source of truth for per-message /model validation.
    if req.normalized == "nous" and req.lookup.lower() in _nous_portal_recommended_names():
        return _accept_with_note(f"Note: `{req.requested}` was not found in the live /v1/models listing "
                                 "but is a current Nous Portal recommendation — accepted.")
    return _reject(f"Model `{req.requested}` was not found in this provider's model listing.{match.suggestion_text}")


def _validate_bedrock(req: _Request) -> Optional[dict[str, Any]]:
    """Bedrock's runtime URL has no /models; discovery goes through the AWS control plane
    (ListFoundationModels + ListInferenceProfiles). Any failure falls through (None)."""
    try:
        from agent.bedrock_adapter import discover_bedrock_models, resolve_bedrock_runtime_region

        region = resolve_bedrock_runtime_region()
        discovered_ids = {m["id"] for m in discover_bedrock_models(region)}
        match = _match_in_catalog(req.requested, list(discovered_ids), suggest_cutoff=0.4)
        if match.exact:
            return _accept()
        # Still accept (custom inference profiles / cross-account access), but warn.
        return _soft_accept(
            f"Note: `{req.requested}` was not found in Bedrock model discovery for {region}. "
            f"It may still work with custom inference profiles or cross-account access."
            f"{match.suggestion_text}"
        )
    except Exception:
        return None


def _validate_external_process(req: _Request) -> Optional[dict[str, Any]]:
    """Process providers have no HTTP listing: the picker's list (``provider_model_ids`` — the
    CLI's live catalog merged with the declared one) plus the profile's short aliases is the whole
    truth, so a listed id is accepted outright and an unlisted one gets the catalog verdict without
    the misleading "endpoint was unreachable" note."""
    from providers import get_provider_profile

    profile = get_provider_profile(req.normalized)
    if profile is None or profile.auth_type != "external_process":
        return None
    if req.lookup.lower() in {k.lower() for k in profile.model_aliases}:
        return _accept()
    catalog = _static_catalog(req.normalized) or list(profile.fallback_models)
    match = _match_in_catalog(req.lookup, catalog, case_insensitive=True)
    if match.exact:
        return _accept()
    return match.verdict(req) or _soft_accept(
        f"Note: `{req.requested}` is not declared by {profile.display_name or profile.name}."
        f"{match.suggestion_text}\n  The model may still work if the local client accepts it.")


def _validate_catalog_fallback(req: _Request) -> dict[str, Any]:
    """/models unreachable: validate against the curated ``provider_model_ids()`` list so gateway
    /model switches keep working while a provider's endpoint is down (otherwise switch_model() would
    fail and the gateway never writes the session override). No catalog → accept with a warning."""
    from hermes_cli import models as _m

    label = _m._PROVIDER_LABELS.get(req.normalized, req.normalized)
    catalog = _static_catalog(req.normalized)
    if not catalog:
        return _soft_accept(f"Note: could not reach the {label} API to validate `{req.requested}`. "
                            "If the service isn't down, this model may not be valid.")
    match = _match_in_catalog(req.lookup, catalog, case_insensitive=True)
    if match.exact:
        return _accept()
    # Same OpenRouter routing-variant rule as the live-listing path.
    if req.normalized == "openrouter":
        variant_base = openrouter_variant_base(req.lookup)
        if variant_base is not None and variant_base.lower() in {m.lower() for m in catalog}:
            return _accept()
    return _soft_accept(
        f"Note: `{req.requested}` was not found in the {label} curated catalog "
        f"and the /models endpoint was unreachable.{match.suggestion_text}"
        f"\n  The model may still work if it exists on the provider."
    )


# ── Orchestrator ─────────────────────────────────────────────────────────

def _is_custom(req: _Request) -> bool:
    return req.normalized == "custom" or req.normalized.startswith("custom:")


def _for(*providers: str) -> Callable[[_Request], bool]:
    return lambda req: req.normalized in providers


# (gate, branch): the branch runs when the gate passes; the first non-None verdict wins. ORDER IS
# BEHAVIOR: moa → whitespace (skipped only for self-hosted providers and a user-configured
# base_url) → OpenRouter preset parse → LM Studio → Ollama native → custom →
# codex/xai static → MiniMax → managed local (staged library) → Anthropic native →
# Anthropic Messages → external process → live listing → Bedrock → curated-catalog fallback (always decides).
_LADDER: tuple[tuple[Callable[[_Request], bool], Callable[[_Request], Optional[dict[str, Any]]]], ...] = (
    (_for("moa"), _validate_moa),
    (lambda req: True, _reject_whitespace),
    (_for("openrouter"), _parse_openrouter_preset),
    (_for("lmstudio"), _validate_lmstudio),
    (lambda req: True, _validate_ollama_native),
    (_is_custom, _validate_custom),
    (_for("openai-codex", "xai-oauth"), _validate_static_catalog),
    (_for("minimax", "minimax-cn"), _validate_minimax),
    (_for("llamacpp", "llama.cpp", "llama-cpp"), _validate_managed_local),
    (_for("anthropic"), _validate_anthropic),
    (lambda req: req.api_mode == "anthropic_messages", _validate_anthropic_messages),
    (lambda req: True, _validate_external_process),
    (lambda req: True, _validate_live_listing),
    # API unreachable — accept and persist, but warn so typos don't silently break things.
    (_for("bedrock"), _validate_bedrock),
    (lambda req: True, _validate_catalog_fallback),
)


def validate_requested_model(
    model_name: str,
    provider: Optional[str],
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    api_mode: Optional[str] = None,
    headers: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Validate a ``/model`` value for the active provider → dict with ``accepted`` (switch now),
    ``persist`` (safe to save to config), ``recognized`` (matched a known provider catalog),
    ``message`` (optional warning / guidance). The requested id is never rewritten: what the user
    selected is what the wire sees."""
    from hermes_cli import models as _m

    requested = (model_name or "").strip()
    normalized = _m.normalize_provider(provider)
    if normalized == "openrouter" and base_url and not base_url_host_matches(base_url, "openrouter.ai"):
        normalized = "custom"
    lookup = requested
    if normalized == "copilot":
        lookup = _m.normalize_copilot_model_id(requested, api_key=api_key) or requested

    if not requested:
        return _reject("Model name cannot be empty.")
    req = _Request(requested, lookup, provider, normalized, api_key, base_url, api_mode, headers)
    for gate, branch in _LADDER:
        if gate(req):
            verdict = branch(req)
            if verdict is not None:
                return verdict
    raise AssertionError("unreachable: _validate_catalog_fallback always decides")
