"""Codex model discovery from API, local cache, and config."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_CODEX_CAPABILITY_CACHE_TTL = 300.0
_CODEX_LIVE_MODEL_ENTRY_CACHE: Dict[str, tuple[float, List[Dict[str, Any]]]] = {}


@lru_cache(maxsize=1)
def _codex_client_version() -> str:
    """Return the installed Codex CLI version for catalog requests."""
    override = os.getenv("CODEX_CLI_VERSION", "").strip()
    if override:
        return override
    binary = shutil.which("codex")
    if binary:
        try:
            completed = subprocess.run(
                [binary, "--version"],
                capture_output=True,
                text=True,
                timeout=1,
                check=False,
            )
            match = re.search(r"\b(\d+\.\d+\.\d+)\b", completed.stdout or "")
            if match:
                return match.group(1)
        except Exception:
            pass
    return "0.0.0"


@dataclass(frozen=True)
class CodexModelCapabilities:
    """Transport capabilities advertised by the OAuth-backed Codex catalog.

    The catalog is also consumed by the Codex CLI.  Hermes historically kept
    only the visible model slugs, which meant it silently lost the transport
    hints needed by newer subscription models such as GPT-5.6 Luna.
    """

    slug: str
    use_responses_lite: bool = False
    prefer_websockets: Optional[bool] = None
    minimal_client_version: Optional[str] = None

    @property
    def should_use_websocket(self) -> bool:
        """Return the safe automatic WebSocket choice for this model.

        Older ``models_cache.json`` files do not contain
        ``prefer_websockets``.  The current Lite-backed Codex models are
        WebSocket-first, so use that as a compatibility fallback while still
        honoring an explicit catalog value when one is available.
        """
        if self.prefer_websockets is not None:
            return self.prefer_websockets
        return self.use_responses_lite


# Keep a small forward-compatible fallback for the current GPT-5.6 preview.
# The live catalog and Codex CLI cache remain authoritative; this only keeps a
# manually configured model usable during a transient catalog/cache miss.
_KNOWN_CODEX_TRANSPORT_CAPABILITIES: Dict[str, CodexModelCapabilities] = {
    "gpt-5.6-sol": CodexModelCapabilities(
        slug="gpt-5.6-sol", use_responses_lite=True, prefer_websockets=True,
    ),
    "gpt-5.6-terra": CodexModelCapabilities(
        slug="gpt-5.6-terra", use_responses_lite=True, prefer_websockets=True,
    ),
    "gpt-5.6-luna": CodexModelCapabilities(
        slug="gpt-5.6-luna", use_responses_lite=True, prefer_websockets=True,
    ),
}

DEFAULT_CODEX_MODELS: List[str] = [
    # GPT-5.6 series (Sol/Terra/Luna + -pro high-effort modes) — GA 2026-07-09
    # (previewed 2026-06-26).
    "gpt-5.6-sol",
    "gpt-5.6-sol-pro",
    "gpt-5.6-terra",
    "gpt-5.6-terra-pro",
    "gpt-5.6-luna",
    "gpt-5.6-luna-pro",
    "gpt-5.5",
    "gpt-5.4-mini",
    "gpt-5.4",
    "gpt-5.3-codex",
    # gpt-5.3-codex-spark is in research preview and is exposed *only* via
    # the Codex CLI / OAuth backend (chatgpt.com/backend-api/codex/models)
    # for ChatGPT Pro subscribers. It is NOT available in the public OpenAI
    # API, so it intentionally stays out of the "openai" provider catalog
    # in hermes_cli/models.py — only the openai-codex (OAuth) provider
    # surfaces it. The Codex backend reports ``supported_in_api: false`` for
    # this slug; that flag describes API availability, not Codex backend
    # availability, so the fetch/cache code paths below intentionally do
    # not filter on it. PR #12994 removed this entry on the assumption it
    # was unsupported — that was wrong; restored here. Keep it in the
    # curated fallback so Pro users still see Spark in `/model` when live
    # discovery is unavailable (offline first run, transient API failure).
    "gpt-5.3-codex-spark",
    # NOTE: gpt-5.2-codex / gpt-5.1-codex-max / gpt-5.1-codex-mini were
    # previously listed here but the chatgpt.com Codex backend returns
    # HTTP 400 "The '<model>' model is not supported when using Codex with
    # a ChatGPT account." for all three on every ChatGPT Pro account we've
    # tested (verified live 2026-05-27). Keeping them in the fallback list
    # leaked dead slugs into /model when live discovery was unavailable
    # (transient API failure, first-run before refresh) and surfaced HTTP 400
    # crashes on selection. The Codex CLI public catalog still references
    # these slugs, which is why they survived previously — but those entries
    # describe the public OpenAI API, not the OAuth-backed Codex backend
    # Hermes uses. Removed here. If OpenAI re-enables them on Codex backend,
    # live discovery will pick them up automatically via _fetch_models_from_api.
]

_FORWARD_COMPAT_TEMPLATE_MODELS: List[tuple[str, tuple[str, ...]]] = [
    ("gpt-5.6-sol", ("gpt-5.5", "gpt-5.4")),
    ("gpt-5.6-sol-pro", ("gpt-5.5", "gpt-5.4")),
    ("gpt-5.6-terra", ("gpt-5.5", "gpt-5.4")),
    ("gpt-5.6-terra-pro", ("gpt-5.5", "gpt-5.4")),
    ("gpt-5.6-luna", ("gpt-5.5", "gpt-5.4")),
    ("gpt-5.6-luna-pro", ("gpt-5.5", "gpt-5.4")),
    ("gpt-5.5", ("gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex")),
    ("gpt-5.4-mini", ("gpt-5.3-codex",)),
    ("gpt-5.4", ("gpt-5.3-codex",)),
    # Surface Spark whenever any compatible Codex template is present so
    # accounts hitting the live endpoint with an older lineup still see
    # Spark in the picker. Backend gates real availability by ChatGPT Pro
    # entitlement; Hermes does not.
    ("gpt-5.3-codex-spark", ("gpt-5.3-codex",)),
]


def _add_forward_compat_models(model_ids: List[str]) -> List[str]:
    """Add Clawdbot-style synthetic forward-compat Codex models.

    If a newer Codex slug isn't returned by live discovery, surface it when an
    older compatible template model is present. This mirrors Clawdbot's
    synthetic catalog / forward-compat behavior for GPT-5 Codex variants.
    """
    ordered: List[str] = []
    seen: set[str] = set()
    for model_id in model_ids:
        if model_id not in seen:
            ordered.append(model_id)
            seen.add(model_id)

    for synthetic_model, template_models in _FORWARD_COMPAT_TEMPLATE_MODELS:
        if synthetic_model in seen:
            continue
        if any(template in seen for template in template_models):
            ordered.append(synthetic_model)
            seen.add(synthetic_model)

    return ordered


def _fetch_model_entries_from_api(access_token: str) -> List[Dict[str, Any]]:
    """Fetch visible model entries from the OAuth-backed Codex API."""
    try:
        import httpx
        client_version = _codex_client_version()
        resp = httpx.get(
            f"https://chatgpt.com/backend-api/codex/models?client_version={client_version}",
            headers={
                "Authorization": f"Bearer {access_token}",
                "User-Agent": f"codex_cli_rs/{client_version} (Hermes Agent)",
                "originator": "codex_cli_rs",
                "version": client_version,
            },
            timeout=10,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        entries = data.get("models", []) if isinstance(data, dict) else []
        if not isinstance(entries, list):
            return []
    except Exception as exc:
        logger.debug("Failed to fetch Codex models from API: %s", exc)
        return []

    sortable: List[tuple[int, Dict[str, Any]]] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        slug = item.get("slug")
        if not isinstance(slug, str) or not slug.strip():
            continue
        slug = slug.strip()
        # Codex CLI's catalog uses ``supported_in_api`` for the public OpenAI
        # API, not for the OAuth-backed Codex backend that this provider uses.
        # Some valid Codex CLI models (for example gpt-5.3-codex-spark) are
        # marked false here but are still accepted by the Codex route.
        visibility = item.get("visibility", "")
        if isinstance(visibility, str) and visibility.strip().lower() in {"hide", "hidden"}:
            continue
        priority = item.get("priority")
        rank = int(priority) if isinstance(priority, (int, float)) else 10_000
        sortable.append((rank, dict(item, slug=slug)))

    sortable.sort(key=lambda x: (x[0], str(x[1].get("slug", ""))))
    return [entry for _, entry in sortable]


def _fetch_models_from_api(access_token: str) -> List[str]:
    """Fetch available model IDs from the Codex API."""
    entries = _fetch_model_entries_from_api(access_token)
    return _add_forward_compat_models([
        str(entry["slug"])
        for entry in entries
        if isinstance(entry.get("slug"), str)
    ])


def _read_default_model(codex_home: Path) -> Optional[str]:
    config_path = codex_home / "config.toml"
    if not config_path.exists():
        return None
    try:
        import tomllib
    except Exception:
        return None
    try:
        payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    model = payload.get("model") if isinstance(payload, dict) else None
    if isinstance(model, str) and model.strip():
        return model.strip()
    return None


def _read_cache_models(codex_home: Path) -> List[str]:
    cache_path = codex_home / "models_cache.json"
    if not cache_path.exists():
        return []
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    entries = raw.get("models") if isinstance(raw, dict) else None
    sortable = []
    if isinstance(entries, list):
        for item in entries:
            if not isinstance(item, dict):
                continue
            slug = item.get("slug")
            if not isinstance(slug, str) or not slug.strip():
                continue
            slug = slug.strip()
            # Do not filter on ``supported_in_api`` here.  It describes the
            # public OpenAI API, while Hermes openai-codex talks to the same
            # OAuth-backed Codex backend as Codex CLI.
            visibility = item.get("visibility")
            if isinstance(visibility, str) and visibility.strip().lower() in {"hide", "hidden"}:
                continue
            priority = item.get("priority")
            rank = int(priority) if isinstance(priority, (int, float)) else 10_000
            sortable.append((rank, slug))

    sortable.sort(key=lambda item: (item[0], item[1]))
    deduped: List[str] = []
    for _, slug in sortable:
        if slug not in deduped:
            deduped.append(slug)
    return deduped


def _read_cache_model_entries(codex_home: Path) -> List[Dict[str, Any]]:
    """Read full model entries from the Codex CLI capability cache."""
    cache_path = codex_home / "models_cache.json"
    if not cache_path.exists():
        return []
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    entries = raw.get("models") if isinstance(raw, dict) else None
    sortable: List[tuple[int, Dict[str, Any]]] = []
    if isinstance(entries, list):
        for item in entries:
            if not isinstance(item, dict):
                continue
            slug = item.get("slug")
            if not isinstance(slug, str) or not slug.strip():
                continue
            slug = slug.strip()
            visibility = item.get("visibility")
            if isinstance(visibility, str) and visibility.strip().lower() in {"hide", "hidden"}:
                continue
            priority = item.get("priority")
            rank = int(priority) if isinstance(priority, (int, float)) else 10_000
            sortable.append((rank, dict(item, slug=slug)))

    sortable.sort(key=lambda item: (item[0], str(item[1].get("slug", ""))))
    return [entry for _, entry in sortable]


def _capabilities_from_entry(entry: Dict[str, Any]) -> CodexModelCapabilities:
    prefer_websockets = entry.get("prefer_websockets")
    if not isinstance(prefer_websockets, bool):
        prefer_websockets = None
    use_responses_lite = entry.get("use_responses_lite")
    if not isinstance(use_responses_lite, bool):
        use_responses_lite = False
    minimal_client_version = entry.get("minimal_client_version")
    if not isinstance(minimal_client_version, str) or not minimal_client_version.strip():
        minimal_client_version = None
    return CodexModelCapabilities(
        slug=str(entry.get("slug") or ""),
        use_responses_lite=use_responses_lite,
        prefer_websockets=prefer_websockets,
        minimal_client_version=minimal_client_version,
    )


def get_codex_model_capabilities(
    model: str,
    access_token: Optional[str] = None,
) -> CodexModelCapabilities:
    """Resolve transport capabilities for one Codex model.

    The local Codex cache is the fast path.  When the caller has a JWT-like
    OAuth token, refresh the catalog as well so newly introduced fields such as
    ``prefer_websockets`` are not lost on older local cache formats.  Failure
    to refresh is deliberately non-fatal: the HTTP path remains available.
    """
    requested = str(model or "").strip()
    slug = requested.rsplit("/", 1)[-1]
    codex_home_str = os.getenv("CODEX_HOME", "").strip() or str(Path.home() / ".codex")
    codex_home = Path(codex_home_str).expanduser()

    local_entries = _read_cache_model_entries(codex_home)
    local_entry = next(
        (entry for entry in local_entries if str(entry.get("slug", "")).strip() == slug),
        None,
    )

    # Access tokens from the Codex OAuth flow are JWTs.  Avoid a live catalog
    # request for synthetic/test tokens and for ordinary API-key providers.
    live_entries: List[Dict[str, Any]] = []
    if isinstance(access_token, str) and access_token.count(".") >= 2:
        token_key = hashlib.sha256(access_token.encode("utf-8")).hexdigest()
        now = time.monotonic()
        cached_live = _CODEX_LIVE_MODEL_ENTRY_CACHE.get(token_key)
        if cached_live and now - cached_live[0] < _CODEX_CAPABILITY_CACHE_TTL:
            live_entries = cached_live[1]
        else:
            live_entries = _fetch_model_entries_from_api(access_token)
            _CODEX_LIVE_MODEL_ENTRY_CACHE[token_key] = (now, live_entries)
    live_entry = next(
        (entry for entry in live_entries if str(entry.get("slug", "")).strip() == slug),
        None,
    )

    if local_entry is None and live_entry is None:
        return _KNOWN_CODEX_TRANSPORT_CAPABILITIES.get(
            slug,
            CodexModelCapabilities(slug=slug),
        )

    entry = dict(local_entry or {})
    if live_entry:
        entry.update(live_entry)
    return _capabilities_from_entry(entry)


def get_codex_model_ids(access_token: Optional[str] = None) -> List[str]:
    """Return available Codex model IDs, trying API first, then local sources.
    
    Resolution order: API (live, if token provided) > config.toml default >
    local cache > hardcoded defaults.
    """
    codex_home_str = os.getenv("CODEX_HOME", "").strip() or str(Path.home() / ".codex")
    codex_home = Path(codex_home_str).expanduser()
    ordered: List[str] = []

    # Try live API if we have a token
    if access_token:
        api_models = _fetch_models_from_api(access_token)
        if api_models:
            return _add_forward_compat_models(api_models)

    # Fall back to local sources
    default_model = _read_default_model(codex_home)
    if default_model:
        ordered.append(default_model)

    for model_id in _read_cache_models(codex_home):
        if model_id not in ordered:
            ordered.append(model_id)

    for model_id in DEFAULT_CODEX_MODELS:
        if model_id not in ordered:
            ordered.append(model_id)

    return _add_forward_compat_models(ordered)
