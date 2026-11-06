"""Scrapling web search + extraction — the 5th keyless ring rung (local, no key).

Why a subprocess
----------------
Scrapling (plus its Playwright/Chromium dependency closure) is installed in a
dedicated venv (``~/.hermes/venvs/scrapling``) that is NOT the Hermes runtime:
the gateway runs a different Python. Importing across that boundary would mean
mutating ``sys.path`` of a live gateway process, so the boundary is kept as a
process boundary instead — the sidecar is spawned with the venv's own
interpreter and speaks JSON over stdin/stdout (the same shape as the bundled
``ddgs`` plugin's worker).

Search source
-------------
The local rung has no search API of its own, so it scrapes DDG Lite
(``lite.duckduckgo.com/lite/``), whose ``robots.txt`` is ``Allow: /``.
``html.duckduckgo.com`` is NOT used: it answers anonymous clients with an
HTTP 202 anomaly challenge. Marginalia and Mojeek were checked and excluded —
their ``robots.txt`` disallows the ``/search`` path.

Extract source
--------------
A direct fetch of the requested URL, with a thin-body escalation to headless
Chromium (the body is a JS shell). Scrapling is only asked to render when the
plain HTTP body is too short to be real content, so the common path never
starts a browser.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from plugins.web._common import (
    BaseWebSearchProvider, document as _page, keyless_extract, keyless_search,
    page_error as _page_error, search_fail, search_ok, setup_schema, web_hit as _row,
)
from plugins.web.scrapling._ddg import parse_ddg_lite_results

logger = logging.getLogger(__name__)

# The sidecar interpreter. A dedicated venv keeps Scrapling's dependency
# closure (Playwright + Chromium) out of the Hermes runtime, and the process
# boundary keeps the two Python versions from meeting.
#
# Resolved lazily (not at import) so the path follows the active Hermes home:
# a test run against an isolated HERMES_HOME must not touch the real one.
SIDECAR_VENV_ENV = "HERMES_SCRAPLING_VENV"
SIDECAR_TIMEOUT_S = int(os.environ.get("HERMES_SCRAPLING_TIMEOUT_S", "60"))

# How to provision the sidecar. The rung is inert until the sidecar exists, so
# the error a user sees must say how to fix it rather than just where it looked.
# Verified on this host: `python3 -m venv` needs the `python3-venv` system
# package (absent on Debian/Ubuntu without it), so use `uv` — which Hermes
# already requires — and `--seed` so the venv has pip.
SIDECAR_INSTALL_HINT = (
    "uv venv --seed ~/.hermes/venvs/scrapling && "
    "~/.hermes/venvs/scrapling/bin/pip install 'scrapling[fetchers]==0.4.11' && "
    "~/.hermes/venvs/scrapling/bin/playwright install chromium"
)


def sidecar_venv() -> str:
    """Directory holding the Scrapling sidecar venv."""
    override = os.environ.get(SIDECAR_VENV_ENV, "").strip()
    if override:
        return os.path.expanduser(override)
    try:
        from hermes_constants import get_hermes_home

        return str(Path(get_hermes_home()) / "venvs" / "scrapling")
    except Exception:
        return os.path.expanduser("~/.hermes/venvs/scrapling")


def sidecar_python() -> str:
    """The sidecar interpreter path."""
    return os.path.join(sidecar_venv(), "bin", "python")


def sidecar_available() -> bool:
    """True when the sidecar interpreter exists on this host."""
    try:
        return os.path.exists(sidecar_python())
    except OSError:
        return False

# The local rung never presents itself as a keyed vendor, so there is no signup
# URL to suggest; the message points at the ring instead.
_LOCAL_HINT = "another web backend via `hermes tools`"


def _run_sidecar(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one sidecar call; returns the worker's JSON envelope.

    The envelope shape is ``{"ok": bool, "results": [...]}`` or
    ``{"ok": false, "error": str}``. Never raises: a sidecar problem must
    surface as a soft rung failure so the ring can fail over.
    """
    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_worker.py")
    if not sidecar_available():
        return {
            "ok": False,
            "error": (
                f"Scrapling sidecar not installed at {sidecar_python()} — "
                f"install it with: {SIDECAR_INSTALL_HINT}"
            ),
        }
    try:
        from tools.environments.local import _sanitize_subprocess_env

        env = _sanitize_subprocess_env(dict(os.environ))
    except Exception:
        env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    try:
        completed = subprocess.run(
            [sidecar_python(), worker],
            input=json.dumps(payload), capture_output=True, text=True,
            timeout=SIDECAR_TIMEOUT_S, env=env, check=False,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"Scrapling sidecar timed out after {SIDECAR_TIMEOUT_S}s"}
    except Exception as exc:
        return {"ok": False, "error": f"Scrapling sidecar failed to start: {exc}"}
    raw = (completed.stdout or "").strip()
    if not raw:
        detail = (completed.stderr or "").strip().splitlines()[-1:] or ["no output"]
        return {"ok": False, "error": f"Scrapling sidecar produced no output: {detail[0][:200]}"}
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"Scrapling sidecar returned invalid JSON: {exc}"}
    if not isinstance(envelope, dict):
        return {"ok": False, "error": "Scrapling sidecar returned an unexpected payload"}
    return envelope


# A local rung failure must be shaped so the ring ADVANCES to the next vendor
# instead of stopping: upstream treats a structured 5xx as "this endpoint fails
# every query from here, while another vendor may still serve it" — exactly
# what a down local sidecar is. The marker also lets the extract path (whose
# failover rule is narrower) recognise a rung-wide outage.
_LOCAL_OUTAGE = "local rung unavailable"


def scrapling_search_keyless(query: str, limit: int = 5) -> dict[str, Any]:
    """Search via the local rung (DDG Lite) — no API key, no vendor rate limit."""
    envelope = _run_sidecar({"op": "search", "query": query, "limit": max(1, int(limit))})
    if not envelope.get("ok"):
        error = str(envelope.get("error") or "Scrapling local search failed")
        # "HTTP 503" is deliberate: the ring's refusal regex advances on a
        # structured 5xx, so a down local rung hands the query to the next
        # vendor instead of failing the whole request.
        return search_fail(
            f"Keyless Scrapling search failed: HTTP 503: {_LOCAL_OUTAGE}: {error}. Use {_LOCAL_HINT}."
        )
    rows = envelope.get("results") or []
    if not rows:
        # An empty parse usually means the search endpoint refused us (anomaly
        # challenge) or changed layout. Hand over rather than report "no
        # results" as a real answer.
        return search_fail(
            f"Keyless Scrapling search failed: HTTP 503: {_LOCAL_OUTAGE}: no results parsed. Use {_LOCAL_HINT}."
        )
    return search_ok([
        _row(r.get("url") or "", r.get("title") or "", r.get("description") or "", r.get("position") or i + 1)
        for i, r in enumerate(rows)
    ])


def scrapling_extract_keyless(urls: list[str]) -> list[dict[str, Any]]:
    """Fetch each URL locally — no API key, no vendor rate limit."""
    envelope = _run_sidecar({"op": "extract", "urls": list(urls)})
    if not envelope.get("ok"):
        error = str(envelope.get("error") or "Scrapling local extract failed")
        return [
            _page_error(u, f"Keyless Scrapling extract failed: HTTP 503: {_LOCAL_OUTAGE}: {error}. Use {_LOCAL_HINT}.")
            for u in urls
        ]
    # Key on the REQUESTED url: the sidecar reports the final url after
    # redirects, which would otherwise leave the requested url unmatched.
    by_url = {r.get("requested_url") or r.get("url"): r for r in (envelope.get("results") or []) if isinstance(r, dict)}
    results: list[dict[str, Any]] = []
    for url in urls:
        row = by_url.get(url) or {}
        if row.get("error"):
            results.append(_page_error(url, str(row["error"])))
            continue
        results.append(_page(url, row.get("title") or "", row.get("text") or ""))
    return results


class ScraplingWebSearchProvider(BaseWebSearchProvider):
    """Local web search + extraction: no API key, no vendor rate limit.

    The provider is the ring entry point; the actual work happens in the
    Scrapling sidecar (a separate venv), reached through a subprocess.
    """

    NAME = "scrapling"
    DISPLAY_NAME = "Scrapling (local)"
    KEY_ENV = ""
    EXTRACT = True
    KEYLESS = True

    def is_available(self) -> bool:
        # No credential exists, so this provider is never "available" in the
        # keyed sense — it is a keyless-tier rung, exactly like the other
        # ring members (BaseWebSearchProvider's default would consult a key
        # env var that does not exist here).
        return False

    def is_keyless_available(self) -> bool:
        from plugins.web.keyless_mcp import keyless_enabled, provider_tier

        return (
            self.KEYLESS
            and keyless_enabled()
            and provider_tier(self.NAME) != "paid"
            and sidecar_available()
        )

    def search(self, query: str, limit: int = 5) -> dict[str, Any]:
        logger.info("Scrapling keyless search: '%s' (limit=%d)", query, limit)
        return keyless_search(self.DISPLAY_NAME, self.NAME, query, limit, logger)

    def extract(self, urls: list[str], **kwargs: Any) -> list[dict[str, Any]]:
        logger.info("Scrapling keyless extract: %d URL(s)", len(urls))
        return keyless_extract(self.DISPLAY_NAME, self.NAME, urls, logger)

    def get_setup_schema(self) -> dict[str, Any]:
        return setup_schema(
            "Scrapling (local)", "free · no key",
            "Local web search + extraction via Scrapling — no API key and no rate limit. "
            f"Needs the sidecar venv: {SIDECAR_INSTALL_HINT}",
        )
