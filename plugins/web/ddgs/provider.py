"""DuckDuckGo search — plugin form (via the ``ddgs`` package).

Subclasses the plugin-facing :class:`agent.web_search_provider.WebSearchProvider`.
The legacy in-tree module ``tools.web_providers.ddgs`` was removed in the
same commit that moved this code under ``plugins/``; this file is now the
canonical implementation.

The ``ddgs`` package is an optional dependency. ``is_available()`` reflects
whether the package is importable; the plugin still registers either way so
``hermes tools`` can prompt the user to install it.

Isolation note (#68096): ``ddgs``/``primp`` can block inside native code while
holding the Python GIL. A ``ThreadPoolExecutor`` + ``future.result(timeout=…)``
cap (see #52118) cannot fire in that state — the waiter never reacquires the
GIL — so the whole Hermes process freezes through Ctrl+C/SIGTERM. Each search
therefore runs in a disposable child process the parent can terminate/kill.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import os
import subprocess
import sys
import time
from typing import Any, Dict, Optional

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

# Overall wall-clock cap for a single ddgs search. The DDGS constructor's
# ``timeout`` only bounds individual HTTP requests; ddgs's multi-engine retry
# loop has no overall cap, so a slow/rate-limited DuckDuckGo response can hang
# the (single, shared) agent loop indefinitely (#36776). Enforce a hard cap
# here by killing a disposable worker process (#68096).
_SEARCH_TIMEOUT_SECS = 30

# How often the parent polls stdout / interrupt flag while waiting.
_POLL_INTERVAL_SECS = 0.1

# After terminate(), wait this long before escalating to kill().
_TERMINATE_GRACE_SECS = 1.0


class _SearchInterrupted(Exception):
    """Raised when tools.interrupt.is_interrupted() trips during a search wait."""


def _run_ddgs_search(query: str, safe_limit: int) -> list[dict[str, Any]]:
    """Run the blocking ddgs query and return normalized hits.

    Module-level (not a closure) so the child worker can import it and so
    tests can patch it for in-process unit tests. ``DDGS(timeout=…)`` bounds
    each individual HTTP request; the overall wall-clock cap is enforced by
    the parent via process timeout (#68096).
    """
    from ddgs import DDGS  # type: ignore

    results: list[dict[str, Any]] = []
    with DDGS(timeout=10) as client:
        for i, hit in enumerate(client.text(query, max_results=safe_limit)):
            if i >= safe_limit:
                break
            url = str(hit.get("href") or hit.get("url") or "")
            results.append(
                {
                    "title": str(hit.get("title", "")),
                    "url": url,
                    "description": str(hit.get("body", "")),
                    "position": i + 1,
                }
            )
    return results


# Optional test-only hook name forwarded to the child (see _search_worker.py).
# Production search() never sets this.
_test_hook: Optional[str] = None

# Last worker Popen started by ``_run_ddgs_search_bounded`` (test reap checks).
_last_worker_proc: Optional[subprocess.Popen] = None


def _plugins_path_entry() -> str:
    """Return the ``sys.path`` entry that makes ``import plugins`` work.

    Prefer the live ``plugins`` package location over counting ``dirname``s from
    this file — that stays correct for source checkouts and site-packages.
    """
    try:
        import plugins as plugins_pkg

        pkg_file = getattr(plugins_pkg, "__file__", None)
        if pkg_file:
            return os.path.dirname(os.path.dirname(os.path.abspath(pkg_file)))
    except Exception:  # noqa: BLE001 — fall through to path-walk fallback
        pass
    return os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    )


def _terminate_and_reap(
    proc: Optional[subprocess.Popen],
    *,
    grace: float = _TERMINATE_GRACE_SECS,
) -> None:
    """Terminate a worker, escalate to kill, and wait so no orphan remains.

    Does not close the parent's pipe ends — the caller must finish any
    ``communicate()``/reader first. Closing stdout while another thread is
    blocked in ``read()`` deadlocks on some platforms.
    """
    if proc is None:
        return

    def _wait_until_dead(seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                return True
            time.sleep(0.05)
        return proc.poll() is not None

    try:
        if proc.poll() is None:
            proc.terminate()
            _wait_until_dead(grace)
        if proc.poll() is None:
            proc.kill()
            if not _wait_until_dead(grace):
                logger.warning("DDGS worker pid=%s did not exit after kill", proc.pid)
    except Exception as exc:  # noqa: BLE001 — best-effort cleanup
        logger.debug("DDGS worker reap error: %s", exc)


def _run_ddgs_search_bounded(query: str, safe_limit: int) -> list[dict[str, Any]]:
    """Run ``_run_ddgs_search`` in a disposable process with a hard deadline.

    The parent never joins the child while it may be inside native code holding
    *its* GIL — it only polls a communicator thread and, on timeout/interrupt,
    terminates the child OS process. Raises ``TimeoutError``,
    ``_SearchInterrupted``, or ``RuntimeError``.
    """
    # Imported lazily so plugin import stays light for ``hermes tools`` probes.
    from tools.interrupt import is_interrupted

    global _last_worker_proc

    request: dict[str, Any] = {"query": query, "safe_limit": safe_limit}
    if _test_hook:
        request["test_hook"] = _test_hook

    from tools.environments.local import _sanitize_subprocess_env

    env = _sanitize_subprocess_env(dict(os.environ))
    if _test_hook:
        env["HERMES_DDGS_ALLOW_TEST_HOOKS"] = "1"

    # Running the worker as a script puts ``plugins/web/ddgs/`` on ``sys.path[0]``,
    # which breaks ``import plugins...``. Prepend the path entry that makes the
    # live ``plugins`` package importable (source tree or site-packages).
    child_pythonpath = env.get("PYTHONPATH", "")
    path_entry = _plugins_path_entry()
    if path_entry and path_entry not in child_pythonpath.split(os.pathsep):
        env["PYTHONPATH"] = (
            path_entry + os.pathsep + child_pythonpath if child_pythonpath else path_entry
        )

    worker_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_search_worker.py")
    # Platform-only spawn knobs — stdin/stdout/stderr must stay as explicit
    # keyword args on the Popen call so scripts/check_subprocess_stdin.py can
    # see them (TUI gateway inherits stdin; #14036).
    extra_kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        # New process group so terminate/kill reach the worker cleanly on Windows.
        extra_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        # Own session so a hung primp/libcurl grandchild can be reaped with the worker.
        extra_kwargs["start_new_session"] = True

    proc = subprocess.Popen(
        [sys.executable, worker_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        # DEVNULL avoids the classic deadlock where a chatty child fills the
        # stderr pipe buffer while the parent only drains stdout.
        stderr=subprocess.DEVNULL,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        **extra_kwargs,
    )
    _last_worker_proc = proc

    # ``communicate`` runs in a side thread so the parent can poll interrupt /
    # deadline without blocking. Killing the child unblocks communicate.
    pool = cf.ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(proc.communicate, json.dumps(request))
    timed_out = False
    interrupted = False
    raw = ""
    try:
        deadline = time.monotonic() + _SEARCH_TIMEOUT_SECS
        while True:
            if is_interrupted():
                interrupted = True
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            try:
                out, _err = fut.result(timeout=min(_POLL_INTERVAL_SECS, remaining))
                raw = out or ""
                break
            except cf.TimeoutError:
                continue
    finally:
        _terminate_and_reap(proc)
        # After kill, communicate should return promptly; don't block forever.
        if not fut.done():
            try:
                out, _err = fut.result(timeout=_TERMINATE_GRACE_SECS)
                if not raw:
                    raw = out or ""
            except Exception:  # noqa: BLE001
                pass
        pool.shutdown(wait=False, cancel_futures=True)

    if interrupted:
        raise _SearchInterrupted("DuckDuckGo search interrupted")
    if timed_out:
        raise TimeoutError(
            f"DuckDuckGo search timed out after {_SEARCH_TIMEOUT_SECS}s"
        )

    raw = raw.strip()
    if not raw:
        raise RuntimeError(
            f"DDGS worker exited without a result (code={proc.poll()})"
        )

    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"DDGS worker returned invalid JSON: {raw[:200]!r}"
        ) from exc

    if not isinstance(envelope, dict):
        raise RuntimeError(f"DDGS worker returned an invalid envelope: {envelope!r}")
    if envelope.get("ok"):
        results = envelope.get("results") or []
        if not isinstance(results, list):
            raise RuntimeError("DDGS worker returned non-list results")
        return results
    raise RuntimeError(str(envelope.get("error") or "DDGS worker failed"))


class DDGSWebSearchProvider(WebSearchProvider):
    """DuckDuckGo HTML-scrape search provider.

    No API key needed. Rate limits are enforced server-side by DuckDuckGo;
    the provider surfaces ``DuckDuckGoSearchException`` and other ddgs errors
    as ``{"success": False, "error": ...}`` rather than raising.
    """

    @property
    def name(self) -> str:
        return "ddgs"

    @property
    def display_name(self) -> str:
        return "DuckDuckGo (ddgs)"

    def is_available(self) -> bool:
        """Return True when the ``ddgs`` package is importable.

        Probes the import once; cheap because Python caches the import. Must
        NOT perform network I/O — runs at tool-registration time and on every
        ``hermes tools`` paint.
        """
        try:
            import ddgs  # noqa: F401

            return True
        except ImportError:
            return False

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a DuckDuckGo search and return normalized results.

        The synchronous ``ddgs`` call runs in a disposable child process with
        a hard wall-clock timeout (``_SEARCH_TIMEOUT_SECS``) so a hung native
        ``primp`` call cannot freeze the Hermes process (#36776, #68096).
        """
        try:
            import ddgs  # type: ignore  # noqa: F401 — availability probe
        except ImportError:
            return {
                "success": False,
                "error": "ddgs package is not installed — run `pip install ddgs`",
            }

        # DDGS().text yields at most `max_results` items; we cap defensively
        # in case the package ignores the hint.
        safe_limit = max(1, int(limit))

        try:
            web_results = _run_ddgs_search_bounded(query, safe_limit)
        except TimeoutError:
            logger.warning(
                "DDGS search timed out after %ds for query: %r",
                _SEARCH_TIMEOUT_SECS,
                query,
            )
            return {
                "success": False,
                "error": (
                    f"DuckDuckGo search timed out after {_SEARCH_TIMEOUT_SECS}s — "
                    "DuckDuckGo may be rate-limiting or slow. Try again later "
                    "or switch to a different search provider."
                ),
            }
        except _SearchInterrupted:
            logger.info("DDGS search interrupted for query: %r", query)
            return {
                "success": False,
                "error": "DuckDuckGo search interrupted",
            }
        except Exception as exc:  # noqa: BLE001 — ddgs raises its own exceptions
            logger.warning("DDGS search error: %s", exc)
            return {"success": False, "error": f"DuckDuckGo search failed: {exc}"}

        logger.info(
            "DDGS search '%s': %d results (limit %d)", query, len(web_results), limit
        )
        return {"success": True, "data": {"web": web_results}}

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "DuckDuckGo (ddgs)",
            "badge": "free · no key · search only",
            "tag": "Search via the ddgs Python package — no API key (pair with any extract provider)",
            "env_vars": [],
            # Trigger `_run_post_setup("ddgs")` after the user picks this row
            # so the ddgs Python package gets pip-installed on first selection.
            "post_setup": "ddgs",
        }
