"""``scripts/check_async_url_guards.py``: blocking ``is_safe_url()`` from inside an
``async def`` is flagged; sync helpers, module-level calls and the awaited
``async_is_safe_url`` wrapper are not.

The full-repo wrapper at the bottom is what actually catches the next
regression: ruff's ASYNC210/220/221/251 rules only see blocking calls written
directly in the async frame, not a synchronous ``socket.getaddrinfo`` hidden
behind a helper, so nothing else in the suite would notice.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_async_url_guards.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_async_url_guards", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _violations(source: str) -> list[tuple[int, str]]:
    return [(lineno, name) for lineno, name, _ in _load().iter_violations(source)]


def test_async_call_is_flagged():
    src = (
        "from tools.url_safety import is_safe_url\n"
        "\n"
        "async def send_image(url):\n"
        "    if not is_safe_url(url):\n"
        "        raise ValueError('blocked')\n"
    )
    assert _violations(src) == [(4, "send_image")]


def test_nested_async_call_is_flagged():
    src = (
        "from tools.url_safety import is_safe_url\n"
        "\n"
        "async def download(url):\n"
        "    async def _guard(target):\n"
        "        return is_safe_url(target)\n"
        "    return await _guard(url)\n"
    )
    assert _violations(src) == [(5, "_guard")]


def test_aliased_import_is_flagged():
    src = (
        "from tools.url_safety import is_safe_url as _is_safe_url\n"
        "\n"
        "async def send(url):\n"
        "    return _is_safe_url(url)\n"
    )
    assert _violations(src) == [(4, "send")]


def test_sync_helper_is_allowed():
    src = (
        "from tools.url_safety import is_safe_url\n"
        "\n"
        "def _guarded(url):\n"
        "    return is_safe_url(url)\n"
        "\n"
        "async def send(url):\n"
        "    return await asyncio.to_thread(_guarded, url)\n"
    )
    assert _violations(src) == []


def test_module_level_call_is_allowed():
    src = "from tools.url_safety import is_safe_url\n\nALLOWED = is_safe_url('https://example.com')\n"
    assert _violations(src) == []


def test_async_wrapper_is_allowed():
    src = (
        "from tools.url_safety import async_is_safe_url\n"
        "\n"
        "async def send_image(url):\n"
        "    if not await async_is_safe_url(url):\n"
        "        raise ValueError('blocked')\n"
    )
    assert _violations(src) == []


def test_full_repo_scan_is_clean():
    """Run the real checker over the shipped tree and require a clean exit, so a
    future async media handler cannot quietly reintroduce the blocking call."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        timeout=120,
        stdin=subprocess.DEVNULL,
    )
    assert result.returncode == 0, f"async URL guard violations:\n{result.stdout}\n{result.stderr}"
