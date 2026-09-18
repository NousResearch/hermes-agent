"""Tests for the ``asyncio.start_unix_server`` platform-gate footgun rule
in ``scripts/check-windows-footguns.py``.

``asyncio.start_unix_server`` (and ``asyncio.start_server``'s
``unix_like=True`` kwarg) bind an AF_UNIX socket and exist only on POSIX —
on Windows they raise ``AttributeError`` at call time and the Proactor
loop cannot serve unix-socket listeners anyway. The rule flags ungated
call sites so they get a platform check plus a Windows fallback (TCP
loopback, or a named pipe via ``loop.start_serving_pipe``), matching the
pattern already used by ``gateway/control_socket.py`` and
``gateway/shutdown_watchdog.py``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LINTER_PATH = REPO_ROOT / "scripts" / "check-windows-footguns.py"


def _load_linter_module():
    """Import the linter script as a module (it's not a package).

    Register the module in sys.modules BEFORE exec_module so that
    ``@dataclass`` can resolve ``cls.__module__`` via
    ``sys.modules.get(cls.__module__).__dict__`` (CPython 3.11+ dataclass
    internals require this).
    """
    spec = importlib.util.spec_from_file_location("check_windows_footguns", LINTER_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_windows_footguns"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def linter():
    return _load_linter_module()


def _find_footgun(linter, name: str):
    """Locate a Footgun by name in the FOOTGUNS list."""
    for fg in linter.FOOTGUNS:
        if fg.name == name:
            return fg
    pytest.fail(f"Footgun rule '{name}' not found in FOOTGUNS")


def _scan_line(linter, line: str, footgun_name: str) -> bool:
    """Return True if the given line triggers the named footgun rule.

    Replicates the relevant checks from scan_file(): suppression marker,
    guard hints, then pattern + post_filter — so the test exercises the
    real detection path.
    """
    fg = _find_footgun(linter, footgun_name)
    if linter.SUPPRESS_MARKER.search(line):
        return False
    if any(hint in line for hint in linter.GUARD_HINTS):
        return False
    code = linter._strip_code(line)
    if not code.strip():
        return False
    match = fg.pattern.search(code)
    if not match:
        return False
    if fg.post_filter is not None:
        try:
            if not fg.post_filter(match, line):
                return False
        except (IndexError, AttributeError):
            return False
    return True


RULE_NAME = (
    "asyncio.start_unix_server / start_server(unix_like=True) "
    "without a platform gate"
)


# ---------------------------------------------------------------------------
# Detection — these SHOULD be flagged
# ---------------------------------------------------------------------------


class TestDetection:
    def test_flags_bare_asyncio_start_unix_server(self, linter):
        line = "    srv = await asyncio.start_unix_server(handler, path=str(sock))"
        assert _scan_line(linter, line, RULE_NAME)

    def test_flags_loop_receiver_form(self, linter):
        line = "    srv = await loop.start_unix_server(handler, path=str(sock))"
        assert _scan_line(linter, line, RULE_NAME)

    def test_flags_start_server_with_unix_like_kwarg(self, linter):
        line = "    srv = await asyncio.start_server(handler, unix_like=True)"
        assert _scan_line(linter, line, RULE_NAME)

    def test_flags_unix_like_with_spaces_around_equals(self, linter):
        line = "    srv = await asyncio.start_server(handler, unix_like = True)"
        assert _scan_line(linter, line, RULE_NAME)


# ---------------------------------------------------------------------------
# Suppression — these should NOT be flagged
# ---------------------------------------------------------------------------


class TestSuppression:
    def test_does_not_flag_plain_tcp_start_server(self, linter):
        line = '    srv = await asyncio.start_server(handler, host="127.0.0.1", port=0)'
        assert not _scan_line(linter, line, RULE_NAME)

    def test_does_not_flag_inline_suppression_marker(self, linter):
        line = (
            "    srv = await asyncio.start_unix_server(handler, path=str(sock))"
            "  # windows-footgun: ok — gated elsewhere"
        )
        assert not _scan_line(linter, line, RULE_NAME)

    def test_does_not_flag_string_literal_mention(self, linter):
        line = '    assert "await asyncio.start_unix_server(" in source'
        assert not _scan_line(linter, line, RULE_NAME)

    def test_does_not_flag_unrelated_receiver(self, linter):
        line = "    srv = await factory.start_server(handler)"
        assert not _scan_line(linter, line, RULE_NAME)


# ---------------------------------------------------------------------------
# Guard hints — same-line platform gates silence the rule
# ---------------------------------------------------------------------------


class TestGuardHints:
    def test_hasattr_asyncio_guard_hint_silences(self, linter):
        line = "    if hasattr(asyncio, 'start_unix_server') and use_unix:"
        assert any(hint in line for hint in linter.GUARD_HINTS)

    def test_os_name_gate_hint_is_registered(self, linter):
        line = '    if os.name == "posix":'
        assert any(hint in line for hint in linter.GUARD_HINTS)


# ---------------------------------------------------------------------------
# Rule registration — the rule must be wired into FOOTGUNS and the linter
# must keep excluding itself (it mentions the pattern in its own docs).
# ---------------------------------------------------------------------------


class TestRuleRegistration:
    def test_rule_present_in_footguns_list(self, linter):
        names = [fg.name for fg in linter.FOOTGUNS]
        assert RULE_NAME in names

    def test_linter_excludes_itself(self, linter):
        assert "scripts/check-windows-footguns.py" in linter.EXCLUDED_FILES
