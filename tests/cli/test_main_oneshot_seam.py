"""Seam + byte-fidelity tests for the main.py god-file slice R1-S1.

C1 (oneshot hard-exit) was extracted from ``hermes_cli/main.py`` (window
105-221 at consensus pin ``5c5f1a6b76...``) into the new module
``hermes_cli/main_oneshot_exit.py``.  ``hermes_cli.main`` keeps an
identity-preserving re-export so every call site, lazy import, and
monkeypatch target resolves exactly as before.

Covered here:

- re-export identity (``hermes_cli.main.<name> is ...main_oneshot_exit.<name>``)
- byte-fidelity of the moved window (golden sha from R1-CONSENSUS.md)
- exit-code mapping of ``_exit_after_oneshot`` / ``_run_and_exit_oneshot``
  (subprocess, because the real path calls ``os._exit``)
- ``_cleanup_oneshot_runtime`` idempotence via the ``_oneshot_cleanup_done``
  flag guard
- the load-bearing finally block: nothing between ``_cleanup_oneshot_runtime()``
  and ``_exit_after_oneshot(rc)`` (#30387/#43055)
"""

import hashlib
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN_PY = REPO_ROOT / "hermes_cli" / "main.py"
ONESHOT_EXIT_PY = REPO_ROOT / "hermes_cli" / "main_oneshot_exit.py"

# Consensus golden shas (R1-CONSENSUS.md §3): window = main.py lines 105-221.
GOLDEN_WINDOW_NO_NL = "bc3a19bdea9f577fd8bd8614f41930a6863f77bba6dd8110f20327c7f331c602"
GOLDEN_WINDOW_PLUS_NL = "a6d81613f0e3049b313db78f4c7a4eef12d28ee14b17e8b46cf4cb89e45f1f8c"

REEXPORTED_NAMES = (
    "_exit_after_oneshot",
    "_cleanup_oneshot_runtime",
    "_run_and_exit_oneshot",
)


@pytest.mark.parametrize("name", REEXPORTED_NAMES)
def test_reimport_identity(name):
    import hermes_cli.main as main_mod
    import hermes_cli.main_oneshot_exit as exit_mod

    assert getattr(main_mod, name) is getattr(exit_mod, name)


def test_extracted_main_has_no_moved_defs():
    src = MAIN_PY.read_text(encoding="utf-8")
    for name in (
        "def _exit_after_oneshot",
        "def _cleanup_oneshot_runtime",
        "def _run_and_exit_oneshot",
    ):
        assert name not in src
    # The flag definition travels with the cluster too.
    assert "_oneshot_cleanup_done = False" not in src


def test_module_window_byte_fidelity():
    """The moved window in the new module matches the consensus golden shas."""
    mod = ONESHOT_EXIT_PY.read_bytes()
    start = mod.index(b"def _exit_after_oneshot")
    moved_slice = mod[start:]
    assert hashlib.sha256(moved_slice).hexdigest() == GOLDEN_WINDOW_PLUS_NL
    assert hashlib.sha256(moved_slice.rstrip(b"\n")).hexdigest() == GOLDEN_WINDOW_NO_NL
    # The def appears exactly once in the module.
    assert mod.count(b"def _run_and_exit_oneshot(") == 1


def test_os_exit_finally_block_is_load_bearing():
    """Nothing may be inserted between cleanup and hard exit (#30387/#43055)."""
    src = ONESHOT_EXIT_PY.read_text(encoding="utf-8")
    # Search inside _run_and_exit_oneshot's body only (the def line also
    # contains the substring "_cleanup_oneshot_runtime()").
    body_start = src.index("def _run_and_exit_oneshot(")
    body = src[body_start:]
    cleanup_idx = body.index("_cleanup_oneshot_runtime()")
    exit_idx = body.index("_exit_after_oneshot(rc)")
    between = body[cleanup_idx + len("_cleanup_oneshot_runtime()") : exit_idx]
    # Allowed: whitespace + the "finally:" statement + its comment, nothing else.
    stripped = "".join(
        line.strip()
        for line in between.splitlines()
        if line.strip() and not line.strip().startswith("#")
    )
    assert stripped in ("finally:", "try:finally:", "finally:")
    # And the call order is preserved: cleanup first, then hard exit.
    assert cleanup_idx < exit_idx


def _run_in_subprocess(program: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", program],
        cwd=REPO_ROOT,
        capture_output=True,
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize(
    ("rc_expr", "expected"),
    [
        ("None", 0),
        ("17", 17),
        ('"boom"', 1),
    ],
)
def test_exit_after_oneshot_exit_code_mapping(rc_expr, expected):
    program = textwrap.dedent(
        f"""
        import hermes_cli.main_oneshot_exit as m

        def fake_exit(rc):
            print("RC", rc)
            raise SystemExit(0)
        m.os._exit = fake_exit
        m._exit_after_oneshot({rc_expr})
        """
    )
    result = _run_in_subprocess(program)
    assert result.returncode == 0, result.stderr.decode()
    assert result.stdout.decode().strip() == f"RC {expected}"


def test_run_and_exit_oneshot_maps_keyboard_interrupt_and_systemexit():
    program = textwrap.dedent(
        """
        import sys, types
        import hermes_cli.main_oneshot_exit as m

        def fake_exit(rc):
            print("RC", rc)
            raise SystemExit(0)
        m.os._exit = fake_exit

        fake_oneshot = types.ModuleType("hermes_cli.oneshot")
        fake_oneshot.run_oneshot = lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt())
        sys.modules["hermes_cli.oneshot"] = fake_oneshot
        m._run_and_exit_oneshot("hello")
        """
    )
    result = _run_in_subprocess(program)
    assert result.returncode == 0, result.stderr.decode()
    assert result.stdout.decode().strip() == "RC 130"


def test_run_and_exit_oneshot_systemexit_non_int_prints_stderr():
    program = textwrap.dedent(
        """
        import sys, types
        import hermes_cli.main_oneshot_exit as m

        def fake_exit(rc):
            print("RC", rc)
            raise SystemExit(0)
        m.os._exit = fake_exit

        fake_oneshot = types.ModuleType("hermes_cli.oneshot")
        fake_oneshot.run_oneshot = lambda *a, **k: (_ for _ in ()).throw(SystemExit("boom"))
        sys.modules["hermes_cli.oneshot"] = fake_oneshot
        m._run_and_exit_oneshot("hello")
        """
    )
    result = _run_in_subprocess(program)
    assert result.returncode == 0, result.stderr.decode()
    assert result.stdout.decode().strip() == "RC 1"
    assert "boom" in result.stderr.decode()


def test_cleanup_oneshot_runtime_runs_each_helper_once(monkeypatch):
    """The flag guard makes _cleanup_oneshot_runtime idempotent."""
    import hermes_cli.main_oneshot_exit as m

    calls = {"terminal": 0, "async": 0, "browser": 0, "mcp": 0, "aux": 0}

    monkeypatch.setitem(
        sys.modules,
        "tools.terminal_tool",
        types.SimpleNamespace(cleanup_all_environments=lambda: calls.__setitem__("terminal", calls["terminal"] + 1)),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.async_delegation",
        types.SimpleNamespace(interrupt_all=lambda reason: calls.__setitem__("async", calls["async"] + 1)),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.browser_tool",
        types.SimpleNamespace(
            _emergency_cleanup_all_sessions=lambda: calls.__setitem__("browser", calls["browser"] + 1)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_tool",
        types.SimpleNamespace(shutdown_mcp_servers=lambda: calls.__setitem__("mcp", calls["mcp"] + 1)),
    )
    monkeypatch.setitem(
        sys.modules,
        "agent.auxiliary_client",
        types.SimpleNamespace(shutdown_cached_clients=lambda: calls.__setitem__("aux", calls["aux"] + 1)),
    )

    # Reset the guard (the module-level flag may be left True by prior tests).
    monkeypatch.setattr(m, "_oneshot_cleanup_done", False)
    m._cleanup_oneshot_runtime()
    first = dict(calls)
    assert first == {"terminal": 1, "async": 1, "browser": 1, "mcp": 1, "aux": 1}

    m._cleanup_oneshot_runtime()
    assert calls == first  # second call is a no-op
