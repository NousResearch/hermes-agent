"""Both concurrent-instance gates must exempt exactly the same process set.

``update_cmd._classify_concurrent_instance`` (via
``_filter_non_gateway_concurrent_instances``) and
``update_cmd_windows._leftover_pausable_gateway_pids`` answer the same question
from two places: *will the updater's own pause machinery stop this live
process, so the gate may let the update proceed?*

The answer is ``gateway.status.looks_like_gateway_runtime_command_line``,
because that is exactly what ``_pause_windows_gateways_for_update`` reaches:
its discovery goes through ``hermes_cli.gateway.find_gateway_pids``, which
scans with ``include_restart_managers=True`` on Windows (no systemd), i.e.
``gateway run`` OR ``gateway restart``, across every launcher shape —
``hermes-gateway.exe``, ``gateway/run.py``, bare ``hermes gateway``.

``_scan_venv_blockers._is_pausable_gateway`` is a DIFFERENT, smaller set (a
``hermes_cli.main``-tail parser used for the venv-scan exemption). Pointing
either gate at it exempts fewer processes than the pause machinery stops at
one site and not the other, which is how the two drifted apart. These tests
pin agreement so a swap at either call site fails here.
"""

from __future__ import annotations

import sys

import pytest

import hermes_cli.update_cmd as update_cmd
import hermes_cli.update_cmd_windows as update_cmd_windows


class _FakeProcess:
    def __init__(self, argv: list[str]) -> None:
        self._argv = argv

    def cmdline(self) -> list[str]:
        return list(self._argv)


class _FakePsutil:
    """Minimal stand-in: both call sites only ever ask for ``Process(pid).cmdline()``."""

    def __init__(self, argv: list[str]) -> None:
        self._argv = argv

    def Process(self, pid):  # noqa: N802 - mirrors the psutil API name
        return _FakeProcess(self._argv)


def _install_fake_psutil(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    fake = _FakePsutil(argv)
    # update_cmd imports psutil inside the function; update_cmd_windows goes
    # through its own _psutil() accessor.
    monkeypatch.setitem(sys.modules, "psutil", fake)
    monkeypatch.setattr(update_cmd_windows, "_psutil", lambda: fake)


# (argv, exempt-because-the-pause-machinery-owns-it)
CASES = [
    # Launcher shapes the Windows pause discovery finds.
    (["C:\\H\\venv\\Scripts\\python.exe", "-m", "hermes_cli.main", "gateway", "run"], True),
    (["C:\\H\\venv\\Scripts\\hermes.exe", "gateway", "run"], True),
    (["C:\\H\\venv\\Scripts\\hermes-gateway.exe"], True),
    (["C:\\H\\venv\\Scripts\\python.exe", "gateway/run.py"], True),
    (["C:\\H\\venv\\Scripts\\hermes.exe", "gateway"], True),  # bare: defaults to run
    (["hermes.exe", "--profile", "work", "gateway", "run"], True),
    # `gateway restart` runs the runtime in-process when there is no service
    # manager, which is the Windows case — include_restart_managers=True.
    (["C:\\H\\venv\\Scripts\\python.exe", "-m", "hermes_cli.main", "gateway", "restart"], True),
    # Never exempt: management subcommands, an operator REPL, a serve backend.
    (["C:\\H\\venv\\Scripts\\hermes.exe", "gateway", "status"], False),
    (["C:\\H\\venv\\Scripts\\hermes.exe", "gateway", "stop"], False),
    (["C:\\H\\venv\\Scripts\\hermes.exe", "dashboard"], False),
    (["C:\\H\\venv\\Scripts\\python.exe", "-m", "hermes_cli.main"], False),
    (["C:\\H\\venv\\Scripts\\python.exe", "-m", "hermes_cli.main", "serve", "--port", "9119"], False),
]


@pytest.mark.parametrize("argv,exempt", CASES, ids=[" ".join(c[0][1:]) or "bare-launcher" for c in CASES])
def test_both_gates_classify_the_same_cmdline_identically(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], exempt: bool
) -> None:
    _install_fake_psutil(monkeypatch, argv)

    cross_platform = update_cmd._classify_concurrent_instance(4242) == "gateway"
    windows = update_cmd_windows._leftover_pausable_gateway_pids(
        [(4242, "python.exe", " ".join(argv))]
    ) == [4242]

    assert cross_platform == windows, (
        "the concurrent-instance gate and the Windows venv-holder gate "
        f"disagree about {argv!r}: exempt-as-gateway is {cross_platform} in "
        f"update_cmd._classify_concurrent_instance and {windows} in "
        "update_cmd_windows._leftover_pausable_gateway_pids. One of them is "
        "letting a live venv holder past with nothing downstream to stop it."
    )
    assert cross_platform is exempt


def test_argv_with_spaces_is_not_re_split_by_a_lossy_join(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Windows install path with a space must not manufacture argv tokens.

    Both gates join live argv with ``subprocess.list2cmdline``, which quotes
    the token; the matcher re-splits quote-aware and strips the quotes back
    off. A plain ``" ".join`` turns ``C:\\Program Files\\...\\python.exe`` into
    two tokens and shifts every following one.
    """
    argv = [
        "C:\\Program Files\\Hermes\\venv\\Scripts\\python.exe",
        "-m",
        "hermes_cli.main",
        "gateway",
        "run",
    ]
    _install_fake_psutil(monkeypatch, argv)

    assert update_cmd._classify_concurrent_instance(4242) == "gateway"
    assert update_cmd_windows._leftover_pausable_gateway_pids(
        [(4242, "python.exe", "scan-prefix-only")]
    ) == [4242]


def test_unreadable_identity_stays_blocking(monkeypatch: pytest.MonkeyPatch) -> None:
    """psutil failure is "unknown", and unknown is not exempt at either gate."""

    class _Exploding:
        def Process(self, pid):  # noqa: N802
            raise RuntimeError("access denied")

    monkeypatch.setitem(sys.modules, "psutil", _Exploding())
    monkeypatch.setattr(update_cmd_windows, "_psutil", lambda: _Exploding())

    assert update_cmd._classify_concurrent_instance(4242) == "unknown"
    assert update_cmd._filter_non_gateway_concurrent_instances([(4242, "python.exe")]) == [
        (4242, "python.exe")
    ]
    # The Windows gate falls back to the scanned cmdline, which is not a gateway.
    assert (
        update_cmd_windows._leftover_pausable_gateway_pids([(4242, "python.exe", "python.exe -m pip")])
        is None
    )
