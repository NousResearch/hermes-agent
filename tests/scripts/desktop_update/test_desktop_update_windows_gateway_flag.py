"""The Windows hand-off's `--gateway` flag, source-level (Linux CI cannot run PowerShell).

`hermes update --gateway` (re)starts the local messaging gateway after the
update. A Desktop served by a remote gateway (#117529) must not ask for that:
the restarted local gateway shares the remote host's channel credentials and
becomes a competing long-poll consumer — Telegram answers the conflict by
rejecting one of the two `getUpdates` callers, taking the production bot
offline. The Desktop passes `-NoGateway` on that path; these tests pin the
script side of the contract the same source-level way
`test_desktop_update_windows_python_handoff.py` guards its invocation rule.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
WINDOWS_PS1 = REPO_ROOT / "scripts" / "desktop-update" / "windows.ps1"


def _handoff_source() -> str:
    """The script with its ``-SelfTest*`` fixture blocks removed (same strip as
    test_desktop_update_windows_python_handoff.py), normalized to LF."""
    source = WINDOWS_PS1.read_text(encoding="utf-8").replace("\r\n", "\n")
    return re.sub(
        r"\n(?P<indent> *)if \(\$SelfTest\w+\) \{.*?\n(?P=indent)\}\n",
        "\n",
        source,
    )


def test_no_gateway_switch_is_declared() -> None:
    assert re.search(r"^\s{4}\[switch\]\$NoGateway,\s*$", _handoff_source(), re.M), (
        "scripts/desktop-update/windows.ps1 must declare [switch]$NoGateway so "
        "a remote-served Desktop (#117529) can opt out of the local gateway "
        "restart."
    )


def test_gateway_flag_is_conditional_not_inline() -> None:
    """`--gateway` may only reach the update argv through $gatewayArg.

    An inline literal would mean someone reintroduced an unconditional local
    gateway restart — the exact regression (#117529) this guards against.
    """
    source = _handoff_source()

    gateway_literals = [line for line in source.splitlines() if '"--gateway"' in line]
    assert len(gateway_literals) == 1, (
        "Expected exactly one \"--gateway\" literal in windows.ps1 (the "
        f"$gatewayArg default); found: {gateway_literals}"
    )
    assert "$gatewayArg = @(\"--gateway\")" in gateway_literals[0]

    assert re.search(r"if \(\$NoGateway\)\s*\{\s*\n\s*\$gatewayArg = @\(\)", source), (
        "-NoGateway must empty $gatewayArg before the update argv is assembled."
    )
    assert "$gatewayArg + @(\"--force\"" in source, (
        "The update argv must be assembled from $gatewayArg so -NoGateway "
        "actually removes --gateway from the invocation."
    )


def test_successful_local_update_restarts_all_gateways_after_verification() -> None:
    """Desktop stops every profile before hand-off, outside update's inventory.

    A successful local hand-off must consequently restore the same all-profile
    set itself. The restart belongs after runtime verification, and a
    remote-served Desktop must retain its ``-NoGateway`` opt-out.
    """
    source = _handoff_source()
    verify = 'Invoke-HermesStep $pythonExe @("-c", $verifyCode) "verify"'
    restart = (
        'Invoke-HermesStep $pythonExe @("-m", "hermes_cli.main", '
        '"gateway", "start", "--all") "gateway restart"'
    )

    assert verify in source
    assert restart in source, (
        "a verified successful Desktop update must restore every gateway "
        "that Desktop stopped before the hand-off"
    )
    assert source.index(verify) < source.index(restart), (
        "gateway restoration must not run before the update runtime verifies"
    )

    restart_block = source[source.index(restart) - 240:source.index(restart) + len(restart)]
    assert "-not $NoGateway" in restart_block, (
        "-NoGateway must keep remote-served Desktop from starting a local "
        "messaging gateway"
    )
