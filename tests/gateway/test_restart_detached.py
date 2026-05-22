"""Regression tests for ``GatewayRunner._launch_detached_restart_command``.

Covers #30342 — in minimal containers the parent gateway may trim PATH during
shutdown cleanup before the detached respawn subprocess runs ``execvp``.  A
bare ``"bash"`` argv[0] then fails with ``No such file or directory: bash``
and the gateway stays down after ``/restart``.  The fix resolves ``bash``
to an absolute path via ``shutil.which`` before spawning, mirroring the
existing ``setsid`` resolution.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

import gateway.run as gateway_run
from tests.gateway.restart_test_helpers import make_restart_runner


@pytest.mark.asyncio
async def test_launch_detached_restart_resolves_bash_to_absolute_path(monkeypatch):
    """When setsid is available, both setsid and bash are passed as absolute paths."""
    if sys.platform == "win32":
        pytest.skip("POSIX-only restart path")

    runner, _adapter = make_restart_runner()
    monkeypatch.setattr(
        gateway_run, "_resolve_hermes_bin", lambda: ["/opt/hermes/bin/hermes"]
    )

    def fake_which(name):
        return {"bash": "/usr/bin/bash", "setsid": "/usr/bin/setsid"}.get(name)

    popen_mock = MagicMock()
    with patch("shutil.which", side_effect=fake_which), patch(
        "subprocess.Popen", popen_mock
    ):
        await runner._launch_detached_restart_command()

    popen_mock.assert_called_once()
    argv = popen_mock.call_args[0][0]
    assert argv[0] == "/usr/bin/setsid"
    assert argv[1] == "/usr/bin/bash"
    assert "bash" not in {argv[0], argv[1]}, (
        f"Bare 'bash' leaked into argv: {argv}"
    )


@pytest.mark.asyncio
async def test_launch_detached_restart_uses_absolute_bash_when_setsid_missing(
    monkeypatch,
):
    """Without setsid, bash is still spawned via its absolute path."""
    if sys.platform == "win32":
        pytest.skip("POSIX-only restart path")

    runner, _adapter = make_restart_runner()
    monkeypatch.setattr(
        gateway_run, "_resolve_hermes_bin", lambda: ["/opt/hermes/bin/hermes"]
    )

    def fake_which(name):
        return {"bash": "/bin/bash"}.get(name)  # setsid → None

    popen_mock = MagicMock()
    with patch("shutil.which", side_effect=fake_which), patch(
        "subprocess.Popen", popen_mock
    ):
        await runner._launch_detached_restart_command()

    popen_mock.assert_called_once()
    argv = popen_mock.call_args[0][0]
    assert argv[0] == "/bin/bash"
    assert argv[0] != "bash"


@pytest.mark.asyncio
async def test_launch_detached_restart_logs_and_returns_when_bash_missing(
    monkeypatch, caplog
):
    """If bash is genuinely unavailable, log the error and skip the spawn."""
    if sys.platform == "win32":
        pytest.skip("POSIX-only restart path")

    runner, _adapter = make_restart_runner()
    monkeypatch.setattr(
        gateway_run, "_resolve_hermes_bin", lambda: ["/opt/hermes/bin/hermes"]
    )

    popen_mock = MagicMock()
    with patch("shutil.which", return_value=None), patch(
        "subprocess.Popen", popen_mock
    ), caplog.at_level("ERROR", logger="gateway.run"):
        await runner._launch_detached_restart_command()

    popen_mock.assert_not_called()
    assert any(
        "bash" in record.getMessage().lower() for record in caplog.records
    )
