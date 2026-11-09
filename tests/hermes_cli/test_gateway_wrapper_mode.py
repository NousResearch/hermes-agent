"""Which Apple platform binary the launchd gateway job spawns.

``launchd_program_arguments`` can emit two shapes. The shipped one is osascript/JXA; the gated one is
``/usr/bin/python3 -c`` — another Apple platform binary, so the job keeps a Local Network identity
(#71206), but one that does not carry osascript's removable-volume denial to the child.
``_launchd_wrapper_mode`` selects between them: ``gateway.launchd_wrapper`` in config first,
``HERMES_LAUNCHD_WRAPPER`` second, osascript when neither says python3.

The three tests split the selection so every path is exercised: environment selects (1), nothing
selects and the conservative default holds (2), config outranks a contradicting environment (3).

The real ``_timestamped_stderr_gateway_command`` is patched out on purpose. It reaches
``hermes_cli._launchers.installation_command``, which reads the real ``~/.hermes/manifest.json`` and
trips ``tests/home_io_guard.py``. This file tests the wrapper choice, not the launcher.
"""
from __future__ import annotations

import plistlib

import pytest

import hermes_cli.gateway as gateway_cli
from hermes_cli import gateway_launchd
from hermes_cli.gateway_runtime_service_identity import _launchd_command


def _generated(monkeypatch, *, config: dict, wrapper_env: str | None) -> tuple[list[str], str]:
    """``(ProgramArguments, plist_text)`` the generator produces for this config+env pair.

    ``load_config`` is patched rather than written to disk: the selector imports it inside the
    function, so the attribute swap is what the code under test reads, and the test does not depend
    on the suite's sandbox config being empty.
    """
    from hermes_cli import config as hermes_config

    monkeypatch.setattr(hermes_config, "load_config", lambda: config)
    # generate_launchd_plist calls the command builder as a module global of gateway_launchd.
    monkeypatch.setattr(gateway_launchd, "_timestamped_stderr_gateway_command", lambda *a, **k: ["true"])
    if wrapper_env is None:
        monkeypatch.delenv("HERMES_LAUNCHD_WRAPPER", raising=False)
    else:
        monkeypatch.setenv("HERMES_LAUNCHD_WRAPPER", wrapper_env)
    text = gateway_cli.generate_launchd_plist()
    return plistlib.loads(text.encode("utf-8"))["ProgramArguments"], text


def test_python3_mode_emits_no_osascript(monkeypatch):
    """``HERMES_LAUNCHD_WRAPPER=python3`` replaces the JXA wrapper wholesale.

    A leftover osascript stage would put the removable-volume denial back in front of the gateway,
    which is the whole reason the mode exists.
    """
    argv, plist = _generated(monkeypatch, config={"gateway": {}}, wrapper_env="python3")

    assert argv[0] == gateway_launchd.LAUNCHD_WRAPPER == "/usr/bin/python3", argv
    assert argv[1] == "-c", argv
    assert "/usr/bin/osascript" not in plist
    assert "$.system(" not in plist


def test_default_mode_stays_osascript(monkeypatch):
    """With no config key and no environment variable the shipped shape is unchanged.

    Upstream default stays conservative: the host opts in, an install does not.
    """
    argv, plist = _generated(monkeypatch, config={"gateway": {}}, wrapper_env=None)

    assert argv[0] == "/usr/bin/osascript", argv
    assert argv[:4] == ["/usr/bin/osascript", "-l", "JavaScript", "-e"], argv
    assert "/usr/bin/python3" not in plist


def test_config_beats_env(monkeypatch):
    """Config outranks the environment, so a stale env var cannot revert a deliberate setting.

    This is the ordering the selector exists for: an env-only choice made the installer read its own
    freshly generated plist as stale, and `hermes gateway start` then reverted the wrapper.
    """
    argv, _ = _generated(
        monkeypatch,
        config={"gateway": {"launchd_wrapper": "python3"}},
        wrapper_env="osascript",
    )

    assert argv[0] == "/usr/bin/python3", argv


@pytest.mark.parametrize("mode", ["python3", "osascript"])
def test_the_installer_rebuilds_either_wrapper_shape(tmp_path, mode):
    """A live job of either shape must verify: the installer recovers its command and rebuilds byte for byte.

    Rebuilding in the shape the plist ALREADY uses is what keeps a running job verified when the config
    key is absent or reverted — otherwise the python3 job reads as ``service_identity_unverified``.
    """
    command = ["/opt/hermes/bin/hermes", "gateway", "run", "--profile", "work"]
    argv = gateway_launchd.launchd_program_arguments(
        command, tmp_path / "gateway.log", tmp_path / "gateway.error.log", mode=mode
    )

    assert _launchd_command(argv) == (command, True)


def test_a_tampered_python3_wrapper_is_refused(tmp_path):
    """An extra argument on the python3 wrapper must fail verification, not ride along.

    The rebuild is the check: only argv the generator itself would emit is accepted.
    """
    argv = gateway_launchd.launchd_program_arguments(
        ["/opt/hermes/bin/hermes", "gateway", "run"],
        tmp_path / "gateway.log",
        tmp_path / "gateway.error.log",
        mode="python3",
    )

    with pytest.raises(ValueError, match="service_identity_unverified"):
        _launchd_command([argv[0], argv[1], argv[2] + " --replace"])
