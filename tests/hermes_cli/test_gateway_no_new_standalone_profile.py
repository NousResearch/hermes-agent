"""A named profile cannot create a NEW standalone gateway — multiplexer running or not (#109417).

One gateway per host serves every profile. ``hermes -p X gateway install|start`` used to refuse only
while a live multiplexer already served X; on a host with no multiplexer running (or one that had not
rescanned yet) it wrote a brand-new per-profile unit. Now every named profile is refused without
``--force`` and pointed at ``hermes gateway install`` (default) / ``hermes gateway migrate --multiplex``;
``--force`` stays the one escape and a ``--force``-installed service stays startable without it. The
dashboard ``/api/gateway/start`` twin (``multiplexed_profile_refusal``) applies the same rule.
"""

from __future__ import annotations
from gateway import systemd_identity
from gateway import systemd_runtime

import argparse
import contextlib
import io
import json
import os

import pytest


@pytest.fixture
def quiet_host(tmp_path, monkeypatch):
    """Two named profiles under one root, no gateway running anywhere, systemd units under tmp."""
    import hermes_cli.gateway as gw
    import hermes_constants

    root = tmp_path / "hermes"
    (root / "config.yaml").parent.mkdir(parents=True)
    (root / "config.yaml").write_text("model: {default: x}\n")
    for name in ("coder", "ops"):
        (root / "profiles" / name).mkdir(parents=True)
        (root / "profiles" / name / "config.yaml").write_text("{}\n")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    monkeypatch.setattr(systemd_runtime, "supports_services", lambda: True)
    monkeypatch.setattr(gw, "_service_backend", lambda: "systemd")
    monkeypatch.setattr(gw, "refuses_container_user_scope_install", lambda system: False)
    monkeypatch.setattr(gw, "_dispatch_via_service_manager_if_s6", lambda v: False)
    monkeypatch.setattr(gw, "is_managed", lambda: False)
    monkeypatch.setattr(gw, "is_termux", lambda: False)
    calls: list[str] = []
    monkeypatch.setattr(gw, "_install_systemd_from_cli", lambda *a, **k: calls.append("install"))
    monkeypatch.setattr(gw, "_service_call", lambda backend, v, system: calls.append(v))

    def use(profile: str | None) -> None:
        home = root if profile is None else root / "profiles" / profile
        monkeypatch.setenv("HERMES_HOME", str(home))
        hermes_constants._default_hermes_root_memo = None

    return root, calls, use


def _run(fn, **ns):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), pytest.raises(SystemExit) as exc:
        fn(argparse.Namespace(system=False, all=False, run_as_user=None, **ns))
    return exc.value.code, out.getvalue()


def test_named_profile_install_and_start_refuse_without_force_when_no_multiplexer_runs(quiet_host):
    import hermes_cli.gateway as gw
    from hermes_cli.web_server_gateway import multiplexed_profile_refusal
    root, calls, use = quiet_host

    # A -> B -> A: the refusal names the right profile from each home and installs nothing.
    for profile in ("coder", "ops", "coder"):
        use(profile)
        for verb in ("install", "start"):
            code, out = _run(getattr(gw, f"_cmd_{verb}"), force=False)
            assert code == gw.GATEWAY_FATAL_CONFIG_EXIT_CODE, (profile, verb)
            assert f"Profile '{profile}' does not get a gateway of its own" in out
            assert "hermes gateway install" in out and "hermes gateway migrate --multiplex" in out
            assert f"hermes -p {profile} gateway install --force" in out
            assert "already serves" not in out  # nothing is running: this is the no-multiplexer form
        assert not systemd_identity.unit_path(system=False).exists()
        # Dashboard twin: same rule, same pointers, `stop` untouched (nothing serves the profile).
        refusal = multiplexed_profile_refusal(profile, "start")
        assert refusal and f"'{profile}'" in refusal and "migrate --multiplex" in refusal and "--force" in refusal
        assert multiplexed_profile_refusal(profile, "stop") is None
    assert calls == []

    # Control: the default profile's own install is unchanged.
    use(None)
    with contextlib.redirect_stdout(io.StringIO()):
        gw._cmd_install(argparse.Namespace(system=False, all=False, run_as_user=None, force=False))
    assert calls == ["install"]

    # Control: a live multiplexer serving the profile still prints the served-by form.
    use("coder")
    (root / "gateway.pid").write_text(json.dumps({"pid": os.getpid(), "hermes_home": str(root)}))
    (root / "gateway_state.json").write_text(json.dumps(
        {"pid": os.getpid(), "hermes_home": str(root), "gateway_state": "running",
         "served_profiles": ["default", "coder"]}))
    import gateway.status as status
    real_cmdline = status._read_process_cmdline
    status._read_process_cmdline = lambda pid: (
        "python -m hermes_cli.main gateway run" if pid == os.getpid() else real_cmdline(pid))
    try:
        code, out = _run(gw._cmd_install, force=False)
    finally:
        status._read_process_cmdline = real_cmdline
    assert code == gw.GATEWAY_FATAL_CONFIG_EXIT_CODE
    assert "The host gateway already serves profile 'coder'" in out and "gateway restart" in out


def test_force_installs_a_separate_profile_gateway_and_its_service_stays_startable(quiet_host):
    import hermes_cli.gateway as gw
    from hermes_cli.web_server_gateway import multiplexed_profile_refusal
    root, calls, use = quiet_host
    use("coder")

    with contextlib.redirect_stdout(io.StringIO()):
        gw._cmd_install(argparse.Namespace(system=False, all=False, run_as_user=None, force=True))
    assert calls == ["install"]

    # The --force-installed fleet member is not NEW: its supervisor (ExecStart carries no --force)
    # and a plain `gateway start` must keep reaching it, CLI and dashboard alike.
    unit = systemd_identity.unit_path(system=False)
    unit.parent.mkdir(parents=True)
    unit.write_text("[Service]\n")
    with contextlib.redirect_stdout(io.StringIO()):
        gw._cmd_start(argparse.Namespace(system=False, all=False, run_as_user=None, force=False))
    assert calls == ["install", "start"]
    assert multiplexed_profile_refusal("coder", "start") is None
