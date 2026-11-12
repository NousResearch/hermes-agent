"""A configured standalone profile may coexist, but never double-bind a served profile."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import host_attach, host_rendezvous as hr


def test_boot_notice_only_labels_configured_standalone_profiles(standalone_home, monkeypatch, caplog):
    from gateway.run import _log_standalone_profiles_at_boot
    from hermes_cli import profiles

    root, solo = standalone_home
    member = root / "profiles" / "member"
    member.mkdir()
    (member / "config.yaml").write_text("{}\n")
    monkeypatch.setattr(profiles, "profiles_to_serve", lambda *a, **kw: [("solo", solo), ("member", member)])
    runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=True), served_profile_names=lambda: ["default"])
    with caplog.at_level("INFO"):
        _log_standalone_profiles_at_boot(runner)
    messages = [r.getMessage() for r in caplog.records if "not served by this gateway" in r.getMessage()]
    assert len(messages) == 1 and "'solo'" in messages[0]

    def broken_roster(*a, **kw):
        raise OSError("unreadable roster")

    monkeypatch.setattr(profiles, "profiles_to_serve", broken_roster)
    _log_standalone_profiles_at_boot(runner)
    assert any(r.levelname == "WARNING" and "boot notice failed" in r.message for r in caplog.records)


@pytest.fixture
def standalone_home(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    home = root / "profiles" / "solo"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text("gateway:\n  standalone: true\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: root)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: root / "profiles")
    return root, home


@pytest.mark.parametrize("served,known", [(False, True), (True, True), (False, False)])
def test_standalone_attach_requires_known_unserved_profile(standalone_home, monkeypatch, served, known):
    root, home = standalone_home
    owner = host_attach.HostGateway(os.getpid() + 1, root,
                                   ("default", "solo") if served else ("default",),
                                   served_known=known)
    monkeypatch.setattr(host_attach, "host_gateway", lambda **kw: owner)
    requests = []
    monkeypatch.setattr(host_attach, "request_serve_profile", lambda *a, **kw: requests.append(a))
    decision = host_attach.decide(home)
    if known and not served:
        assert decision.outcome == host_attach.START
    else:
        assert decision.outcome == host_attach.REFUSE
        assert decision.transient
        if served:
            assert "rescan-profiles" in decision.message
    assert requests == [], "an opted-out profile must never ask the host to serve it"


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("served,known", [(False, True), (True, True), (False, False)])
def test_standalone_lock_loser_requires_known_unserved_profile(
    standalone_home, monkeypatch, capsys, served, known,
):
    import fcntl
    from gateway import run

    root, home = standalone_home
    owner = host_attach.HostGateway(os.getpid() + 1, root,
                                   ("default", "solo") if served else ("default",),
                                   served_known=known)
    monkeypatch.setattr(host_attach, "host_gateway", lambda **kw: owner)
    monkeypatch.setattr(run, "get_hermes_home", lambda: home)
    monkeypatch.setattr(host_attach, "request_serve_profile", lambda *a, **kw: None)
    hr.ensure_host_state_dir()
    with open(hr.lock_path(hr.ROLE_GATEWAY), "a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if known and not served:
            run._claim_host_gateway_role()
        else:
            with pytest.raises(SystemExit) as exc:
                run._claim_host_gateway_role()
            assert exc.value.code == 75
            if served:
                assert "rescan-profiles" in capsys.readouterr().out


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("host_name", ["default", "member"])
def test_standalone_owner_cannot_hide_a_live_multiplexer(standalone_home, monkeypatch, host_name):
    """A starts first, host starts beside A, then B opts out before the host rescans."""
    import fcntl
    from gateway import run, status

    root, home = standalone_home
    first = root / "profiles" / "first"
    first.mkdir()
    (first / "config.yaml").write_text("gateway:\n  standalone: true\n")
    host_home = root if host_name == "default" else root / "profiles" / host_name
    host_home.mkdir(exist_ok=True)
    (host_home / "config.yaml").write_text("{}\n")
    owner = host_attach.HostGateway(os.getpid() + 1, first, ("first",), standalone=True)
    host_pid = os.getpid() + 2
    monkeypatch.setattr(host_attach, "host_gateway", lambda **kw: owner)
    monkeypatch.setattr(status, "live_gateway_pid_for_home",
                        lambda h: host_pid if Path(h) == host_home else None)
    identity = {"pid": host_pid, "hermes_home": str(host_home), "profile": host_name,
                "multiplex": True, "served_profiles": ["default", "solo", host_name]}
    monkeypatch.setattr(host_attach, "_identify", lambda h: identity if h == host_home else None)
    monkeypatch.setattr(run, "get_hermes_home", lambda: home)
    hr.ensure_host_state_dir()
    with open(hr.lock_path(hr.ROLE_GATEWAY), "a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        for answer in (identity, None, {**identity, "pid": host_pid + 1}):
            monkeypatch.setattr(host_attach, "_identify", lambda h: answer)
            decision = host_attach.decide(home)
            assert decision.outcome == host_attach.REFUSE
            assert decision.transient
            with pytest.raises(SystemExit) as exc:
                run._claim_host_gateway_role()
            assert exc.value.code == 75
            run._claim_host_gateway_role(force=True)
            # The original lock owner may exit without stopping the multiplexer.
            with monkeypatch.context() as m:
                m.setattr(host_attach, "host_gateway", lambda **kw: None)
                assert host_attach.decide(home).outcome == host_attach.REFUSE
        # Once the live host confirms removal, both entry points permit coexistence.
        identity["served_profiles"] = [host_name]
        monkeypatch.setattr(host_attach, "_identify", lambda h: identity)
        assert host_attach.decide(home).outcome == host_attach.START
        run._claim_host_gateway_role()
