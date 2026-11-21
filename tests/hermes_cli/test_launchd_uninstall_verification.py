"""`hermes gateway uninstall` on launchd must verify launchd really dropped the job.

`launchd_uninstall()` booted the job out with ``check=False`` and unlinked the plist either
way (#143026). A boot-out that failed (the macOS 26+ ``EIO`` flake) is indistinguishable from
an already-unloaded job by exit code alone, so the plist disappeared while launchd still held
the job in memory: ``KeepAlive`` kept respawning the gateway, ``hermes gateway status`` read
the loaded job and reported it supervised, and nothing on disk could restore the definition.
The uninstall now re-asks launchd before unlinking, so these pins cover both directions:
a still-loaded job keeps its plist (and fails the command), a dropped one is removed as before.
"""

from types import SimpleNamespace

import pytest

import hermes_cli.gateway as gateway_cli
from hermes_cli import gateway_launchd, gateway_service_owner

LABEL = "ai.hermes.gateway"
DOMAIN = "gui/501"


@pytest.fixture()
def uninstall(tmp_path, monkeypatch):
    """The launchd uninstall path with every ``launchctl`` call recorded instead of run."""
    state = SimpleNamespace(
        plist=tmp_path / "LaunchAgents" / f"{LABEL}.plist",
        # ``print`` exits to say whether launchd still holds the job after each boot-out.
        print_codes=[],
        calls=[],
    )
    state.plist.parent.mkdir(parents=True, exist_ok=True)
    state.plist.write_text("<plist/>", encoding="utf-8")

    monkeypatch.setattr(gateway_service_owner, "refuse_foreign_launchd_plist", lambda verb: None)
    monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: state.plist)
    # The uninstall body reads the label/domain through the launchd module itself, and a domain
    # probe would burn one of the scripted ``launchctl print`` answers.
    monkeypatch.setattr(gateway_launchd, "get_launchd_label", lambda: LABEL)
    monkeypatch.setattr(gateway_launchd, "_launchd_domain", lambda: DOMAIN)
    monkeypatch.setattr(gateway_cli, "get_launchd_label", lambda: LABEL)
    monkeypatch.setattr(gateway_cli, "_launchd_domain", lambda: DOMAIN)

    def fake_run(cmd, *a, **k):
        state.calls.append(cmd)
        if cmd[:2] == ["launchctl", "print"]:
            code = state.print_codes.pop(0) if state.print_codes else 113
            return SimpleNamespace(returncode=code, stdout="", stderr="")
        return SimpleNamespace(returncode=5, stdout="", stderr="EIO")

    monkeypatch.setattr("subprocess.run", fake_run)
    # The retry pause only exists to ride out the EIO flake; the loop's shape is what is pinned.
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    return state


def _bootouts(state) -> int:
    return sum(1 for cmd in state.calls if cmd[:2] == ["launchctl", "bootout"])


def test_plist_survives_a_bootout_launchd_did_not_honour(uninstall, capsys):
    """Job still loaded after every attempt: keep the definition and fail the command."""
    uninstall.print_codes = [0, 0, 0]

    with pytest.raises(SystemExit) as exc:
        gateway_cli.launchd_uninstall()

    out = capsys.readouterr().out
    assert exc.value.code == 1
    assert uninstall.plist.exists(), "a still-loaded job must keep its definition"
    assert "Service uninstalled" not in out
    assert "still has" in out and LABEL in out
    assert "hermes gateway uninstall" in out, "the error must say how to finish the job"
    assert _bootouts(uninstall) == 3, "the EIO flake is retried, not trusted once"


def test_uninstall_removes_the_plist_once_launchd_dropped_the_job(uninstall, capsys):
    """Normal path: boot-out succeeds, launchd no longer knows the label, plist goes."""
    uninstall.print_codes = [113]

    gateway_cli.launchd_uninstall()

    out = capsys.readouterr().out
    assert not uninstall.plist.exists()
    assert "✓ Service uninstalled" in out
    assert _bootouts(uninstall) == 1


def test_a_retried_bootout_that_lands_still_uninstalls(uninstall, capsys):
    """The known EIO flake answers the first attempt; the retry drops the job."""
    uninstall.print_codes = [0, 113]

    gateway_cli.launchd_uninstall()

    assert not uninstall.plist.exists()
    assert "✓ Service uninstalled" in capsys.readouterr().out
    assert _bootouts(uninstall) == 2


def test_an_unanswerable_launchctl_does_not_block_the_uninstall(uninstall, monkeypatch, capsys):
    """No launchctl (or a timeout) must not strand the definition: nothing can supervise it."""
    def fake_run(cmd, *a, **k):
        uninstall.calls.append(cmd)
        if cmd[:2] == ["launchctl", "print"]:
            raise FileNotFoundError("launchctl")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    gateway_cli.launchd_uninstall()

    assert not uninstall.plist.exists()
    assert "✓ Service uninstalled" in capsys.readouterr().out


def test_an_already_unloaded_job_still_uninstalls(uninstall, capsys):
    """Uninstalling a service that was never loaded stays a success (no regression)."""
    uninstall.print_codes = [3]

    gateway_cli.launchd_uninstall()

    assert not uninstall.plist.exists()
    assert "✓ Service uninstalled" in capsys.readouterr().out


def test_a_missing_plist_with_a_running_job_is_reported_not_swallowed(uninstall, capsys):
    """The orphan state itself: no definition on disk, launchd still running the job."""
    uninstall.plist.unlink()
    uninstall.print_codes = [0, 0, 0]

    with pytest.raises(SystemExit):
        gateway_cli.launchd_uninstall()

    out = capsys.readouterr().out
    assert "Service uninstalled" not in out
    assert "still has" in out
