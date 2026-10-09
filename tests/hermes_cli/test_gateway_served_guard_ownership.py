"""The CLI served-profile guard only counts a host gateway that is OURS and NOT our own process.

``host_multiplexer_serving`` reads the HOST-wide rendezvous record, so two readers were fooled:

* #121352 — two Hermes tenants (separate ``HERMES_HOME`` roots) on one host each expose a profile
  named ``default``; tenant B's guard saw tenant A's multiplexer "serving default" and refused
  with exit 78, parking B's launchd unit. A host gateway under ANOTHER tenant root never serves us.
* #120871 — a standalone fleet member (``-p argus gateway run``, no default gateway) publishes the
  host record itself; ``named_profile_served_by_running_multiplexer`` counted the profile's OWN
  process as "a multiplexer serves you" and ``gateway restart`` refused, pointing at ``-p default``.

Control: a satellite whose own home is NOT the host record's home, while the same tenant's live
default multiplexer lists it, is still refused without ``--force``.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest

import hermes_constants
from gateway import host_attach
from gateway.host_attach import HostGateway


def _use_home(monkeypatch, fake_home: Path, hermes_home: Path) -> None:
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    monkeypatch.setattr("gateway.status._get_process_hermes_home", lambda: hermes_home)


def _publish_host(monkeypatch, home: Path, served: tuple[str, ...]) -> HostGateway:
    gateway = HostGateway(4242, home, served)
    monkeypatch.setattr(host_attach, "host_gateway_serving",
                        lambda profile, **kw: gateway if gateway.serves(profile) else None)
    return gateway


def _refusal(gw) -> tuple[bool, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        refused = gw._named_profile_refused_under_multiplexer()
    return refused, buf.getvalue()


@pytest.fixture
def gw(monkeypatch):
    from hermes_cli import gateway as gw
    monkeypatch.setattr(gw, "_is_service_installed", lambda: True)
    return gw


def test_foreign_tenants_host_gateway_never_serves_this_home(tmp_path, monkeypatch, gw):
    """#121352 (T2 vs a second tenant home): tenant A's multiplexer serves A's 'default' and 'coder';
    tenant B's default AND B's coder start beside it without --force."""
    fake_home = tmp_path / "fakehome"
    root_a = fake_home / "hermes-a" / "home"
    root_b = fake_home / "hermes-b" / "home"
    _use_home(monkeypatch, fake_home, root_b)
    _publish_host(monkeypatch, root_a, ("default", "coder"))

    assert gw.host_multiplexer_serving("default") is None
    assert gw._served_by_another_host_gateway() is None
    assert gw.named_profile_served_by_running_multiplexer("coder") is False
    refused, out = _refusal(gw)
    assert (refused, out) == (False, "")


def test_own_standalone_gateway_is_not_a_multiplexer_refusing_its_restart(tmp_path, monkeypatch, gw):
    """#120871 (T1/T6 standalone fleet): argus's own host record is ours to restart."""
    fake_home = tmp_path / "fakehome"
    own_home = fake_home / ".hermes" / "profiles" / "argus"
    _use_home(monkeypatch, fake_home, own_home)
    _publish_host(monkeypatch, own_home, ("argus",))

    assert gw._served_by_another_host_gateway() is None
    assert gw.named_profile_served_by_running_multiplexer() is False
    refused, out = _refusal(gw)
    assert (refused, out) == (False, "")
    gw._guard_named_profile_under_multiplexer()  # must not sys.exit(78)


def test_satellite_served_by_same_tenants_multiplexer_is_still_refused(tmp_path, monkeypatch, gw):
    """Control: the default multiplexer of OUR tenant lists coder -> refused, hint names the owner."""
    fake_home = tmp_path / "fakehome"
    root = fake_home / ".hermes"
    _use_home(monkeypatch, fake_home, root / "profiles" / "coder")
    _publish_host(monkeypatch, root, ("default", "coder"))

    assert gw.named_profile_served_by_running_multiplexer() is True
    refused, out = _refusal(gw)
    assert refused is True
    assert "hermes -p default gateway restart" in out
    with pytest.raises(SystemExit) as exc:
        with redirect_stdout(io.StringIO()):
            gw._guard_named_profile_under_multiplexer()
    assert exc.value.code == gw.GATEWAY_FATAL_CONFIG_EXIT_CODE


def test_cron_status_restart_hint_names_the_profile_hosting_the_scheduler(tmp_path, monkeypatch, gw):
    """#120871: `hermes -p argus cron status` on a standalone fleet points at argus, not default."""
    from gateway.host_topology import HostGatewayTopology
    from hermes_cli import cron as cron_mod

    fake_home = tmp_path / "fakehome"
    own_home = fake_home / ".hermes" / "profiles" / "argus"
    _use_home(monkeypatch, fake_home, own_home)
    _publish_host(monkeypatch, own_home, ("argus",))
    monkeypatch.setattr(cron_mod, "_active_cron_provider_name", lambda: "builtin")
    monkeypatch.setattr("profiles.current.get_active_profile_name", lambda: "argus")
    monkeypatch.setattr("gateway.host_topology.host_gateway_serving",
                        lambda name=None: HostGatewayTopology(4242, ("argus",), "host_record"))
    seen = []

    class _Stop(Exception):
        pass

    def _capture(pids, restart_command="hermes gateway restart"):
        seen.append(restart_command)
        raise _Stop

    monkeypatch.setattr(cron_mod, "_print_ticker_health", _capture)
    with pytest.raises(_Stop), redirect_stdout(io.StringIO()):
        cron_mod.cron_status()
    assert seen == ["hermes --profile argus gateway restart"]
