"""UX and contract tests for ``secret_source_blocker`` (issue #159818)."""

from __future__ import annotations

from pathlib import Path

import pytest

from gateway.webapp_listener import _public_url as accepted_public_url
from hermes_cli import gateway_migrate_backends_plan as plan_mod
from hermes_cli.web_server_boundary import public_hosts_engage_gate, public_url_hosts
from tests.hermes_cli.test_gateway_migrate_backends_apply import FakeHost

_GUIDE = Path(__file__).resolve().parents[3] / "website" / "docs" / "user-guide" / "features" / "web-dashboard.md"


def _write_config(home: Path, body: str) -> None:
    (home / "config.yaml").write_text("model:\n  default: fake-model\n" + body, encoding="utf-8")


def _host_on(home: Path, *, bind: str = "0.0.0.0", port: int = 9119, supervisor: str = "systemd") -> FakeHost:
    # FakeHost._reaches() does not alias "localhost" to probe_host("localhost") == "127.0.0.1".
    host = FakeHost(home, host="127.0.0.1" if bind == "localhost" else bind, supervisor=supervisor)
    host.argv = [
        "/venv/bin/python", "-m", "hermes_cli.main", "dashboard", "--no-open",
        "--host", bind, "--port", str(port),
    ]
    return host


def test_secret_source_blocker_explains_why_and_local_lan_options(home):
    """Issue #159818 repro: LAN/Desktop-only user with an enabled external secret source gets a clear
    explanation of what webapp.public_url is, why Hermes asks, and a valid loopback example."""
    _write_config(home, "secrets:\n  bitwarden:\n    enabled: true\n")
    plan = plan_mod.build_plan(_host_on(home, bind="0.0.0.0", port=9119))

    assert plan.exit_code() == plan_mod.EXIT_BLOCKED
    assert len(plan.blockers) == 1
    blocker = plan.blockers[0]
    assert "external secret source is enabled" in blocker
    assert "HERMES_DASHBOARD_PUBLIC_URL" in blocker
    assert "never reads vault values" in blocker
    assert "browser-facing" in blocker
    assert "http:// or https://" in blocker
    assert "sign-in callbacks" in blocker
    assert "Host/Origin" in blocker
    assert "internet-routable" in blocker
    assert "LAN" in blocker
    assert "Desktop-only" in blocker
    assert "http://127.0.0.1:9119" in blocker
    assert "hermes -p default config set webapp.public_url http://127.0.0.1:9119" in blocker
    assert blocker.endswith("Nothing changed.")


@pytest.mark.parametrize(("bind", "port", "expected_url", "expected_arg"), [
    ("0.0.0.0", 9119, "http://127.0.0.1:9119", "http://127.0.0.1:9119"),
    ("127.0.0.1", 8443, "http://127.0.0.1:8443", "http://127.0.0.1:8443"),
    ("192.168.1.50", 9120, "http://127.0.0.1:9120", "http://127.0.0.1:9120"),
    ("localhost", 9119, "http://localhost:9119", "http://localhost:9119"),
    ("LOCALHOST", 8080, "http://localhost:8080", "http://localhost:8080"),
    ("::", 9119, "http://[::1]:9119", "'http://[::1]:9119'"),
    ("::1", 8080, "http://[::1]:8080", "'http://[::1]:8080'"),
])
def test_suggested_loopback_url_matches_backend_and_unblocks_migration(
    home, bind, port, expected_url, expected_arg,
):
    """The loopback URL the blocker suggests is valid for webapp.public_url, does not engage the auth
    gate on loopback, and unblocks build_plan once configured."""
    _write_config(home, "secrets:\n  vault:\n    enabled: true\n")
    blocked = plan_mod.build_plan(_host_on(home, bind=bind, port=port))

    assert blocked.exit_code() == plan_mod.EXIT_BLOCKED
    assert expected_url in blocked.blockers[0]
    assert f"config set webapp.public_url {expected_arg}" in blocked.blockers[0]
    if bind.lower() == "localhost":
        assert "127.0.0.1" not in blocked.blockers[0]
    assert accepted_public_url(expected_url) == expected_url
    assert public_hosts_engage_gate(public_url_hosts(expected_url)) is False

    _write_config(home, f"secrets:\n  vault:\n    enabled: true\nwebapp:\n  public_url: {expected_url}\n")
    unblocked = plan_mod.build_plan(_host_on(home, bind=bind, port=port))

    assert unblocked.exit_code() == plan_mod.EXIT_OK
    assert unblocked.blockers == []


def test_secret_source_blocker_does_not_suggest_config_set_on_nix(home):
    """Nix-managed installs refuse `hermes config set`, so the blocker points at Nix configuration."""
    _write_config(home, "secrets:\n  vault:\n    enabled: true\n")
    plan = plan_mod.build_plan(_host_on(home, bind="0.0.0.0", port=9119, supervisor="nix-systemd"))

    assert plan.exit_code() == plan_mod.EXIT_BLOCKED
    assert len(plan.blockers) == 1
    blocker = plan.blockers[0]
    assert "external secret source is enabled" in blocker
    assert "Nix configuration" in blocker
    assert "http://127.0.0.1:9119" in blocker
    assert "config set" not in blocker


@pytest.mark.parametrize("marker", ["nixos", "home-manager", ""])
def test_secret_source_blocker_uses_nix_guidance_for_managed_home(home, monkeypatch, marker):
    """A .managed home (even with a manual backend) refuses `hermes config set`, so point at Nix configuration."""
    monkeypatch.delenv("HERMES_MANAGED", raising=False)
    (home / ".managed").write_text(marker, encoding="utf-8")
    _write_config(home, "secrets:\n  vault:\n    enabled: true\n")

    plan = plan_mod.build_plan(_host_on(home, bind="0.0.0.0", port=9119, supervisor="manual"))
    assert plan.exit_code() == plan_mod.EXIT_BLOCKED
    assert len(plan.blockers) == 1
    blocker = plan.blockers[0]
    assert "Nix configuration" in blocker
    assert "http://127.0.0.1:9119" in blocker
    assert "config set" not in blocker

    direct = plan_mod.secret_source_blocker(FakeHost(home))
    assert direct is not None
    assert "Nix configuration" in direct
    assert "config set" not in direct


def test_secret_source_blocker_respects_other_managed_and_opt_out_markers(home, monkeypatch):
    """Non-Nix managed markers also suppress `config set`; explicit opt-out (`false`) keeps `config set`."""
    monkeypatch.delenv("HERMES_MANAGED", raising=False)
    _write_config(home, "secrets:\n  vault:\n    enabled: true\n")

    (home / ".managed").write_text("apt", encoding="utf-8")
    managed_plan = plan_mod.build_plan(_host_on(home, bind="0.0.0.0", port=9119, supervisor="manual"))
    assert managed_plan.exit_code() == plan_mod.EXIT_BLOCKED
    assert "apt configuration" in managed_plan.blockers[0]
    assert "config set" not in managed_plan.blockers[0]

    (home / ".managed").write_text("false", encoding="utf-8")
    unmanaged_plan = plan_mod.build_plan(_host_on(home, bind="0.0.0.0", port=9119, supervisor="manual"))
    assert unmanaged_plan.exit_code() == plan_mod.EXIT_BLOCKED
    assert "config set webapp.public_url http://127.0.0.1:9119" in unmanaged_plan.blockers[0]


def test_secret_source_blocker_uses_kept_webapp_port_with_keep_webapp_address(home):
    """With --keep-webapp-address, the loopback example uses the already-enabled Webapp's port."""
    _write_config(
        home,
        "secrets:\n  vault:\n    enabled: true\nwebapp:\n  enabled: true\n  host: 127.0.0.1\n  port: 9200\n",
    )
    plan = plan_mod.build_plan(_host_on(home, bind="0.0.0.0", port=9119), keep_webapp=True)

    assert plan.exit_code() == plan_mod.EXIT_BLOCKED
    assert "http://127.0.0.1:9200" in plan.blockers[0]
    assert "http://127.0.0.1:9119" not in plan.blockers[0]


def test_secret_source_blocker_names_no_port_while_the_webapp_is_enabled_elsewhere(home):
    """The Webapp already on 9200 blocks a move onto the old backend's 9119 until --keep-webapp-address keeps
    it on 9200, so an example on 9119 would leave webapp.public_url on the port that re-run retires. With no
    port known it prints no command either: saved as is, the placeholder would keep the Webapp off."""
    _write_config(
        home,
        "secrets:\n  vault:\n    enabled: true\nwebapp:\n  enabled: true\n  host: 127.0.0.1\n  port: 9200\n",
    )
    plan = plan_mod.build_plan(_host_on(home, bind="0.0.0.0", port=9119))

    blocker = next(b for b in plan.blockers if "external secret source" in b)
    assert "http://127.0.0.1:<port>" in blocker
    assert "9119" not in blocker
    assert "config set" not in blocker


def test_secret_source_blocker_ignores_keep_webapp_address_on_a_by_hand_move(home):
    """A by-hand move refuses --keep-webapp-address, so its example never names the kept Webapp's port."""
    _write_config(
        home,
        "secrets:\n  vault:\n    enabled: true\nwebapp:\n  enabled: true\n  host: 127.0.0.1\n  port: 9200\n",
    )
    plan = plan_mod.build_plan(_host_on(home, bind="0.0.0.0", port=9119, supervisor="nix-systemd"), keep_webapp=True)

    blocker = next(b for b in plan.blockers if "external secret source" in b)
    assert "http://127.0.0.1:<port>" in blocker
    assert "9200" not in blocker
    assert "9119" not in blocker


def test_secret_source_blocker_direct_call_without_address_uses_port_placeholder(home):
    """Direct 1-arg callers stay compatible and never guess port 9119 or print a placeholder command."""
    _write_config(home, "secrets:\n  vault:\n    enabled: true\n")
    blocker = plan_mod.secret_source_blocker(FakeHost(home))

    assert blocker is not None
    assert "external secret source is enabled" in blocker
    assert "http://127.0.0.1:<port>" in blocker
    assert "http://127.0.0.1:9119" not in blocker
    assert "config set" not in blocker


def test_secret_source_blocker_keeps_fail_closed_and_url_validation_invariants(home):
    """Control checks: disabled secret sources never block; bare IP without scheme is still rejected;
    valid LAN and HTTPS URLs unblock the plan."""
    _write_config(home, "secrets:\n  vault:\n    enabled: false\n")
    assert plan_mod.secret_source_blocker(FakeHost(home)) is None

    _write_config(home, "secrets:\n  vault:\n    enabled: true\nwebapp:\n  public_url: 127.0.0.1\n")
    bare_ip = plan_mod.build_plan(_host_on(home, bind="0.0.0.0", port=9119))
    assert bare_ip.exit_code() == plan_mod.EXIT_BLOCKED
    assert any("webapp.public_url must be an http:// or https://" in b for b in bare_ip.blockers)

    for valid_url in ("http://mybox.local:9119", "https://hermes.example.com"):
        _write_config(home, f"secrets:\n  vault:\n    enabled: true\nwebapp:\n  public_url: {valid_url}\n")
        ok = plan_mod.build_plan(_host_on(home, bind="0.0.0.0", port=9119))
        assert ok.exit_code() == plan_mod.EXIT_OK, valid_url
        assert ok.blockers == [], valid_url


def test_migration_docs_explain_secret_source_public_url_guidance():
    text = " ".join(_GUIDE.read_text(encoding="utf-8-sig").split())
    assert "when an external secret source is enabled under `secrets:`" in text
    assert "the planner never reads vault values" in text
    assert "without exposing anything to the internet" in text
