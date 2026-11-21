"""Interactive-flow tests for the Hermes Desktop setup wizard entry.

All prompts print through the hermes_cli.setup facade (monkeypatched here);
webapp_access, mdns, caddy and netops seams are injected — no network, no
system changes. Config-writing tests use the conftest's isolated HERMES_HOME.
"""
import pytest

from hermes_cli.webapp_publish import flow


@pytest.fixture
def prompts(monkeypatch):
    """Capture output; drive prompt/prompt_choice/prompt_yes_no from queues."""
    state = {"printed": [], "prompt_answers": [], "choice_answers": [], "yesno_answers": []}

    def record(kind):
        def fn(*args, **kwargs):
            state["printed"].append((kind, " ".join(str(a) for a in args)))
        return fn

    for name in ("print_info", "print_success", "print_warning", "print_error", "print_header"):
        monkeypatch.setattr(f"hermes_cli.setup.{name}", record(name))

    def prompt(question, default="", **kwargs):
        state["printed"].append(("prompt", question))
        return state["prompt_answers"].pop(0) if state["prompt_answers"] else default

    def prompt_choice(question, choices, default=0, **kwargs):
        state["printed"].append(("choice", question))
        return state["choice_answers"].pop(0) if state["choice_answers"] else default

    def prompt_yes_no(question, default=False, **kwargs):
        state["printed"].append(("yesno", question))
        return state["yesno_answers"].pop(0) if state["yesno_answers"] else default

    monkeypatch.setattr("hermes_cli.setup.prompt", prompt)
    monkeypatch.setattr("hermes_cli.setup.prompt_choice", prompt_choice)
    monkeypatch.setattr("hermes_cli.setup.prompt_yes_no", prompt_yes_no)
    return state


@pytest.fixture
def happy_seams(monkeypatch):
    """Everything succeeds; record the webapp_access calls."""
    calls = {"register": [], "serve": [], "plan": [], "built": 0, "restarted": 0, "nous_only": 0,
             "state": [], "cleanup": []}
    monkeypatch.setattr(flow, "_webapp_supported", lambda: True)
    monkeypatch.setattr(flow, "_supported_host", lambda: "")
    monkeypatch.setattr(flow, "_switch_mode_cleanup", lambda mode: calls["cleanup"].append(mode))
    monkeypatch.setattr(flow, "_installed_blocks", lambda blocks: list(blocks))
    monkeypatch.setattr("hermes_cli.webapp_publish.state.save_state",
                        lambda home, st: calls["state"].append(st))
    monkeypatch.setattr("hermes_cli.webapp_access.nous_signed_in", lambda: True)
    monkeypatch.setattr("hermes_cli.webapp_access.access_status", lambda: {"enabled": False})
    monkeypatch.setattr("hermes_cli.webapp_access.lan_address", lambda: "192.168.1.5")

    def plan(mode, public_url=None):
        calls["plan"].append((mode, public_url))
        if mode == "lan":
            return "0.0.0.0", 9120, "", "http://192.168.1.5:9120/auth/callback"
        return "127.0.0.1", 9120, public_url, f"{public_url}/auth/callback"

    monkeypatch.setattr("hermes_cli.webapp_access.plan", plan)
    monkeypatch.setattr("hermes_cli.webapp_access.register", lambda cb: calls["register"].append(cb))
    monkeypatch.setattr("hermes_cli.webapp_access.serve",
                        lambda host, url, port=None: calls["serve"].append((host, url, port)))
    monkeypatch.setattr(flow, "_build_bundle", lambda: calls.__setitem__("built", calls["built"] + 1) or True)
    monkeypatch.setattr(flow, "_ensure_nous_only", lambda: calls.__setitem__("nous_only", calls["nous_only"] + 1))
    monkeypatch.setattr(flow, "_offer_restart", lambda: calls.__setitem__("restarted", calls["restarted"] + 1) or False)
    monkeypatch.setattr(flow, "_configure_local_redirect",
                        lambda name, ip, port: calls.setdefault("redirect", []).append((name, ip, port)) or True)
    monkeypatch.setattr("hermes_cli.webapp_publish.mdns.current_hostname", lambda conf: None)
    monkeypatch.setattr("hermes_cli.webapp_publish.mdns.resolves", lambda name, ip: True)
    return calls


# ── preflight / entry ────────────────────────────────────────────────────────

def test_preflight_refuses_without_webapp_support(prompts, monkeypatch):
    monkeypatch.setattr(flow, "_webapp_supported", lambda: False)
    flow.interactive_setup()
    assert any(kind == "print_error" for kind, _ in prompts["printed"])
    assert not prompts["choice_answers"] and not any(k == "choice" for k, _ in prompts["printed"])


def test_preflight_refuses_without_nous_signin(prompts, monkeypatch):
    monkeypatch.setattr(flow, "_webapp_supported", lambda: True)
    monkeypatch.setattr("hermes_cli.webapp_access.nous_signed_in", lambda: False)
    flow.interactive_setup()
    assert any("Nous account" in text for kind, text in prompts["printed"] if kind == "print_error")


def test_already_configured_decline_stops(prompts, happy_seams, monkeypatch):
    monkeypatch.setattr("hermes_cli.webapp_access.access_status",
                        lambda: {"enabled": True, "mode": "public", "public_url": "https://desk.example.com", "port": 9120})
    prompts["yesno_answers"] = [False]  # decline reconfigure
    flow.interactive_setup()
    assert happy_seams["plan"] == []
    assert not any(kind == "choice" for kind, _ in prompts["printed"])


def test_cancel_choice_does_nothing(prompts, happy_seams):
    prompts["choice_answers"] = [2]
    flow.interactive_setup()
    assert happy_seams["plan"] == []


def test_preflight_refuses_unsupported_host(prompts, monkeypatch):
    monkeypatch.setattr(flow, "_supported_host", lambda: "Linux")
    flow.interactive_setup()
    assert any("needs Linux" in text for kind, text in prompts["printed"] if kind == "print_error")
    assert not any(kind == "choice" for kind, _ in prompts["printed"])


def test_mode_choice_runs_cleanup_for_chosen_mode(prompts, happy_seams, monkeypatch):
    monkeypatch.setattr("hermes_cli.webapp_publish.mdns.publish_local_name",
                        lambda name, ip: {"published": True, "instructions": [], "backup": None})
    prompts["choice_answers"] = [0]
    flow.interactive_setup()
    assert happy_seams["cleanup"] == ["local"]
    assert happy_seams["state"] and happy_seams["state"][0]["mode"] == "local"


def test_local_flow_avahi_consent_declined_keeps_identity(prompts, happy_seams, monkeypatch):
    published = []
    monkeypatch.setattr("hermes_cli.webapp_publish.mdns.publish_local_name",
                        lambda name, ip: published.append(name) or {"published": True, "instructions": []})
    prompts["choice_answers"] = [0]
    prompts["yesno_answers"] = [False]  # decline the mDNS identity change
    flow.interactive_setup()
    assert published == []  # never touched avahi
    assert happy_seams["serve"] == [("0.0.0.0", "", 9120)]  # webapp still configured
    assert any("direct IP URL" in text for _, text in prompts["printed"])
    assert happy_seams["state"][0]["mdns_name"] is None


def test_local_flow_avahi_consent_given_publishes(prompts, happy_seams, monkeypatch):
    published = []
    monkeypatch.setattr("hermes_cli.webapp_publish.mdns.publish_local_name",
                        lambda name, ip: published.append(name) or {"published": True, "instructions": [], "backup": "/tmp/bak"})
    prompts["choice_answers"] = [0]
    prompts["yesno_answers"] = [True]
    flow.interactive_setup()
    assert published == ["hermes-desktop.local"]
    assert happy_seams["state"][0]["avahi_backup"] == "/tmp/bak"


# ── local mode ───────────────────────────────────────────────────────────────

def test_local_flow_happy_path(prompts, happy_seams, monkeypatch):
    published = {}
    monkeypatch.setattr("hermes_cli.webapp_publish.mdns.publish_local_name",
                        lambda name, ip: published.update(name=name, ip=ip) or {"published": True, "instructions": []})
    prompts["choice_answers"] = [0]  # local mode
    prompts["prompt_answers"] = ["hermes-desktop.local"]
    flow.interactive_setup()
    assert published == {"name": "hermes-desktop.local", "ip": "192.168.1.5"}
    assert happy_seams["plan"] == [("lan", None)]
    # Portal rejects .local callbacks: sign-in registers the LAN IP origin.
    assert happy_seams["register"] == ["http://192.168.1.5:9120/auth/callback"]
    assert happy_seams["serve"] == [("0.0.0.0", "", 9120)]
    assert happy_seams["redirect"] == [("hermes-desktop.local", "192.168.1.5", 9120)]
    assert happy_seams["built"] == 1 and happy_seams["nous_only"] == 1
    assert any("http://192.168.1.5:9120" in text for _, text in prompts["printed"])
    assert any("http://hermes-desktop.local" in text for _, text in prompts["printed"])


def test_local_flow_reprompts_invalid_hostname(prompts, happy_seams, monkeypatch):
    monkeypatch.setattr("hermes_cli.webapp_publish.mdns.publish_local_name",
                        lambda name, ip: {"published": True, "instructions": []})
    prompts["choice_answers"] = [0]
    prompts["prompt_answers"] = ["desk.example.com", "bad name.local", "desk.local"]
    flow.interactive_setup()
    assert happy_seams["register"] == ["http://192.168.1.5:9120/auth/callback"]
    assert happy_seams["redirect"] == [("desk.local", "192.168.1.5", 9120)]


def test_local_flow_no_lan_address_aborts(prompts, happy_seams, monkeypatch):
    monkeypatch.setattr("hermes_cli.webapp_access.lan_address", lambda: None)
    prompts["choice_answers"] = [0]
    flow.interactive_setup()
    assert happy_seams["plan"] == []
    assert any(kind == "print_error" for kind, _ in prompts["printed"])


def test_local_flow_mdns_failure_prints_instructions(prompts, happy_seams, monkeypatch):
    monkeypatch.setattr("hermes_cli.webapp_publish.mdns.publish_local_name",
                        lambda name, ip: {"published": False, "reason": "no root",
                                          "instructions": ["sudo systemctl restart avahi-daemon"]})
    prompts["choice_answers"] = [0]
    flow.interactive_setup()
    assert any("sudo systemctl restart avahi-daemon" in text for _, text in prompts["printed"])
    assert happy_seams["serve"] == [("0.0.0.0", "", 9120)]  # webapp still configured


def test_local_flow_registration_failure_stops(prompts, happy_seams, monkeypatch):
    from hermes_cli.webapp_access import WebappAccessError
    monkeypatch.setattr("hermes_cli.webapp_publish.mdns.publish_local_name",
                        lambda name, ip: {"published": True, "instructions": []})

    def boom(cb):
        raise WebappAccessError("portal refused")

    monkeypatch.setattr("hermes_cli.webapp_access.register", boom)
    prompts["choice_answers"] = [0]
    flow.interactive_setup()
    assert happy_seams["serve"] == [] and happy_seams["built"] == 0


# ── domain mode ──────────────────────────────────────────────────────────────

@pytest.fixture
def domain_seams(monkeypatch):
    monkeypatch.setattr("hermes_cli.webapp_publish.netops.public_ip", lambda: "203.0.113.7")
    monkeypatch.setattr("hermes_cli.webapp_publish.netops.verify_a_record",
                        lambda host, ip: (True, {"1.1.1.1": [ip]}))
    monkeypatch.setattr("hermes_cli.webapp_publish.caddy.caddy_installed", lambda: True)
    monkeypatch.setattr(flow, "_configure_caddy", lambda name, port: True)
    monkeypatch.setattr(flow, "_verify_https_gate", lambda name: None)


def test_domain_flow_happy_path(prompts, happy_seams, domain_seams):
    prompts["choice_answers"] = [1]
    prompts["prompt_answers"] = ["desk.example.com"]
    flow.interactive_setup()
    assert happy_seams["plan"] == [("public", "https://desk.example.com")]
    assert happy_seams["register"] == ["https://desk.example.com/auth/callback"]
    assert happy_seams["serve"] == [("127.0.0.1", "https://desk.example.com", 9120)]
    assert any("https://desk.example.com" in text for _, text in prompts["printed"])
    assert any("203.0.113.7" in text for _, text in prompts["printed"])  # IP shown for the A record
    assert happy_seams["cleanup"] == ["domain"]
    assert happy_seams["state"][0]["mode"] == "domain"
    assert happy_seams["state"][0]["public_url"] == "https://desk.example.com"


def test_domain_flow_dns_cancel_stops_before_registration(prompts, happy_seams, domain_seams, monkeypatch):
    monkeypatch.setattr("hermes_cli.webapp_publish.netops.verify_a_record",
                        lambda host, ip: (False, {}))
    prompts["choice_answers"] = [1, 3]  # domain mode, then Cancel in the DNS loop
    prompts["prompt_answers"] = ["desk.example.com"]
    flow.interactive_setup()
    assert happy_seams["register"] == [] and happy_seams["serve"] == []


def test_domain_flow_dns_skip_verification_proceeds_with_warning(prompts, happy_seams, domain_seams, monkeypatch):
    monkeypatch.setattr("hermes_cli.webapp_publish.netops.verify_a_record",
                        lambda host, ip: (False, {}))
    prompts["choice_answers"] = [1, 2]  # domain mode, then Skip verification
    prompts["prompt_answers"] = ["desk.example.com"]
    flow.interactive_setup()
    assert happy_seams["register"] == ["https://desk.example.com/auth/callback"]
    assert any(kind == "print_warning" for kind, _ in prompts["printed"])


def test_domain_flow_missing_caddy_nonroot_aborts(prompts, happy_seams, domain_seams, monkeypatch):
    monkeypatch.setattr("hermes_cli.webapp_publish.caddy.caddy_installed", lambda: False)
    monkeypatch.setattr(flow.os, "geteuid", lambda: 1000)
    prompts["choice_answers"] = [1]
    prompts["prompt_answers"] = ["desk.example.com"]
    flow.interactive_setup()
    assert happy_seams["plan"] == []
    assert any("apt-get install" in text for _, text in prompts["printed"])


def test_domain_flow_resolves_none_port_from_config(prompts, happy_seams, domain_seams, monkeypatch):
    from types import SimpleNamespace

    def plan(mode, public_url=None):
        return "127.0.0.1", None, public_url, f"{public_url}/auth/callback"

    monkeypatch.setattr("hermes_cli.webapp_access.plan", plan)
    monkeypatch.setattr("gateway.webapp_listener.webapp_settings", lambda home: SimpleNamespace(port=9120))
    monkeypatch.setattr("hermes_cli.webapp_access.gateway_home", lambda: "/tmp/fake-home")
    configured = []
    monkeypatch.setattr(flow, "_configure_caddy", lambda name, port: configured.append(port) or True)
    prompts["choice_answers"] = [1]
    prompts["prompt_answers"] = ["desk.example.com"]
    flow.interactive_setup()
    assert configured == [9120]
    assert happy_seams["serve"] == [("127.0.0.1", "https://desk.example.com", 9120)]


def test_domain_flow_rejects_local_hostname(prompts, happy_seams, domain_seams):
    prompts["choice_answers"] = [1]
    prompts["prompt_answers"] = ["hermes-desktop.local", "desk.example.com"]
    flow.interactive_setup()
    assert happy_seams["register"] == ["https://desk.example.com/auth/callback"]


# ── Nous-only enforcement ────────────────────────────────────────────────────

def test_nous_only_disables_basic_provider(prompts, monkeypatch):
    from hermes_cli.config import load_config, save_config
    cfg = load_config()
    cfg.setdefault("dashboard", {})["basic_auth"] = {"username": "admin", "password_hash": "x"}
    save_config(cfg)
    prompts["yesno_answers"] = [True]
    flow._ensure_nous_only()
    assert "basic" in load_config()["plugins"]["disabled"]


def test_nous_only_decline_keeps_provider(prompts, monkeypatch):
    from hermes_cli.config import load_config, save_config
    cfg = load_config()
    cfg.setdefault("dashboard", {})["basic_auth"] = {"username": "admin"}
    save_config(cfg)
    prompts["yesno_answers"] = [False]
    flow._ensure_nous_only()
    assert "basic" not in (load_config().get("plugins") or {}).get("disabled", [])


def test_nous_only_noop_without_basic_creds(prompts):
    from hermes_cli.config import load_config
    before = load_config()
    flow._ensure_nous_only()
    assert load_config() == before
    assert not any(kind == "print_warning" for kind, _ in prompts["printed"])


def test_nous_only_disables_self_hosted_oidc(prompts):
    from hermes_cli.config import load_config, save_config
    cfg = load_config()
    cfg.setdefault("dashboard", {}).setdefault("oauth", {})["self_hosted"] = {
        "issuer": "https://keycloak.example.com", "client_id": "x"}
    save_config(cfg)
    prompts["yesno_answers"] = [True]
    flow._ensure_nous_only()
    disabled = load_config()["plugins"]["disabled"]
    assert "self-hosted" in disabled and "basic" not in disabled