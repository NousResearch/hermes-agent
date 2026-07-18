from __future__ import annotations

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import yaml

from gateway.config import PlatformConfig
from plugins.platforms.a2a import adapter, auth, cli, client_state, config, setup


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_setup_preserves_unrelated_config_and_pins_empty_toolsets(hermes_home):
    existing = {
        "model": "example/model",
        "platforms": {"telegram": {"enabled": True}},
        "platform_toolsets": {"telegram": ["web"]},
    }
    (hermes_home / "config.yaml").write_text(yaml.safe_dump(existing), encoding="utf-8")

    setup.ensure_a2a_platform_config(public_url="https://agent.example.test/a2a")

    saved = yaml.safe_load((hermes_home / "config.yaml").read_text(encoding="utf-8"))
    assert saved["model"] == "example/model"
    assert saved["platforms"]["telegram"] == {"enabled": True}
    assert saved["platform_toolsets"]["telegram"] == ["web"]
    assert saved["platform_toolsets"]["a2a"] == []
    assert saved["platforms"]["a2a"]["enabled"] is True
    assert saved["platforms"]["a2a"]["extra"]["public_url"] == "https://agent.example.test/a2a"


def test_setup_resets_explicit_a2a_toolsets_to_empty_without_clobbering_others(hermes_home):
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"platform_toolsets": {"a2a": ["web"], "telegram": ["web"]}}),
        encoding="utf-8",
    )

    setup.ensure_a2a_platform_config()

    saved = yaml.safe_load((hermes_home / "config.yaml").read_text(encoding="utf-8"))
    assert saved["platform_toolsets"]["a2a"] == []
    assert saved["platform_toolsets"]["telegram"] == ["web"]


def test_principal_mapping_is_server_owned_and_contains_no_token(hermes_home):
    token = setup.add_principal("laptop", profile="reviewer")
    raw = (hermes_home / "config.yaml").read_text(encoding="utf-8")
    cfg = config.load_a2a_settings()

    assert token not in raw
    assert cfg.principals["laptop"] == {
        "credential_ref": "inbound:laptop",
        "profile": "reviewer",
    }


def test_named_peer_stores_only_credential_reference_in_config(hermes_home):
    token = "peer-token-with-more-than-thirty-two-characters"
    setup.add_peer("norbert", url="https://norbert.example.test/a2a", token=token)

    raw = (hermes_home / "config.yaml").read_text(encoding="utf-8")
    cfg = config.load_a2a_settings()
    assert token not in raw
    assert cfg.peers["norbert"]["credential_ref"] == "outbound:norbert"
    assert cfg.peers["norbert"]["url"] == "https://norbert.example.test/a2a"
    assert len(cfg.peers["norbert"]["generation"]) >= 24


def test_duplicate_principal_is_rejected_without_rotating_working_credential(hermes_home):
    original = setup.add_principal("laptop", profile="reviewer")

    with pytest.raises(ValueError, match="already exists; use credential rotate"):
        setup.add_principal("laptop", profile="other")

    assert auth.verify_inbound_token("inbound:laptop", original)
    assert config.load_a2a_settings().principals["laptop"]["profile"] == "reviewer"


def test_duplicate_peer_is_rejected_without_replacing_working_credential(hermes_home):
    original = "original-peer-token-with-more-than-thirty-two-characters"
    setup.add_peer("norbert", url="https://norbert.example.test/a2a", token=original)

    with pytest.raises(ValueError, match="already exists; remove it before adding"):
        setup.add_peer(
            "norbert",
            url="https://replacement.example.test/a2a",
            token="replacement-peer-token-with-more-than-thirty-two-characters",
        )

    assert auth.load_outbound_token("outbound:norbert") == original
    assert config.load_a2a_settings().peers["norbert"]["url"] == "https://norbert.example.test/a2a"


@pytest.mark.parametrize("kind", ["principal", "peer"])
def test_new_name_config_save_failure_leaves_no_new_credential(hermes_home, monkeypatch, kind):
    original_update = config.update_a2a_config
    calls = 0

    def fail_second_update(mutator):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated config failure")
        return original_update(mutator)

    monkeypatch.setattr(config, "update_a2a_config", fail_second_update)

    with pytest.raises(OSError, match="simulated config failure"):
        if kind == "principal":
            setup.add_principal("new-principal", profile="reviewer")
        else:
            setup.add_peer(
                "new-peer",
                url="https://new.example.test/a2a",
                token="new-peer-token-with-more-than-thirty-two-characters",
            )

    summary = auth.credential_summary()
    assert f"inbound:new-principal" not in summary["inbound"]
    assert f"outbound:new-peer" not in summary["outbound"]


@pytest.mark.parametrize(
    "url",
    [
        "http://example.test/a2a",
        "file:///tmp/socket",
        "https://user:pass@example.test/a2a",
        "https://example.test/a2a#fragment",
        "https://example.test/a2a?token=nope",
        "https://example.test:notaport/a2a",
    ],
)
def test_peer_url_rejects_unsafe_shapes(url):
    with pytest.raises(ValueError):
        config.validate_peer_url(url)


def test_peer_url_allows_https_and_loopback_http():
    assert config.validate_peer_url("https://agent.example.test/a2a") == "https://agent.example.test/a2a"
    assert config.validate_peer_url("http://127.0.0.1:9999") == "http://127.0.0.1:9999"
    assert config.validate_peer_url("http://localhost:9999") == "http://localhost:9999"


def test_settings_repr_does_not_include_unknown_secret_fields(hermes_home):
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "platforms": {
                    "a2a": {
                        "enabled": True,
                        "extra": {
                            "token": "must-not-appear",
                            "principals": {
                                "laptop": {
                                    "credential_ref": "inbound:laptop",
                                    "profile": "reviewer",
                                    "token": "nested-principal-secret",
                                }
                            },
                            "peers": {
                                "norbert": {
                                    "credential_ref": "outbound:norbert",
                                    "url": "https://norbert.example.test/a2a",
                                    "token": "nested-peer-secret",
                                }
                            },
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    assert "must-not-appear" not in repr(config.load_a2a_settings())
    assert "nested-principal-secret" not in repr(config.load_a2a_settings())
    assert "nested-peer-secret" not in repr(config.load_a2a_settings())


def test_remove_peer_and_principal_delete_config_and_credentials(hermes_home):
    setup.add_principal("laptop", profile="main")
    setup.add_peer(
        "norbert",
        url="https://norbert.example.test/a2a",
        token="peer-token-with-more-than-thirty-two-characters",
    )

    assert setup.remove_principal("laptop")
    assert setup.remove_peer("norbert")
    cfg = config.load_a2a_settings()
    assert cfg.principals == {}
    assert cfg.peers == {}


def test_plugin_registers_platform_and_local_cli():
    calls = {"platform": [], "cli": [], "skill": []}

    class Context:
        def register_platform(self, **kwargs):
            calls["platform"].append(kwargs)

        def register_cli_command(self, **kwargs):
            calls["cli"].append(kwargs)

        def register_skill(self, *args):
            calls["skill"].append(args)

    adapter.register(Context())

    assert calls["platform"][0]["name"] == "a2a"
    assert "hermes-agent[a2a]" in calls["platform"][0]["install_hint"]
    assert calls["platform"][0]["agent_tool_policy"] == "explicit"
    assert calls["cli"][0]["name"] == "a2a"
    assert calls["skill"][0][0] == "a2a-peer"
    assert calls["skill"][0][1].is_file()
    assert "no tools" in calls["platform"][0]["platform_hint"].lower()


def test_cli_exposes_configuration_and_named_peer_commands():
    parser = argparse.ArgumentParser()
    cli.register_cli(parser)

    assert parser.parse_args(["status"]).a2a_command == "status"
    assert parser.parse_args(["setup"]).a2a_command == "setup"
    assert parser.parse_args(["peer", "list"]).peer_command == "list"
    assert parser.parse_args(["principal", "list"]).principal_command == "list"
    assert parser.parse_args(["credential", "rotate", "laptop"]).credential_command == "rotate"
    assert parser.parse_args(["card", "norbert"]).peer == "norbert"
    ask = parser.parse_args(["ask", "norbert", "hello", "--new-context", "--json"])
    assert ask.peer == "norbert"
    assert ask.new_context is True
    assert ask.json is True
    assert parser.parse_args(["get", "norbert", "task-1"]).task_id == "task-1"
    assert parser.parse_args(["list", "norbert"]).peer == "norbert"
    assert parser.parse_args(["cancel", "norbert", "task-1"]).task_id == "task-1"


def test_missing_sdk_and_unconfigured_adapter_fail_closed(monkeypatch):
    monkeypatch.setattr(adapter, "A2A_SDK_AVAILABLE", False)
    assert adapter.check_requirements() is False

    cfg = PlatformConfig(enabled=True, extra={})
    assert adapter.validate_config(cfg) is False


def test_adapter_rejects_principal_mapping_without_matching_hash(hermes_home, monkeypatch):
    monkeypatch.setattr(adapter, "_current_profile_name", lambda: "default")
    setup.ensure_a2a_platform_config(public_url="https://agent.example.test/a2a")
    cfg = PlatformConfig(
        enabled=True,
        extra={
            "public_url": "https://agent.example.test/a2a",
            "principals": {
                "laptop": {
                    "credential_ref": "inbound:laptop",
                    "profile": "default",
                }
            }
        },
    )
    assert adapter.validate_config(cfg) is False

    setup.add_principal("laptop", profile="default")
    assert adapter.validate_config(cfg) is True


def test_status_never_prints_token(hermes_home, capsys):
    inbound = setup.add_principal("laptop", profile="reviewer")
    outbound = "peer-token-with-more-than-thirty-two-characters"
    setup.add_peer("norbert", url="https://norbert.example.test/a2a", token=outbound)

    assert cli.dispatch(argparse.Namespace(a2a_command="status")) == 0
    output = capsys.readouterr().out
    assert inbound not in output
    assert outbound not in output


@pytest.mark.parametrize("kind", ["principal", "peer"])
def test_concurrent_duplicate_setup_has_one_winner_and_consistent_state(
    hermes_home, monkeypatch, kind
):
    original_ensure = setup._ensure_a2a_platform_config_unlocked
    first_inside = threading.Event()
    release_first = threading.Event()
    calls_lock = threading.Lock()
    calls = 0

    def gated_ensure(*, public_url=None):
        nonlocal calls
        with calls_lock:
            calls += 1
            position = calls
        if position == 1:
            first_inside.set()
            assert release_first.wait(timeout=2)
        return original_ensure(public_url=public_url)

    monkeypatch.setattr(setup, "_ensure_a2a_platform_config_unlocked", gated_ensure)

    def add():
        if kind == "principal":
            return setup.add_principal("duplicate", profile="default")
        return setup.add_peer(
            "duplicate",
            url="https://peer.example.test/a2a",
            token="peer-token-with-more-than-thirty-two-characters",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(add)
        assert first_inside.wait(timeout=2)
        second = executor.submit(add)
        time.sleep(0.05)
        assert calls == 1
        release_first.set()
        results = []
        for future in (first, second):
            try:
                results.append(future.result(timeout=2))
            except ValueError as exc:
                results.append(exc)

    assert sum(isinstance(result, ValueError) for result in results) == 1
    settings = config.load_a2a_settings()
    summary = auth.credential_summary()
    if kind == "principal":
        assert list(settings.principals) == ["duplicate"]
        assert summary["inbound"] == ["inbound:duplicate"]
    else:
        assert list(settings.peers) == ["duplicate"]
        assert summary["outbound"] == ["outbound:duplicate"]


@pytest.mark.parametrize("kind", ["principal", "peer"])
def test_remove_config_failure_restores_deleted_credential(hermes_home, monkeypatch, kind):
    if kind == "principal":
        setup.add_principal("rollback", profile="default")
    else:
        setup.add_peer(
            "rollback",
            url="https://peer.example.test/a2a",
            token="peer-token-with-more-than-thirty-two-characters",
        )
        generation = config.load_a2a_settings().peers["rollback"]["generation"]
        completed = client_state.try_begin_request(
            "rollback", generation, "completed-owner", new_context=False
        )
        assert completed is not None
        assert client_state.complete_request(
            "rollback",
            generation,
            completed,
            context_id="rollback-context",
            task_id="rollback-task",
        )
        active = client_state.try_begin_request(
            "rollback", generation, "active-owner", new_context=False
        )
        assert active is not None
        state_before = client_state.get_peer_state("rollback")

    def fail_config_write(_mutator):
        raise OSError("simulated config failure")

    monkeypatch.setattr(config, "update_a2a_config", fail_config_write)
    with pytest.raises(OSError, match="simulated config failure"):
        if kind == "principal":
            setup.remove_principal("rollback")
        else:
            setup.remove_peer("rollback")

    settings = config.load_a2a_settings()
    summary = auth.credential_summary()
    if kind == "principal":
        assert "rollback" in settings.principals
        assert "inbound:rollback" in summary["inbound"]
    else:
        assert "rollback" in settings.peers
        assert "outbound:rollback" in summary["outbound"]
        assert client_state.get_peer_state("rollback") == state_before
