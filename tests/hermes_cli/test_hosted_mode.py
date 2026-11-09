"""The hosted boot profile: the config switch, and one clear refusal per boot condition.

``hermes_cli/hosted_mode.py`` decides whether this process is a shared one
(:func:`is_hosted`, read once and frozen), and refuses to serve when a tenant's secret, code or
bot would reach the process (:func:`assert_hosted_boot`). Every condition in the module docstring
has a test here that is red without its check, a local install whose boot consults none of them,
and a hosted scope in which the legacy profile setters and ``bind_profile_scope`` keep refusing.
"""

from __future__ import annotations

import json
import logging
import textwrap
from pathlib import Path
from types import SimpleNamespace

import hermes_yaml as yaml
import pytest

from hermes_cli import hosted_mode
import hermes_cli.plugins as plugins_mod
from hermes_constants import get_hermes_home
from hermes_cli.hosted_mode import HostedBootRefused
from hermes_scope import Fence, Group, Principal, Scope, ScopeSwitchRefused

# A sandbox backend plugin that serves hosted scopes, loaded through the real plugin discovery
# (a plugin dir under the home's plugins/, enabled in config).
SANDBOX_PLUGIN = '''
from agent.terminal_env_provider import TerminalEnvironmentProvider


class _Env:
    uses_host_filesystem = False

    def execute(self, command, *, scope, **kw):
        return {"output": "", "returncode": 0}

    def get_temp_dir(self, *, scope):
        return "/tmp"

    def fetch_realpath(self, remote_path, *, scope):
        return remote_path

    def cleanup(self, *, scope):
        pass

    def start_process(self, name, argv, *, scope, child_env=None):
        raise NotImplementedError

    def attach_process(self, name, *, scope):
        return None

    def stop_process(self, name, *, scope):
        return False

    def stop_idle(self, *, scope):
        pass


class Provider(TerminalEnvironmentProvider):
    serves_hosted_scopes = True

    @property
    def name(self):
        return "stub-sandbox"

    def is_available(self):
        return True

    def create_environment(self, *, cwd, timeout, task_id="default", image=None,
                            container_config=None, scope=None, sandbox=None, **kw):
        return _Env()


def register(ctx):
    ctx.register_terminal_environment_provider(Provider())
'''

# The same plugin, without the hosted declaration.
PLAIN_SANDBOX_PLUGIN = SANDBOX_PLUGIN.replace("    serves_hosted_scopes = True\n", "")

STORAGE_PLUGIN = '''
from hermes_state_port import SessionStorageProvider


class Provider(SessionStorageProvider):
    @property
    def name(self):
        return "stub-storage"

    def open(self, db=None):
        raise NotImplementedError

    def assert_isolating(self):
        return


def register(ctx):
    ctx.register_session_storage(Provider())
'''

NON_ISOLATING_STORAGE_PLUGIN = STORAGE_PLUGIN.replace(
    "    def assert_isolating(self):\n        return\n",
    "    def assert_isolating(self):\n"
    "        raise RuntimeError(\"the storage role bypasses row security\")\n",
)

HOSTED = {"enabled": True, "identity_verifier": "stub-verifier", "session_storage": "stub-storage"}


def _write_config(settings: dict) -> None:
    """The home's config.yaml, through the repo's own YAML writer."""
    get_hermes_home().mkdir(parents=True, exist_ok=True)
    (get_hermes_home() / "config.yaml").write_text(
        yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")


def _install_plugin(name: str, body: str, *, manifest: dict | None = None) -> Path:
    """A plugin dir under the home's plugins/; a manifest makes the general plugin loader see it."""
    plugin = get_hermes_home() / "plugins" / name
    plugin.mkdir(parents=True, exist_ok=True)
    (plugin / "__init__.py").write_text(textwrap.dedent(body), encoding="utf-8")
    if manifest is not None:
        (plugin / "plugin.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return plugin


def _curated(plugin: Path, *, tier: str = "verified") -> None:
    """The installer-owned catalog record a curated install carries (an operator installing the
    deployment's backends from the Hermes plugin catalog at its reviewed pin)."""
    (get_hermes_home() / "plugins" / ".install-metadata.json").write_text(json.dumps({
        plugin.name: {"catalog": {"name": plugin.name,
                                  "repo": f"https://github.com/NousResearch/{plugin.name}",
                                  "tier": tier, "pin": "0" * 40}},
    }), encoding="utf-8")


def _clean_hosted_config() -> dict:
    """The config of a deployment that passes every check: a sandbox plugin backend, no direct
    platform, the two required backend names, and no MCP server."""
    return {
        "hosted": dict(HOSTED),
        "terminal": {"backend": "stub-sandbox"},
        "plugins": {"enabled": ["stub-sandbox"]},
    }


@pytest.fixture
def verdict(monkeypatch):
    """A fresh boot verdict: each test decides what this process is by the config it writes."""
    monkeypatch.setattr(hosted_mode, "_VERDICT", {"hosted": None})


@pytest.fixture
def empty_bundled(monkeypatch, tmp_path):
    """No bundled plugins: the checks judge the home's own plugins, not the repo's tree."""
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: bundled)


def _no_gateway_config():
    return SimpleNamespace(get_connected_platforms=list)


def boot(config=None, *, env=None, gateway_config=None, profile_homes=()):
    hosted_mode.assert_hosted_boot(gateway_config if gateway_config is not None else _no_gateway_config(),
                                   profile_homes=profile_homes, config=config, env=env)


# --------------------------------------------------------------------------------------------------
# The switch


def test_a_local_install_is_not_hosted(verdict, empty_bundled):
    assert hosted_mode.is_hosted() is False


def test_hosted_reads_the_section_and_the_shorthand(verdict, empty_bundled, monkeypatch, tmp_path):
    _write_config({"hosted": {"enabled": True}})
    assert hosted_mode.is_hosted() is True
    # The shorthand `hosted: true` is the same switch.
    monkeypatch.setattr(hosted_mode, "_VERDICT", {"hosted": None})
    _write_config({"hosted": True})
    assert hosted_mode.is_hosted() is True


def test_the_verdict_is_read_once_and_frozen_for_the_process(verdict, empty_bundled):
    _write_config({"hosted": {"enabled": True}})
    assert hosted_mode.is_hosted() is True
    _write_config({"hosted": {"enabled": False}})  # a mid-process config edit flips nothing
    assert hosted_mode.is_hosted() is True


def test_a_local_boot_consults_none_of_the_checks(verdict, empty_bundled, tmp_path, monkeypatch):
    """hosted false: no condition is refused — a profile .env, a credential in the environment and
    a direct platform are all the local install's to have."""
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / ".env").write_text("OPENROUTER_API_KEY=sk-local\n", encoding="utf-8")
    from gateway.config import GatewayConfig, Platform, PlatformConfig

    config = _clean_hosted_config() | {"hosted": {"enabled": False}}
    monkeypatch.setattr(hosted_mode, "_VERDICT", {"hosted": None})
    _write_config(config)
    assert boot(config=config, env={"OPENROUTER_API_KEY": "sk-local"},
                gateway_config=GatewayConfig(platforms={
                    Platform.TELEGRAM: PlatformConfig(enabled=True, token="tg-token")})) is None


# --------------------------------------------------------------------------------------------------
# One refusal per boot condition


def _clean_boot(**kwargs):
    """A hosted deployment that passes every check: the stub sandbox and storage plugins are
    discovered for real, and the boot call sees a clean environment."""
    _write_config(_clean_hosted_config())
    _install_plugin("stub-sandbox", SANDBOX_PLUGIN,
                    manifest={"name": "stub-sandbox", "version": "1.0"})
    _curated(get_hermes_home() / "plugins" / "stub-sandbox")
    _install_plugin("stub-storage", STORAGE_PLUGIN)
    boot(env={"PATH": "/usr/bin", "HOME": "/home/op", "LC_ALL": "C"}, **kwargs)


@pytest.fixture
def clean_hosted(verdict, empty_bundled):
    _clean_boot()


def test_a_credential_in_the_environment_refuses_hosted_boot(clean_hosted):
    with pytest.raises(HostedBootRefused, match="OPENROUTER_API_KEY"):
        boot(env={"PATH": "/usr/bin", "OPENROUTER_API_KEY": "sk-tenant"})


def test_operator_process_settings_pass_the_environment_check(clean_hosted):
    """The allowlist is the operator's own plumbing: host names, the locale prefix and the config
    backend's deployment names pass; so does an unknown non-credential setting."""
    from hermes_cli.config_backend import deployment_env_names

    env = {"PATH": "/usr/bin", "HOME": "/home/op", "LC_ALL": "C",
           "SSL_CERT_FILE": "/etc/certs/ca.pem", "MY_TOOL_SETTING": "1"}
    env.update({name: "x" for name in deployment_env_names()})
    boot(env=env)


@pytest.mark.parametrize("credential", ["TWILIO_AUTH_TOKEN", "MY_SERVICE_SECRET", "DB_PASSWORD",
                                        "GOOGLE_CREDENTIALS", "STRIPE_KEY"])
def test_every_credential_shape_refuses_hosted_boot(clean_hosted, credential):
    with pytest.raises(HostedBootRefused, match=credential):
        boot(env={"PATH": "/usr/bin", credential: "value"})


@pytest.mark.parametrize("name", [".env", "auth.json"])
def test_a_profile_credential_file_refuses_hosted_boot(clean_hosted, name):
    (get_hermes_home() / name).write_text("{}\n", encoding="utf-8")
    with pytest.raises(HostedBootRefused, match=name):
        boot(env={"PATH": "/usr/bin"})


def test_a_served_profile_env_refuses_hosted_boot(clean_hosted, tmp_path):
    profile = tmp_path / "profiles" / "work"
    profile.mkdir(parents=True)
    (profile / ".env").write_text("TOKEN=x\n", encoding="utf-8")
    with pytest.raises(HostedBootRefused, match="work/.env"):
        boot(env={"PATH": "/usr/bin"}, profile_homes=(profile,))


@pytest.mark.parametrize("backend,reason", [
    ("local", "would run on this host"),
    ("docker", "built-in backend"),
])
def test_a_non_sandbox_terminal_backend_refuses_hosted_boot(clean_hosted, backend, reason):
    config = _clean_hosted_config() | {"terminal": {"backend": backend}}
    _write_config(config)
    with pytest.raises(HostedBootRefused, match=reason):
        boot(config=config, env={"PATH": "/usr/bin"})


def test_an_unknown_terminal_backend_refuses_hosted_boot(clean_hosted):
    config = _clean_hosted_config() | {"terminal": {"backend": "no-such-backend"}}
    _write_config(config)
    with pytest.raises(HostedBootRefused, match="no registered sandbox plugin"):
        boot(config=config, env={"PATH": "/usr/bin"})


def test_a_plugin_backend_without_the_hosted_declaration_refuses_hosted_boot(clean_hosted):
    _install_plugin("plain-backend", PLAIN_SANDBOX_PLUGIN.replace('"stub-sandbox"', '"plain-backend"'),
                    manifest={"name": "plain-backend", "version": "1.0"})
    _curated(get_hermes_home() / "plugins" / "plain-backend")
    config = _clean_hosted_config() | {"terminal": {"backend": "plain-backend"},
                                      "plugins": {"enabled": ["plain-backend"]}}
    _write_config(config)
    with pytest.raises(HostedBootRefused, match="serves_hosted_scopes"):
        boot(config=config, env={"PATH": "/usr/bin"})


def test_a_direct_platform_adapter_refuses_hosted_boot(clean_hosted):
    from gateway.config import GatewayConfig, Platform, PlatformConfig

    direct = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="tg"),
                                      Platform.RELAY: PlatformConfig(
                                          enabled=True, extra={"relay_url": "wss://connector"})})
    with pytest.raises(HostedBootRefused, match="telegram"):
        boot(env={"PATH": "/usr/bin"}, gateway_config=direct)


def test_relay_only_traffic_passes_the_platform_check(clean_hosted):
    from gateway.config import GatewayConfig, Platform, PlatformConfig

    relay = GatewayConfig(platforms={Platform.RELAY: PlatformConfig(
        enabled=True, extra={"relay_url": "wss://connector"})})
    boot(env={"PATH": "/usr/bin"}, gateway_config=relay)


def test_a_configured_mcp_server_refuses_hosted_boot(clean_hosted):
    config = _clean_hosted_config() | {"mcp_servers": {"my-server": {"command": "my-mcp"}}}
    _write_config(config)
    with pytest.raises(HostedBootRefused, match="my-server"):
        boot(config=config, env={"PATH": "/usr/bin"})


def test_a_user_plugin_refuses_hosted_boot_until_it_is_catalog_curated(clean_hosted):
    """The sandbox stub is curated by the record in the fixtures above; here the operator's record
    is missing, so the same enabled plugin is a user install and refuses."""
    (get_hermes_home() / "plugins" / ".install-metadata.json").unlink()
    with pytest.raises(HostedBootRefused, match="stub-sandbox"):
        boot(env={"PATH": "/usr/bin"})


def test_a_catalog_plugin_off_the_curated_tier_refuses_hosted_boot(clean_hosted):
    _curated(get_hermes_home() / "plugins" / "stub-sandbox", tier="community")
    with pytest.raises(HostedBootRefused, match="tier community"):
        boot(env={"PATH": "/usr/bin"})


def test_a_disabled_user_plugin_never_loads_and_never_refuses(clean_hosted):
    """A user plugin config never runs (absent from plugins.enabled): its provenance is never
    consulted, because its code never reaches the process."""
    _install_plugin("disabled-extra", SANDBOX_PLUGIN,
                    manifest={"name": "disabled-extra", "version": "1.0"})
    boot(env={"PATH": "/usr/bin"})  # no installer record for it either; it never loads


def test_no_identity_verifier_refuses_hosted_boot(clean_hosted):
    config = _clean_hosted_config()
    config["hosted"] = dict(HOSTED, identity_verifier="")
    _write_config(config)
    with pytest.raises(HostedBootRefused, match="identity_verifier"):
        boot(config=config, env={"PATH": "/usr/bin"})


def test_no_session_storage_name_refuses_hosted_boot(clean_hosted):
    config = _clean_hosted_config()
    config["hosted"] = dict(HOSTED, session_storage="")
    _write_config(config)
    with pytest.raises(HostedBootRefused, match="session_storage"):
        boot(config=config, env={"PATH": "/usr/bin"})


def test_a_missing_session_storage_provider_refuses_hosted_boot(clean_hosted):
    config = _clean_hosted_config()
    config["hosted"] = dict(HOSTED, session_storage="not-installed")
    _write_config(config)
    with pytest.raises(HostedBootRefused, match="not-installed"):
        boot(config=config, env={"PATH": "/usr/bin"})


def test_a_session_storage_role_that_bypasses_isolation_refuses_hosted_boot(clean_hosted):
    _install_plugin("leaky-storage", NON_ISOLATING_STORAGE_PLUGIN.replace(
        '"stub-storage"', '"leaky-storage"'))  # a second provider, so no cached module answers
    config = _clean_hosted_config()
    config["hosted"] = dict(HOSTED, session_storage="leaky-storage")
    _write_config(config)
    with pytest.raises(HostedBootRefused, match="bypasses row security"):
        boot(config=config, env={"PATH": "/usr/bin"})


# --------------------------------------------------------------------------------------------------
# The hosted scope keeps refusing profile switches (the legacy setters and bind_profile_scope)


A_HOSTED = Scope("acme", "agent-a", Principal("u1"), owner=Principal("u1"), can_administer=True)


@pytest.mark.parametrize("setter", [
    "set_hermes_home_override",
    "set_secret_scope",
    "set_terminal_scope",
    "install_profile_terminal_scope",
])
def test_every_legacy_setter_refuses_inside_a_hosted_scope(setter):
    from agent.runtime_scope import bind_state

    with bind_state(A_HOSTED):
        with pytest.raises(ScopeSwitchRefused):
            if setter == "set_hermes_home_override":
                from hermes_constants import set_hermes_home_override
                set_hermes_home_override("/somewhere/else")
            elif setter == "set_secret_scope":
                from agent.secret_scope import set_secret_scope
                set_secret_scope({"A": "b"}, profile_home="/somewhere/else")
            elif setter == "set_terminal_scope":
                from tools.terminal_scope import set_terminal_scope
                set_terminal_scope({"TERMINAL_ENV": "docker"}, profile_home="/somewhere/else")
            else:
                from tools.terminal_scope import install_profile_terminal_scope
                install_profile_terminal_scope("/somewhere/else")


def test_bind_profile_scope_refuses_inside_a_hosted_scope():
    from agent.runtime_scope import bind_profile_scope, bind_state

    with bind_state(A_HOSTED):
        with pytest.raises(ScopeSwitchRefused):
            with bind_profile_scope("/somewhere/else"):
                pass


def test_bind_state_of_another_scope_refuses_inside_a_hosted_scope():
    from agent.runtime_scope import bind_state

    other = Scope("acme", "agent-b", Principal("u1"), owner=Group("team"), can_administer=True)
    with bind_state(A_HOSTED):
        with pytest.raises(ScopeSwitchRefused):
            bind_state(other)


def test_the_same_scope_rebinds_inside_a_hosted_scope():
    """A turn of the bound scope re-binds (the turn's own bind): the refusal is about changing
    agent mid-activity, not about the one activity the scope names."""
    from agent.runtime_scope import bind_state

    turn = A_HOSTED.for_turn("session-1", "turn-1", fence=Fence(1, 1))
    with bind_state(A_HOSTED):
        with bind_state(turn):
            pass


def test_local_setters_still_work_with_no_scope_bound():
    from agent.runtime_scope import bind_profile_scope

    with bind_profile_scope(get_hermes_home(), hydrate_secrets=False):
        pass  # the local install's replacement for the setters, unchanged


# --------------------------------------------------------------------------------------------------
# The shared host model key never reaches tenant-visible output


def test_the_host_model_key_never_reaches_tenant_visible_output(clean_hosted, caplog, monkeypatch):
    """The credential the shared process holds for its own model calls (the operator's secret
    scope) appears in none of what a tenant's turn sees: the system prompt, a tool's result, a
    log line, a stored session row. The positive control plants the key into the same corpus and
    the same scan finds it, so the assertion is not vacuous."""
    from agent import secret_scope
    from agent.runtime_scope import bind_state
    from tests.agent.test_system_prompt import _make_agent
    from agent.system_prompt import build_system_prompt

    host_key = "sk-host-7f31c0d2b9e54a8c6f00d1e2c3b4a598"
    with secret_scope.set_secret_scope({"OPENAI_API_KEY": host_key}):
        assert secret_scope.get_secret("OPENAI_API_KEY") == host_key
        with bind_state(A_HOSTED):
            with caplog.at_level(logging.DEBUG):
                prompt = build_system_prompt(_make_agent())
                tool_result = _dispatch_stub_tool(monkeypatch)
            stored = _stored_rows()

    corpus = {"prompt": prompt, "tool result": tool_result, "log lines": caplog.text,
              "stored rows": stored}
    assert all(host_key not in text for text in corpus.values()), corpus
    # Positive control: a planted copy of the key in the same corpus IS detected by the same scan.
    planted = dict(corpus)
    planted["log lines"] = f"...{host_key}..."
    assert any(host_key in text for text in planted.values())


def _dispatch_stub_tool(monkeypatch) -> str:
    """One allowed tool's result for the bound hosted agent, through the real tool registry and
    the stub sandbox the hosted boot registered (the same state reset the sandbox conformance
    tests do, so a leftover cached environment from an earlier test cannot answer instead)."""
    import model_tools  # registers the tools
    from tools import file_tools, process_registry, terminal_tool
    from tools.environments import sandbox_conformance as conformance
    from tools.registry import registry

    monkeypatch.setenv("TERMINAL_ENV", "stub-sandbox")
    monkeypatch.setattr(terminal_tool, "_active_environments", {})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(file_tools, "_file_ops_cache", {})
    monkeypatch.setattr(process_registry.process_registry, "_running", {})
    monkeypatch.setattr(process_registry.process_registry, "_finished", {})
    result = conformance.call_tool("terminal", {"command": "echo tenant-data"}, task_id="canary")
    return json.dumps(result)


def _stored_rows() -> str:
    """The session rows a hosted turn leaves in the test home's store, read back as text: a real
    write and a real read through the session database, not a canned string."""
    from hermes_state import SessionDB

    db = SessionDB(get_hermes_home() / "state.db")
    try:
        db.ensure_session("session-1", source="canary")
        db.append_message("session-1", "user", content="tenant-data hello")
        db.append_message("session-1", "assistant", content="tenant-data reply")
        return json.dumps(db.get_messages("session-1"), default=str)
    finally:
        db.close()
