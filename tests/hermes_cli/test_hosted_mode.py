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
    """The installer-owned catalog record a curated local install carries (an operator installing
    a backend from the Hermes plugin catalog at its reviewed pin). A hosted boot never consults
    it — in hosted mode nothing under the deployment home loads in-process, whatever its record
    says — so this is the shape a LOCAL install's trust still reads."""
    repo = f"https://github.com/NousResearch/{plugin.name}"
    sha = "a" * 40
    metadata_path = get_hermes_home() / "plugins" / ".install-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig")) if metadata_path.exists() else {}
    metadata[plugin.name] = {"source": repo, "revision": sha, "pinned": True,
                             "catalog": {"name": plugin.name, "repo": repo,
                                         "tier": tier, "pin": sha, "sha": sha}}
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")


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
    """No bundled plugins: the checks judge the home's own plugins, not the repo's tree. The
    session-storage kind's bundled directory moves beside it — a hosted boot resolves named
    providers only from the bundled tree, so the storage stubs install there."""
    from plugins import session_storage

    bundled = tmp_path / "bundled"
    (bundled / "session_storage").mkdir(parents=True)
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: bundled)
    monkeypatch.setattr(session_storage, "_PLUGINS_DIR", bundled / "session_storage")


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


def _bundled_plugin(name: str, body: str) -> Path:
    """A plugin installed in the image's bundled plugins tree — the only plugins a hosted boot
    accepts in-process (a deployment's own plugins are baked into its image by the operator)."""
    plugin = plugins_mod.get_bundled_plugins_dir() / name
    plugin.mkdir(parents=True, exist_ok=True)
    (plugin / "__init__.py").write_text(textwrap.dedent(body), encoding="utf-8")
    (plugin / "plugin.yaml").write_text(
        yaml.safe_dump({"name": name, "version": "1.0"}, sort_keys=False), encoding="utf-8")
    return plugin


def _bundled_provider(name: str, body: str) -> Path:
    """A named session-storage provider installed in the kind's BUNDLED directory — where a
    hosted boot resolves its providers from (the deployment home's own dirs are refused)."""
    from plugins import session_storage

    provider = session_storage._PLUGINS_DIR / name
    provider.mkdir(parents=True, exist_ok=True)
    (provider / "__init__.py").write_text(textwrap.dedent(body), encoding="utf-8")
    return provider


def _clean_boot(**kwargs):
    """A hosted deployment that passes every check: the stub sandbox and storage plugins are
    baked into the image's bundled tree (where a hosted deployment's own plugins live) and are
    discovered for real, and the boot call sees a clean environment."""
    _write_config(_clean_hosted_config())
    _bundled_plugin("stub-sandbox", SANDBOX_PLUGIN)
    _bundled_provider("stub-storage", STORAGE_PLUGIN)
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
    _bundled_plugin("plain-backend", PLAIN_SANDBOX_PLUGIN.replace('"stub-sandbox"', '"plain-backend"'))
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


def test_a_home_plugin_refuses_hosted_boot_whatever_its_install_record(clean_hosted):
    """The same sandbox plugin, this time as a home install carrying a perfectly curated catalog
    record (verified tier, at its reviewed pin): the boot still refuses it. An install record
    lives in the tenant-writable home, so it is never consulted for hosted trust — a hosted
    deployment's own plugins are baked into its image, and a home plugin is a user plugin."""
    home_copy = _install_plugin("home-sandbox", SANDBOX_PLUGIN,
                                manifest={"name": "home-sandbox", "version": "1.0"})
    _curated(home_copy)
    config = _clean_hosted_config() | {"plugins": {"enabled": ["stub-sandbox", "home-sandbox"]}}
    _write_config(config)
    with pytest.raises(HostedBootRefused, match="home-sandbox"):
        boot(env={"PATH": "/usr/bin"})


def test_a_disabled_user_plugin_never_loads_and_never_refuses(clean_hosted):
    """A user plugin config never runs (absent from plugins.enabled): its provenance is never
    consulted, because its code never reaches the process."""
    _install_plugin("disabled-extra", SANDBOX_PLUGIN,
                    manifest={"name": "disabled-extra", "version": "1.0"})
    boot(env={"PATH": "/usr/bin"})  # no installer record for it either; it never loads


# --------------------------------------------------------------------------------------------------
# Discovery stays off until the import-free preflight passed


MARKER_PLUGIN = '''
from pathlib import Path

Path(__file__).with_name("imported").write_text("ran")


def register(ctx):
    pass
'''


def test_plugin_discovery_stays_off_until_the_boot_preflight_passes(verdict, empty_bundled):
    """Hosted: a would-load user plugin's code must not run just because something read config —
    a config read discovers and imports plugins, so discovery stays suppressed until the
    import-free preflight passed, and the preflight refuses the uncurated plugin before its
    module ever imports."""
    _write_config({"hosted": {"enabled": True}, "plugins": {"enabled": ["marker-probe"]}})
    plugin = _install_plugin("marker-probe", MARKER_PLUGIN,
                             manifest={"name": "marker-probe", "version": "1.0"})
    from hermes_cli.plugins import discover_plugins, get_plugin_manager

    discover_plugins()  # the preflight has not passed: a no-op, not an import
    assert get_plugin_manager()._discovered is False
    assert not (plugin / "imported").exists()
    with pytest.raises(HostedBootRefused, match="marker-probe"):
        hosted_mode.hosted_boot_preflight(env={"PATH": "/usr/bin"})
    assert not (plugin / "imported").exists()  # the refusal came before any import


def test_discovery_runs_once_the_boot_preflight_passed(verdict, empty_bundled):
    """A hosted boot that passes the preflight loads plugins exactly as before: the gate opens."""
    _write_config({"hosted": {"enabled": True}})
    hosted_mode.hosted_boot_preflight(env={"PATH": "/usr/bin"})
    assert hosted_mode.plugin_loading_permitted() is True
    from hermes_cli.plugins import discover_plugins, get_plugin_manager

    discover_plugins()
    assert get_plugin_manager()._discovered is True


def test_after_the_boot_records_its_config_no_hosted_read_touches_the_files(clean_hosted, monkeypatch):
    """The boot's first read is its only one: once the preflight recorded the config it resolved,
    the verdict and every boot check consult that object — the config reader itself must not be
    reached again."""
    def _no_reads(*args, **kwargs):
        raise AssertionError("a hosted read re-read the profile's config files")

    monkeypatch.setattr(hosted_mode, "_effective_config", _no_reads)
    assert hosted_mode.is_hosted() is True  # the recorded verdict, no I/O
    boot(env={"PATH": "/usr/bin"})  # the recorded config, no I/O


# --------------------------------------------------------------------------------------------------
# A named provider that exists only in the deployment home's writable dirs


def test_a_user_dir_session_storage_provider_is_refused_before_import(clean_hosted):
    """A named backend that exists only as a manifest-less directory in the home's plugin dirs
    (the shape the shared named-provider loaders import directly): in a hosted process it is
    refused BEFORE import, with the reason, and its code never runs."""
    probe = get_hermes_home() / "plugins" / "userdir-storage"
    probe.mkdir(parents=True)
    (probe / "__init__.py").write_text(textwrap.dedent('''
        from pathlib import Path

        from hermes_state_port import SessionStorageProvider

        Path(__file__).with_name("imported").write_text("ran")


        class Probe(SessionStorageProvider):
            @property
            def name(self):
                return "userdir-storage"

            def open(self, db=None):
                raise NotImplementedError

            def assert_isolating(self):
                return
    '''), encoding="utf-8")
    config = _clean_hosted_config()
    config["hosted"] = dict(HOSTED, session_storage="userdir-storage")
    _write_config(config)
    with pytest.raises(HostedBootRefused, match="userdir-storage"):
        boot(config=config, env={"PATH": "/usr/bin"})
    assert not (probe / "imported").exists()  # its code never ran


# --------------------------------------------------------------------------------------------------
# Provenance: one rule — only what the deployment image ships loads in-process


def test_the_boot_lists_every_refused_plugin_in_one_refusal(clean_hosted):
    """Two uncurated home plugins would load: the preflight refuses once, naming each — an
    operator reading the boot error sees everything the process refuses, not just the first."""
    for name in ("probe-a", "probe-b"):
        _install_plugin(name, SANDBOX_PLUGIN, manifest={"name": name, "version": "1.0"})
    config = _clean_hosted_config() | {"plugins": {"enabled": ["stub-sandbox", "probe-a", "probe-b"]}}
    _write_config(config)
    with pytest.raises(HostedBootRefused) as refusal:
        boot(env={"PATH": "/usr/bin"})
    assert "probe-a" in str(refusal.value) and "probe-b" in str(refusal.value)


def test_a_symlink_in_the_bundled_dir_is_not_a_bundled_plugin(clean_hosted, tmp_path):
    """A symlink planted in the bundled plugins dir (or any dir whose realpath leaves that tree)
    does not get the bundled trust: the boot judges it by provenance, and it refuses. The
    linked tree lives outside the home's own plugin dirs, so only the bundled scan sees it —
    the bundled label is the only thing that could carry it."""
    target = tmp_path / "elsewhere" / "impostor"
    target.mkdir(parents=True)
    (target / "plugin.yaml").write_text("name: impostor\nversion: '1.0'\n", encoding="utf-8")
    (target / "__init__.py").write_text("def register(ctx):\n    pass\n", encoding="utf-8")
    (plugins_mod.get_bundled_plugins_dir() / "impostor").symlink_to(target)
    config = _clean_hosted_config() | {"plugins": {"enabled": ["stub-sandbox", "impostor"]}}
    _write_config(config)
    with pytest.raises(HostedBootRefused, match="impostor"):
        boot(env={"PATH": "/usr/bin"})


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
    _bundled_provider("leaky-storage", NON_ISOLATING_STORAGE_PLUGIN.replace(
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
