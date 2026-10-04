"""Local Git → real PM admission → native setup → fresh-process load/init."""
from contextlib import nullcontext
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import memory_setup, plugin_catalog
from tests.pm._fixtures import _wheel

PROVIDER = '''from agent.memory_provider import MemoryProvider
from pathlib import Path
import json
class FixtureProvider(MemoryProvider):
    name = "featured-local"
    def is_available(self): return True
    def initialize(self, config, hermes_home):
        import fixture_sdk
        assert fixture_sdk.__version__ == "1.0"
        assert json.loads(Path(hermes_home, "native.json").read_text()) == {"mode": "local"}
        self.ready = True
    def shutdown(self): pass
    def get_tool_schemas(self): return []
    def get_config_schema(self):
        return [{"key": "mode", "choices": ["local", "remote"], "default": "local"}]
    def save_config(self, values, hermes_home):
        from hermes_cli.config import load_config
        assert load_config()["memory"]["provider"] == "previous"
        Path(hermes_home, "native.json").write_text(json.dumps(values))
def register(ctx): ctx.register_memory_provider(FixtureProvider())
'''
HOOK = '''
    def post_setup(self, hermes_home, config):
        from hermes_cli.config import save_config
        assert config["memory"]["provider"] == "previous"
        self.save_config({"mode": "local"}, hermes_home)
        config["memory"]["provider"] = self.name
        save_config(config)
'''


@pytest.mark.parametrize("outcome", ["configure", "hook", "cancel", "save-failure", "bad-pin",
                                     "decline-deps", "conflict", "nonprovider", "decline-install"])
def test_local_git_install_configure_is_scoped(tmp_path, monkeypatch, request, outcome):
    import plugins.memory as memory
    from agent import secret_scope
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from hermes_cli.config import atomic_config_write, load_config
    from hermes_cli import plugins_cmd_catalog, plugins_cmd_install
    from pm.environments import selected_venv
    from pm import paths

    uv = shutil.which("uv")
    assert uv, "real PM admission requires uv"
    root = tmp_path / ".hermes"
    homes = (root, root / "profiles" / "second") if outcome == "configure" else (root,)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("UV_OFFLINE", "1")
    monkeypatch.setenv("UV_PYTHON_DOWNLOADS", "never")
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "uv-cache"))
    for home in homes:
        home.mkdir(parents=True, exist_ok=True)
        atomic_config_write(home / "config.yaml", {"memory": {"provider": "previous"},
                                                  "plugins": {"disabled": ["featured-local"]}})
    monkeypatch.setattr(memory, "_MEMORY_PLUGINS_DIR", tmp_path / "no-bundled")
    monkeypatch.setattr(memory, "_iter_entry_points", lambda: [])
    core, wheels = tmp_path / "core", tmp_path / "wheels"
    core.mkdir()
    wheels.mkdir()
    _wheel(wheels, "fixture_sdk", "1.0")
    (core / "pyproject.toml").write_text(
        '[project]\nname="fixture-core"\nversion="1"\nrequires-python=">=3.14"\n'
        '[tool.uv]\npackage=false\nno-index=true\n'
        f'find-links=[{json.dumps(wheels.as_posix())}]\n')
    subprocess.run([uv, "lock", "--python", sys.executable], cwd=core, env=os.environ,
                   capture_output=True, check=True, timeout=60)
    monkeypatch.setattr(paths, "repo_root", lambda: core)
    ensure = importlib.import_module("pm.install")
    monkeypatch.setattr(ensure, "lazy_installs_allowed", lambda: True)
    monkeypatch.setattr("pm._uv._toolchain", lambda **kw: (Path(uv), Path(sys.executable)))
    monkeypatch.setattr("pm.client.is_runtime", lambda: True)
    ensure.sync_venv(explicit=True)
    import pm.client
    sync = pm.client.sync_venv
    transactions = []
    def record_sync(*args, **kwargs):
        transactions.append(type(kwargs.get("plugins")).__name__)
        return sync(*args, **kwargs)
    monkeypatch.setattr(pm.client, "sync_venv", record_sync)

    repo = tmp_path / "repo"
    package = repo / "provider"
    package.mkdir(parents=True)
    source = PROVIDER
    if outcome == "hook":
        source = source.replace("def register(ctx):", HOOK + "\ndef register(ctx):")
    elif outcome == "save-failure":
        source = source.replace('Path(hermes_home, "native.json").write_text(json.dumps(values))',
                                'raise OSError("native save refused")')
    elif outcome == "nonprovider":
        source = '"""Memory-related tool, not a selectable provider."""\ndef register(ctx): pass\n'
    (package / "__init__.py").write_text(source)
    requirement = '"fixture_sdk==1.0", "fixture_sdk==2.0"' if outcome == "conflict" else '"fixture_sdk==1.0"'
    (package / "plugin.yaml").write_text(
        "name: featured-local\nversion: 1.0.0\ndescription: Fixture\n"
        f"python_dependencies: [{requirement}]\n")
    env = {**os.environ, "GIT_AUTHOR_NAME": "Fixture", "GIT_AUTHOR_EMAIL": "fixture@example.test",
           "GIT_COMMITTER_NAME": "Fixture", "GIT_COMMITTER_EMAIL": "fixture@example.test"}
    for args in (("init", "-q"), ("add", "."), ("commit", "-qm", "fixture")):
        subprocess.run(["git", *args], cwd=repo, env=env, check=True, capture_output=True)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    entry = plugin_catalog.PluginCatalogEntry("featured-local", repo.as_uri(), sha, "Fixture", "Fixture",
                                            category="memory", subdir="provider", featured=True)
    if outcome == "bad-pin":
        entry.sha = "0" * 40
    monkeypatch.setattr(plugin_catalog, "load_catalog", lambda *a: [entry])
    monkeypatch.setattr(plugin_catalog, "fetch_live_catalog", lambda **k: None)
    monkeypatch.setattr(plugin_catalog, "load_removed_list", lambda *a: [])
    # Consent inputs only; publication, admission, resolver and package install are real.
    monkeypatch.setattr(plugins_cmd_install.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(plugins_cmd_install.sys.stdout, "isatty", lambda: True)
    prompts = []
    def answer(prompt):
        prompts.append(prompt)
        return "n" if outcome == "decline-deps" else "y"
    monkeypatch.setattr("builtins.input", answer)
    # Do not contact a live gateway or Desktop while testing next-session semantics.
    monkeypatch.setattr("hermes_cli.plugins_activation.activate_plugin_now", lambda *a, **kw: {
        "gateway_reloaded": False, "activation": None, "restart_required": True})
    installs = []
    def select(title, items, **kw):
        if title.startswith("Install"):
            installs.append(title)
            return 0 if outcome == "decline-install" else 1
        if title == "Memory provider setup":
            return next(i for i, (label, desc) in enumerate(items) if "featured-local" in label)
        return -1 if outcome == "cancel" else 0
    monkeypatch.setattr(memory_setup, "_curses_select", select)

    previous_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    request.addfinalizer(lambda: secret_scope.set_multiplex_active(previous_multiplex))
    # One successful A→B→A proves scope; refusal paths need only their own home.
    for owner in (*homes, root) if outcome == "configure" else homes:
        token = set_hermes_home_override(owner)
        secrets = secret_scope.set_secret_scope({}, profile_home=str(owner))
        other_before = {home: (home / "config.yaml").read_bytes() for home in homes if home != owner}
        try:
            installed = owner / "plugins" / entry.name
            prompts.clear()
            transactions.clear()
            config_before = (owner / "config.yaml").read_bytes()
            environment_before = selected_venv(core)
            if not installed.exists():
                with pytest.raises(SystemExit) if outcome == "bad-pin" else nullcontext() as error:
                    memory_setup.cmd_setup(SimpleNamespace())
                if error is not None:
                    assert error.value.code == 1
                assert len(prompts) == (0 if outcome in {"bad-pin", "decline-install"} else 1)
                expected = [] if outcome in {"bad-pin", "decline-install"} else ["StagedUpdate"]
                if outcome not in {"bad-pin", "decline-install", "decline-deps"}:
                    expected.append("Selection")
                assert transactions == expected
            current = load_config()
            if outcome not in {"bad-pin", "decline-install"}:
                assert (installed / "__init__.py").read_text() == source
                assert plugins_cmd_catalog.catalog_install_record(installed)["sha"] == sha
            if outcome not in {"bad-pin", "decline-install", "decline-deps", "conflict"}:
                assert entry.name in current["plugins"]["enabled"]
                assert entry.name not in current["plugins"]["disabled"]
            if outcome not in {"configure", "hook"}:
                assert current["memory"]["provider"] == "previous"
                assert not (owner / "native.json").exists()
                if outcome in {"decline-deps", "conflict", "bad-pin", "decline-install"}:
                    assert entry.name not in current.get("plugins", {}).get("enabled", [])
                    assert selected_venv(core) == environment_before
                    assert (owner / "config.yaml").read_bytes() == config_before
                continue
            assert memory.find_provider_dir(entry.name) == installed
            assert current["memory"]["provider"] == entry.name
            assert json.loads((owner / "native.json").read_text()) == {"mode": "local"}
            # Fresh interpreter from the PM-selected environment. Only core imports
            # borrow the test interpreter's paths; fixture_sdk must come from PM.
            python = selected_venv(core) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            probe = f'''
import sys
sys.path.extend({sys.path!r})
from pathlib import Path
import fixture_sdk
assert Path(fixture_sdk.__file__).is_relative_to(Path(sys.prefix))
import plugins.memory as memory
memory._MEMORY_PLUGINS_DIR = Path({str(tmp_path / "no-bundled")!r})
memory._iter_entry_points = lambda: []
from hermes_cli.config import load_config
config = load_config()
provider = memory.load_memory_provider(config["memory"]["provider"])
assert provider is not None
provider.initialize(config, {str(owner)!r})
assert provider.ready
provider.shutdown()
print("loaded and initialized")
'''
            result = subprocess.run([str(python), "-I", "-c", probe],
                env={**os.environ, "HERMES_HOME": str(owner)}, cwd=tmp_path,
                capture_output=True, text=True, timeout=30)
            assert result.returncode == 0, result.stderr
            assert result.stdout.strip() == "loaded and initialized"
        finally:
            secret_scope.reset_secret_scope(secrets)
            reset_hermes_home_override(token)
            assert all((home / "config.yaml").read_bytes() == raw for home, raw in other_before.items())
    assert len(installs) == len(homes)


def test_discovered_provider_with_load_failure_is_not_offered_for_reinstall(tmp_path, monkeypatch):
    import plugins.memory as memory
    provider = memory_setup.get_hermes_home() / "plugins" / "featured-broken"
    provider.mkdir(parents=True)
    (provider / "__init__.py").write_text('"""MemoryProvider with a missing SDK."""\nraise ImportError("fixture")\n')
    (provider / "plugin.yaml").write_text("name: featured-broken\n")
    monkeypatch.setattr(memory, "_MEMORY_PLUGINS_DIR", tmp_path / "no-bundled")
    monkeypatch.setattr(memory, "_iter_entry_points", lambda: [])
    entry = plugin_catalog.PluginCatalogEntry("featured-broken", "https://github.com/example/broken",
        "a" * 40, "Fixture", "Fixture", category="memory", featured=True)
    monkeypatch.setattr(plugin_catalog, "load_catalog", lambda *a: [entry])
    monkeypatch.setattr(plugin_catalog, "fetch_live_catalog", lambda **k: None)
    monkeypatch.setattr(plugin_catalog, "load_removed_list", lambda *a: [])
    def select(title, items, **kw):
        assert "install required" not in str(items[0]).lower()
        assert "featured-broken" in items[0][0]
        return -1
    monkeypatch.setattr(memory_setup, "_curses_select", select)
    memory_setup.cmd_setup(SimpleNamespace())
