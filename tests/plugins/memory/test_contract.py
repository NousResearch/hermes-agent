"""plugins.memory.contract: companion files, schema translation and saves, OAuth runners for external packages."""

import json
import os
import textwrap
import time
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest
import yaml

from plugins import memory
from plugins.memory import contract


INERT = '# MemoryProvider\nraise AssertionError("no runtime initialization")\n'
RAW = '''
    import json
    from pathlib import Path
    from agent.memory_provider import MemoryProvider

    class Probe(MemoryProvider):
        name = "probe"
        def is_available(self): return True
        def initialize(self, session_id, **kwargs): raise AssertionError("must not initialize")
        def get_tool_schemas(self): return []
        def get_config_schema(self): return [
            {"key": "dependent", "default": "fallback", "when": {"enabled": True}},
            {"key": "enabled", "type": "boolean", "default": True},
            {"key": "count", "type": "integer", "default": 3, "minimum": 0, "maximum": 9},
            {"key": "token", "secret": True, "env_var": "PROBE_TOKEN"},
        ]
'''
NATIVE_SAVE = '''
        def save_config(self, values, hermes_home):
            Path(hermes_home, "native-calls.json").write_text(json.dumps(values))
'''
DECLARED = '''
    from plugins.memory.config_schema import ProviderConfigSchema, ProviderField
    CONFIG_SCHEMA = ProviderConfigSchema(name="probe", label="A", fields=(
        ProviderField(key="endpoint", label="Endpoint"),
        ProviderField(key="count", label="Count", kind="number", default="3"),
        ProviderField(key="api_key", label="Key", kind="secret", env_key="PROBE_KEY"),
    ))
'''
HOME_FLOW = '''
    import os, time
    from pathlib import Path
    STATES = {}
    def get_flow_status(*, hermes_home):
        return {"state": STATES.get(hermes_home, "idle"), "pid": os.getpid()}
    def start_loopback_flow_background(*, hermes_home):
        STATES[hermes_home] = "pending"
        Path(hermes_home, "receipt").write_text(str(os.getpid()))
        while not Path(hermes_home, "release").exists():
            time.sleep(.02)
        STATES[hermes_home] = "connected"
'''
LEGACY_FLOW = '''
    import os
    from pathlib import Path
    def get_flow_status():
        Path(os.environ["HERMES_HOME"], "child-pid").write_text(str(os.getpid()))
        return {"state": "idle", "detail": "private-" + os.environ["HERMES_HOME"]}
    start_loopback_flow_background = get_flow_status
'''
MARKED_FLOW = LEGACY_FLOW.replace("    def get_flow_status", '''    Path(os.environ["HERMES_HOME"], "imported-" + str(os.getpid())).touch()
    def get_flow_status''', 1)
FLAKY_FLOW = '''
    CALLS = []
    def get_flow_status():
        CALLS.append(1)
        if len(CALLS) == 1:
            raise RuntimeError("transient")
        return {"state": "pending"}
    def start_loopback_flow_background():
        return {"state": "pending"}
'''


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_ENABLE_PROJECT_PLUGINS", raising=False)
    monkeypatch.setattr(memory, "_iter_entry_points", lambda: [])
    monkeypatch.setattr(contract, "_SCHEMA_CACHE", {})
    (home / "config.yaml").write_text("memory:\n  provider: ''\n")
    return home


@pytest.fixture
def multiplex():
    from agent import secret_scope

    secret_scope.set_multiplex_active(True)
    try:
        yield
    finally:
        contract.shutdown_oauth()
        secret_scope.set_multiplex_active(False)


def _install(home, runtime=INERT, **companions):
    plugin = home / "plugins/probe"
    plugin.mkdir(parents=True, exist_ok=True)
    (plugin / "__init__.py").write_text(textwrap.dedent(runtime))
    for filename, source in companions.items():
        (plugin / filename).write_text(textwrap.dedent(source))
    return plugin


def _wait(check):
    deadline = time.monotonic() + 15
    while not check():
        assert time.monotonic() < deadline
        time.sleep(.02)


def test_for_provider_distinguishes_bundled_external_and_unknown(home, tmp_path, monkeypatch):
    assert contract.for_provider("holographic") is None is contract.for_provider("unknown")  # bundled keeps main's paths
    plugin = _install(home)
    external = contract.for_provider("probe")
    assert external.directory == plugin and external.companion("oauth_flow.py") is None
    package = tmp_path / "ep_probe"
    package.mkdir()
    (package / "__init__.py").write_text(textwrap.dedent(RAW))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(memory, "_iter_entry_points", lambda: [EntryPoint("ep_probe", "ep_probe", memory.ENTRY_POINTS_GROUP)])
    assert contract.for_provider("ep_probe").directory == package


def test_declared_schema_wins_and_follows_source_bytes_not_timestamps(home):
    # A same-size reinstall keeps the mtime; the bytes decide, and a caller holding the old schema keeps it.
    path = _install(home, RAW, **{"config_schema.py": DECLARED}) / "config_schema.py"
    stamp = path.stat()
    external = contract.for_provider("probe")
    retained = external.declared_schema()
    assert external.describe()["label"] == "A" and [f["kind"] for f in external.describe()["fields"]] == ["text", "number", "secret"]
    for label in ("B", "C"):
        path.write_text(textwrap.dedent(DECLARED).replace('label="A"', f'label="{label}"'))
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        assert external.describe()["label"] == label and retained.label == "A"
    path.write_text("raise ImportError('missing schema dependency')\n")
    raw = external.describe()  # the raw schema in Desktop kinds
    assert [f["kind"] for f in raw["fields"]] == ["text", "bool", "number", "secret"] and raw["label"] == "Probe"


def test_partial_save_clears_blank_keys_and_keeps_a_blank_secret(home):
    from dotenv import dotenv_values

    _install(home, **{"config_schema.py": DECLARED})
    stored = home / "probe/config.json"
    stored.parent.mkdir()
    stored.write_text(json.dumps({"endpoint": "old", "count": 1, "extra": "keep"}))
    (home / ".env").write_text("PROBE_KEY=old-key\n")
    external = contract.for_provider("probe")
    assert external.save({"endpoint": "", "api_key": "  "}) is True
    assert json.loads(stored.read_text()) == {"count": 1, "extra": "keep"}
    assert dotenv_values(home / ".env")["PROBE_KEY"] == "old-key"
    for invalid in ({"count": "many"}, {"unknown": "x"}):
        with pytest.raises(ValueError):
            external.save({"count": "5", **invalid})
    assert json.loads(stored.read_text()) == {"count": 1, "extra": "keep"}
    assert external.save({"count": "5", "api_key": "new-key"}) is True
    assert json.loads(stored.read_text())["count"] == 5 and dotenv_values(home / ".env")["PROBE_KEY"] == "new-key"
    fields = {f["key"]: f for f in external.describe()["fields"]}
    assert fields["api_key"]["value"] == "" and fields["api_key"]["is_set"] is True and fields["count"]["value"] == "5"


def test_raw_schema_saves_through_the_writer_the_provider_has(home):
    _install(home, RAW + NATIVE_SAVE)
    external = contract.for_provider("probe")
    form = external.describe()
    assert form["capabilities"] == {"save_without_activation": True, "supports_partial_updates": False, "requires_full_form": True}
    fields = {f["key"]: f for f in form["fields"]}
    assert fields["enabled"]["value"] == "true" and fields["count"]["value"] == "3"
    assert fields["token"] == {**fields["token"], "value": "", "default": "", "is_set": False}
    assert fields["dependent"]["when"] == {"enabled": True}  # the Desktop evaluates conditions itself
    for invalid in ({"count": "many"}, {"count": "12"}, {"unknown": "x"}):
        with pytest.raises(ValueError):
            external.save({"enabled": "true", **invalid})
    assert not (home / "native-calls.json").exists()
    external.save({"enabled": "false", "count": "4", "token": " "})  # the writer gets the whole form, a blank secret is skipped
    assert json.loads((home / "native-calls.json").read_text()) == {"dependent": "fallback", "enabled": False, "count": 4}
    assert external.save({"count": "7"}) is True  # missing fields come from stored values, then defaults
    assert json.loads((home / "native-calls.json").read_text()) == {"dependent": "fallback", "enabled": True, "count": 7}

    _install(home, RAW)  # the inherited no-op writer means host-owned config.yaml storage
    external = contract.for_provider("probe")
    assert external.describe()["capabilities"]["supports_partial_updates"] is True
    assert external.save({"count": "7"}) is True
    config = yaml.safe_load((home / "config.yaml").read_text())["memory"]
    assert config["probe"]["count"] == 7 and config["provider"] == ""


@pytest.mark.parametrize("payload,expected", [
    ({"state": "pending", "detail": "private"}, {"state": "pending", "detail": "Waiting for browser consent"}),
    ({"state": "idle", "connected": True, "auth": "apikey"}, {"state": "idle", "detail": "", "connected": True, "auth": "apikey"}),
    ({"state": "weird", "connected": "yes", "auth": "token", "home": "/x"}, {"state": "error", "detail": "Authorization did not complete", "auth": None}),
])
def test_normalize_status_keeps_only_the_public_fields(payload, expected):
    assert contract.normalize_status(payload) == expected


def test_hooks_with_hermes_home_run_in_a_thread_of_this_process_and_pin_while_pending(home, multiplex):
    plugin = _install(home, **{"oauth_flow.py": HOME_FLOW})
    external = contract.for_provider("probe")
    assert external.oauth(home, start=True)["state"] in ("idle", "pending")
    _wait(lambda: external.oauth(home, start=False)["state"] == "pending")
    assert (home / "receipt").read_text() == str(os.getpid())
    (plugin / "oauth_flow.py").write_text(textwrap.dedent(LEGACY_FLOW))  # a replaced package must not reset the flow
    assert external.oauth(home, start=True) == {"state": "pending", "detail": "Waiting for browser consent"}
    (home / "release").touch()
    _wait(lambda: external.oauth(home, start=False)["state"] == "connected")
    assert not contract._pending  # a settled flow releases its runner


def test_zero_argument_hooks_run_in_a_profile_bound_child(home, multiplex):
    _install(home, **{"oauth_flow.py": LEGACY_FLOW})
    external = contract.for_provider("probe")
    status = external.oauth(home, start=False)
    assert status == {"state": "idle", "detail": ""}
    assert (home / "child-pid").read_text() != str(os.getpid())
    assert external.oauth(home, start=True) == status and not contract._pending
    (home / "plugins/probe/oauth_flow.py").write_text("raise ImportError('private')\n")
    with pytest.raises(contract.CompanionError):
        external.oauth(home, start=False)


def test_zero_argument_companion_is_read_not_imported_by_the_server_process(home, multiplex):
    _install(home, **{"oauth_flow.py": MARKED_FLOW})
    external = contract.for_provider("probe")
    for start in (False, True):
        assert external.oauth(home, start=start)["state"] == "idle"
    assert not (home / f"imported-{os.getpid()}").exists()
    assert (home / f"imported-{(home / 'child-pid').read_text()}").exists()  # only the child ran the module body


def test_a_failing_status_call_keeps_the_pending_runner(home, multiplex):
    _install(home, **{"oauth_flow.py": FLAKY_FLOW})
    external, key = contract.for_provider("probe"), (str(home), "probe")
    assert external.oauth(home, start=True)["state"] == "pending"
    runner = contract._pending[key]
    assert external.oauth(home, start=False)["state"] == "error"
    assert contract._pending[key] is runner
    assert external.oauth(home, start=False)["state"] == "pending"
