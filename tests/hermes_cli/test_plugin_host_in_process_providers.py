"""plugins.isolation: host: a model-provider profile that needs the Hermes process loads there only when the
operator names its plugin in ``plugins.host.in_process_providers``.

Every test goes through the real ``providers`` lookup with a throwaway home and real plugin directories, so
the profile-extraction child really runs. Each plugin's module body appends ``<pid> <host flag>`` to a marker
file: a line with this test's pid means the plugin's code ran in the calling process.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import textwrap
from pathlib import Path

import hermes_yaml as yaml
import pytest

from hermes_cli import plugins as plugins_mod

_MARK = textwrap.dedent("""
    import os
    with open({marker!r}, "a", encoding="utf-8") as _f:
        _f.write(f"{{os.getpid()}} {{os.environ.get('HERMES_PLUGIN_HOST_PROCESS')}}\\n")
    from providers import register_provider
    from providers.base import ProviderProfile
""")

# A profile that builds its own client: the extraction child reports it as one that cannot run in the host.
_OWN_CLIENT = _MARK + textwrap.dedent("""
    class Local(ProviderProfile):
        def create_client(self, **kwargs):
            return {{"pid": os.getpid()}}

    register_provider(Local(name={name!r}, base_url="acp://{name}"))
""")

# An ordinary profile: it runs in the plugin host.
_PLAIN = _MARK + textwrap.dedent("""
    class Hosted(ProviderProfile):
        def build_extra_body(self, *, session_id=None, **context):
            return {{"pid": os.getpid()}}

    register_provider(Hosted(name={name!r}, base_url="https://{name}.example/v1"))
""")

# One of each in one plugin.
_PAIR = _MARK + textwrap.dedent("""
    class Hosted(ProviderProfile):
        def build_extra_body(self, *, session_id=None, **context):
            return {{"pid": os.getpid()}}

    class Local(ProviderProfile):
        def create_client(self, **kwargs):
            return {{"pid": os.getpid()}}

    register_provider(Hosted(name={name!r} + "-hosted", base_url="https://{name}.example/v1"))
    register_provider(Local(name={name!r} + "-local", base_url="acp://{name}"))
""")

# No create_client override: inside the extraction child the plugin writes its own answer on the child's
# protocol output (the one pipe other than stderr), marking its profile as needing the Hermes process.
_SELF_MARKED = _MARK + textwrap.dedent("""
    import json, stat, sys
    if os.environ.get("HERMES_PLUGIN_HOST_PROCESS") == "1" and "--extract-profiles" in sys.argv:
        err = os.fstat(2)
        for fd in range(3, 256):
            try:
                st = os.fstat(fd)
            except OSError:
                continue
            if stat.S_ISFIFO(st.st_mode) and (st.st_ino, st.st_dev) != (err.st_ino, err.st_dev):
                answer = {{"profiles": [{{"name": {name!r}, "in_process": "chosen by the plugin itself"}}]}}
                os.write(fd, json.dumps(answer).encode())
                os._exit(0)

    register_provider(ProviderProfile(name={name!r}, base_url="https://{name}.example/v1"))
""")


def _reset_providers() -> None:
    import providers

    providers._HOME_LAYERS.clear()
    for mod in [m for m in sys.modules if m.startswith("_hermes_user_provider")]:
        del sys.modules[mod]


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    _reset_providers()
    yield home
    host = getattr(plugins_mod.get_plugin_manager(), "_plugin_host_instance", None)
    if host is not None:
        host.shutdown()
    _reset_providers()


def _config(home: Path, **plugins) -> None:
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": plugins}), encoding="utf-8")


def _plugin(home: Path, name: str, source: str, *, layout: str = "nested") -> Path:
    """Install *source* as plugin *name*; returns the marker file its module body appends to."""
    directory = home / "plugins" / "model-providers" / name if layout == "nested" else home / "plugins" / name
    directory.mkdir(parents=True)
    marker = home / f"ran-{name}"
    (directory / "plugin.yaml").write_text(f"name: {name}\nkind: model-provider\n", encoding="utf-8")
    (directory / "__init__.py").write_text(source.format(marker=str(marker), name=name), encoding="utf-8")
    return marker


def _pids(marker: Path) -> list[str]:
    return [line.split()[0] for line in marker.read_text(encoding="utf-8").splitlines()] if marker.exists() else []


def _messages(caplog, needle: str) -> list[str]:
    return [r.getMessage() for r in caplog.records if needle in r.getMessage()]


def _assert_refused(name: str, marker: Path, caplog, reason: str) -> None:
    import providers

    profile = providers.get_provider_profile(name)
    pids = _pids(marker)
    assert pids, "the extraction child should have run the plugin"
    assert str(os.getpid()) not in pids, "the plugin's module ran in the calling process"
    assert profile is None
    error = providers.plugin_load_error(name)
    assert reason in error and "plugins.isolation: in_process" in error
    assert f"add {name!r} to plugins.host.in_process_providers" in error
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and name in r.getMessage()]
    assert [w.getMessage() for w in warnings] == [error]


# ── Refused unless the operator names the plugin ─────────────────────────────


@pytest.mark.platforms("any")  # the extraction child is a separate process: its env/home resolution is per-OS
@pytest.mark.parametrize("layout", ["nested", "flat"])
def test_a_profile_that_builds_its_own_client_is_refused_and_its_code_never_runs_here(home, caplog, layout):
    marker = _plugin(home, "ownclient", _OWN_CLIENT, layout=layout)
    _config(home, isolation="host")
    with caplog.at_level(logging.WARNING, logger="providers"):
        _assert_refused("ownclient", marker, caplog, "overrides create_client()")


@pytest.mark.platforms("posix")  # finds the child's protocol pipe with os.fstat
def test_a_plugin_that_marks_its_own_profile_as_in_process_is_refused_the_same_way(home, caplog):
    marker = _plugin(home, "selfmark", _SELF_MARKED)
    _config(home, isolation="host")
    with caplog.at_level(logging.WARNING, logger="providers"):
        _assert_refused("selfmark", marker, caplog, "chosen by the plugin itself")


@pytest.mark.platforms("any")
def test_the_refused_profiles_sibling_keeps_running_in_the_plugin_host(home, caplog):
    import providers

    marker = _plugin(home, "pairprov", _PAIR)
    _config(home, isolation="host")
    with caplog.at_level(logging.WARNING, logger="providers"):
        hosted = providers.get_provider_profile("pairprov-hosted")
        local = providers.get_provider_profile("pairprov-local")
    assert str(os.getpid()) not in _pids(marker), "the plugin's module ran in the calling process"
    assert local is None
    assert hosted is not None and type(hosted).__module__ == "hermes_cli.plugin_host_profiles"
    assert hosted.build_extra_body(session_id="s")["pid"] != os.getpid()
    error = providers.plugin_load_error("pairprov")
    assert "'pairprov-local'" in error and "pairprov-hosted" not in error
    assert len(_messages(caplog, "pairprov-local")) == 1


@pytest.mark.platforms("any")
def test_a_cached_answer_that_marks_a_profile_in_process_is_refused_without_an_import(home, caplog):
    """An extraction cached by an earlier build, or earlier by the plugin, still only leads to a refusal."""
    import providers

    marker = _plugin(home, "cachedprov", _PLAIN)
    _config(home, isolation="host")
    assert providers.get_provider_profile("cachedprov") is not None
    [cache] = (home / "cache" / "plugin_host" / "model-providers").glob("cachedprov-*.json")
    data = json.loads(cache.read_text(encoding="utf-8"))
    [answer] = [value for value in data.values() if isinstance(value, dict) and "profiles" in value]
    answer["profiles"] = [{"name": "cachedprov", "in_process": "recorded by an earlier build"}]
    cache.write_text(json.dumps(data), encoding="utf-8")
    runs = len(_pids(marker))

    _reset_providers()  # a fresh process: nothing of the plugin loaded yet
    with caplog.at_level(logging.WARNING, logger="providers"):
        _assert_refused("cachedprov", marker, caplog, "recorded by an earlier build")
    assert len(_pids(marker)) == runs, "the cached answer is read; no child ran again"


# ── The operator's list ───────────────────────────────────────────────────────


@pytest.mark.platforms("any")
def test_a_plugin_named_in_the_list_loads_that_profile_in_process(home, caplog):
    import providers

    marker = _plugin(home, "ownclient", _OWN_CLIENT)
    _config(home, isolation="host", host={"in_process_providers": ["ownclient"]})
    with caplog.at_level(logging.WARNING, logger="providers"):
        profile = providers.get_provider_profile("ownclient")
    assert profile is not None and profile.create_client()["pid"] == os.getpid()
    assert str(os.getpid()) in _pids(marker)
    assert providers.plugin_load_error("ownclient") == ""
    [notice] = _messages(caplog, "ownclient")
    assert "create_client" in notice and "plugins.host.in_process_providers names this plugin" in notice


@pytest.mark.platforms("any")
def test_the_list_lets_in_only_the_plugins_it_names(home, caplog):
    import providers

    allowed = _plugin(home, "ownclient", _OWN_CLIENT)
    other = _plugin(home, "otherclient", _OWN_CLIENT)
    plain = _plugin(home, "plainprov", _PLAIN)
    _config(home, isolation="host", host={"in_process_providers": ["ownclient", "plainprov", "notinstalled"]})
    with caplog.at_level(logging.WARNING, logger="providers"):
        assert providers.get_provider_profile("ownclient") is not None
        _assert_refused("otherclient", other, caplog, "overrides create_client()")
        hosted = providers.get_provider_profile("plainprov")
    assert str(os.getpid()) in _pids(allowed)
    # Naming a plugin with no such profile changes nothing: it stays in the host.
    assert hosted is not None and type(hosted).__module__ == "hermes_cli.plugin_host_profiles"
    assert str(os.getpid()) not in _pids(plain)


# ── The list comes from whoever set the isolation ────────────────────────────
# One Hermes process serving several profiles pins plugins.isolation in managed scope; a profile's own
# config.yaml must not be able to let its own provider plugin into that process.


@pytest.fixture
def managed(home, tmp_path, monkeypatch):
    directory = tmp_path / "managed"
    directory.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(directory))
    yield directory
    from hermes_cli import managed_scope

    managed_scope.invalidate_managed_cache()


def _managed_config(directory: Path, **plugins) -> None:
    import hermes_cli.config as cfg
    from hermes_cli import managed_scope

    (directory / "config.yaml").write_text(yaml.safe_dump({"plugins": plugins}), encoding="utf-8")
    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    managed_scope.invalidate_managed_cache()


@pytest.mark.platforms("any")
@pytest.mark.parametrize("managed_list", ["unset", "empty"])
def test_a_profile_cannot_list_its_own_plugin_when_managed_scope_pins_isolation(home, managed, caplog,
                                                                                 managed_list):
    marker = _plugin(home, "ownclient", _OWN_CLIENT)
    _config(home, host={"in_process_providers": ["ownclient"]})
    pinned = {"host": {"in_process_providers": []}} if managed_list == "empty" else {}
    _managed_config(managed, isolation="host", **pinned)
    with caplog.at_level(logging.WARNING, logger="providers"):
        _assert_refused("ownclient", marker, caplog, "overrides create_client()")
    import providers

    error = providers.plugin_load_error("ownclient")
    assert f"in the managed config ({managed / 'config.yaml'})" in error
    assert "a profile's own list is ignored" in error


@pytest.mark.platforms("any")
def test_a_plugin_listed_in_managed_scope_beside_the_pinned_isolation_loads_as_before(home, managed, caplog):
    import providers

    marker = _plugin(home, "ownclient", _OWN_CLIENT)
    _config(home)
    _managed_config(managed, isolation="host", host={"in_process_providers": ["ownclient"]})
    with caplog.at_level(logging.WARNING, logger="providers"):
        profile = providers.get_provider_profile("ownclient")
    assert profile is not None and profile.create_client()["pid"] == os.getpid()
    assert str(os.getpid()) in _pids(marker)
    [notice] = _messages(caplog, "ownclient")
    assert "plugins.host.in_process_providers names this plugin" in notice


@pytest.mark.platforms("any")
def test_isolation_set_in_the_profile_config_reads_the_list_beside_it(home, managed, caplog):
    """A managed scope that pins something else leaves the isolation and its list to the profile."""
    import providers

    marker = _plugin(home, "ownclient", _OWN_CLIENT)
    _config(home, isolation="host", host={"in_process_providers": ["ownclient"]})
    _managed_config(managed, scan_on_install=True)
    profile = providers.get_provider_profile("ownclient")
    assert profile is not None and profile.create_client()["pid"] == os.getpid()
    assert str(os.getpid()) in _pids(marker)


# ── Controls: the same on the base and with this change ──────────────────────


def test_default_isolation_loads_the_plugin_in_process(home):
    import providers

    marker = _plugin(home, "ownclient", _OWN_CLIENT)
    _config(home, enabled=[])
    profile = providers.get_provider_profile("ownclient")
    assert profile is not None and profile.create_client()["pid"] == os.getpid()
    assert _pids(marker) == [str(os.getpid())]


@pytest.mark.platforms("any")
def test_a_disabled_plugin_runs_nowhere_under_host(home):
    import providers

    marker = _plugin(home, "ownclient", _OWN_CLIENT)
    _config(home, isolation="host", disabled=["ownclient"])
    assert providers.get_provider_profile("ownclient") is None
    assert not marker.exists()


@pytest.mark.platforms("any")
def test_an_ordinary_provider_plugin_stays_in_the_plugin_host(home):
    import providers

    marker = _plugin(home, "plainprov", _PLAIN)
    _config(home, isolation="host")
    profile = providers.get_provider_profile("plainprov")
    assert profile is not None and type(profile).__module__ == "hermes_cli.plugin_host_profiles"
    assert profile.build_extra_body(session_id="s")["pid"] != os.getpid()
    assert _pids(marker) and str(os.getpid()) not in _pids(marker)
    assert providers.plugin_load_error("plainprov") == ""


@pytest.mark.platforms("any")
def test_a_recalled_plugin_is_refused_before_any_child_starts_even_when_listed(home):
    import providers

    marker = _plugin(home, "ownclient", _OWN_CLIENT)
    _config(home, isolation="host", host={"in_process_providers": ["ownclient"]})
    (home / "cache").mkdir(exist_ok=True)
    (home / "cache" / "plugin-catalog.json").write_text(
        json.dumps({"entries": [], "removed": [{"name": "ownclient", "reason": "withdrawn by its author"}]}),
        encoding="utf-8")
    assert providers.get_provider_profile("ownclient") is None
    assert not marker.exists()
    assert providers.plugin_load_error("ownclient").startswith("removed from the Hermes plugin catalog")


@pytest.mark.platforms("any")
def test_bundled_providers_that_build_their_own_client_still_load_under_host(home):
    import providers

    _config(home, isolation="host")
    for name in ("gemini", "solstice"):
        profile = providers.get_provider_profile(name)
        assert profile is not None and type(profile).__module__.startswith("plugins.model_providers.")
