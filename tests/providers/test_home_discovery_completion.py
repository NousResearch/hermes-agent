"""Home scans must finish before auth callbacks can request the same layer again."""
from pathlib import Path
from unittest.mock import patch

import pytest


def _plugin(home, name="completion-probe", before="", after=""):
    directory = home / "plugins" / "model-providers" / name
    directory.mkdir(parents=True)
    (directory / "__init__.py").write_text(
        before
        + "from providers import register_provider\n"
        + "from providers.base import ProviderProfile\n"
        + f"register_provider(ProviderProfile(name={name!r}, aliases=({(name + '-alias')!r},), "
        + f"env_vars=('COMPLETION_PROBE_KEY',), base_url='https://{name}.invalid/v1'))\n"
        + after,
        encoding="utf-8",
    )
    return directory


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "bootstrap"))
    import providers
    import hermes_cli.auth as auth
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    import copy
    from hermes_cli import auth_plugin_providers
    monkeypatch.setattr(providers, "_HOME_LAYERS", {})
    monkeypatch.setattr(auth, "PROVIDER_REGISTRY", copy.deepcopy(auth.PROVIDER_REGISTRY))
    monkeypatch.setattr(auth_plugin_providers, "PLUGIN_MIRRORED_PROVIDERS",
                        set(auth_plugin_providers.PLUGIN_MIRRORED_PROVIDERS))
    home = tmp_path / "active"
    token = set_hermes_home_override(home)
    try:
        yield providers, auth, home
    finally:
        reset_hermes_home_override(token)


def test_completed_scan_is_not_reentered_by_auth(isolated_home, record_property):
    import sys
    providers, auth, home = isolated_home
    _plugin(home)
    recursions = []
    traced_files = {providers.__file__, auth.sync_plugin_provider_registry.__code__.co_filename}

    def trace(frame, event, arg):
        if frame.f_code.co_filename not in traced_files:
            return None
        # Inspect the existing catch path too: deep exception events may not be delivered.
        if ((event == "exception" and isinstance(arg[1], RecursionError))
                or (event == "line" and isinstance(sys.exception(), RecursionError))):
            recursions.append((frame.f_code.co_name, frame.f_lineno))
        return trace

    with (patch.object(providers, "_scan_home_layer", wraps=providers._scan_home_layer) as scan,
          patch.object(providers, "_sync_auth_registry", wraps=providers._sync_auth_registry) as sync):
        previous_trace = sys.gettrace()
        sys.settrace(trace)
        try:
            profiles = providers.list_providers()
        finally:
            trace_active = sys.gettrace() is trace
            sys.settrace(previous_trace)
        record_property("cold_scans", scan.call_count)
        record_property("cold_syncs", sync.call_count)
        record_property("recursion_exception_events", len(recursions))
        record_property("trace_remained_active", trace_active)
        assert scan.call_count == 1
        assert sync.call_count == 1
        assert trace_active
        assert recursions == []
        assert any(p.name == "completion-probe" for p in profiles)
        row = auth.PROVIDER_REGISTRY["completion-probe"]
        assert auth.PROVIDER_REGISTRY["completion-probe-alias"] is row
        assert row.inference_base_url == "https://completion-probe.invalid/v1"
        assert providers.get_provider_profile("completion-probe-alias").name == row.id
        providers.list_providers()
        assert scan.call_count == 1  # warm lookups do not scan again


def test_import_callback_observes_partial_layer_then_auth_is_healed(isolated_home):
    providers, auth, home = isolated_home
    early = _plugin(
        home, "aaa-callback-probe",
        before="from providers import list_providers\n"
               "from hermes_cli.auth import sync_plugin_provider_registry, PROVIDER_REGISTRY\n"
               "partial_names = {p.name for p in list_providers()}\n"
               "sync_plugin_provider_registry()\n"
               "partial_auth = set(PROVIDER_REGISTRY)\n",
    )
    _plugin(home, "zzz-callback-probe")
    with patch.object(providers, "_scan_home_layer", wraps=providers._scan_home_layer) as scan:
        profiles = providers.list_providers()
        assert scan.call_count == 1
    import sys
    from hermes_constants import hermes_home_key
    module = sys.modules[providers._user_module_name(early, hermes_home_key(home))]
    names = {"aaa-callback-probe", "zzz-callback-probe"}
    assert names.isdisjoint(module.partial_names)
    assert names.isdisjoint(module.partial_auth)
    assert names <= {p.name for p in profiles}
    for name in names:
        assert auth.PROVIDER_REGISTRY[name + "-alias"] is auth.PROVIDER_REGISTRY[name]
