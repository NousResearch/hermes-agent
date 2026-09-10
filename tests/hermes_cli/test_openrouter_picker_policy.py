"""Fixed catalog fixtures exercise profile policy through real picker assembly."""

from types import SimpleNamespace
from pathlib import Path
import json
import os
import subprocess
import sys

import pytest

from hermes_cli import config, models, inventory, model_switch_providers
from hermes_cli.models import _fetch_live_catalog_index
from hermes_cli.models_openrouter_policy import (
    explicitly_zero_priced, openrouter_free_only, openrouter_picker_models,
)
from hermes_constants import set_hermes_home_override, reset_hermes_home_override


CURATED = [
    ("vendor/paid", "recommended"),
    ("vendor/zero", ""),
    ("vendor/tag:free", ""),
    ("vendor/label", "free"),
    ("vendor/unknown", "free context"),
]
LIVE = [
    {"id": "vendor/paid", "pricing": {"prompt": "0.01", "completion": "0.02"}},
    {"id": "vendor/zero", "pricing": {"prompt": "0", "completion": "0.0"}},
    {"id": "vendor/tag:free", "pricing": {"prompt": "0.01", "completion": "0"}},
    {"id": "vendor/label", "pricing": {"prompt": "0", "completion": "0"}},
    {"id": "vendor/unknown", "pricing": {"prompt": "0"}},
]


@pytest.fixture
def profile(tmp_path, monkeypatch):
    token = set_hermes_home_override(tmp_path)
    monkeypatch.setattr(models, "_openrouter_catalog_cache", None)
    monkeypatch.setattr(models, "OPENROUTER_MODELS", CURATED)
    monkeypatch.setattr(models, "get_preferred_silent_default_model", lambda _provider: "vendor/paid")
    monkeypatch.setattr(models, "_seed_reasoning_caps", lambda *_a: None)
    monkeypatch.setattr("hermes_cli.model_catalog.get_curated_openrouter_models", lambda: CURATED)
    monkeypatch.setattr(models, "_fetch_live_catalog_index",
                        lambda *_a, **_kw: (LIVE, {row["id"]: row for row in LIVE}))
    monkeypatch.setattr(config, "load_config", config.load_config_readonly)
    try:
        yield tmp_path
    finally:
        reset_hermes_home_override(token)


def _policy(home, value):
    (home / "config.yaml").write_text(f"models:\n  openrouter:\n    free_only: {value}\n")


@pytest.mark.parametrize("price", [
    {}, {"prompt": "0"}, {"completion": "0"},
    {"prompt": True, "completion": False},
    {"prompt": "1e-9999", "completion": "0"},
    {"prompt": "-1", "completion": "0"},
    {"prompt": "NaN", "completion": "0"},
    {"prompt": "Infinity", "completion": "0"},
    {"prompt": "invalid", "completion": "0"},
])
def test_unknown_and_nonzero_prices_are_not_free(price):
    assert explicitly_zero_priced(price) is False


@pytest.mark.parametrize("price", [
    {"prompt": 0, "completion": "0.000"},
    {"prompt": "0e-9999", "completion": 0.0},
])
def test_explicit_finite_zero_prices(price):
    assert explicitly_zero_priced(price) is True


@pytest.mark.parametrize("invalid", ['"false"', "1", "null", "[]"])
def test_profile_setting_rejects_non_boolean_in_real_config(profile, invalid):
    _policy(profile, invalid)
    with pytest.raises(ValueError, match="must be a YAML boolean"):
        openrouter_picker_models()
    assert any(issue.severity == "error" and "free_only" in issue.message
               for issue in config.validate_config_structure(config.load_config_readonly()))


def test_policy_default_and_toggling_leave_raw_slug_resolution_unrestricted(profile):
    assert openrouter_free_only() is False
    assert [mid for mid, _ in openrouter_picker_models()] == [mid for mid, _ in CURATED]
    _policy(profile, "true")
    assert [mid for mid, _ in openrouter_picker_models()] == ["vendor/zero", "vendor/label"]
    assert models._find_openrouter_slug("vendor/paid") == "vendor/paid"
    _policy(profile, "false")
    assert "vendor/paid" in [mid for mid, _ in openrouter_picker_models()]


def test_one_live_response_populates_separate_raw_and_free_choices(profile, monkeypatch):
    calls = []
    def fetch(*_args, **_kwargs):
        calls.append(True)
        return LIVE, {row["id"]: row for row in LIVE}
    monkeypatch.setattr(models, "_fetch_live_catalog_index", fetch)
    assert "vendor/paid" in models.model_ids()
    _policy(profile, "true")
    assert [mid for mid, _ in openrouter_picker_models()] == ["vendor/zero", "vendor/label"]
    assert len(calls) == 1


def test_unavailable_catalog_uses_exact_offline_annotations(profile, monkeypatch):
    _policy(profile, "true")
    monkeypatch.setattr(models, "_fetch_live_catalog_index", lambda *_a, **_kw: None)
    assert [mid for mid, _ in openrouter_picker_models()] == ["vendor/tag:free", "vendor/label"]
    assert [mid for mid, _ in models.fetch_openrouter_models()] == [mid for mid, _ in CURATED]


def test_authoritative_empty_replaces_warm_cache_and_survives_outage(profile, monkeypatch):
    _policy(profile, "true")
    assert openrouter_picker_models()
    monkeypatch.setattr(models, "_fetch_live_catalog_index", lambda *_a, **_kw: ([], {}))
    assert openrouter_picker_models(force_refresh=True) == []
    monkeypatch.setattr(models, "_fetch_live_catalog_index", lambda *_a, **_kw: None)
    assert openrouter_picker_models(force_refresh=True) == []
    assert openrouter_picker_models() == []


def test_no_eligible_live_rows_never_use_free_suffix_or_stale_generic_cache(profile, monkeypatch):
    _policy(profile, "true")
    paid = LIVE[:1] + LIVE[2:3] + LIVE[4:]
    monkeypatch.setattr(models, "_fetch_live_catalog_index",
                        lambda *_a, **_kw: (paid, {row["id"]: row for row in paid}))
    assert openrouter_picker_models() == []
    monkeypatch.setattr(models, "_fetch_live_catalog_index", lambda *_a, **_kw: ([], {}))
    monkeypatch.setattr(models, "_load_provider_models_cache",
                        lambda: {"openrouter": {"models": ["vendor/paid"]}})
    rows = [_row()]
    from hermes_cli.models_openrouter_policy import apply_openrouter_picker_policy
    apply_openrouter_picker_policy(rows, force_refresh=True)
    assert rows[0]["models"] == []


def test_catalog_json_preserves_tiny_positive_numeric_prices(profile, monkeypatch):
    from io import BytesIO

    payload = b'{"data":[{"id":"vendor/zero","pricing":{"prompt":1e-9999,"completion":0}}]}'
    # Exercise real HTTP JSON decoding, not an already rounded Python price fixture.
    monkeypatch.setattr(models, "_urlopen_model_catalog_request", lambda *_a, **_kw: BytesIO(payload))
    raw, index = _fetch_live_catalog_index("https://catalog.invalid/models", 1,
                                          models._urlopen_model_catalog_request, parse_float=str)
    assert index["vendor/zero"]["pricing"]["prompt"] == "1e-9999"
    assert explicitly_zero_priced(index["vendor/zero"]["pricing"]) is False


def test_shared_catalog_float_types_and_real_openrouter_reasoning_seed(profile, monkeypatch):
    from io import BytesIO
    from hermes_cli.models_reasoning_caps import _seed_reasoning_caps

    payload = (b'{"data":[{"id":"vendor/zero","metadata_number":0.5,'
               b'"pricing":{"prompt":1e-9999,"completion":0},'
               b'"supported_parameters":["tools","reasoning"],'
               b'"reasoning":{"mandatory":true,"supported_efforts":["low","high"]}}]}')
    opener = lambda *_a, **_kw: BytesIO(payload)
    _, index = _fetch_live_catalog_index("https://catalog.invalid/models", 1, opener)
    # Other consumers, including AI Gateway, retain ordinary JSON numeric types.
    assert type(index["vendor/zero"]["metadata_number"]) is float
    monkeypatch.setattr(models, "_urlopen_model_catalog_request", opener)
    monkeypatch.setattr(models, "_fetch_live_catalog_index", _fetch_live_catalog_index)
    monkeypatch.setattr(models, "_seed_reasoning_caps", _seed_reasoning_caps)
    monkeypatch.setattr(models, "_openrouter_reasoning_caps_cache", None)
    assert models.fetch_openrouter_models(free_only=True) == []
    expected = {"supports_reasoning": True, "supported_efforts": ["low", "high"], "mandatory": True}
    assert models._openrouter_reasoning_caps_cache["vendor/zero"] == expected
    stored = json.loads((profile / "cache" / "reasoning_caps.json").read_text())
    assert stored[models._OPENROUTER_CATALOG_URL]["caps"]["vendor/zero"] == expected


@pytest.mark.parametrize("body", [b'{}', b'[]', b'{"data":null}', b'{"data":{}}'])
def test_malformed_catalog_is_unavailable_instead_of_authoritative_empty(body):
    from io import BytesIO

    assert _fetch_live_catalog_index("https://catalog.invalid/models", 1,
                                     lambda *_a, **_kw: BytesIO(body)) is None


def test_cache_is_scoped_to_profile_and_policy(profile, tmp_path, monkeypatch):
    _policy(profile, "true")
    assert openrouter_picker_models()
    second = tmp_path / "other-profile"
    second.mkdir()
    _policy(second, "true")
    token = set_hermes_home_override(second)
    try:
        monkeypatch.setattr(models, "_fetch_live_catalog_index", lambda *_a, **_kw: ([], {}))
        assert openrouter_picker_models() == []
    finally:
        reset_hermes_home_override(token)
    assert [mid for mid, _ in openrouter_picker_models()] == ["vendor/zero", "vendor/label"]


def _cold_catalog(profile, *, live_rows=None, force_refresh=False):
    """Read the real profile disk cache from a fresh interpreter with synthetic transport."""
    probe = r'''
import json, sys
sys.path.insert(0, sys.argv[2])
from hermes_cli import models, model_catalog
settings = json.loads(sys.argv[1])
calls = []
def fetch(*args, **kwargs):
    calls.append(True)
    rows = settings["live_rows"]
    return None if rows is None else (rows, {row["id"]: row for row in rows})
models._fetch_live_catalog_index = fetch
models._seed_reasoning_caps = lambda *args: None
models.get_preferred_silent_default_model = lambda provider: "vendor/zero"
model_catalog.get_curated_openrouter_models = lambda: settings["curated"]
print(json.dumps({
    "raw": models.fetch_openrouter_models(force_refresh=settings["force_refresh"]),
    "free": models.fetch_openrouter_models(free_only=True, force_refresh=settings["force_refresh"]),
    "calls": len(calls),
}))
'''
    user = profile / "isolated-home"
    temp = user / "temp"
    temp.mkdir(parents=True, exist_ok=True)
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
           "HOME": str(user), "HERMES_HOME": str(profile), "USERPROFILE": str(user),
           "LOCALAPPDATA": str(user / "AppData/Local"), "APPDATA": str(user / "AppData/Roaming"),
           "TEMP": str(temp), "TMP": str(temp), "TMPDIR": str(temp)}
    if "SYSTEMROOT" in os.environ:
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", probe, json.dumps({
            "live_rows": live_rows, "curated": CURATED, "force_refresh": force_refresh,
        }), str(Path(models.__file__).resolve().parent.parent)],
        cwd=profile,
        env=env,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("legacy_cache", [False, True])
def test_cold_disk_cache_separates_policy_and_refreshes_legacy_free_view(profile, monkeypatch, legacy_cache):
    monkeypatch.setattr(models, "get_preferred_silent_default_model", lambda _provider: "vendor/zero")
    raw = models.fetch_openrouter_models()
    free = models.fetch_openrouter_models(free_only=True)
    assert ("vendor/zero", "default") in free  # eligibility cannot be recovered from this label
    if legacy_cache:
        path = models._openrouter_catalog_disk_path()
        payload = json.loads(path.read_text())
        payload.pop("schema_version")
        payload.pop("free_curated")
        path.write_text(json.dumps(payload))

    cold = _cold_catalog(profile, live_rows=LIVE)
    assert cold["raw"] == [list(row) for row in raw]
    assert cold["free"] == [list(row) for row in free]
    assert cold["calls"] == int(legacy_cache)


@pytest.mark.parametrize("force_refresh", [False, True])
@pytest.mark.parametrize("live_rows", [[], LIVE[:1]])
def test_empty_free_disk_cache_survives_cold_process_and_refresh_outage(
    profile, monkeypatch, live_rows, force_refresh,
):
    monkeypatch.setattr(models, "_fetch_live_catalog_index",
                        lambda *_a, **_kw: (live_rows, {row["id"]: row for row in live_rows}))
    assert models.fetch_openrouter_models(free_only=True) == []
    raw = models.fetch_openrouter_models()

    cold = _cold_catalog(profile, force_refresh=force_refresh)
    assert cold["free"] == []
    assert cold["raw"] == [list(row) for row in raw]
    assert cold["calls"] == (2 if force_refresh else 0)


def _row():
    return dict(slug="openrouter", name="OpenRouter", is_current=True,
                models=["vendor/paid", "vendor/unknown"], total_models=2, source="hermes")


def _isolate_provider_sources(monkeypatch):
    """Inject an account catalog at the source; leave finalization/inventory code real."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(model_switch_providers, "_build_curated_lists", lambda *_a: {})
    monkeypatch.setattr(model_switch_providers, "_collect_authed_provider_slugs", lambda *_a: [])
    monkeypatch.setattr(model_switch_providers, "_lap_builtin_rows", lambda b, *_a: b.results.append(_row()))
    for name in ("_lap_overlay_rows", "_lap_canonical_rows",
                 "_lap_user_provider_rows", "_lap_bare_custom_row", "_lap_custom_provider_rows"):
        monkeypatch.setattr(model_switch_providers, name, lambda *_a: None)
    monkeypatch.setattr(inventory, "_local_runtime_row", lambda *_a: None)
    monkeypatch.setattr(inventory, "_moa_provider_row", lambda *_a: None)


@pytest.mark.parametrize("configured_policy", [None, "false"])
def test_default_picker_populates_openrouter_when_provider_cache_is_empty(
    profile, monkeypatch, configured_policy,
):
    if configured_policy is not None:
        _policy(profile, configured_policy)
    build_builtin = model_switch_providers._lap_builtin_rows
    _isolate_provider_sources(monkeypatch)
    monkeypatch.setattr(model_switch_providers, "_lap_builtin_rows", build_builtin)
    monkeypatch.setattr(model_switch_providers, "_iter_builtin_candidates", lambda *_a: [
        ("openrouter", "openrouter", SimpleNamespace(name="OpenRouter"), ()),
    ])
    monkeypatch.setattr(model_switch_providers, "_any_env", lambda *_a: True)
    monkeypatch.setattr("agent.models_dev.get_provider_info", lambda *_a: None)
    monkeypatch.setattr(models, "cached_provider_model_ids", lambda *_a: [])

    rows = model_switch_providers.list_picker_providers(current_provider="openrouter")
    assert len(rows) == 1
    assert rows[0]["slug"] == "openrouter"
    assert rows[0]["models"] == [mid for mid, _ in CURATED]
    assert rows[0]["catalog_authoritative"] is True
    assert rows[0]["free_only"] is False


@pytest.mark.parametrize("initial_policy", [False, True])
def test_finalized_rows_follow_changed_policy_and_skip_matching_policy(
    profile, monkeypatch, initial_policy,
):
    from hermes_cli.models_openrouter_policy import apply_openrouter_picker_policy

    fetch = models.fetch_openrouter_models
    calls = []
    def record_fetch(**kwargs):
        calls.append(kwargs["free_only"])
        return fetch(**kwargs)
    monkeypatch.setattr(models, "fetch_openrouter_models", record_fetch)
    rows = [_row()]
    for policy in (initial_policy, not initial_policy):
        _policy(profile, str(policy).lower())
        apply_openrouter_picker_policy(rows)
        expected = ["vendor/zero", "vendor/label"] if policy else [mid for mid, _ in CURATED]
        assert rows[0]["models"] == expected
        assert rows[0]["catalog_authoritative"] is True
        assert rows[0]["free_only"] is policy
        apply_openrouter_picker_policy(rows)
    assert calls == [initial_policy, not initial_policy]


def test_shared_gateway_cli_inventory_and_options_keep_active_paid_model(profile, monkeypatch):
    _policy(profile, "true")
    _isolate_provider_sources(monkeypatch)
    kwargs = dict(current_provider="openrouter", current_model="vendor/paid")
    rows = model_switch_providers.list_authenticated_providers(**kwargs, max_models=1)
    assert rows[0]["models"] == ["vendor/zero"]
    assert rows[0]["total_models"] == 2
    gateway = model_switch_providers.list_picker_providers(**kwargs)
    ctx = inventory.ConfigContext("openrouter", "vendor/paid", "", {}, [])
    payload = inventory.build_models_payload(ctx)
    for name in ("_apply_picker_hints", "_apply_pricing", "_apply_capabilities", "_apply_featured",
                 "_prewarm_pricing_async"):
        monkeypatch.setattr(inventory, name, lambda *_a, **_kw: None)
    options = inventory.build_model_options_payload(ctx)
    for result in (gateway, payload["providers"], options["providers"]):
        assert result[0]["models"] == ["vendor/zero", "vendor/label"]
    assert payload["model"] == options["model"] == "vendor/paid"
    assert payload["provider"] == options["provider"] == "openrouter"


@pytest.mark.parametrize("free_only", [False, True])
def test_inventory_filters_late_unconfigured_current_row(profile, monkeypatch, free_only):
    _policy(profile, str(free_only).lower())
    _isolate_provider_sources(monkeypatch)
    proxy = {"slug": "custom:proxy", "is_user_defined": True, "is_current": False,
             "models": ["vendor/zero"], "total_models": 1}
    monkeypatch.setattr(model_switch_providers, "_lap_builtin_rows", lambda b, *_a: b.results.append(proxy))
    monkeypatch.setattr(inventory, "_append_unconfigured_rows", lambda *_a, **_kw: [_row()])
    ctx = inventory.ConfigContext("openrouter", "vendor/paid", "", {}, [])
    payload = inventory.build_models_payload(ctx, include_unconfigured=True)
    row = next(row for row in payload["providers"] if row["slug"] == "openrouter")
    expected = ["vendor/label"] if free_only else [mid for mid, _ in CURATED if mid != "vendor/zero"]
    assert row["models"] == expected
    assert proxy["models"] == ["vendor/zero"]
    assert payload["model"] == "vendor/paid"


def test_cli_empty_policy_row_does_not_reprobe_unrestricted_catalog(profile, monkeypatch):
    from hermes_cli.cli_model_switch_mixin import CLIModelSwitchMixin

    monkeypatch.setattr(models, "provider_model_ids", lambda *_a, **_kw: pytest.fail("unrestricted probe"))
    row = {**_row(), "models": [], "catalog_authoritative": True}
    cli = SimpleNamespace(_model_picker_state={"stage": "provider", "providers": [row]},
                          _invalidate=lambda **_kw: None)
    CLIModelSwitchMixin._handle_model_picker_selection(cli)
    assert cli._model_picker_state["model_list"] == []
    assert cli._model_picker_state["stage"] == "model"


def test_setup_uses_same_policy_without_reinserting_current_paid_model(profile, monkeypatch, capsys):
    from hermes_cli import model_setup_flows

    _policy(profile, "true")
    monkeypatch.setattr(model_setup_flows, "_ensure_flow_api_key", lambda *_a, **_kw: ("test", "test", False))
    monkeypatch.setattr("hermes_cli.models_pricing.get_pricing_for_provider", lambda *_a, **_kw: {})
    captured = {}
    def select(ids, **kwargs):
        captured.update(ids=ids, **kwargs)
        return None
    monkeypatch.setattr("hermes_cli.auth._prompt_model_selection", select)
    monkeypatch.setattr(model_setup_flows, "_finish_model", lambda *_a, **_kw: None)
    model_setup_flows._model_flow_openrouter(config.load_config_readonly(), current_model="vendor/paid")
    assert captured["ids"] == ["vendor/zero", "vendor/label"]
    assert captured["current_model"] == "vendor/paid"
    assert "Current model: vendor/paid" in capsys.readouterr().out
