"""Compatibility views must preserve route identity and source precedence."""

from copy import deepcopy

import pytest

from hermes_cli.config import (
    atomic_config_write,
    get_compatible_custom_providers,
    get_custom_provider_context_length,
    get_custom_provider_extra_headers,
    get_custom_provider_tls_settings,
    load_config,
    load_config_readonly,
)
from hermes_cli.inventory import load_picker_context
from hermes_cli.runtime_provider_custom import (
    _get_named_custom_provider,
    named_custom_provider_config_path,
)


def _entry(url, *, second=False, **overrides):
    label = "two" if second else "one"
    return {
        "name": "Shared Route",
        "base_url": url,
        "model": "fixture-model",
        "context_length": 128000 if second else 32000,
        "ssl_ca_cert": f"synthetic-ca-{label}.pem",
        "ssl_verify": not second,
        "extra_headers": {"X-Test-Route": label},
        **overrides,
    }


def _sources(shape, one, two):
    entries = deepcopy([one, two])
    if shape == "legacy-only":
        return {"custom_providers": entries}
    providers = {}
    for key, entry in zip(("route-one", "route-two"), entries):
        entry["api"] = entry.pop("base_url")
        entry["default_model"] = entry.pop("model")
        providers[key] = entry
    if shape == "providers-only":
        return {"providers": providers}
    return {"custom_providers": [deepcopy(one)], "providers": {"route-two": providers["route-two"]}}


def _assert_metadata(url, expected, config, view):
    for kwargs in ({}, {"config": config}, {"custom_providers": view}):
        assert get_custom_provider_context_length("fixture-model", url, **kwargs) == expected.get("context_length")
        assert get_custom_provider_tls_settings(url, **kwargs) == {
            key: expected[key] for key in ("ssl_ca_cert", "ssl_verify") if key in expected
        }
        assert get_custom_provider_extra_headers(url, **kwargs) == expected.get("extra_headers", {})
        assert get_custom_provider_context_length("", url, **kwargs) is None


@pytest.mark.parametrize("source_shape", ["legacy-only", "providers-only", "mixed"])
@pytest.mark.parametrize(
    "url_one,url_two,distinct",
    [
        pytest.param("https://api.example.test/Tenant-A/v1", "https://api.example.test/tenant-a/v1", True,
                     id="path-case-distinct"),
        pytest.param("https://api.example.test/v1?Tenant=A", "https://api.example.test/v1?tenant=A", True,
                     id="query-key-case-distinct"),
        pytest.param("https://api.example.test/v1?tenant=A", "https://api.example.test/v1?tenant=a", True,
                     id="query-value-case-distinct"),
        pytest.param("https://API.example.test/v1", "https://api.example.test/v1", False,
                     id="hostname-case-equivalent"),
        pytest.param("https://api.example.test/v1", "https://api.example.test/v1", False,
                     id="exact-duplicate"),
    ],
)
def test_loaded_custom_provider_route_metadata(tmp_path, monkeypatch, source_shape, url_one, url_two, distinct):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    one, two = _entry(url_one), _entry(url_two, second=True)
    raw = _sources(source_shape, one, two)
    raw["model"] = {"provider": "custom", "default": "fixture-model", "base_url": url_two}
    config_path = tmp_path / "config.yaml"
    atomic_config_write(config_path, raw)
    disk_before = config_path.read_bytes()
    cached = load_config_readonly()
    snapshot = deepcopy(cached)
    expected = [deepcopy(one)]
    if source_shape == "providers-only":
        expected[0]["provider_key"] = "route-one"
    if distinct:
        expected.append(deepcopy(two))
        if source_shape != "legacy-only":
            expected[1]["provider_key"] = "route-two"

    for _ in range(2):
        loaded = load_config()
        loaded_before = deepcopy(loaded)
        view = get_compatible_custom_providers()
        picker = load_picker_context()
        assert view == expected
        assert get_compatible_custom_providers(loaded) == expected
        assert get_compatible_custom_providers(cached) == expected
        assert picker.custom_providers == expected
        assert (picker.current_provider, picker.current_model, picker.current_base_url) == (
            "custom", "fixture-model", url_two,
        )
        _assert_metadata(url_one, one, loaded, view)
        _assert_metadata(url_two, two if distinct else one, loaded, view)
        _assert_metadata("https://missing.example.test/v1", {}, loaded, view)
        _assert_metadata("", {}, loaded, view)
        if distinct and source_shape != "legacy-only":
            named = _get_named_custom_provider("route-two", config=loaded, resolve_credentials=False)
            assert named is not None
            assert named["base_url"] == url_two
            assert named["provider_key"] == "route-two"
            assert named_custom_provider_config_path("route-two", loaded) == ("providers", "route-two")
        assert loaded == loaded_before
        assert cached == snapshot
        assert load_config_readonly() == snapshot
        assert config_path.read_bytes() == disk_before
        for section in ("custom_providers", "providers"):
            if section in raw:
                assert cached[section] == raw[section]
    if source_shape == "providers-only":
        assert not cached.get("custom_providers")


def _url_case(case_id, url_one, url_two, *, distinct):
    one, two = _entry(url_one), _entry(url_two, second=True)
    return pytest.param({"custom_providers": [one, two]}, [one, two] if distinct else [one], id=case_id)


@pytest.mark.parametrize(
    "config,expected",
    [
        _url_case("scheme-host-single-slash", "HTTPS://API.example.test/v1/", "https://api.example.test/v1",
                  distinct=False),
        _url_case("http-default-port", "http://api.example.test:80/v1", "http://api.example.test/v1",
                  distinct=False),
        _url_case("https-default-port", "https://api.example.test:443/v1", "https://api.example.test/v1",
                  distinct=False),
        _url_case("non-default-port", "https://api.example.test:444/v1", "https://api.example.test/v1",
                  distinct=True),
        _url_case("fragment", "https://api.example.test/v1#One", "https://api.example.test/v1#Two",
                  distinct=False),
        _url_case("repeated-trailing-slash", "https://api.example.test/v1//", "https://api.example.test/v1/",
                  distinct=True),
        _url_case("repeated-internal-slash", "https://api.example.test/a//v1", "https://api.example.test/a/v1",
                  distinct=True),
        _url_case("query-path-slash", "https://api.example.test/v1/?q=a", "https://api.example.test/v1?q=a",
                  distinct=True),
        _url_case("query-value-slash", "https://api.example.test/v1?q=a/", "https://api.example.test/v1?q=a",
                  distinct=True),
        _url_case("empty-query", "https://api.example.test/v1?", "https://api.example.test/v1",
                  distinct=True),
        _url_case("percent-escape-case", "https://api.example.test/%2Fv1", "https://api.example.test/%2fv1",
                  distinct=True),
        _url_case("userinfo-case", "https://Fixture@api.example.test/v1", "https://fixture@api.example.test/v1",
                  distinct=True),
        _url_case("ipv6-zone-case", "https://[fe80::1%25Zone]/v1", "https://[fe80::1%25zone]/v1",
                  distinct=True),
        _url_case("invalid-port-fallback", "https://api.example.test:bad/Tenant", "https://api.example.test:bad/tenant",
                  distinct=True),
        _url_case("whitespace-fallback", "https://api.example.test/Tenant A", "https://api.example.test/tenant a",
                  distinct=True),
        _url_case("placeholder-fallback", "{Tenant}/v1", "{tenant}/v1", distinct=True),
        _url_case("exact-fallback", "{Tenant}/v1", "{Tenant}/v1", distinct=False),
        pytest.param(
            {"custom_providers": [_entry("https://api.example.test/v1"),
                                  _entry("https://api.example.test/v1", name=" shared route ", model=" FIXTURE-MODEL ")]},
            [_entry("https://api.example.test/v1")], id="name-model-case-equivalent",
        ),
        pytest.param(
            {"custom_providers": [_entry("https://api.example.test/v1"),
                                  _entry("https://api.example.test/v1", second=True, name="Other Route")]},
            [_entry("https://api.example.test/v1"), _entry("https://api.example.test/v1", second=True, name="Other Route")],
            id="different-name",
        ),
        pytest.param(
            {"custom_providers": [_entry("https://api.example.test/v1"),
                                  _entry("https://api.example.test/v1", second=True, model="other-model")]},
            [_entry("https://api.example.test/v1"), _entry("https://api.example.test/v1", second=True, model="other-model")],
            id="different-model",
        ),
        pytest.param(
            {"providers": {"Route-One": {"name": "First", "api": "https://api.example.test/First", "context_length": 32000},
                           "route-one": {"name": "Second", "api": "https://api.example.test/Second", "context_length": 128000}}},
            [{"name": "First", "base_url": "https://api.example.test/First", "provider_key": "Route-One", "context_length": 32000}],
            id="provider-key-case-precedes-url",
        ),
        pytest.param({}, [], id="explicit-empty-config"),
        pytest.param({"custom_providers": [], "providers": {}}, [], id="empty-sources"),
        pytest.param({"custom_providers": None, "providers": None}, [], id="null-sources"),
        pytest.param({"custom_providers": [None, "bad", {}, {"name": "Missing URL"},
                                          {"name": "Empty URL", "base_url": ""},
                                          {"name": "Bad URL", "base_url": "not-a-url"},
                                          {"base_url": "https://api.example.test/v1"}]}, [], id="invalid-legacy-entries"),
        pytest.param({"providers": {"bad": None, "no-url": {"name": "Missing URL"},
                                    "invalid": {"api": "not-a-url"}}}, [], id="invalid-providers-entries"),
        pytest.param({"providers": {"route-one": {"api": "https://api.example.test/v1"}}},
                     [{"name": "route-one", "base_url": "https://api.example.test/v1", "provider_key": "route-one"}],
                     id="provider-name-key-default"),
        pytest.param({"custom_providers": [{"name": "Shared Route", "base_url": "https://api.example.test/v1"},
                                          {"name": "Shared Route", "base_url": "https://api.example.test/v1", "model": ""}]},
                     [{"name": "Shared Route", "base_url": "https://api.example.test/v1"}], id="absent-empty-model"),
        pytest.param({"providers": {"route-one": {"api": "https://api.example.test/v1", "default_model": "fixture-model"}}},
                     [{"name": "route-one", "base_url": "https://api.example.test/v1", "provider_key": "route-one", "model": "fixture-model"}],
                     id="default-model"),
        pytest.param({"providers": {"route-one": {"api": "https://api.example.test/v1", "default_model": ""}}},
                     [{"name": "route-one", "base_url": "https://api.example.test/v1", "provider_key": "route-one"}],
                     id="empty-default-model"),
        pytest.param({"providers": {"route-one": {"api": "https://api.example.test/v1", "enabled": False},
                                    "route-two": {"api": "https://api.example.test/v2", "enabled": "off"}}},
                     [], id="disabled-providers"),
        pytest.param({"custom_providers": "malformed", "providers": {"route-two": {"api": "https://api.example.test/v2"}}},
                     [{"name": "route-two", "base_url": "https://api.example.test/v2", "provider_key": "route-two"}],
                     id="malformed-legacy-valid-providers"),
    ],
)
def test_custom_provider_compatibility_identity_controls(config, expected):
    before = deepcopy(config)
    assert get_compatible_custom_providers(config) == expected
    assert get_compatible_custom_providers(config) == expected
    assert config == before
