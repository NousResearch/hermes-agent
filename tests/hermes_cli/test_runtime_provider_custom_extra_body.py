"""Endpoint-match lift of a named custom provider's extra_body: a runtime resolved through a
NON-custom provider name (``model.provider: openai-api``, local bypass) whose base_url equals a
``providers:`` entry's base_url must carry that entry's ``extra_body`` — the rung never reads the
entry otherwise, so e.g. ``max_tokens`` silently reverted to the server default."""

from hermes_cli import runtime_provider as rp

_BIFROST_URL = "http://192.168.100.180:30900/v1"


def _write_bifrost_config(tmp_path, monkeypatch, *, model_base_url=_BIFROST_URL, extra_body=True):
    """Isolated HERMES_HOME: ``model.provider: openai-api`` plus a named custom ``bifrost`` entry
    whose base_url is ``_BIFROST_URL`` (optionally equal to ``model.base_url``)."""
    from hermes_cli import config as _cfg

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-test-key")
    entry_extra = "    extra_body:\n      max_tokens: 8000\n" if extra_body else ""
    (hermes_home / "config.yaml").write_text(
        "model:\n"
        "  provider: openai-api\n"
        f"  base_url: {model_base_url}\n"
        "  default: cake/claude-opus\n"
        "providers:\n"
        "  bifrost:\n"
        f"    base_url: {_BIFROST_URL}\n"
        "    api_key: dummy-test-key\n" + entry_extra
    )
    _cfg._LOAD_CONFIG_CACHE.clear()
    _cfg._RAW_CONFIG_CACHE.clear()


def test_extra_body_lifted_when_base_url_matches_named_custom_entry(tmp_path, monkeypatch):
    """``model.provider: openai-api`` pointed at a named custom entry's endpoint must still carry
    the entry's extra_body."""
    _write_bifrost_config(tmp_path, monkeypatch)

    resolved = rp.resolve_runtime_provider(requested="openai-api", target_model="cake/claude-opus")

    assert resolved["base_url"] == _BIFROST_URL
    assert resolved["request_overrides"]["extra_body"] == {"max_tokens": 8000}
    # Only request_overrides is lifted; extra_headers may carry credentials — never re-applied here.
    assert not resolved.get("extra_headers")


def test_extra_body_not_doubled_when_requested_is_the_entry(tmp_path, monkeypatch):
    """The named-custom rung already applies the entry's extra_body; the endpoint-match lift must
    leave an existing value untouched."""
    _write_bifrost_config(tmp_path, monkeypatch)

    resolved = rp.resolve_runtime_provider(requested="bifrost", target_model="cake/claude-opus")

    assert resolved["provider"] == "custom"
    assert resolved["request_overrides"] == {"extra_body": {"max_tokens": 8000}}


def test_no_extra_body_when_base_url_matches_no_entry(tmp_path, monkeypatch):
    """An endpoint no configured entry owns resolves exactly as before — no request_overrides."""
    _write_bifrost_config(tmp_path, monkeypatch, model_base_url="http://unmatched.example:9999/v1")

    resolved = rp.resolve_runtime_provider(requested="openai-api", target_model="cake/claude-opus")

    assert resolved["base_url"] == "http://unmatched.example:9999/v1"
    assert not resolved.get("request_overrides")


def test_no_extra_body_when_entry_has_none(tmp_path, monkeypatch):
    """An entry without extra_body adds no request_overrides on the endpoint-match path."""
    _write_bifrost_config(tmp_path, monkeypatch, extra_body=False)

    resolved = rp.resolve_runtime_provider(requested="openai-api", target_model="cake/claude-opus")

    assert resolved["base_url"] == _BIFROST_URL
    assert not resolved.get("request_overrides")
