"""parse_model_input routes bare user-declared provider prefixes (providers:/custom_providers:)."""

from hermes_cli.models import parse_model_input

ROUTER = "providers:\n  myrouter:\n    base_url: http://127.0.0.1:20128/v1\n    api_key: sk-test\n"


def _home(tmp_path, monkeypatch, yaml):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(yaml)


def test_bare_providers_key_is_split(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch, ROUTER)
    assert parse_model_input("myrouter:gpt-5", "openrouter") == ("custom:myrouter", "gpt-5")


def test_bare_custom_providers_name_is_split(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch, "custom_providers:\n  - name: My Router\n    base_url: http://x/v1\n")
    assert parse_model_input("my-router:ocg/ox-alpha", "openrouter") == ("custom:my-router", "ocg/ox-alpha")


def test_caller_supplied_ids_are_the_only_source(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch, ROUTER)
    assert parse_model_input("myrouter:gpt-5", "openrouter", custom_ids={"custom"}) == ("openrouter", "myrouter:gpt-5")


def test_undeclared_prefix_stays_whole(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch, ROUTER)
    assert parse_model_input("anthropic/claude-3.5-sonnet:beta", "openrouter") == (
        "openrouter", "anthropic/claude-3.5-sonnet:beta")


def test_builtin_provider_prefix_still_wins(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch, ROUTER)
    assert parse_model_input("zai:glm-5.2", "myrouter") == ("zai", "glm-5.2")
