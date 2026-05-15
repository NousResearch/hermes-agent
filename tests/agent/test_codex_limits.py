import json

from agent import codex_limits


def test_codex_limits_app_server_fixture_formats_model_buckets():
    provider, payload = codex_limits.fixture_payload("app-server")

    state = codex_limits.normalize(provider, payload)
    pretty = codex_limits.format_pretty(state)

    assert state["source"]["provider"] == "app_server"
    assert [bucket["limit_id"] for bucket in state["rate_limits"]] == ["codex", "codex_model"]
    assert state["rate_limits"][0]["five_h"]["remaining_percent"] == 75
    assert "Rate limits remaining: 5h 75% reset" in pretty
    assert "GPT-Codex model bucket: 5h 89% reset" in pretty


def test_codex_limits_wham_fixture_normalizes_used_to_remaining():
    provider, payload = codex_limits.fixture_payload("wham")

    state = codex_limits.normalize(provider, payload)

    bucket = state["rate_limits"][0]
    assert state["source"]["provider"] == "codex_wham"
    assert bucket["five_h"]["remaining_percent"] == 76
    assert bucket["week"]["remaining_percent"] == 93


def test_codex_limits_json_contains_no_raw_auth_material():
    provider, payload = codex_limits.fixture_payload("app-server")

    encoded = json.dumps(codex_limits.normalize(provider, payload))

    assert "access_token" not in encoded
    assert "refresh_token" not in encoded
    assert "account_id" not in encoded
    assert "chatgpt_account_id" not in encoded
