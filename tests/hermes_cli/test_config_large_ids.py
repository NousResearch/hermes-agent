import json
import pytest
from hermes_cli.config import load_config, save_config, _deep_merge
from hermes_cli.web_server_config import _normalize_config_for_web, _denormalize_config_from_web


def test_normalize_config_for_web_large_snowflake_to_string(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    raw = {
        "discord": {
            "home_channel": 1532136816336044092,
            "nested": {"chat_id": 1532136816336044092},
            "allowed_chats": [1532136816336044092, 12345],
        },
        "custom_large_id": 10000000000000000000,
        "normal_int": 8080,
        "is_enabled": True,
    }
    normalized = _normalize_config_for_web(raw)

    assert normalized["discord"]["home_channel"] == "1532136816336044092"
    assert normalized["discord"]["nested"]["chat_id"] == "1532136816336044092"
    assert normalized["discord"]["allowed_chats"] == ["1532136816336044092", "12345"]
    assert normalized["custom_large_id"] == "10000000000000000000"
    assert normalized["normal_int"] == 8080
    assert normalized["is_enabled"] is True

    # Check JSON serialization produces strings, not bare numbers >= 2^53
    payload = json.dumps(normalized)
    assert '"home_channel": "1532136816336044092"' in payload
    assert '"custom_large_id": "10000000000000000000"' in payload


def test_denormalize_config_preserves_disk_integer_type(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    save_config({
        "discord": {
            "home_channel": 1532136816336044092,
            "allowed_chats": [1532136816336044092],
        }
    })

    # Web sends back strings
    incoming = {
        "discord": {
            "home_channel": "1532136816336044092",
            "allowed_chats": ["1532136816336044092"],
        }
    }
    denormalized = _denormalize_config_from_web(incoming)
    assert denormalized["discord"]["home_channel"] == 1532136816336044092
    assert isinstance(denormalized["discord"]["home_channel"], int)
    assert denormalized["discord"]["allowed_chats"] == [1532136816336044092]
    assert isinstance(denormalized["discord"]["allowed_chats"][0], int)


def test_denormalize_config_refuses_bare_number_exceeding_safe_int(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    save_config({
        "discord": {
            "home_channel": 1532136816336044092,
        }
    })

    # Client sends back a corrupted bare number >= 2^53
    corrupted_incoming = {
        "discord": {
            "home_channel": 1532136816336044000,
        }
    }
    with pytest.raises(ValueError, match="exceeds JavaScript safe integer limit"):
        _denormalize_config_from_web(corrupted_incoming)


def test_full_roundtrip_unrelated_edit_does_not_corrupt_snowflake(monkeypatch, tmp_path):
    """Reproduces issue #123439 where an unrelated edit from the config UI
    previously corrupted 19-digit snowflake integers into rounded doubles."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    initial_config = {
        "discord": {
            "home_channel": 1532136816336044092,
        },
        "model": {"default": "model-a"},
    }
    save_config(initial_config)

    # 1. GET /api/config produces normalized output
    get_payload = _normalize_config_for_web(load_config())
    serialized = json.dumps(get_payload)

    # 2. Browser parses JSON, edits unrelated key (model), stringifies back
    browser_data = json.loads(serialized)
    browser_data["model"] = "model-b"

    # 3. PUT /api/config denormalizes and merges over disk
    existing = load_config()
    incoming = _denormalize_config_from_web(browser_data, disk_cfg=existing)
    merged = _deep_merge(existing, incoming)
    save_config(merged)

    # 4. Verify disk config retains exact 19-digit snowflake
    reloaded = load_config()
    assert reloaded["discord"]["home_channel"] == 1532136816336044092
    assert isinstance(reloaded["discord"]["home_channel"], int)
    assert reloaded["model"]["default"] == "model-b"
