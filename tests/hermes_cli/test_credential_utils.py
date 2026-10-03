"""Tests for hermes_cli/credential_utils.py — credential utility helpers."""


def test_mask_key_returns_stars():
    from hermes_cli.credential_utils import mask_api_key
    result = mask_api_key("sk-abc123def456")
    assert "*" in result
    assert "sk-" in result[:5] or result == "****"
