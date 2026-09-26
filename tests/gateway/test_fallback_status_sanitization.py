from gateway import status


def test_sanitize_fallback_status_rejects_private_reason_and_nonfinite_cooldown():
    sanitized = status._sanitize_fallback_status({
        "active": {"provider": "safe", "model": "model", "credential": "redacted-token"},
        "chain": [{"provider": "safe", "model": "model", "base_url": "https://private"}],
        "cooldown_until": float("nan"),
        "reason": "https://private.invalid/error",
    })

    assert sanitized == {
        "active": {"provider": "safe", "model": "model"},
        "chain": [{"provider": "safe", "model": "model"}],
    }
