"""Per-source gateway toolset authorization tests."""

from types import SimpleNamespace

from gateway.run import GatewayRunner


BASE = {
    "platform_toolsets": {
        "slack": ["web", "browser", "computer_use"],
        "discord": ["web", "computer_use"],
    }
}


def _runner():
    runner = object.__new__(GatewayRunner)
    runner._adapter_for_source = lambda source: None
    return runner


def _source(user_id=None, user_id_alt=None):
    return SimpleNamespace(user_id=user_id, user_id_alt=user_id_alt)


def _config(rule):
    return {
        **BASE,
        "gateway": {
            "toolset_user_allowlists": {
                "slack": {"computer_use": rule},
            }
        },
    }


def _resolve(config, source, platform="slack"):
    return GatewayRunner._resolve_enabled_toolsets_for_source(
        _runner(), config, source, platform
    )


def test_allowed_user_keeps_restricted_toolset():
    result = _resolve(_config(["U_ALLOWED"]), _source("U_ALLOWED"))
    assert "computer_use" in result


def test_other_user_loses_only_restricted_toolset():
    result = _resolve(_config(["U_ALLOWED"]), _source("U_OTHER"))
    assert "computer_use" not in result
    assert "browser" in result
    assert "web" in result


def test_missing_identity_denies_restricted_toolset():
    result = _resolve(_config(["U_ALLOWED"]), _source())
    assert "computer_use" not in result


def test_alternate_authenticated_identity_can_match():
    result = _resolve(_config(["U_ALT"]), _source("U_PRIMARY", "U_ALT"))
    assert "computer_use" in result


def test_malformed_toolset_allowlist_denies_that_toolset():
    result = _resolve(_config("U_ALLOWED"), _source("U_ALLOWED"))
    assert "computer_use" not in result
    assert "browser" in result


def test_malformed_platform_rules_deny_all_platform_toolsets():
    config = {
        **BASE,
        "gateway": {"toolset_user_allowlists": {"slack": "bad"}},
    }
    assert _resolve(config, _source("U_ALLOWED")) == []


def test_malformed_root_rules_deny_all_platform_toolsets():
    config = {
        **BASE,
        "gateway": {"toolset_user_allowlists": "bad"},
    }
    assert _resolve(config, _source("U_ALLOWED")) == []


def test_absent_rules_preserve_existing_resolution():
    result = _resolve(BASE, _source("U_OTHER"))
    assert "computer_use" in result


def test_slack_rules_do_not_change_other_platforms():
    result = _resolve(_config(["U_ALLOWED"]), _source("U_OTHER"), "discord")
    assert "computer_use" in result
