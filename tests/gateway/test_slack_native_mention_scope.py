"""Native-only channel fallback follows the served Slack profile, not launch env."""

import agent.secret_scope as scope
from tests.gateway.test_slack_native_mention_only_channels import _make


def test_native_only_fallback_is_profile_scoped_a_b_a(monkeypatch, tmp_path):
    key = "SLACK_NATIVE_MENTION_ONLY_CHANNELS"
    monkeypatch.setenv(key, "C_LAUNCH")
    previous = scope._MULTIPLEX_ACTIVE
    scope.set_multiplex_active(True)
    try:
        for profile, values, expected in (
            ("a", {key: "C_A"}, {"C_A"}),
            ("b", {}, set()),
            ("a", {key: "C_A"}, {"C_A"}),
        ):
            token = scope.set_secret_scope(values, profile_home=str(tmp_path / profile))
            try:
                assert _make()._slack_native_mention_only_channels() == expected
            finally:
                scope.reset_secret_scope(token)
    finally:
        scope.set_multiplex_active(previous)
