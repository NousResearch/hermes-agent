"""Quoted booleans in ``gateway.strict`` / ``gateway.trust_recent_files`` mean what they say.

The gateway reads config.yaml raw, so ``strict: "false"`` reaches the media policy as the string
``"false"``; truthiness of a non-empty string turned it into strict mode (and ``trust_recent_files:
"off"`` into trusting recent files). Both the env bridge and the routed-profile readers must parse
the token the same way the env var is parsed."""

from __future__ import annotations

import pytest

from gateway import media_policy
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def clean_policy_env(monkeypatch):
    for _, env in media_policy._FLAG_ENVS:
        monkeypatch.delenv(env, raising=False)


@pytest.mark.parametrize(
    ("strict", "trust", "want_strict", "want_trust"),
    [("false", "false", False, False), ("off", "no", False, False), ("true", "yes", True, True)],
)
def test_env_bridge_parses_quoted_flags(clean_policy_env, strict, trust, want_strict, want_trust):
    media_policy.apply_media_policy_env({"gateway": {"strict": strict, "trust_recent_files": trust}})

    assert media_policy.media_delivery_strict() is want_strict
    assert media_policy.media_delivery_trust_recent() is want_trust


def test_routed_profile_reads_quoted_flags(tmp_path, clean_policy_env):
    home = tmp_path / "profiles" / "routed"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(
        'gateway:\n  strict: "false"\n  trust_recent_files: "false"\n', encoding="utf-8")

    token = set_hermes_home_override(str(home))
    try:
        assert media_policy.media_delivery_strict() is False
        assert media_policy.media_delivery_trust_recent() is False
    finally:
        reset_hermes_home_override(token)
