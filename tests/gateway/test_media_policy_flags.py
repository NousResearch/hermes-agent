"""``gateway.strict`` and ``gateway.trust_recent_files`` must read the same way on every path:
the routed profile's own config, the env readers, and the bridge that copies the launch
profile's policy into the process env. A hand-edited ``config.yaml`` spells the flags as
strings (``strict: "false"``), and ``bool("false")`` is True — the disagreement this file pins."""

from __future__ import annotations

from pathlib import Path

import pytest
import hermes_yaml as yaml

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def homes(tmp_path, monkeypatch):
    launch = tmp_path / ".hermes"
    routed = launch / "profiles" / "b"
    routed.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    for var in ("HERMES_MEDIA_DELIVERY_STRICT", "HERMES_MEDIA_TRUST_RECENT_FILES"):
        monkeypatch.delenv(var, raising=False)
    return launch, routed


def _under(home: Path, fn):
    token = set_hermes_home_override(str(home))
    try:
        return fn()
    finally:
        reset_hermes_home_override(token)


def _write_gateway(home: Path, cfg: dict) -> None:
    (home / "config.yaml").write_text(
        yaml.safe_dump({"gateway": cfg}), encoding="utf-8"
    )


def test_quoted_false_is_false_on_the_config_path(homes):
    """The routed profile's own config path: ``strict: "false"`` / ``trust_recent_files: "false"``
    (yaml.safe_dump of a str writes exactly the quoted spelling a hand edit gets)."""
    _, routed = homes
    from gateway import media_policy

    _write_gateway(routed, {"strict": "false", "trust_recent_files": "false"})
    assert _under(routed, media_policy.media_delivery_strict) is False
    assert _under(routed, media_policy.media_delivery_trust_recent) is False


def test_quoted_truthy_strings_are_true_on_the_config_path(homes):
    _, routed = homes
    from gateway import media_policy

    _write_gateway(routed, {"strict": "on", "trust_recent_files": "1"})
    assert _under(routed, media_policy.media_delivery_strict) is True
    assert _under(routed, media_policy.media_delivery_trust_recent) is True


@pytest.mark.parametrize(
    "cfg_value", [False, True, "false", "true", "0", "1", "off", "no"]
)
def test_config_path_agrees_with_the_env_path_for_every_spelling(
    homes, monkeypatch, cfg_value
):
    """Invariant: for any spelling, the routed profile's config answer equals the env reader's
    answer for the same value, and the plain bool spelling keeps its current reading."""
    _, routed = homes
    from gateway import media_policy

    _write_gateway(routed, {"strict": cfg_value, "trust_recent_files": cfg_value})
    monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", str(cfg_value))
    monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_FILES", str(cfg_value))

    assert (
        _under(routed, media_policy.media_delivery_strict)
        is media_policy.media_delivery_strict()
    )
    assert (
        _under(routed, media_policy.media_delivery_trust_recent)
        is media_policy.media_delivery_trust_recent()
    )


def test_unknown_value_keeps_the_knob_default(homes):
    """An unreadable value is not a silent flip: strict stays opt-in off, trust stays opt-out on
    (the env readers' existing rule for unknown values)."""
    _, routed = homes
    from gateway import media_policy

    _write_gateway(routed, {"strict": "maybe", "trust_recent_files": "maybe"})
    assert _under(routed, media_policy.media_delivery_strict) is False
    assert _under(routed, media_policy.media_delivery_trust_recent) is True


def test_unset_config_keeps_the_knob_default(homes):
    _, routed = homes
    from gateway import media_policy

    _write_gateway(routed, {})
    assert _under(routed, media_policy.media_delivery_strict) is False
    assert _under(routed, media_policy.media_delivery_trust_recent) is True


@pytest.mark.parametrize(
    "env_value,expected",
    [
        ("1", True),
        ("true", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("off", False),
        ("", False),
        ("maybe", False),  # unknown values keep the opt-in default
    ],
)
def test_strict_env_reader_is_unchanged(homes, monkeypatch, env_value, expected):
    from gateway import media_policy

    monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", env_value)
    assert media_policy.media_delivery_strict() is expected


@pytest.mark.parametrize(
    "env_value,expected",
    [
        ("0", False),
        ("false", False),
        ("no", False),
        ("off", False),
        ("", False),
        ("1", True),
        ("true", True),
        ("yes", True),
        ("on", True),
        ("maybe", True),  # unknown values keep the opt-out default
    ],
)
def test_trust_recent_env_reader_is_unchanged(homes, monkeypatch, env_value, expected):
    from gateway import media_policy

    monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_FILES", env_value)
    assert media_policy.media_delivery_trust_recent() is expected


def test_bridge_copies_the_same_reading_into_the_launch_env(homes):
    """The launch profile's bridge must bridge the quoted "false" as 0, so a shell that only
    sees the env reads the policy the operator wrote."""
    launch, _ = homes
    from gateway import media_policy
    from hermes_cli.config import load_config

    _write_gateway(launch, {"strict": "false", "trust_recent_files": "false"})
    media_policy.apply_media_policy_env(load_config())
    import os

    assert os.environ["HERMES_MEDIA_DELIVERY_STRICT"] == "0"
    assert os.environ["HERMES_MEDIA_TRUST_RECENT_FILES"] == "0"
    # The env readers now answer the same as the config path did.
    assert media_policy.media_delivery_strict() is False
    assert media_policy.media_delivery_trust_recent() is False


def test_bridge_still_bridges_plain_bools(homes):
    launch, _ = homes
    from gateway import media_policy
    from hermes_cli.config import load_config
    import os

    _write_gateway(launch, {"strict": True, "trust_recent_files": False})
    media_policy.apply_media_policy_env(load_config())
    assert os.environ["HERMES_MEDIA_DELIVERY_STRICT"] == "1"
    assert os.environ["HERMES_MEDIA_TRUST_RECENT_FILES"] == "0"
