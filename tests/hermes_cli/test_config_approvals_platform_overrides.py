"""``approvals.platform_overrides`` is a registered, free-form config key (#88184).

The section is a ``{}`` default in ``DEFAULT_CONFIG`` so a per-platform mode can be set with
``hermes config set approvals.platform_overrides.telegram manual`` without the validator refusing
the platform name as a typo. Lives beside ``test_config.py`` (not in it) so the new assertions do
not grow that module past its code-health FILE_LINES cap.
"""


def test_platform_overrides_defaults_to_empty_mapping():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["approvals"]["platform_overrides"] == {}


def test_platform_override_keys_are_recognized_config_keys():
    from hermes_cli.config import _validate_config_key

    assert _validate_config_key("approvals.platform_overrides.telegram") == (True, None)
    assert _validate_config_key("approvals.platform_overrides.telegram.mode") == (True, None)


def test_unknown_approvals_key_still_gets_a_suggestion():
    from hermes_cli.config import _validate_config_key

    known, suggestion = _validate_config_key("approvals.platform_override")
    assert known is False
    assert suggestion == "approvals.platform_overrides"
