from hermes_cli.model_switch import parse_model_flags, parse_model_flags_detailed


def test_parse_model_flags_detailed_supports_once():
    parsed = parse_model_flags_detailed("sonnet --provider anthropic --once")

    assert parsed.model_input == "sonnet"
    assert parsed.explicit_provider == "anthropic"
    assert parsed.is_global is False
    assert parsed.force_refresh is False
    assert parsed.is_session is False
    assert parsed.is_once is True


def test_parse_model_flags_legacy_wrapper_strips_once():
    model_input, provider, is_global, force_refresh, is_session = parse_model_flags("sonnet --once")

    assert model_input == "sonnet"
    assert provider == ""
    assert is_global is False
    assert force_refresh is False
    assert is_session is False
