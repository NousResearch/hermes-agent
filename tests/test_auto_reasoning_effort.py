from hermes_constants import parse_reasoning_effort, resolve_auto_reasoning_config


def test_parse_reasoning_effort_accepts_auto_marker():
    assert parse_reasoning_effort("auto") == {"enabled": True, "effort": "auto"}


def test_auto_reasoning_uses_low_for_simple_status_question():
    cfg = {"enabled": True, "effort": "auto"}
    messages = [{"role": "user", "content": "koks effort nustatytas default profiliui?"}]

    assert resolve_auto_reasoning_config(cfg, messages) == {"enabled": True, "effort": "low"}


def test_auto_reasoning_uses_medium_for_design_and_config_changes():
    cfg = {"enabled": True, "effort": "auto"}
    messages = [{"role": "user", "content": "noriu auto effort pakeitimo Hermes confige"}]

    assert resolve_auto_reasoning_config(cfg, messages) == {"enabled": True, "effort": "medium"}


def test_auto_reasoning_uses_high_for_debug_security_and_architecture():
    cfg = {"enabled": True, "effort": "auto"}
    messages = [{"role": "user", "content": "padaryk root cause debugging security architekturos sprendimui"}]

    assert resolve_auto_reasoning_config(cfg, messages) == {"enabled": True, "effort": "high"}


def test_auto_reasoning_leaves_explicit_effort_unchanged():
    cfg = {"enabled": True, "effort": "medium"}

    assert resolve_auto_reasoning_config(cfg, []) is cfg
