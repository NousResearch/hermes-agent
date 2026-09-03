"""`collect_unknown_config_keys` — doctor warns on config.yaml typos.

Walks the raw (on-disk) user config and reports dotted paths for keys that
do not exist anywhere in DEFAULT_CONFIG. Known keys (present at any depth)
are never reported; underscore-prefixed internal keys are skipped.
"""

from hermes_cli.doctor import collect_unknown_config_keys


def test_known_keys_not_reported():
    # Every key here exists in DEFAULT_CONFIG (verified against defaults).
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    raw = {}
    for section, value in DEFAULT_CONFIG.items():
        if isinstance(value, dict) and not section.startswith("_"):
            raw[section] = {}
            for k, v in list(value.items())[:2]:
                if not k.startswith("_"):
                    raw[section][k] = v
        elif not section.startswith("_"):
            raw[section] = value
    assert collect_unknown_config_keys(raw) == []


def test_unknown_top_level_key_reported():
    raw = {"tui_compact": True}
    assert collect_unknown_config_keys(raw) == ["tui_compact"]


def test_unknown_nested_key_reported_with_dotted_path():
    raw = {"display": {"tui_statusbar": "off"}}
    assert collect_unknown_config_keys(raw) == ["display.tui_statusbar"]


def test_known_section_with_unknown_child_reports_only_child():
    raw = {"display": {"compact": True, "definitely_not_a_real_key": 1}}
    findings = collect_unknown_config_keys(raw)
    assert findings == ["display.definitely_not_a_real_key"]


def test_underscore_keys_skipped():
    raw = {"_internal": 1, "display": {"_private": 2}}
    assert collect_unknown_config_keys(raw) == []


def test_non_dict_input_returns_empty():
    assert collect_unknown_config_keys(None) == []
    assert collect_unknown_config_keys([1, 2]) == []
    assert collect_unknown_config_keys("nope") == []


def test_empty_config_no_findings():
    assert collect_unknown_config_keys({}) == []


def test_multiple_unknown_keys_all_found():
    raw = {
        "modle": {"default": "x"},      # typo'd section reported as a whole
        "display": {"compct": True},    # typo'd key in known section
    }
    found = set(collect_unknown_config_keys(raw))
    # Unknown sections are reported at the section level, not descended into.
    assert found == {"modle", "display.compct"}
