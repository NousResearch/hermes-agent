from agent.skill_commands import (
    _normalize_cyber_vc_analyst_instruction,
    _normalize_cyber_vc_company_instruction,
    _normalize_cyber_vc_compare_instruction,
    _normalize_cyber_vc_competitors_instruction,
    _normalize_cyber_vc_theme_instruction,
    _normalize_cyber_vc_triage_instruction,
    _normalize_skill_user_instruction,
)


def test_company_mode_shorthand_expands_to_company_prompt():
    text = _normalize_cyber_vc_analyst_instruction("company Red Access Security")

    assert text.startswith("Analyze Red Access Security as an early-stage cybersecurity investment.")
    assert "Use company mode." in text
    assert "Research depth: standard." in text
    assert "verification pass" in text
    assert "Recommendation now:" in text
    assert "Want me to save the full company memo to the vault?" in text


def test_theme_mode_shorthand_expands_to_theme_prompt():
    text = _normalize_cyber_vc_analyst_instruction("theme SOC Automation / AI SOC")

    assert text.startswith("Analyze SOC Automation / AI SOC as a cybersecurity venture investment theme.")
    assert "Use theme mode, not company mode." in text
    assert "Research depth: standard." in text
    assert "representative companies" in text
    assert "Theme view:" in text
    assert "Want me to save this theme memo to the vault?" in text


def test_compare_mode_shorthand_expands_to_comparison_prompt():
    text = _normalize_cyber_vc_analyst_instruction("compare Red Access Security vs Noma Security")

    assert text.startswith("Compare Red Access Security vs Noma Security as cybersecurity investment opportunities.")
    assert "Research depth: standard." in text
    assert "verify ranking deltas" in text
    assert "Winner now:" in text
    assert "Want me to save this comparison to the vault" in text


def test_triage_mode_shorthand_expands_to_fast_ic_prompt():
    text = _normalize_cyber_vc_analyst_instruction("triage Red Access Security")

    assert text.startswith("Triage Red Access Security as an early-stage cybersecurity investment.")
    assert "fast IC-style read" in text
    assert "Research depth: standard." in text
    assert "Quick read:" in text
    assert "Want me to save this triage note to the vault" in text


def test_competitors_mode_shorthand_expands_to_landscape_prompt():
    text = _normalize_cyber_vc_analyst_instruction("competitors browser security")

    assert text.startswith("Map the competitive landscape for browser security as a cybersecurity investment category.")
    assert "Use competitors mode." in text
    assert "Research depth: standard." in text
    assert "Category read:" in text
    assert "Want me to save this competitor landscape to the vault?" in text


def test_depth_prefix_is_preserved_for_structured_modes():
    text = _normalize_cyber_vc_analyst_instruction("deep theme SOC Automation / AI SOC")

    assert "Research depth: deep." in text
    assert text.startswith("Analyze SOC Automation / AI SOC as a cybersecurity venture investment theme.")


def test_direct_company_skill_shorthand_expands():
    text = _normalize_cyber_vc_company_instruction("Red Access Security")

    assert text.startswith("Analyze Red Access Security as an early-stage cybersecurity investment.")
    assert "Use company mode." in text


def test_direct_theme_skill_shorthand_expands():
    text = _normalize_cyber_vc_theme_instruction("quick SOC Automation / AI SOC")

    assert text.startswith("Analyze SOC Automation / AI SOC as a cybersecurity venture investment theme.")
    assert "Research depth: quick." in text


def test_direct_compare_skill_shorthand_expands():
    text = _normalize_cyber_vc_compare_instruction("Red Access Security vs Noma Security")

    assert text.startswith("Compare Red Access Security vs Noma Security as cybersecurity investment opportunities.")
    assert "Winner now:" in text


def test_direct_triage_skill_shorthand_expands():
    text = _normalize_cyber_vc_triage_instruction("Red Access Security")

    assert text.startswith("Triage Red Access Security as an early-stage cybersecurity investment.")
    assert "Quick read:" in text


def test_direct_competitors_skill_shorthand_expands():
    text = _normalize_cyber_vc_competitors_instruction("deep browser security")

    assert text.startswith("Map the competitive landscape for browser security as a cybersecurity investment category.")
    assert "Research depth: deep." in text


def test_non_structured_prompt_is_left_unchanged():
    raw = "Analyze Red Access Security as an early-stage cybersecurity investment."

    assert _normalize_cyber_vc_analyst_instruction(raw) == raw


def test_only_cyber_vc_analyst_skill_uses_structured_normalization():
    raw = "company Red Access Security"

    assert _normalize_skill_user_instruction("/cyber-vc-analyst", "cyber-vc-analyst", raw) != raw
    assert _normalize_skill_user_instruction("/other-skill", "other-skill", raw) == raw


def test_direct_cyber_vc_skills_are_normalized():
    assert (
        _normalize_skill_user_instruction("/cyber-vc-company", "cyber-vc-company", "Red Access Security")
        != "Red Access Security"
    )
    assert (
        _normalize_skill_user_instruction("/cyber-vc-theme", "cyber-vc-theme", "SOC Automation / AI SOC")
        != "SOC Automation / AI SOC"
    )
