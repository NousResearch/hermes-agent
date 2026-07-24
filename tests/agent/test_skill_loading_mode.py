import logging

from agent.skill_preprocessing import resolve_skill_loading_mode


def test_skill_loading_mode_defaults_to_eager():
    assert resolve_skill_loading_mode({}) == "eager"


def test_skill_loading_mode_accepts_routed_case_insensitively():
    assert resolve_skill_loading_mode({"loading": " Routed "}) == "routed"


def test_unknown_skill_loading_mode_falls_back_and_warns(caplog):
    with caplog.at_level(logging.WARNING):
        mode = resolve_skill_loading_mode({"loading": "semantic"})

    assert mode == "eager"
    assert "Unknown skills.loading value" in caplog.text
