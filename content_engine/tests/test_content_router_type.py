from content_router import content_type_for


def test_comparison_detected():
    assert content_type_for({"title": "Clubcard vs Nectar vs Aldi"}) == "comparison"


def test_framework_detected():
    assert content_type_for({"pillar": "framework", "title": "The 6-minute session plan"}) == "framework"


def test_list_is_infographic():
    assert content_type_for({"title": "5 ways to debug faster"}) == "infographic"


def test_story_default_scene():
    assert content_type_for({"pillar": "coach_life", "title": "Coach spotlight: Sarah"}) == "scene"


def test_explicit_format_wins():
    assert content_type_for({"format": "hero", "title": "anything"}) == "hero"
