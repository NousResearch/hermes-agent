from prompt_engine import build_image_prompt, BRAND_STYLE_MAP


def test_brand_map_has_all_brands():
    for b in ("plenishd", "coachos", "matchdaymaestro", "sahil_twitter", "sahil_linkedin"):
        assert b in BRAND_STYLE_MAP


def test_prompt_includes_rules_and_text():
    draft = {"brand": "matchdaymaestro", "platform": "twitter", "title": "Don't watch. Predict.",
             "pillar": "live_predictions", "body_text": "In-match calls. Speed bonus."}
    prompt, aspect, model, expected = build_image_prompt(draft)
    assert "do NOT display" in prompt
    assert "Don't watch" in prompt or "Don't Watch" in prompt
    assert expected
    assert model in ("seedream45", "flux_ultra", "nano_banana")


def test_textless_photo_returns_no_expected():
    draft = {"brand": "plenishd", "platform": "instagram", "title": "Snap your fridge",
             "pillar": "voice", "format": "scene", "visual_text": "none"}
    prompt, aspect, model, expected = build_image_prompt(draft)
    assert expected == []
