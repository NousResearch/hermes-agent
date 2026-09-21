# content_engine/tests/test_config_pillars.py
from config import BRANDS


def test_twitter_educational_pillars():
    p = BRANDS["sahil_twitter"]["content_pillars"]
    for pillar in ("Agent Build Notes","Harness Tuning","Paper Takes","Radar Finds","AI Patterns"):
        assert pillar in p
    assert BRANDS["sahil_twitter"]["posts_per_week"] >= 21        # 3-5/day
    assert BRANDS["sahil_twitter"]["educational_mix"] == 0.65

def test_linkedin_educational_pillars():
    p = BRANDS["sahil_linkedin"]["content_pillars"]
    for pillar in ("Agentic Systems in Practice","AI Engineering Notes","Research to Practice","Tooling Signals"):
        assert pillar in p
    assert BRANDS["sahil_linkedin"]["posts_per_week"] >= 14        # 2-3/day
