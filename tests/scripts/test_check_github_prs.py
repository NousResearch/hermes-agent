import json
from pathlib import Path
from scripts.check_github_prs import derive_recommendation

def test_derive_recommendation_conflicting():
    recs = derive_recommendation("123", "CONFLICTING", [], "", [])
    assert any("Rebase" in r for r in recs)

def test_derive_recommendation_ci_failing():
    recs = derive_recommendation("123", "MERGEABLE", ["build_test"], "", [])
    assert any("Fix failing CI" in r for r in recs)

def test_derive_recommendation_duplicate_comment():
    comments = [{"body": "Duplicate of #15676.", "author": {"login": "alt-glitch"}}]
    recs = derive_recommendation("123", "MERGEABLE", [], "", comments)
    assert any("#15676" in r for r in recs)
