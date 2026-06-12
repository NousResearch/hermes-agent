"""Tests for completion-auditor's deterministic claim classifier."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_classifier():
    plugin_dir = _repo_root() / "plugins" / "completion-auditor"
    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        sys.modules["hermes_plugins"] = ns
    for key in list(sys.modules):
        if key.startswith("hermes_plugins.completion_auditor_classifier_under_test"):
            del sys.modules[key]
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.completion_auditor_classifier_under_test.classifier",
        plugin_dir / "classifier.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hermes_plugins.completion_auditor_classifier_under_test.classifier"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "text",
    [
        "Hermes plugins are opt-in. You can enable one from config.yaml.",
        "Plan: add a plugin skeleton, then write tests, then run pytest.",
        "I will run the tests next.",
        "테스트는 다음에 실행할게.",
        "If tests pass, we can call this done.",
        "테스트가 통과하면 완료라고 볼 수 있어.",
        "Blocked because credentials are missing.",
        "Blocked because credentials are missing, but updated config.yaml.",
        "인증 정보가 없어서 진행할 수 없어.",
        "```\nTests passed: 21 passed.\n```",
        "Example output: `Tests passed: 21 passed.`",
    ],
)
def test_false_positive_fixtures_return_none(text):
    mod = _load_classifier()
    assert mod.classify_response(text) is None


@pytest.mark.parametrize(
    ("text", "claim_type"),
    [
        ("Updated config.yaml and kept the change scoped.", "modified"),
        ("Created plugins/completion-auditor/classifier.py.", "created"),
        ("Implemented the deterministic classifier.", "implemented"),
        ("Tests passed: 21 passed.", "tested"),
        ("Verification passed with pytest.", "verified"),
        ("Deployed the worker and published the release notes.", "deployed"),
        ("작업 완료했어.", "completed"),
        ("테스트 통과했어.", "tested"),
        ("設定を更新しました。", "modified"),
        ("テスト成功しました。", "tested"),
    ],
)
def test_completion_claim_fixtures_are_classified(text, claim_type):
    mod = _load_classifier()
    claim = mod.classify_response(text)
    assert claim is not None
    assert claim.claim_type == claim_type
    assert claim.claim_text


def test_scope_prefers_path_like_token():
    mod = _load_classifier()
    claim = mod.classify_response("Updated plugins/completion-auditor/report.py.")
    assert claim is not None
    assert claim.claim_type == "modified"
    assert claim.claim_scope == "plugins/completion-auditor/report.py"


def test_first_real_claim_skips_plan_sentence():
    mod = _load_classifier()
    claim = mod.classify_response(
        "Plan: run pytest next. Updated plugins/completion-auditor/classifier.py."
    )
    assert claim is not None
    assert claim.claim_type == "modified"
    assert claim.claim_scope == "plugins/completion-auditor/classifier.py"


def test_separate_blocker_sentence_does_not_hide_later_real_claim():
    mod = _load_classifier()
    claim = mod.classify_response(
        "Blocked because credentials are missing. Updated config.yaml."
    )
    assert claim is not None
    assert claim.claim_type == "modified"
    assert claim.claim_scope == "config.yaml"
