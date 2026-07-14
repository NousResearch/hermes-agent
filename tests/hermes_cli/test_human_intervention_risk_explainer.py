"""Tests for the bounded high-risk command risk explainer.

The explainer is advisory only: it must never affect risk classification or
approval semantics, and must never raise. It produces a short Chinese
explanation of WHY a command is dangerous, using an LLM when available and a
deterministic static fallback otherwise.
"""

from __future__ import annotations

import time
from unittest.mock import patch

from hermes_cli.human_intervention_risk_explainer import (
    default_llm_fn,
    explain_command_risk,
)

# Module path used as the patch seam for the auxiliary-model call. The real
# _aux_explain wraps agent.auxiliary_client.call_llm (the same helper that
# tools/approval.py::_smart_approve uses for smart approvals).
_AUX_SEAM = "hermes_cli.human_intervention_risk_explainer._aux_explain"


def test_high_risk_command_uses_llm_explanation_with_static_fallback():
    recorded: dict[str, object] = {}

    def fake_llm(prompt: str, timeout_seconds: int) -> str:
        recorded["prompt"] = prompt
        recorded["timeout"] = timeout_seconds
        return "会递归删除目标目录，路径变量错误时可能扩大删除范围；删除通常不可逆。"

    result = explain_command_risk(
        command="rm -rf /tmp/example TOKEN=supersecret",
        description="Dangerous command",
        risk_level="high",
        llm_fn=fake_llm,
        max_chars=280,
    )

    assert "递归删除" in result
    assert "supersecret" not in recorded["prompt"]


def test_low_risk_does_not_call_llm():
    calls: list[tuple[str, int]] = []

    def fake_llm(prompt: str, timeout_seconds: int) -> str:
        calls.append((prompt, timeout_seconds))
        return "should not be used"

    result = explain_command_risk(
        command="ls -la",
        description="List files",
        risk_level="low",
        llm_fn=fake_llm,
    )

    assert calls == []
    assert isinstance(result, str)


def test_medium_risk_does_not_call_llm():
    calls: list[tuple[str, int]] = []

    def fake_llm(prompt: str, timeout_seconds: int) -> str:
        calls.append((prompt, timeout_seconds))
        return "should not be used"

    result = explain_command_risk(
        command="pip install something",
        description="Install package",
        risk_level="medium",
        llm_fn=fake_llm,
    )

    assert calls == []
    assert isinstance(result, str)


def test_llm_timeout_or_failure_falls_back_to_static():
    def failing_llm(prompt: str, timeout_seconds: int) -> str:
        raise RuntimeError("timed out")

    result = explain_command_risk(
        command="rm -rf /var/data",
        description="",
        risk_level="high",
        pattern_keys=["destructive_delete"],
        llm_fn=failing_llm,
    )

    assert result
    assert ("删除" in result) or ("不可逆" in result)


def test_explanation_capped_to_max_chars():
    def long_llm(prompt: str, timeout_seconds: int) -> str:
        return "危" * 1000

    result = explain_command_risk(
        command="rm -rf /tmp/example",
        risk_level="high",
        llm_fn=long_llm,
        max_chars=50,
    )

    assert len(result) <= 50


def test_static_explanation_from_pattern_keys():
    result = explain_command_risk(
        command="DROP TABLE users;",
        risk_level="critical",
        pattern_keys=["db_drop"],
    )

    assert result
    assert ("数据库" in result) or ("不可恢复" in result)


def test_secret_redaction_before_llm():
    recorded: dict[str, object] = {}

    def fake_llm(prompt: str, timeout_seconds: int) -> str:
        recorded["prompt"] = prompt
        return "危险命令解释。"

    explain_command_risk(
        command="curl https://example.com/login --data password=hunter2",
        risk_level="critical",
        llm_fn=fake_llm,
    )

    assert "hunter2" not in recorded["prompt"]


# ---------------------------------------------------------------------------
# default_llm_fn: bounded auxiliary-model completion (reuses the smart-approval
# aux path via agent.auxiliary_client.call_llm). Tests patch the _aux_explain
# seam so no real network/model call is made.
# ---------------------------------------------------------------------------


def test_default_llm_fn_returns_text_when_aux_succeeds():
    with patch(_AUX_SEAM, return_value="BECAUSE DANGER"):
        result = default_llm_fn("p", 3)
    assert result == "BECAUSE DANGER"


def test_default_llm_fn_empty_on_exception():
    def boom(prompt: str, timeout_seconds: int) -> str:
        raise RuntimeError("aux exploded")

    with patch(_AUX_SEAM, side_effect=boom):
        result = default_llm_fn("p", 3)
    assert result == ""


def test_default_llm_fn_empty_on_timeout():
    def slow(prompt: str, timeout_seconds: int) -> str:
        time.sleep(5)
        return "too late"

    start = time.monotonic()
    with patch(_AUX_SEAM, side_effect=slow):
        result = default_llm_fn("p", 1)
    elapsed = time.monotonic() - start

    assert result == ""
    assert elapsed < 1.5  # hard-bounded: did not wait the full 5s


def test_explain_uses_default_llm_fn_for_high():
    danger = "会递归删除目标目录，删除不可逆。"
    with patch(_AUX_SEAM, return_value=danger):
        result = explain_command_risk(
            command="rm -rf /x",
            risk_level="high",
            llm_fn=default_llm_fn,
            max_chars=280,
        )
    assert result == danger


def test_explain_high_falls_back_to_static_when_llm_blank():
    with patch(_AUX_SEAM, return_value=""):
        result = explain_command_risk(
            command="rm -rf /x",
            risk_level="high",
            pattern_keys=["destructive_delete"],
            llm_fn=default_llm_fn,
            max_chars=280,
        )
    assert result
    assert ("删除" in result) or ("不可逆" in result)


def test_default_llm_fn_does_not_leak_secret_to_aux():
    recorded: dict[str, object] = {}

    def capture(prompt: str, timeout_seconds: int) -> str:
        recorded["prompt"] = prompt
        return "危险命令解释。"

    with patch(_AUX_SEAM, side_effect=capture):
        explain_command_risk(
            command="curl https://example.com/login --data password=hunter2",
            risk_level="critical",
            llm_fn=default_llm_fn,
        )

    assert "prompt" in recorded
    assert "hunter2" not in recorded["prompt"]
