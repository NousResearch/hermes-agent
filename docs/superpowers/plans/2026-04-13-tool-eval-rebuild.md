# Tool Eval Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the `tool_eval/` harness that tests OpenAI-compatible models on hermes-agent tool calling quality.

**Architecture:** Five files in `tool_eval/`: `scorer.py` (scoring engine reconstructed from cached bytecode), `run_eval.py` (CLI runner), `test_cases.json` (~51 test cases), `hermes_context.md` (system prompt injection), `tool_primer.md` (tool-call format cheatsheet). Unit tests live in `tool_eval/tests/test_scorer.py`. Scoring is 100% objective — no LLM-as-judge.

**Tech Stack:** Python 3.11+, `openai` package (already in pyproject.toml), `python-dotenv`, `rich` (already in deps), `pytest` for scorer unit tests.

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `tool_eval/scorer.py` | Scoring engine: parse responses, evaluate criteria |
| Create | `tool_eval/tests/test_scorer.py` | Unit tests for scorer |
| Create | `tool_eval/run_eval.py` | CLI runner: send prompts, collect results, print summary |
| Create | `tool_eval/test_cases.json` | 51 test cases across 11 tool categories |
| Create | `tool_eval/hermes_context.md` | Hermes operational context for `--hermes-context` |
| Create | `tool_eval/tool_primer.md` | OpenAI tool-call format cheatsheet for `--tool-primer` |

---

## Task 1: scorer.py — TestResult + extraction helpers

**Files:**
- Create: `tool_eval/tests/test_scorer.py`
- Create: `tool_eval/scorer.py`

- [ ] **Step 1: Write the failing tests**

Create `tool_eval/tests/__init__.py` (empty) and `tool_eval/tests/test_scorer.py`:

```python
"""Unit tests for tool_eval/scorer.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import pytest
from tool_eval.scorer import (
    TestResult,
    _safe_first_choice,
    _extract_tool_calls,
    _has_text_content,
    _text_content,
    _is_infra_error,
)


# --- _safe_first_choice ---

def test_safe_first_choice_normal():
    raw = {"choices": [{"message": {"content": "hello"}}]}
    assert _safe_first_choice(raw) == {"message": {"content": "hello"}}


def test_safe_first_choice_null_choices():
    assert _safe_first_choice({"choices": None}) is None


def test_safe_first_choice_empty_list():
    assert _safe_first_choice({"choices": []}) is None


def test_safe_first_choice_missing_key():
    assert _safe_first_choice({}) is None


def test_safe_first_choice_not_dict():
    assert _safe_first_choice("not a dict") is None


# --- _extract_tool_calls ---

def _make_tool_response(name: str, args: dict) -> dict:
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args)
                    }
                }]
            }
        }]
    }


def test_extract_tool_calls_standard_shape():
    raw = _make_tool_response("terminal", {"command": "ls"})
    calls = _extract_tool_calls(raw)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "terminal"
    assert calls[0]["function"]["arguments"]["command"] == "ls"


def test_extract_tool_calls_empty_when_null_choices():
    raw = {"choices": None, "error": {"message": "rate limit"}}
    assert _extract_tool_calls(raw) == []


def test_extract_tool_calls_empty_when_no_tools():
    raw = {"choices": [{"message": {"content": "hello", "tool_calls": None}}]}
    assert _extract_tool_calls(raw) == []


def test_extract_tool_calls_multiple():
    raw = {
        "choices": [{
            "message": {
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "todo", "arguments": '{"todos": []}'}},
                    {"id": "c2", "type": "function", "function": {"name": "memory", "arguments": '{"action": "add", "target": "user"}'}},
                ]
            }
        }]
    }
    calls = _extract_tool_calls(raw)
    assert len(calls) == 2
    assert calls[0]["function"]["name"] == "todo"
    assert calls[1]["function"]["name"] == "memory"


def test_extract_tool_calls_malformed_json_args():
    raw = {
        "choices": [{
            "message": {
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": "NOT JSON"}}]
            }
        }]
    }
    calls = _extract_tool_calls(raw)
    assert len(calls) == 1
    assert calls[0]["function"]["arguments"] == {}


# --- _has_text_content / _text_content ---

def test_text_content_extracts_string():
    raw = {"choices": [{"message": {"content": "Sure, I can help!"}}]}
    assert _text_content(raw) == "Sure, I can help!"


def test_text_content_empty_when_null():
    raw = {"choices": [{"message": {"content": None}}]}
    assert _text_content(raw) == ""


def test_has_text_content_true():
    raw = {"choices": [{"message": {"content": "hello"}}]}
    assert _has_text_content(raw) is True


def test_has_text_content_false_for_tool_call():
    raw = {"choices": [{"message": {"content": None, "tool_calls": [{"function": {"name": "todo"}}]}}]}
    assert _has_text_content(raw) is False


# --- _is_infra_error ---

def test_is_infra_error_rate_limit():
    raw = {"choices": None, "error": {"message": "rate limit exceeded", "code": 429}}
    assert _is_infra_error(raw) is True


def test_is_infra_error_502():
    raw = {"choices": None, "error": {"message": "Bad Gateway", "code": "502"}}
    assert _is_infra_error(raw) is True


def test_is_infra_error_all_null():
    raw = {"id": None, "choices": None, "model": None}
    assert _is_infra_error(raw) is True


def test_is_infra_error_false_for_normal():
    raw = {"choices": [{"message": {"content": "hello"}}], "id": "chatcmpl-123"}
    assert _is_infra_error(raw) is False


def test_is_infra_error_false_for_tool_call():
    raw = _make_tool_response("terminal", {"command": "ls"})
    raw["id"] = "chatcmpl-abc"
    assert _is_infra_error(raw) is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/eric/Docker/Stack/hermes-agent
python -m pytest tool_eval/tests/test_scorer.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'tool_eval'` or similar import error.

- [ ] **Step 3: Create tool_eval/__init__.py and scorer.py**

Create `tool_eval/__init__.py` (empty):
```python
```

Create `tool_eval/scorer.py`:

```python
"""Scoring engine for tool_eval harness.

Scoring: 40 pts tool name(s) correct + 60 pts argument quality.
All scoring is structural/parsing-based — no LLM-as-judge.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TestResult:
    """Result of a single test evaluation."""

    test_id: str
    category: str
    description: str
    score: int
    passed: bool
    details: Dict[str, Any]
    model_calls: int
    raw_response: Dict
    error: Optional[str] = None
    retries: int = 0
    is_infra_error: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "category": self.category,
            "description": self.description,
            "score": self.score,
            "passed": self.passed,
            "details": self.details,
            "model_calls": self.model_calls,
            "error": self.error,
            "retries": self.retries,
            "is_infra_error": self.is_infra_error,
            "raw_response": self._trunc(str(self.raw_response), 500),
        }

    @staticmethod
    def _trunc(text: str, max_len: int = 200) -> str:
        return text if len(text) <= max_len else text[:max_len] + "..."


def _safe_first_choice(raw: Any) -> Optional[Dict]:
    """Safely extract the first choice from a model response, handling None/empty lists."""
    if not isinstance(raw, dict):
        return None
    choices = raw.get("choices")
    if not isinstance(choices, list) or len(choices) == 0:
        return None
    first = choices[0]
    if first is None or not isinstance(first, dict):
        return None
    return first


def _extract_tool_calls(raw: Any) -> List[Dict]:
    """Extract tool call list from a model response dict.

    Handles both OpenAI chat.completion shape and raw dict shapes.
    Returns list of dicts with 'function' key containing name + arguments.
    """
    if not isinstance(raw, dict):
        return []

    choice = _safe_first_choice(raw)
    if choice:
        message = choice.get("message") or {}
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            return _parse_tool_calls_list(tool_calls)

    # Fallback: top-level tool_calls
    tool_calls = raw.get("tool_calls")
    if isinstance(tool_calls, list):
        return _parse_tool_calls_list(tool_calls)

    return []


def _parse_tool_calls_list(tool_calls: list) -> List[Dict]:
    result = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        func = tc.get("function") or {}
        name = func.get("name", "")
        args = func.get("arguments", "{}")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                args = {}
        result.append({"function": {"name": name, "arguments": args}})
    return result


def _has_text_content(raw: Any) -> bool:
    """Check if model produced any textual content (not tool calls or refusals)."""
    return bool(_text_content(raw).strip())


def _text_content(raw: Any) -> str:
    """Extract text content from model response."""
    if not isinstance(raw, dict):
        return ""
    choice = _safe_first_choice(raw)
    if not choice:
        return ""
    message = choice.get("message") or {}
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _check_arg_values(actual: Dict, expected: Dict) -> Tuple[bool, Dict]:
    """Check if specific arg values match (with type awareness)."""
    details: Dict = {}
    all_passed = True
    for key, exp_val in expected.items():
        act_val = actual.get(key)
        if isinstance(exp_val, str) and isinstance(act_val, str):
            passed = act_val.lower() == exp_val.lower()
        else:
            passed = act_val == exp_val
        details[key] = {"expected": exp_val, "actual": act_val, "passed": passed}
        if not passed:
            all_passed = False
    return all_passed, details


def _check_type_compliance(actual: Dict, expected: Dict) -> Tuple[bool, Dict]:
    """Verify argument types match expected."""
    type_map = {
        "int": int, "str": str, "bool": bool,
        "float": float, "list": list, "dict": dict,
    }
    details: Dict = {}
    all_passed = True
    for key, type_name in expected.items():
        act_val = actual.get(key)
        expected_type = type_map.get(type_name)
        if expected_type is None:
            continue
        passed = isinstance(act_val, expected_type)
        details[key] = {
            "expected_type": type_name,
            "actual_type": type(act_val).__name__,
            "passed": passed,
        }
        if not passed:
            all_passed = False
    return all_passed, details


def _check_no_extra_params(actual: Dict, schema: Dict) -> Tuple[bool, str]:
    """Check that model didn't invent parameters not in schema.

    Some models add hallucinated args. This catches that.
    """
    allowed = set(schema.get("parameters", {}).get("properties", {}).keys())
    allowed.discard("_raw_string")
    allowed.discard("_parse_error")
    extra = set(actual.keys()) - allowed
    if extra:
        return False, f"Hallucinated args: {sorted(extra)}"
    return True, ""


def _check_list_field(actual: Dict, field: str, min_items: int) -> Tuple[bool, str]:
    """Check that a list field exists and has the minimum number of items.

    Returns (passed, details).
    """
    val = actual.get(field)
    if not isinstance(val, list):
        return False, f"field='{field}' missing or actual_type={type(val).__name__}"
    actual_count = len(val)
    passed = actual_count >= min_items
    return passed, f"field='{field}' min_items={min_items} actual_count={actual_count}"


def _is_infra_error(raw: Any) -> bool:
    """Detect upstream infrastructure errors (rate limits, 502s, null responses).

    These are NOT model failures — the model never received the prompt.
    """
    if not isinstance(raw, dict):
        return False
    choices = raw.get("choices")
    error = raw.get("error")

    # choices is null/empty AND error dict present with rate/502 signal
    if (not choices) and isinstance(error, dict):
        message = str(error.get("message", "")).lower()
        code = str(error.get("code", ""))
        if "rate" in message or "502" in message or code == "502":
            return True

    # All primary fields null simultaneously
    if not any([raw.get("id"), raw.get("choices"), raw.get("model")]):
        return True

    return False


def _score_single_args_ratio(actual_args: Dict, spec: Dict) -> Tuple[float, Dict]:
    """Score arguments for a single matched tool call.

    Returns (ratio 0.0-1.0, details). All active criteria equally weighted.
    """
    criteria_scores: List[float] = []
    details: Dict = {}

    required_args = spec.get("required_args", [])
    if required_args:
        present = [k for k in required_args if k in actual_args]
        ratio = len(present) / len(required_args)
        criteria_scores.append(ratio)
        details["required_args"] = {"required": required_args, "present": present, "ratio": ratio}

    arg_values = spec.get("arg_values", {})
    if arg_values:
        _, val_details = _check_arg_values(actual_args, arg_values)
        ratio = sum(1 for v in val_details.values() if v["passed"]) / len(val_details)
        criteria_scores.append(ratio)
        details["arg_values"] = val_details

    optional_args = spec.get("optional_args", [])
    if optional_args:
        present = [k for k in optional_args if k in actual_args]
        ratio = len(present) / len(optional_args)
        criteria_scores.append(ratio)
        details["optional_args"] = {"optional": optional_args, "present": present, "ratio": ratio}

    arg_substring_checks = spec.get("arg_substring_checks", {})
    if arg_substring_checks:
        results = {}
        for key, substr in arg_substring_checks.items():
            val = str(actual_args.get(key, "")).lower()
            passed = substr.lower() in val
            results[key] = {"substring": substr, "passed": passed}
        ratio = sum(1 for v in results.values() if v["passed"]) / len(results)
        criteria_scores.append(ratio)
        details["arg_substring_checks"] = results

    arg_types = spec.get("arg_types", {})
    if arg_types:
        _, type_details = _check_type_compliance(actual_args, arg_types)
        if type_details:
            ratio = sum(1 for v in type_details.values() if v["passed"]) / len(type_details)
            criteria_scores.append(ratio)
            details["arg_types"] = type_details

    no_extra_params = spec.get("no_extra_params")
    if no_extra_params:
        passed, msg = _check_no_extra_params(actual_args, no_extra_params)
        criteria_scores.append(1.0 if passed else 0.0)
        details["no_extra_params"] = {"passed": passed, "message": msg}

    list_field_check = spec.get("list_field_check")
    if list_field_check:
        field = list_field_check.get("field", "")
        min_items = list_field_check.get("min_items", 1)
        passed, msg = _check_list_field(actual_args, field, min_items)
        criteria_scores.append(1.0 if passed else 0.0)
        details["list_field_check"] = {"passed": passed, "message": msg}

    if not criteria_scores:
        details["_criteria_summary"] = "No detailed arg scoring criteria — full credit"
        return 1.0, details

    return round(sum(criteria_scores) / len(criteria_scores), 4), details


def score_test(test_case: Dict, raw_response: Dict) -> TestResult:
    """Score a single test case against the model's response.

    Returns a TestResult with score 0-100.
    """
    test_id = test_case.get("id", "unknown")
    category = test_case.get("category", "Unknown")
    description = test_case.get("description", "")
    expected = test_case.get("expected", {})

    if _is_infra_error(raw_response):
        return TestResult(
            test_id=test_id, category=category, description=description,
            score=0, passed=False,
            details={"error": "infra_error"},
            model_calls=0, raw_response=raw_response,
            error="Infrastructure error (rate limit / 502)",
            is_infra_error=True,
        )

    tool_calls = _extract_tool_calls(raw_response)
    details: Dict = {}

    # --- no_tool_calls ---
    if expected.get("no_tool_calls"):
        hallucinated = [tc["function"]["name"] for tc in tool_calls]
        passed = len(tool_calls) == 0
        for name in hallucinated:
            details[f"hallucinated_{name}"] = {"passed": False}
        score = 100 if passed else 0
        return TestResult(
            test_id=test_id, category=category, description=description,
            score=score, passed=passed, details=details,
            model_calls=1, raw_response=raw_response,
            error=None if passed else f"unexpected_tool_calls: {hallucinated}",
        )

    # --- text_no_call ---
    if expected.get("text_no_call"):
        has_text = _has_text_content(raw_response)
        no_calls = len(tool_calls) == 0
        passed = has_text and no_calls
        details["text_no_call"] = {"has_text": has_text, "no_tool_calls": no_calls, "passed": passed}
        return TestResult(
            test_id=test_id, category=category, description=description,
            score=100 if passed else 0, passed=passed, details=details,
            model_calls=1, raw_response=raw_response,
            error=None if passed else "Expected text response with no tool calls",
        )

    # --- has_text / text_contains ---
    if expected.get("has_text") or expected.get("text_contains"):
        text = _text_content(raw_response)
        has_text = bool(text.strip())
        if not has_text:
            return TestResult(
                test_id=test_id, category=category, description=description,
                score=0, passed=False, details={"has_text": False},
                model_calls=1, raw_response=raw_response,
                error="Model did not return any text content",
            )
        substr = expected.get("text_contains", "")
        if substr:
            passed = substr.lower() in text.lower()
            details[f"text_contains_{substr[:20]}"] = {"passed": passed, "substring": substr}
            score = 100 if passed else 0
        else:
            passed = True
            score = 100
        return TestResult(
            test_id=test_id, category=category, description=description,
            score=score, passed=passed, details=details,
            model_calls=1, raw_response=raw_response,
            error=None if passed else f"Text did not contain: {substr!r}",
        )

    # --- unexpected_tool_calls: model must NOT call listed tools ---
    unexpected_tools = expected.get("unexpected_tool_calls", [])
    if unexpected_tools:
        called_names = {tc["function"]["name"] for tc in tool_calls}
        violations = [n for n in unexpected_tools if n in called_names]
        passed = len(violations) == 0
        details["unexpected_tool_calls"] = {"forbidden": unexpected_tools, "violations": violations, "passed": passed}
        return TestResult(
            test_id=test_id, category=category, description=description,
            score=100 if passed else 0, passed=passed, details=details,
            model_calls=1, raw_response=raw_response,
            error=None if passed else f"Called forbidden tools: {violations}",
        )

    # --- Standard tool call scoring ---
    if not tool_calls:
        return TestResult(
            test_id=test_id, category=category, description=description,
            score=0, passed=False, details={"no_tool_calls_made": True},
            model_calls=1, raw_response=raw_response,
            error="Model did not call any tools",
        )

    tool_names = {tc["function"]["name"] for tc in tool_calls}
    fail_reason = None

    # function_count (exact)
    expected_count = expected.get("function_count")
    if expected_count is not None:
        actual_count = len(tool_calls)
        count_met = actual_count == expected_count
        details["function_count"] = {"expected": expected_count, "actual": actual_count, "passed": count_met}
        if not count_met:
            return TestResult(
                test_id=test_id, category=category, description=description,
                score=0, passed=False, details=details,
                model_calls=1, raw_response=raw_response,
                error=f"Expected exactly {expected_count} calls, got {actual_count}",
            )

    # function_counts_at_least
    min_count = expected.get("function_counts_at_least")
    if min_count is not None:
        actual_count = len(tool_calls)
        count_met = actual_count >= min_count
        details[f"at_least_{min_count}_calls"] = {
            "expected": f">= {min_count} calls, got {actual_count}",
            "passed": count_met,
        }
        if not count_met:
            return TestResult(
                test_id=test_id, category=category, description=description,
                score=0, passed=False, details=details,
                model_calls=1, raw_response=raw_response,
                error=f"Expected >= {min_count} calls, got {actual_count}",
            )

    # Resolve expected tool name(s)
    expected_names: List[str] = (
        expected.get("function_names")
        or ([expected["function_name"]] if expected.get("function_name") else [])
    )

    # --- Tool name score (40 pts) ---
    name_score = 40
    if expected_names:
        matched_tools = [n for n in expected_names if n in tool_names]
        missing_tools = [n for n in expected_names if n not in tool_names]
        extra_tools = sorted(tool_names - set(expected_names))
        name_ratio = len(matched_tools) / len(expected_names)
        name_score = int(round(name_ratio * 40))
        details["function_names"] = {
            "expected": expected_names,
            "actual": sorted(tool_names),
            "matched": matched_tools,
            "missing": missing_tools,
            "extra": extra_tools,
            "score": name_score,
        }
        if missing_tools:
            fail_reason = f"Tool '{missing_tools[0]}' not called"

    # --- Argument score (60 pts) ---
    args_spec = expected.get("arguments", {})
    args_score = 60

    if args_spec and expected_names:
        per_tool_ratios: List[float] = []
        for i, exp_name in enumerate(expected_names):
            found = next(
                (tc for tc in tool_calls if tc["function"]["name"] == exp_name),
                None,
            )
            if found is None:
                details[f"args_{exp_name}"] = {"passed": False, "error": f"Tool '{exp_name}' not called"}
                per_tool_ratios.append(0.0)
                continue

            spec = args_spec if len(expected_names) == 1 else args_spec.get(exp_name, {})
            actual_args = found["function"].get("arguments") or {}
            if not isinstance(actual_args, dict):
                actual_args = {}

            ratio, arg_details = _score_single_args_ratio(actual_args, spec)
            per_tool_ratios.append(ratio)
            details[f"call_{i}_{exp_name}"] = arg_details

        avg_ratio = sum(per_tool_ratios) / len(per_tool_ratios) if per_tool_ratios else 1.0
        args_score = int(round(avg_ratio * 60))

    score = name_score + args_score
    passed = score >= 60 and not (expected_names and missing_tools)

    return TestResult(
        test_id=test_id, category=category, description=description,
        score=score, passed=passed, details=details,
        model_calls=1, raw_response=raw_response,
        error=fail_reason,
    )


def score_debug_fixture(test_case: Dict) -> TestResult:
    """Score a test case using its own 'expected' as the model response.

    This is for --debug mode: verifies the scoring engine gives 100 on
    a perfect response by feeding the expected output back through scorer.
    """
    expected = test_case.get("expected", {})

    # Text-only test cases
    if expected.get("no_tool_calls") or expected.get("text_no_call"):
        raw_response = {
            "id": "debug-fixture",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "I cannot do that — the requested tool is not available.",
                    "tool_calls": None,
                }
            }],
        }
        return score_test(test_case, raw_response)

    if expected.get("has_text") or expected.get("text_contains"):
        text = expected.get("text_contains", "Here is the information you requested.")
        raw_response = {
            "id": "debug-fixture",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": text,
                    "tool_calls": None,
                }
            }],
        }
        return score_test(test_case, raw_response)

    if expected.get("unexpected_tool_calls"):
        raw_response = {
            "id": "debug-fixture",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "I'll handle this without using those tools.",
                    "tool_calls": None,
                }
            }],
        }
        return score_test(test_case, raw_response)

    # Tool call test cases
    expected_names: List[str] = (
        expected.get("function_names")
        or ([expected["function_name"]] if expected.get("function_name") else [])
    )
    args_spec = expected.get("arguments", {})

    tool_calls_raw = []
    for i, func_name in enumerate(expected_names):
        spec = args_spec if len(expected_names) == 1 else args_spec.get(func_name, {})
        args: Dict = {}

        for k in spec.get("required_args", []):
            args[k] = spec.get("arg_values", {}).get(k, f"test_{k}")
        for k, v in spec.get("arg_values", {}).items():
            args[k] = v
        for k, substr in spec.get("arg_substring_checks", {}).items():
            if k not in args:
                args[k] = substr

        lfc = spec.get("list_field_check")
        if lfc:
            field = lfc["field"]
            min_items = lfc.get("min_items", 1)
            if field not in args:
                args[field] = [
                    {"id": f"t{j}", "content": f"task {j}", "status": "pending"}
                    for j in range(min_items)
                ]

        tool_calls_raw.append({
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": func_name, "arguments": json.dumps(args)},
        })

    raw_response = {
        "id": "debug-fixture",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls_raw,
            }
        }],
    }
    return score_test(test_case, raw_response)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/eric/Docker/Stack/hermes-agent
python -m pytest tool_eval/tests/test_scorer.py -v
```

Expected: All tests pass (green).

- [ ] **Step 5: Commit**

```bash
git add tool_eval/__init__.py tool_eval/scorer.py tool_eval/tests/__init__.py tool_eval/tests/test_scorer.py
git commit -m "feat(tool_eval): add scorer with extraction helpers and unit tests"
```

---

## Task 2: scorer.py — argument scoring + score_test tests

**Files:**
- Modify: `tool_eval/tests/test_scorer.py` (append)

- [ ] **Step 1: Append argument scoring tests to test_scorer.py**

```python
# Append to tool_eval/tests/test_scorer.py

from tool_eval.scorer import (
    _check_arg_values,
    _check_type_compliance,
    _check_no_extra_params,
    _check_list_field,
    _score_single_args_ratio,
    score_test,
    score_debug_fixture,
)


# --- _check_arg_values ---

def test_check_arg_values_match():
    passed, details = _check_arg_values({"action": "create"}, {"action": "create"})
    assert passed
    assert details["action"]["passed"]


def test_check_arg_values_case_insensitive():
    passed, details = _check_arg_values({"action": "CREATE"}, {"action": "create"})
    assert passed


def test_check_arg_values_mismatch():
    passed, _ = _check_arg_values({"action": "delete"}, {"action": "create"})
    assert not passed


def test_check_arg_values_missing_key():
    passed, details = _check_arg_values({}, {"action": "create"})
    assert not passed
    assert details["action"]["actual"] is None


# --- _check_type_compliance ---

def test_check_type_compliance_pass():
    passed, _ = _check_type_compliance({"count": 3}, {"count": "int"})
    assert passed


def test_check_type_compliance_fail():
    passed, details = _check_type_compliance({"count": "3"}, {"count": "int"})
    assert not passed
    assert details["count"]["actual_type"] == "str"


def test_check_type_compliance_list():
    passed, _ = _check_type_compliance({"items": [1, 2]}, {"items": "list"})
    assert passed


# --- _check_no_extra_params ---

def test_no_extra_params_pass():
    schema = {"parameters": {"properties": {"command": {}, "timeout": {}}}}
    passed, msg = _check_no_extra_params({"command": "ls"}, schema)
    assert passed
    assert msg == ""


def test_no_extra_params_fail():
    schema = {"parameters": {"properties": {"command": {}}}}
    passed, msg = _check_no_extra_params({"command": "ls", "secret_flag": True}, schema)
    assert not passed
    assert "secret_flag" in msg


# --- _check_list_field ---

def test_check_list_field_pass():
    passed, msg = _check_list_field({"items": [1, 2, 3]}, "items", 3)
    assert passed
    assert "3" in msg


def test_check_list_field_too_short():
    passed, msg = _check_list_field({"items": [1]}, "items", 3)
    assert not passed


def test_check_list_field_missing():
    passed, msg = _check_list_field({}, "items", 1)
    assert not passed
    assert "missing" in msg


# --- _score_single_args_ratio ---

def test_score_single_args_all_criteria():
    spec = {
        "required_args": ["action", "target"],
        "arg_values": {"action": "add"},
        "list_field_check": {"field": "todos", "min_items": 2},
    }
    actual = {"action": "add", "target": "memory", "todos": [{"id": "1"}, {"id": "2"}]}
    ratio, details = _score_single_args_ratio(actual, spec)
    assert ratio == 1.0


def test_score_single_args_partial_credit():
    spec = {"required_args": ["action", "content"]}
    actual = {"action": "add"}  # missing content
    ratio, details = _score_single_args_ratio(actual, spec)
    assert ratio == 0.5


def test_score_single_args_no_criteria():
    ratio, details = _score_single_args_ratio({"anything": "goes"}, {})
    assert ratio == 1.0
    assert "No detailed arg scoring criteria" in details["_criteria_summary"]


# --- score_test (full integration) ---

def test_score_test_no_tool_calls_pass():
    test_case = {
        "id": "t1", "category": "test", "description": "no tool",
        "expected": {"no_tool_calls": True},
    }
    raw = {"id": "x", "choices": [{"message": {"content": "Sure!", "tool_calls": None}}]}
    result = score_test(test_case, raw)
    assert result.score == 100
    assert result.passed


def test_score_test_no_tool_calls_fail():
    test_case = {
        "id": "t1", "category": "test", "description": "no tool",
        "expected": {"no_tool_calls": True},
    }
    raw = _make_tool_response("terminal", {"command": "ls"})
    raw["id"] = "x"
    result = score_test(test_case, raw)
    assert result.score == 0
    assert not result.passed


def test_score_test_tool_name_match():
    test_case = {
        "id": "t2", "category": "terminal", "description": "run ls",
        "expected": {"function_name": "terminal", "arguments": {}},
    }
    raw = _make_tool_response("terminal", {"command": "ls"})
    raw["id"] = "x"
    result = score_test(test_case, raw)
    assert result.score == 100
    assert result.passed


def test_score_test_wrong_tool_name():
    test_case = {
        "id": "t3", "category": "terminal", "description": "run ls",
        "expected": {"function_name": "terminal"},
    }
    raw = _make_tool_response("web_search", {"query": "ls command"})
    raw["id"] = "x"
    result = score_test(test_case, raw)
    assert result.score == 0
    assert not result.passed


def test_score_test_full_arg_scoring():
    test_case = {
        "id": "t4", "category": "memory", "description": "add memory",
        "expected": {
            "function_name": "memory",
            "arguments": {
                "required_args": ["action", "target"],
                "arg_values": {"action": "add", "target": "user"},
            },
        },
    }
    raw = _make_tool_response("memory", {"action": "add", "target": "user", "content": "Eric lives in Colorado"})
    raw["id"] = "x"
    result = score_test(test_case, raw)
    assert result.score == 100
    assert result.passed


def test_score_test_infra_error():
    test_case = {"id": "t5", "category": "test", "description": "x", "expected": {"function_name": "terminal"}}
    raw = {"choices": None, "error": {"message": "rate limit exceeded"}}
    result = score_test(test_case, raw)
    assert result.is_infra_error
    assert result.score == 0


# --- score_debug_fixture ---

def test_debug_fixture_tool_call_scores_100():
    test_case = {
        "id": "d1", "category": "terminal", "description": "terminal test",
        "expected": {
            "function_name": "terminal",
            "arguments": {
                "required_args": ["command"],
                "arg_values": {"command": "ls /tmp"},
            },
        },
    }
    result = score_debug_fixture(test_case)
    assert result.score == 100, f"Expected 100, got {result.score}: {result.details}"


def test_debug_fixture_no_tool_calls_scores_100():
    test_case = {
        "id": "d2", "category": "test", "description": "no tool",
        "expected": {"no_tool_calls": True},
    }
    result = score_debug_fixture(test_case)
    assert result.score == 100, f"Expected 100, got {result.score}"


def test_debug_fixture_list_field_scores_100():
    test_case = {
        "id": "d3", "category": "todo", "description": "create todos",
        "expected": {
            "function_name": "todo",
            "arguments": {
                "list_field_check": {"field": "todos", "min_items": 3},
            },
        },
    }
    result = score_debug_fixture(test_case)
    assert result.score == 100, f"Expected 100, got {result.score}: {result.details}"
```

- [ ] **Step 2: Run the expanded test suite**

```bash
cd /home/eric/Docker/Stack/hermes-agent
python -m pytest tool_eval/tests/test_scorer.py -v
```

Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add tool_eval/tests/test_scorer.py
git commit -m "test(tool_eval): add full scorer test coverage for arg scoring and score_test"
```

---

## Task 3: hermes_context.md and tool_primer.md

**Files:**
- Create: `tool_eval/hermes_context.md`
- Create: `tool_eval/tool_primer.md`

- [ ] **Step 1: Create hermes_context.md**

```markdown
# Hermes Agent — Operational Context

You are Hermes, a self-improving AI agent. You operate autonomously to complete tasks for users.

## Core Behavior

- Use tools proactively and precisely. Don't ask for clarification unless genuinely ambiguous.
- Call the right tool for the job. Use `read_file` instead of `terminal` for reading files. Use `search_files` instead of `terminal` for searching.
- When given a task, call the appropriate tool immediately with complete arguments.
- Only use `terminal` for operations that have no dedicated tool (process management, running scripts, compiling code, etc.).

## Tool Selection Rules

- Reading files → `read_file`
- Writing files → `write_file`
- Editing files → `patch`
- Searching file contents → `search_files`
- Running shell commands → `terminal`
- Web search → `web_search`
- Fetching URLs → `web_extract`
- Managing tasks → `todo`
- Saving persistent facts → `memory`
- Scheduling jobs → `cronjob`
- Sending platform messages → `send_message`
- Managing skills → `skill_manage`

## Response Style

- Be concise and direct.
- Don't explain what you're about to do — just do it.
- Don't add caveats unless they're essential.
```

- [ ] **Step 2: Create tool_primer.md**

```markdown
# OpenAI Tool Call Format

When calling tools, your response must use the `tool_calls` field in the assistant message. Do NOT put tool calls in the text content.

## Correct Format

```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "tool_name",
        "arguments": "{\"arg1\": \"value1\", \"arg2\": 42}"
      }
    }
  ]
}
```

## Rules

1. `arguments` must be a JSON **string** (not an object) — serialize it with `json.dumps`.
2. Only include arguments defined in the tool's schema. Do not add extra arguments.
3. Required arguments must always be present.
4. Argument types must match the schema exactly (string, integer, boolean, array, object).
5. To call multiple tools, include multiple entries in the `tool_calls` array.

## Common Mistakes

- ❌ Putting tool call JSON in `content` as text
- ❌ Adding `arguments` as a dict instead of a JSON string
- ❌ Inventing argument names not in the schema
- ❌ Omitting required arguments
```

- [ ] **Step 3: Commit**

```bash
git add tool_eval/hermes_context.md tool_eval/tool_primer.md
git commit -m "feat(tool_eval): add hermes_context.md and tool_primer.md injection files"
```

---

## Task 4: test_cases.json — file category (10 tests)

**Files:**
- Create: `tool_eval/test_cases.json` (start with file category)

- [ ] **Step 1: Create test_cases.json with file tests**

```json
[
  {
    "id": "file_read_basic",
    "category": "file",
    "description": "Read a file using read_file tool",
    "prompt": "Read the file at /etc/hostname and tell me its contents.",
    "available_tools": ["read_file"],
    "expected": {
      "function_name": "read_file",
      "arguments": {
        "required_args": ["path"],
        "arg_values": {"path": "/etc/hostname"}
      }
    }
  },
  {
    "id": "file_read_with_offset",
    "category": "file",
    "description": "Read a specific range of lines from a file",
    "prompt": "Read lines 10 through 30 of the file /var/log/syslog.",
    "available_tools": ["read_file"],
    "expected": {
      "function_name": "read_file",
      "arguments": {
        "required_args": ["path", "offset"],
        "arg_values": {"path": "/var/log/syslog"},
        "arg_types": {"offset": "int"}
      }
    }
  },
  {
    "id": "file_write_new",
    "category": "file",
    "description": "Write content to a new file",
    "prompt": "Create a file at /tmp/hello.txt with the content: Hello, world!",
    "available_tools": ["write_file"],
    "expected": {
      "function_name": "write_file",
      "arguments": {
        "required_args": ["path", "content"],
        "arg_values": {"path": "/tmp/hello.txt"},
        "arg_substring_checks": {"content": "Hello, world!"}
      }
    }
  },
  {
    "id": "file_write_python_script",
    "category": "file",
    "description": "Write a Python script to a file",
    "prompt": "Write a Python script to /tmp/greet.py that prints 'Hello from hermes'.",
    "available_tools": ["write_file"],
    "expected": {
      "function_name": "write_file",
      "arguments": {
        "required_args": ["path", "content"],
        "arg_values": {"path": "/tmp/greet.py"},
        "arg_substring_checks": {"content": "print"}
      }
    }
  },
  {
    "id": "file_patch_replace",
    "category": "file",
    "description": "Replace a string in a file using patch tool",
    "prompt": "In the file /tmp/config.txt which contains the line 'debug=false', change 'debug=false' to 'debug=true'.",
    "available_tools": ["patch"],
    "expected": {
      "function_name": "patch",
      "arguments": {
        "required_args": ["path", "old_string", "new_string"],
        "arg_values": {
          "path": "/tmp/config.txt",
          "old_string": "debug=false",
          "new_string": "debug=true"
        }
      }
    }
  },
  {
    "id": "file_patch_default_mode",
    "category": "file",
    "description": "patch tool defaults to replace mode",
    "prompt": "In /tmp/app.py, replace the line 'VERSION = \"1.0\"' with 'VERSION = \"2.0\"'.",
    "available_tools": ["patch"],
    "expected": {
      "function_name": "patch",
      "arguments": {
        "required_args": ["path", "old_string", "new_string"],
        "arg_values": {
          "old_string": "VERSION = \"1.0\"",
          "new_string": "VERSION = \"2.0\""
        }
      }
    }
  },
  {
    "id": "file_search_content",
    "category": "file",
    "description": "Search file contents by regex pattern",
    "prompt": "Search for all lines containing 'TODO' in the /home/eric/Docker/Stack/hermes-agent directory.",
    "available_tools": ["search_files"],
    "expected": {
      "function_name": "search_files",
      "arguments": {
        "required_args": ["pattern"],
        "arg_values": {"target": "content"},
        "arg_substring_checks": {"pattern": "TODO"}
      }
    }
  },
  {
    "id": "file_search_by_filename",
    "category": "file",
    "description": "Find files by glob pattern",
    "prompt": "Find all Python files in /home/eric/Docker/Stack/hermes-agent/tools/.",
    "available_tools": ["search_files"],
    "expected": {
      "function_name": "search_files",
      "arguments": {
        "required_args": ["pattern"],
        "arg_values": {"target": "files"},
        "arg_substring_checks": {"pattern": ".py"}
      }
    }
  },
  {
    "id": "file_no_tool_for_math",
    "category": "file",
    "description": "File tools should not be called for non-file tasks",
    "prompt": "What is 15 multiplied by 7?",
    "available_tools": ["read_file", "write_file", "patch", "search_files"],
    "expected": {
      "no_tool_calls": true
    }
  },
  {
    "id": "file_read_not_terminal",
    "category": "file",
    "description": "Prefer read_file over terminal for reading files",
    "prompt": "Show me the contents of /etc/os-release.",
    "available_tools": ["read_file", "terminal"],
    "expected": {
      "function_name": "read_file",
      "arguments": {
        "required_args": ["path"],
        "arg_values": {"path": "/etc/os-release"}
      }
    }
  }
]
```

- [ ] **Step 2: Validate JSON is parseable**

```bash
python3 -c "import json; data=json.load(open('tool_eval/test_cases.json')); print(f'Loaded {len(data)} tests')"
```

Expected: `Loaded 10 tests`

- [ ] **Step 3: Commit**

```bash
git add tool_eval/test_cases.json
git commit -m "feat(tool_eval): add file category test cases (10 tests)"
```

---

## Task 5: test_cases.json — terminal + todo categories (11 tests)

**Files:**
- Modify: `tool_eval/test_cases.json` (append)

- [ ] **Step 1: Append terminal tests (4) to test_cases.json**

Add to the JSON array:

```json
  {
    "id": "terminal_list_files",
    "category": "terminal",
    "description": "Run ls command via terminal",
    "prompt": "List the files in the /tmp directory using a shell command.",
    "available_tools": ["terminal"],
    "expected": {
      "function_name": "terminal",
      "arguments": {
        "required_args": ["command"],
        "arg_substring_checks": {"command": "/tmp"}
      }
    }
  },
  {
    "id": "terminal_check_python_version",
    "category": "terminal",
    "description": "Check Python version via terminal",
    "prompt": "What version of Python is installed? Run the appropriate shell command to check.",
    "available_tools": ["terminal"],
    "expected": {
      "function_name": "terminal",
      "arguments": {
        "required_args": ["command"],
        "arg_substring_checks": {"command": "python"}
      }
    }
  },
  {
    "id": "terminal_create_directory",
    "category": "terminal",
    "description": "Create a directory via terminal",
    "prompt": "Create a directory called 'test_output' inside /tmp using a shell command.",
    "available_tools": ["terminal"],
    "expected": {
      "function_name": "terminal",
      "arguments": {
        "required_args": ["command"],
        "arg_substring_checks": {"command": "test_output"}
      }
    }
  },
  {
    "id": "terminal_run_python_script",
    "category": "terminal",
    "description": "Run an existing Python script via terminal",
    "prompt": "Run the Python script at /tmp/greet.py using a shell command.",
    "available_tools": ["terminal"],
    "expected": {
      "function_name": "terminal",
      "arguments": {
        "required_args": ["command"],
        "arg_substring_checks": {"command": "greet.py"}
      }
    }
  },
```

- [ ] **Step 2: Append todo tests (7) to test_cases.json**

```json
  {
    "id": "todo_create_tasks",
    "category": "todo",
    "description": "Create a multi-step todo list from a self-contained prompt",
    "prompt": "I need to set up a new Python project. Create a todo list with these steps: 1) Create a virtual environment with python -m venv .venv, 2) Install dependencies with pip install -r requirements.txt, 3) Run the tests with pytest, 4) Commit the results with git commit -m 'initial setup'.",
    "available_tools": ["todo"],
    "expected": {
      "function_name": "todo",
      "arguments": {
        "list_field_check": {"field": "todos", "min_items": 4},
        "arg_types": {"todos": "list"}
      }
    }
  },
  {
    "id": "todo_create_single",
    "category": "todo",
    "description": "Create a single todo item",
    "prompt": "Add a task to my todo list: 'Review pull request #42 in the hermes-agent repo'.",
    "available_tools": ["todo"],
    "expected": {
      "function_name": "todo",
      "arguments": {
        "list_field_check": {"field": "todos", "min_items": 1},
        "arg_types": {"todos": "list"}
      }
    }
  },
  {
    "id": "todo_read_list",
    "category": "todo",
    "description": "Read the current todo list by calling todo with no arguments",
    "prompt": "What's on my todo list right now?",
    "available_tools": ["todo"],
    "expected": {
      "function_name": "todo",
      "arguments": {}
    }
  },
  {
    "id": "todo_update_with_merge",
    "category": "todo",
    "description": "Update an existing todo item using merge=true",
    "prompt": "I have a todo list with item id='setup-venv' (content='Create virtual environment', status='pending'). Mark that item as in_progress. Use merge mode so you don't replace the whole list.",
    "available_tools": ["todo"],
    "expected": {
      "function_name": "todo",
      "arguments": {
        "required_args": ["todos"],
        "arg_values": {"merge": true},
        "list_field_check": {"field": "todos", "min_items": 1}
      }
    }
  },
  {
    "id": "todo_complete_task",
    "category": "todo",
    "description": "Mark a todo item as completed",
    "prompt": "I have a todo list with one item: id='write-tests', content='Write unit tests for scorer.py', status='in_progress'. I just finished writing the tests. Mark it as completed using merge mode.",
    "available_tools": ["todo"],
    "expected": {
      "function_name": "todo",
      "arguments": {
        "required_args": ["todos"],
        "arg_values": {"merge": true}
      }
    }
  },
  {
    "id": "todo_replace_list",
    "category": "todo",
    "description": "Replace the entire todo list with a fresh plan",
    "prompt": "Replace my entire todo list with a new 3-item plan: 1) Read the spec doc, 2) Write the implementation, 3) Run the tests.",
    "available_tools": ["todo"],
    "expected": {
      "function_name": "todo",
      "arguments": {
        "list_field_check": {"field": "todos", "min_items": 3}
      }
    }
  },
  {
    "id": "todo_not_needed_simple",
    "category": "todo",
    "description": "Simple questions should not trigger todo tool",
    "prompt": "What is the capital of France?",
    "available_tools": ["todo"],
    "expected": {
      "no_tool_calls": true
    }
  },
```

- [ ] **Step 3: Validate JSON**

```bash
python3 -c "import json; data=json.load(open('tool_eval/test_cases.json')); print(f'Loaded {len(data)} tests')"
```

Expected: `Loaded 21 tests`

- [ ] **Step 4: Commit**

```bash
git add tool_eval/test_cases.json
git commit -m "feat(tool_eval): add terminal (4) and todo (7) test cases"
```

---

## Task 6: test_cases.json — memory + web categories (11 tests)

**Files:**
- Modify: `tool_eval/test_cases.json` (append)

- [ ] **Step 1: Append memory tests (5)**

```json
  {
    "id": "memory_add_user_pref",
    "category": "memory",
    "description": "Save a user preference to memory",
    "prompt": "Remember that my name is Eric and I prefer concise responses.",
    "available_tools": ["memory"],
    "expected": {
      "function_name": "memory",
      "arguments": {
        "required_args": ["action", "target"],
        "arg_values": {"action": "add", "target": "user"},
        "arg_substring_checks": {"content": "Eric"}
      }
    }
  },
  {
    "id": "memory_add_env_fact",
    "category": "memory",
    "description": "Save an environment fact to memory store",
    "prompt": "Remember that the project uses Python 3.12 and lives at /home/eric/Docker/Stack/hermes-agent.",
    "available_tools": ["memory"],
    "expected": {
      "function_name": "memory",
      "arguments": {
        "required_args": ["action", "target", "content"],
        "arg_values": {"action": "add", "target": "memory"}
      }
    }
  },
  {
    "id": "memory_replace_entry",
    "category": "memory",
    "description": "Replace an existing memory entry",
    "prompt": "Update my memory: I previously noted my timezone was UTC but it's actually America/Denver. Find the old entry containing 'UTC' and replace it with 'America/Denver'.",
    "available_tools": ["memory"],
    "expected": {
      "function_name": "memory",
      "arguments": {
        "required_args": ["action", "target", "old_text"],
        "arg_values": {"action": "replace"},
        "arg_substring_checks": {"old_text": "UTC"}
      }
    }
  },
  {
    "id": "memory_remove_entry",
    "category": "memory",
    "description": "Remove a memory entry by identifying text",
    "prompt": "Forget the memory entry about my old API key — it contains the text 'sk-old-key-prefix'. Remove it.",
    "available_tools": ["memory"],
    "expected": {
      "function_name": "memory",
      "arguments": {
        "required_args": ["action", "target", "old_text"],
        "arg_values": {"action": "remove"},
        "arg_substring_checks": {"old_text": "sk-old-key-prefix"}
      }
    }
  },
  {
    "id": "memory_not_needed",
    "category": "memory",
    "description": "Don't use memory for ephemeral task state",
    "prompt": "I just finished step 1 of my task. Move on to step 2.",
    "available_tools": ["memory"],
    "expected": {
      "no_tool_calls": true
    }
  },
```

- [ ] **Step 2: Append web tests (6)**

```json
  {
    "id": "web_search_basic",
    "category": "web",
    "description": "Search the web for a topic",
    "prompt": "Search the web for the latest release of Python.",
    "available_tools": ["web_search"],
    "expected": {
      "function_name": "web_search",
      "arguments": {
        "required_args": ["query"],
        "arg_substring_checks": {"query": "Python"}
      }
    }
  },
  {
    "id": "web_search_technical",
    "category": "web",
    "description": "Search for technical documentation",
    "prompt": "Search the web for how to use asyncio.gather in Python.",
    "available_tools": ["web_search"],
    "expected": {
      "function_name": "web_search",
      "arguments": {
        "required_args": ["query"],
        "arg_substring_checks": {"query": "asyncio"}
      }
    }
  },
  {
    "id": "web_extract_url",
    "category": "web",
    "description": "Extract content from a URL",
    "prompt": "Extract the content from https://docs.python.org/3/library/asyncio.html",
    "available_tools": ["web_extract"],
    "expected": {
      "function_name": "web_extract",
      "arguments": {
        "required_args": ["urls"],
        "arg_types": {"urls": "list"},
        "list_field_check": {"field": "urls", "min_items": 1}
      }
    }
  },
  {
    "id": "web_extract_multiple_urls",
    "category": "web",
    "description": "Extract content from multiple URLs",
    "prompt": "Extract content from these two pages: https://docs.python.org/3/library/json.html and https://docs.python.org/3/library/os.html",
    "available_tools": ["web_extract"],
    "expected": {
      "function_name": "web_extract",
      "arguments": {
        "required_args": ["urls"],
        "list_field_check": {"field": "urls", "min_items": 2}
      }
    }
  },
  {
    "id": "web_search_not_terminal",
    "category": "web",
    "description": "Use web_search instead of terminal for web queries",
    "prompt": "Look up the current version of the openai Python package.",
    "available_tools": ["web_search", "terminal"],
    "expected": {
      "function_name": "web_search",
      "arguments": {
        "required_args": ["query"]
      }
    }
  },
  {
    "id": "web_extract_pdf",
    "category": "web",
    "description": "Extract content from a PDF URL",
    "prompt": "Extract the text from this arxiv paper PDF: https://arxiv.org/pdf/2303.08774.pdf",
    "available_tools": ["web_extract"],
    "expected": {
      "function_name": "web_extract",
      "arguments": {
        "required_args": ["urls"],
        "list_field_check": {"field": "urls", "min_items": 1}
      }
    }
  },
```

- [ ] **Step 3: Validate JSON**

```bash
python3 -c "import json; data=json.load(open('tool_eval/test_cases.json')); print(f'Loaded {len(data)} tests')"
```

Expected: `Loaded 32 tests`

- [ ] **Step 4: Commit**

```bash
git add tool_eval/test_cases.json
git commit -m "feat(tool_eval): add memory (5) and web (6) test cases"
```

---

## Task 7: test_cases.json — cron + messaging + skills categories (11 tests)

**Files:**
- Modify: `tool_eval/test_cases.json` (append)

- [ ] **Step 1: Append cron tests (5)**

```json
  {
    "id": "cron_create_daily",
    "category": "cron",
    "description": "Create a recurring daily cron job",
    "prompt": "Schedule a cron job that runs every day at 9am to check the system disk usage and send a summary to Telegram. Name it 'disk-check'.",
    "available_tools": ["cronjob"],
    "expected": {
      "function_name": "cronjob",
      "arguments": {
        "required_args": ["action", "prompt", "schedule"],
        "arg_values": {"action": "create"},
        "arg_substring_checks": {"name": "disk"}
      }
    }
  },
  {
    "id": "cron_create_one_shot",
    "category": "cron",
    "description": "Create a one-shot future cron job",
    "prompt": "In 30 minutes, run a one-time job that searches the web for 'Python 3.14 release notes' and sends the result to Telegram.",
    "available_tools": ["cronjob"],
    "expected": {
      "function_name": "cronjob",
      "arguments": {
        "required_args": ["action", "prompt", "schedule"],
        "arg_values": {"action": "create"}
      }
    }
  },
  {
    "id": "cron_list_jobs",
    "category": "cron",
    "description": "List all cron jobs",
    "prompt": "Show me all my scheduled cron jobs.",
    "available_tools": ["cronjob"],
    "expected": {
      "function_name": "cronjob",
      "arguments": {
        "required_args": ["action"],
        "arg_values": {"action": "list"}
      }
    }
  },
  {
    "id": "cron_remove_job",
    "category": "cron",
    "description": "Remove a cron job by ID",
    "prompt": "Delete the cron job with id 'disk-check-abc123'.",
    "available_tools": ["cronjob"],
    "expected": {
      "function_name": "cronjob",
      "arguments": {
        "required_args": ["action", "job_id"],
        "arg_values": {"action": "remove", "job_id": "disk-check-abc123"}
      }
    }
  },
  {
    "id": "cron_pause_job",
    "category": "cron",
    "description": "Pause an active cron job",
    "prompt": "Pause the cron job with id 'weather-report-xyz789'.",
    "available_tools": ["cronjob"],
    "expected": {
      "function_name": "cronjob",
      "arguments": {
        "required_args": ["action", "job_id"],
        "arg_values": {"action": "pause", "job_id": "weather-report-xyz789"}
      }
    }
  },
```

- [ ] **Step 2: Append messaging tests (3)**

```json
  {
    "id": "send_message_telegram",
    "category": "messaging",
    "description": "Send a message to the Telegram home channel",
    "prompt": "Send 'Build complete — all tests passed' to Telegram.",
    "available_tools": ["send_message"],
    "expected": {
      "function_name": "send_message",
      "arguments": {
        "required_args": ["target", "message"],
        "arg_values": {"action": "send"},
        "arg_substring_checks": {
          "target": "telegram",
          "message": "Build complete"
        }
      }
    }
  },
  {
    "id": "send_message_list_targets",
    "category": "messaging",
    "description": "List available messaging targets before sending to a specific channel",
    "prompt": "Send a message to the #dev-alerts channel on Discord, but first check what channels are available.",
    "available_tools": ["send_message"],
    "expected": {
      "function_name": "send_message",
      "arguments": {
        "arg_values": {"action": "list"}
      }
    }
  },
  {
    "id": "send_message_discord",
    "category": "messaging",
    "description": "Send a message to a specific Discord channel",
    "prompt": "Send 'Deployment to staging is complete' to discord:#engineering.",
    "available_tools": ["send_message"],
    "expected": {
      "function_name": "send_message",
      "arguments": {
        "required_args": ["target", "message"],
        "arg_substring_checks": {
          "target": "discord",
          "message": "staging"
        }
      }
    }
  },
```

- [ ] **Step 3: Append skills tests (3)**

```json
  {
    "id": "skill_create",
    "category": "skills",
    "description": "Create a new skill",
    "prompt": "Create a new skill called 'deploy-checklist' with this content:\n---\nname: deploy-checklist\ndescription: Use before deploying to production\n---\n\n# Deploy Checklist\n1. Run tests\n2. Review diff\n3. Tag the release\n4. Push to main",
    "available_tools": ["skill_manage"],
    "expected": {
      "function_name": "skill_manage",
      "arguments": {
        "required_args": ["action", "name", "content"],
        "arg_values": {"action": "create", "name": "deploy-checklist"}
      }
    }
  },
  {
    "id": "skill_patch",
    "category": "skills",
    "description": "Patch an existing skill using old_string/new_string",
    "prompt": "Update the skill 'deploy-checklist': find the line '3. Tag the release' and replace it with '3. Tag the release with git tag v<version>'.",
    "available_tools": ["skill_manage"],
    "expected": {
      "function_name": "skill_manage",
      "arguments": {
        "required_args": ["action", "name", "old_string", "new_string"],
        "arg_values": {
          "action": "patch",
          "name": "deploy-checklist",
          "old_string": "3. Tag the release"
        }
      }
    }
  },
  {
    "id": "skill_delete",
    "category": "skills",
    "description": "Delete a skill by name",
    "prompt": "Delete the skill named 'deploy-checklist'.",
    "available_tools": ["skill_manage"],
    "expected": {
      "function_name": "skill_manage",
      "arguments": {
        "required_args": ["action", "name"],
        "arg_values": {"action": "delete", "name": "deploy-checklist"}
      }
    }
  },
```

- [ ] **Step 4: Validate JSON**

```bash
python3 -c "import json; data=json.load(open('tool_eval/test_cases.json')); print(f'Loaded {len(data)} tests')"
```

Expected: `Loaded 43 tests`

- [ ] **Step 5: Commit**

```bash
git add tool_eval/test_cases.json
git commit -m "feat(tool_eval): add cron (5), messaging (3), and skills (3) test cases"
```

---

## Task 8: test_cases.json — vision, image, tts categories (8 tests)

**Files:**
- Modify: `tool_eval/test_cases.json` (append)

- [ ] **Step 1: Append vision tests (3)**

```json
  {
    "id": "vision_analyze_url",
    "category": "vision",
    "description": "Analyze an image from a URL",
    "prompt": "Analyze this image and describe what you see: https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/240px-PNG_transparency_demonstration_1.png",
    "available_tools": ["vision_analyze"],
    "expected": {
      "function_name": "vision_analyze",
      "arguments": {
        "required_args": ["image_url", "question"],
        "arg_substring_checks": {"image_url": "wikimedia"}
      }
    }
  },
  {
    "id": "vision_analyze_local",
    "category": "vision",
    "description": "Analyze a local image file",
    "prompt": "Analyze the image at /tmp/screenshot.png and describe its contents.",
    "available_tools": ["vision_analyze"],
    "expected": {
      "function_name": "vision_analyze",
      "arguments": {
        "required_args": ["image_url", "question"],
        "arg_values": {"image_url": "/tmp/screenshot.png"}
      }
    }
  },
  {
    "id": "vision_identify_object",
    "category": "vision",
    "description": "Answer a specific question about an image",
    "prompt": "Look at this photo and tell me what breed of dog is shown: https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg",
    "available_tools": ["vision_analyze"],
    "expected": {
      "function_name": "vision_analyze",
      "arguments": {
        "required_args": ["image_url", "question"],
        "arg_substring_checks": {"question": "dog"}
      }
    }
  },
```

- [ ] **Step 2: Append image generation tests (3)**

```json
  {
    "id": "image_generate_basic",
    "category": "image",
    "description": "Generate an image from a text prompt",
    "prompt": "Generate an image of a red fox sitting in a snowy forest.",
    "available_tools": ["image_generate"],
    "expected": {
      "function_name": "image_generate",
      "arguments": {
        "required_args": ["prompt"],
        "arg_substring_checks": {"prompt": "fox"}
      }
    }
  },
  {
    "id": "image_generate_with_aspect",
    "category": "image",
    "description": "Generate a portrait-orientation image",
    "prompt": "Generate a portrait-oriented image of a lighthouse at sunset on a rocky coastline.",
    "available_tools": ["image_generate"],
    "expected": {
      "function_name": "image_generate",
      "arguments": {
        "required_args": ["prompt"],
        "arg_values": {"aspect_ratio": "portrait"},
        "arg_substring_checks": {"prompt": "lighthouse"}
      }
    }
  },
  {
    "id": "image_generate_square",
    "category": "image",
    "description": "Generate a square image",
    "prompt": "Create a square image of a minimalist geometric logo with blue and white colors.",
    "available_tools": ["image_generate"],
    "expected": {
      "function_name": "image_generate",
      "arguments": {
        "required_args": ["prompt"],
        "arg_values": {"aspect_ratio": "square"}
      }
    }
  },
```

- [ ] **Step 3: Append TTS tests (2)**

```json
  {
    "id": "tts_basic",
    "category": "tts",
    "description": "Convert text to speech",
    "prompt": "Say this out loud: 'Hello, your deployment completed successfully.'",
    "available_tools": ["text_to_speech"],
    "expected": {
      "function_name": "text_to_speech",
      "arguments": {
        "required_args": ["text"],
        "arg_substring_checks": {"text": "deployment"}
      }
    }
  },
  {
    "id": "tts_announcement",
    "category": "tts",
    "description": "Convert a longer announcement to speech",
    "prompt": "Read this announcement aloud: 'The system will be down for maintenance on Saturday from 2am to 4am UTC. Please save your work before that time.'",
    "available_tools": ["text_to_speech"],
    "expected": {
      "function_name": "text_to_speech",
      "arguments": {
        "required_args": ["text"],
        "arg_substring_checks": {"text": "maintenance"}
      }
    }
  }
```

- [ ] **Step 4: Close the JSON array and validate**

The final entry must not have a trailing comma. Validate:

```bash
python3 -c "import json; data=json.load(open('tool_eval/test_cases.json')); cats={}; [cats.update({d['category']: cats.get(d['category'],0)+1}) for d in data]; print(f'Total: {len(data)} tests'); print(cats)"
```

Expected output:
```
Total: 51 tests
{'file': 10, 'terminal': 4, 'todo': 7, 'memory': 5, 'web': 6, 'cron': 5, 'messaging': 3, 'skills': 3, 'vision': 3, 'image': 3, 'tts': 2}
```

- [ ] **Step 5: Commit**

```bash
git add tool_eval/test_cases.json
git commit -m "feat(tool_eval): add vision (3), image (3), and tts (2) test cases — 51 total"
```

---

## Task 9: run_eval.py

**Files:**
- Create: `tool_eval/run_eval.py`

- [ ] **Step 1: Create run_eval.py**

```python
#!/usr/bin/env python3
"""Tool eval runner for hermes-agent.

Usage:
    python tool_eval/run_eval.py --model "anthropic/claude-sonnet-4" --openrouter
    python tool_eval/run_eval.py --model "gpt-4o" --base-url https://api.openai.com/v1 --api-key sk-...
    python tool_eval/run_eval.py --debug   # verify scorer correctness, no API calls
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Resolve paths relative to this file
TOOL_EVAL_DIR = Path(__file__).parent
REPO_ROOT = TOOL_EVAL_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

# Load .env before importing openai
def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
        for candidate in [
            TOOL_EVAL_DIR / ".env",
            REPO_ROOT / ".env",
            Path.home() / ".env",
        ]:
            if candidate.exists():
                load_dotenv(candidate)
                break
    except ImportError:
        pass

_load_dotenv()

import openai
from tool_eval.scorer import TestResult, score_test, score_debug_fixture

try:
    from rich.console import Console
    from rich.table import Table
    from rich import print as rprint
    _rich = True
    console = Console()
except ImportError:
    _rich = False
    console = None


def _load_test_cases(path: Path) -> list:
    with open(path) as f:
        return json.load(f)


def _build_tool_schemas(tool_names: list) -> list:
    """Load tool schemas from the hermes registry for the given tool names."""
    try:
        # Import registry without triggering the full agent boot
        from tools.registry import registry
        schemas = []
        for name in tool_names:
            entry = registry.get(name)
            if entry and hasattr(entry, "schema"):
                schemas.append(entry.schema)
        if schemas:
            return schemas
    except Exception:
        pass

    # Fallback: minimal passthrough schemas so the model knows tools exist
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"Tool: {name}",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
        for name in tool_names
    ]


def _build_tool_schemas_openai_format(tool_names: list) -> list:
    """Return tool schemas in OpenAI tools array format."""
    try:
        from tools.registry import registry
        schemas = []
        for name in tool_names:
            entry = registry.get(name)
            if entry and hasattr(entry, "schema"):
                schemas.append({"type": "function", "function": entry.schema})
                continue
            schemas.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"Tool: {name}",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            })
        return schemas
    except Exception:
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"Tool: {name}",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
            for name in tool_names
        ]


def _load_injection_file(path: Path) -> str:
    if path.exists():
        return path.read_text()
    return ""


def _run_single_test(
    client: openai.OpenAI,
    test_case: dict,
    model: str,
    hermes_context: str,
    tool_primer: str,
    rate_limit: float,
    max_retries: int = 3,
) -> TestResult:
    """Send a single test case to the model and score the response."""
    system_parts = []
    if hermes_context:
        system_parts.append(hermes_context)
    if tool_primer:
        system_parts.append(tool_primer)
    system_prompt = "\n\n---\n\n".join(system_parts) if system_parts else "You are a helpful assistant."

    tool_names = test_case.get("available_tools", [])
    tools = _build_tool_schemas_openai_format(tool_names) if tool_names else None

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": test_case["prompt"]},
    ]

    retries = 0
    last_error = None
    raw_response = {}

    while retries <= max_retries:
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": 0,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            response = client.chat.completions.create(**kwargs)
            raw_response = response.model_dump()
            break
        except openai.RateLimitError as e:
            retries += 1
            last_error = str(e)
            if retries <= max_retries:
                time.sleep(rate_limit * 2)
            else:
                raw_response = {"choices": None, "error": {"message": str(e), "code": 429}}
        except Exception as e:
            last_error = str(e)
            raw_response = {"choices": None, "error": {"message": str(e)}}
            break

        time.sleep(rate_limit)

    result = score_test(test_case, raw_response)
    result.retries = retries
    return result


def _print_result(result: TestResult, verbose: bool = False) -> None:
    status = "PASS" if result.passed else ("INFRA" if result.is_infra_error else "FAIL")
    color = "green" if result.passed else ("yellow" if result.is_infra_error else "red")

    if _rich:
        console.print(
            f"  [{color}]{status:5}[/{color}] {result.score:3}/100  "
            f"[dim]{result.test_id}[/dim]"
            + (f"  [dim red]{result.error}[/dim red]" if result.error and not result.passed else "")
        )
    else:
        print(f"  {status:5} {result.score:3}/100  {result.test_id}"
              + (f"  -- {result.error}" if result.error and not result.passed else ""))

    if verbose and result.details:
        for k, v in result.details.items():
            print(f"         {k}: {v}")


def _print_summary(results: list[TestResult], model: str) -> None:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    infra_errors = sum(1 for r in results if r.is_infra_error)
    avg_score = sum(r.score for r in results) / total if total else 0

    # Category breakdown
    cats: dict = {}
    for r in results:
        if r.category not in cats:
            cats[r.category] = {"total": 0, "passed": 0, "score_sum": 0}
        cats[r.category]["total"] += 1
        if r.passed:
            cats[r.category]["passed"] += 1
        cats[r.category]["score_sum"] += r.score

    print()
    if _rich:
        table = Table(title=f"Results — {model}")
        table.add_column("Category", style="cyan")
        table.add_column("Pass", justify="right")
        table.add_column("Total", justify="right")
        table.add_column("Avg Score", justify="right")
        for cat, data in sorted(cats.items()):
            avg = data["score_sum"] / data["total"]
            color = "green" if data["passed"] == data["total"] else ("yellow" if data["passed"] > 0 else "red")
            table.add_row(cat, f"[{color}]{data['passed']}[/{color}]", str(data["total"]), f"{avg:.0f}")
        table.add_section()
        table.add_row("TOTAL", f"{passed}", str(total), f"{avg_score:.0f}")
        console.print(table)
    else:
        print(f"=== Results: {model} ===")
        for cat, data in sorted(cats.items()):
            avg = data["score_sum"] / data["total"]
            print(f"  {cat:15} {data['passed']}/{data['total']}  avg={avg:.0f}")
        print(f"  {'TOTAL':15} {passed}/{total}  avg={avg_score:.0f}")

    if infra_errors:
        print(f"\n  WARNING: {infra_errors} infra errors (rate limits/502s) — scores may be lower than actual")
    print()


def _run_debug_mode(test_cases: list, categories: Optional[set]) -> int:
    """Feed gold fixtures back through scorer — all must score 100."""
    print("=== DEBUG MODE: verifying scorer on gold fixtures ===\n")
    failures = []
    for tc in test_cases:
        if categories and tc.get("category") not in categories:
            continue
        result = score_debug_fixture(tc)
        if result.score != 100:
            failures.append(result)
            print(f"  FAIL  {result.test_id}: score={result.score}, details={result.details}")
        else:
            print(f"  OK    {result.test_id}")

    print()
    if failures:
        print(f"FAIL: {len(failures)} fixture(s) scored < 100. Scorer has a bug.")
        return 1
    print(f"OK: All {len(test_cases)} fixtures score 100.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Tool eval runner for hermes-agent")
    parser.add_argument("--model", default="", help="Model ID (e.g. anthropic/claude-sonnet-4)")
    parser.add_argument("--base-url", default="", help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default="", help="API key (fallback: env vars)")
    parser.add_argument("--openrouter", action="store_true", help="Use OpenRouter (sets base-url, reads OPENROUTER_API_KEY)")
    parser.add_argument("--rate-limit", type=float, default=3.0, help="Seconds between requests (default: 3)")
    parser.add_argument("--hermes-context", action="store_true", help="Inject hermes_context.md into system prompt")
    parser.add_argument("--tool-primer", action="store_true", help="Inject tool_primer.md into system prompt")
    parser.add_argument("--vision", action="store_true", help="Include vision_analyze tests (opt-in)")
    parser.add_argument("--image", action="store_true", help="Include image_generate tests (opt-in)")
    parser.add_argument("--tts", action="store_true", help="Include text_to_speech tests (opt-in)")
    parser.add_argument("--debug", action="store_true", help="Debug mode: verify scorer on gold fixtures, no API calls")
    parser.add_argument("--category", default="", help="Run only tests in this category")
    parser.add_argument("--test-id", default="", help="Run a single test by ID")
    parser.add_argument("--verbose", action="store_true", help="Print per-test scoring details")
    parser.add_argument("--json-output", default="", help="Write results JSON to this file path")
    args = parser.parse_args()

    test_cases = _load_test_cases(TOOL_EVAL_DIR / "test_cases.json")

    # Filter by opt-in categories
    excluded_by_default = set()
    if not args.vision:
        excluded_by_default.add("vision")
    if not args.image:
        excluded_by_default.add("image")
    if not args.tts:
        excluded_by_default.add("tts")

    test_cases = [tc for tc in test_cases if tc.get("category") not in excluded_by_default]

    # Filter by --category or --test-id
    if args.test_id:
        test_cases = [tc for tc in test_cases if tc["id"] == args.test_id]
    elif args.category:
        test_cases = [tc for tc in test_cases if tc.get("category") == args.category]

    if not test_cases:
        print("No test cases matched the given filters.")
        sys.exit(1)

    # Debug mode
    if args.debug:
        categories = {args.category} if args.category else None
        sys.exit(_run_debug_mode(test_cases, categories))

    # Require model for live runs
    if not args.model:
        parser.error("--model is required (unless using --debug)")

    # Configure OpenAI client
    base_url = args.base_url
    api_key = args.api_key

    if args.openrouter:
        base_url = "https://openrouter.ai/api/v1"
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")

    if not api_key:
        api_key = (
            os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
            or "no-key"
        )

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = openai.OpenAI(**client_kwargs)

    # Load injection files
    hermes_context = _load_injection_file(TOOL_EVAL_DIR / "hermes_context.md") if args.hermes_context else ""
    tool_primer = _load_injection_file(TOOL_EVAL_DIR / "tool_primer.md") if args.tool_primer else ""

    # Run
    print(f"\nRunning {len(test_cases)} tests against {args.model}\n")
    results: list[TestResult] = []

    for tc in test_cases:
        result = _run_single_test(
            client=client,
            test_case=tc,
            model=args.model,
            hermes_context=hermes_context,
            tool_primer=tool_primer,
            rate_limit=args.rate_limit,
        )
        results.append(result)
        _print_result(result, verbose=args.verbose)
        time.sleep(args.rate_limit)

    _print_summary(results, args.model)

    if args.json_output:
        out_path = Path(args.json_output)
        out_path.write_text(json.dumps([r.to_dict() for r in results], indent=2))
        print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify run_eval.py is importable**

```bash
cd /home/eric/Docker/Stack/hermes-agent
python3 -c "import tool_eval.run_eval; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add tool_eval/run_eval.py
git commit -m "feat(tool_eval): add run_eval.py CLI runner with all flags"
```

---

## Task 10: Verify --debug mode end-to-end

**Files:** none (read-only verification)

- [ ] **Step 1: Run --debug mode against all non-opt-in tests**

```bash
cd /home/eric/Docker/Stack/hermes-agent
python3 tool_eval/run_eval.py --debug
```

Expected: All tests print `OK`, final line: `OK: All 43 fixtures score 100.`

- [ ] **Step 2: Run --debug with opt-in flags**

```bash
python3 tool_eval/run_eval.py --debug --vision --image --tts
```

Expected: All 51 tests print `OK`, final line: `OK: All 51 fixtures score 100.`

- [ ] **Step 3: Run scorer unit tests one final time**

```bash
python -m pytest tool_eval/tests/test_scorer.py -v
```

Expected: All pass.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat(tool_eval): rebuild complete — scorer, runner, 51 test cases, debug mode verified"
```

---

## Self-Review

### Spec Coverage

| Spec Requirement | Task |
|---|---|
| scorer.py with TestResult + all helpers | Tasks 1–2 |
| 40 pts name + 60 pts args scoring | Task 1 scorer.py |
| All scoring criteria (no_tool_calls, text_contains, etc.) | Task 1 score_test |
| --debug mode feeding gold fixtures | Task 2 score_debug_fixture |
| hermes_context.md + tool_primer.md | Task 3 |
| 51 test cases across 11 categories | Tasks 4–8 |
| run_eval.py with all flags | Task 9 |
| --vision / --image / --tts opt-in | Task 9 (excluded_by_default filter) |
| --openrouter shortcut | Task 9 |
| .env auto-load from 3 locations | Task 9 (_load_dotenv) |
| --debug exits non-zero on failure | Task 9 (_run_debug_mode returns 1) |
| Unit tests for scorer | Tasks 1–2 |

All spec requirements covered. ✓
