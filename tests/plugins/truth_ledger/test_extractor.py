from __future__ import annotations

import asyncio
import importlib.util
import json
import time
import types
from pathlib import Path


class _FakeStructuredResult:
    def __init__(self, *, parsed, provider="openai-api", model="gpt-5"):
        self.parsed = parsed
        self.provider = provider
        self.model = model


class _FakeLLM:
    def __init__(self, behavior):
        self._behavior = behavior
        self.calls = []

    def complete_structured(self, **kwargs):
        self.calls.append(kwargs)
        return self._behavior(**kwargs)


class _FakeCtx:
    def __init__(self, llm):
        self.llm = llm


def _load_extractor_module():
    repo_root = Path(__file__).resolve().parents[3]
    plugin_dir = repo_root / "plugins" / "truth-ledger"
    module_path = plugin_dir / "extractor.py"
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.truth_ledger.extractor",
        module_path,
        submodule_search_locations=[str(plugin_dir)],
    )
    assert spec is not None
    assert spec.loader is not None

    import sys

    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        sys.modules["hermes_plugins"] = ns
    if "hermes_plugins.truth_ledger" not in sys.modules:
        pkg = types.ModuleType("hermes_plugins.truth_ledger")
        pkg.__path__ = [str(plugin_dir)]
        sys.modules["hermes_plugins.truth_ledger"] = pkg

    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "hermes_plugins.truth_ledger"
    sys.modules["hermes_plugins.truth_ledger.extractor"] = mod
    spec.loader.exec_module(mod)
    return mod


def _envelope(**overrides):
    payload = {
        "schema_name": "truth-ledger.source-envelope.v1",
        "schema_version": 1,
        "captured_at": "2026-07-19T00:00:00Z",
        "profile": "default",
        "session_id": "session-1",
        "turn_id": "turn-1",
        "origin": {
            "platform": "cli",
            "conversation_id": "conv-1",
            "thread_id": "thread-1",
            "speaker_id": "user-1",
        },
        "input": {"user_message": "Keep responses concise."},
        "output": {"assistant_response": "Understood."},
        "attempt_count": 0,
    }
    payload.update(overrides)
    return payload


def _candidate(value: object = "concise"):
    return {
        "schema_version": 1,
        "operation": "assert",
        "fact": {
            "scope": "user",
            "kind": "preference",
            "subject": "platform-user:cli:user-1",
            "key": "response.style",
            "value": value,
        },
        "evidence": {
            "type": "user_stated",
            "speaker_id": "user-1",
            "session_id": "session-1",
            "turn_id": "turn-1",
            "conversation_id": "conv-1",
            "thread_id": "thread-1",
        },
        "confidence": 0.98,
    }


def _run(coro):
    return asyncio.run(coro)


def _fact_candidates_schema_from_repo() -> dict:
    repo_root = Path(__file__).resolve().parents[3]
    schema_path = repo_root / "plugins" / "truth-ledger" / "schemas" / "fact-candidates-v1.schema.json"
    return json.loads(schema_path.read_text(encoding="utf-8"))


def test_extract_success_dedupes_and_defaults_to_no_override():
    extractor = _load_extractor_module()

    def _ok(**_kwargs):
        return _FakeStructuredResult(
            parsed={
                "schema_name": "truth-ledger.fact-candidates.v1",
                "facts": [_candidate(), _candidate()],
            }
        )

    llm = _FakeLLM(_ok)
    ctx = _FakeCtx(llm)

    out = _run(extractor.extract_candidates(ctx=ctx, envelope=_envelope()))

    assert out["status"] == "ok"
    assert len(out["facts"]) == 1
    assert out["facts"][0]["operation"] == "assert"
    assert out["extraction"]["provider"] == "openai-api"
    assert out["extraction"]["model"] == "gpt-5"
    call = llm.calls[0]
    assert call["json_schema"] == _fact_candidates_schema_from_repo()
    assert call["json_mode"] is False
    assert "provider" not in call
    assert "model" not in call


def test_extract_canonicalizes_user_subject_from_trusted_origin():
    extractor = _load_extractor_module()
    candidate = _candidate()
    candidate["fact"]["subject"] = "spoofed-user"
    llm = _FakeLLM(
        lambda **_kwargs: _FakeStructuredResult(
            parsed={"schema_name": "truth-ledger.fact-candidates.v1", "facts": [candidate]}
        )
    )

    out = _run(extractor.extract_candidates(ctx=_FakeCtx(llm), envelope=_envelope()))

    assert out["status"] == "ok"
    assert out["facts"][0]["subject"] == "platform-user:cli:user-1"


def test_extract_canonicalizes_response_style_key_alias():
    extractor = _load_extractor_module()
    candidate = _candidate()
    candidate["fact"]["key"] = "response.default_conciseness"
    llm = _FakeLLM(
        lambda **_kwargs: _FakeStructuredResult(
            parsed={"schema_name": "truth-ledger.fact-candidates.v1", "facts": [candidate]}
        )
    )

    out = _run(extractor.extract_candidates(ctx=_FakeCtx(llm), envelope=_envelope()))

    assert out["status"] == "ok"
    assert out["facts"][0]["key"] == "response.style"


def test_extract_canonicalizes_response_style_scalar_value():
    extractor = _load_extractor_module()
    candidate = _candidate(value="concise by default")
    llm = _FakeLLM(
        lambda **_kwargs: _FakeStructuredResult(
            parsed={"schema_name": "truth-ledger.fact-candidates.v1", "facts": [candidate]}
        )
    )

    out = _run(extractor.extract_candidates(ctx=_FakeCtx(llm), envelope=_envelope()))

    assert out["status"] == "ok"
    assert out["facts"][0]["value"] == "concise"


def test_extract_canonicalizes_contextual_response_style_value():
    extractor = _load_extractor_module()
    candidate = _candidate(value={"context": "engineering topics", "verbosity": "detailed"})
    candidate["operation"] = "supersede"
    llm = _FakeLLM(
        lambda **_kwargs: _FakeStructuredResult(
            parsed={"schema_name": "truth-ledger.fact-candidates.v1", "facts": [candidate]}
        )
    )

    out = _run(extractor.extract_candidates(ctx=_FakeCtx(llm), envelope=_envelope()))

    assert out["status"] == "ok"
    assert out["facts"][0]["value"] == "detailed"


def test_extract_canonicalizes_workflow_and_rollout_aliases():
    extractor = _load_extractor_module()
    raw_facts = []
    for key, value, scope, kind in (
        ("proposals.presentation_order", "options_first", "user", "workflow"),
        ("proposals.delivery_format", "google_docs_with_links", "user", "workflow"),
        (
            "rollout.review_requirement",
            "An independent exact-commit review is required before rollout.",
            "project",
            "constraint",
        ),
        (
            "merge.approval_requirement",
            "Do not merge without explicit approval.",
            "project",
            "constraint",
        ),
        (
            "default_profile_enablement.approval_requirement",
            "Do not enable the default profile without explicit approval.",
            "project",
            "constraint",
        ),
    ):
        candidate = _candidate(value=value)
        candidate["fact"].update({"key": key, "scope": scope, "kind": kind})
        raw_facts.append(candidate)

    llm = _FakeLLM(
        lambda **_kwargs: _FakeStructuredResult(
            parsed={"schema_name": "truth-ledger.fact-candidates.v1", "facts": raw_facts}
        )
    )

    out = _run(extractor.extract_candidates(ctx=_FakeCtx(llm), envelope=_envelope()))

    assert out["status"] == "ok"
    assert [(fact["key"], fact["value"]) for fact in out["facts"]] == [
        ("proposal.presentation_order", "options first"),
        ("proposal.delivery_format", "Google Docs with links"),
        ("rollout.independent_exact_commit_review_required", True),
        ("rollout.merge_requires_explicit_approval", True),
        ("rollout.default_profile_change_requires_explicit_approval", True),
    ]


def test_extract_canonicalizes_negative_requirements_and_contextual_styles():
    extractor = _load_extractor_module()
    raw_facts = []
    for key, value, scope, kind in (
        (
            "rollout.independent_exact_commit_review_required",
            "not required",
            "project",
            "constraint",
        ),
        (
            "rollout.merge_requires_explicit_approval",
            "false",
            "project",
            "constraint",
        ),
        (
            "rollout.default_profile_change_requires_explicit_approval",
            False,
            "project",
            "constraint",
        ),
        ("response.style.engineering_review", "detailed by default", "user", "preference"),
        ("response.style.slack_progress", "concise by default", "user", "preference"),
    ):
        candidate = _candidate(value=value)
        candidate["fact"].update({"key": key, "scope": scope, "kind": kind})
        raw_facts.append(candidate)

    llm = _FakeLLM(
        lambda **_kwargs: _FakeStructuredResult(
            parsed={"schema_name": "truth-ledger.fact-candidates.v1", "facts": raw_facts}
        )
    )

    out = _run(extractor.extract_candidates(ctx=_FakeCtx(llm), envelope=_envelope()))

    assert out["status"] == "ok"
    assert [(fact["key"], fact["value"]) for fact in out["facts"]] == [
        ("rollout.independent_exact_commit_review_required", False),
        ("rollout.merge_requires_explicit_approval", False),
        ("rollout.default_profile_change_requires_explicit_approval", False),
        ("response.style.engineering_review", "detailed"),
        ("response.style.slack_progress", "concise"),
    ]


def test_extract_rejects_noncanonical_requirement_and_contextual_style_values():
    extractor = _load_extractor_module()

    for key, value in (
        ("rollout.merge_requires_explicit_approval", 0),
        ("rollout.merge_requires_explicit_approval", "sometimes"),
        ("proposal.presentation_order", "start with options"),
        ("proposal.delivery_format", 1),
        ("response.style.engineering_review", "verbose"),
        ("response.style.slack_progress", {"verbosity": "concise"}),
    ):
        candidate = _candidate(value=value)
        candidate["fact"]["key"] = key
        llm = _FakeLLM(
            lambda **_kwargs: _FakeStructuredResult(
                parsed={"schema_name": "truth-ledger.fact-candidates.v1", "facts": [candidate]}
            )
        )

        out = _run(extractor.extract_candidates(ctx=_FakeCtx(llm), envelope=_envelope()))

        assert out["status"] == "dead_letter"
        assert out["reason"] == "schema_mismatch"
        assert out["dead_letter"]["reason_code"] == "schema_mismatch"


def test_extractor_prompt_declares_response_style_value_contract():
    extractor = _load_extractor_module()
    llm = _FakeLLM(
        lambda **_kwargs: _FakeStructuredResult(
            parsed={"schema_name": "truth-ledger.fact-candidates.v1", "facts": []}
        )
    )

    out = _run(extractor.extract_candidates(ctx=_FakeCtx(llm), envelope=_envelope()))

    assert out["status"] == "none"
    instructions = llm.calls[0]["instructions"]
    assert "response.style value must be exactly concise or detailed" in instructions
    assert "Use canonical key response.style only for an unqualified global response preference" in instructions
    assert "response.style.engineering_review" in instructions
    assert "response.style.slack_progress" in instructions
    assert "proposal.presentation_order" in instructions
    assert "proposal.delivery_format" in instructions
    assert "rollout.independent_exact_commit_review_required" in instructions
    assert "rollout.merge_requires_explicit_approval" in instructions
    assert "rollout.default_profile_change_requires_explicit_approval" in instructions
    assert "Requirement values must be JSON booleans true or false" in instructions
    assert "Context-specific response.style.* values must also be exactly concise or detailed" in instructions
    assert out["extraction"]["prompt_version"] == 5


def test_default_timeout_is_bounded_above_observed_structured_latency():
    extractor = _load_extractor_module()
    llm = _FakeLLM(
        lambda **_kwargs: _FakeStructuredResult(
            parsed={"schema_name": "truth-ledger.fact-candidates.v1", "facts": []}
        )
    )

    out = _run(extractor.extract_candidates(ctx=_FakeCtx(llm), envelope=_envelope()))

    assert out["status"] == "none"
    assert llm.calls[0]["timeout"] == 30.0


def test_extract_canonicalizes_timezone_kind():
    extractor = _load_extractor_module()
    candidate = _candidate(value="UTC")
    candidate["fact"]["key"] = "timezone"
    candidate["fact"]["kind"] = "environment"
    llm = _FakeLLM(
        lambda **_kwargs: _FakeStructuredResult(
            parsed={"schema_name": "truth-ledger.fact-candidates.v1", "facts": [candidate]}
        )
    )

    out = _run(extractor.extract_candidates(ctx=_FakeCtx(llm), envelope=_envelope()))

    assert out["status"] == "ok"
    assert out["facts"][0]["kind"] == "preference"


def test_extract_preserves_non_user_subject():
    extractor = _load_extractor_module()
    candidate = _candidate()
    candidate["fact"].update(
        {"scope": "project", "subject": "projects:truth-ledger", "key": "status"}
    )
    llm = _FakeLLM(
        lambda **_kwargs: _FakeStructuredResult(
            parsed={"schema_name": "truth-ledger.fact-candidates.v1", "facts": [candidate]}
        )
    )

    out = _run(extractor.extract_candidates(ctx=_FakeCtx(llm), envelope=_envelope()))

    assert out["status"] == "ok"
    assert out["facts"][0]["subject"] == "projects:truth-ledger"


def test_extract_none_when_llm_returns_empty_fact_list():
    extractor = _load_extractor_module()

    llm = _FakeLLM(
        lambda **_kwargs: _FakeStructuredResult(
            parsed={"schema_name": "truth-ledger.fact-candidates.v1", "facts": []}
        )
    )
    out = _run(extractor.extract_candidates(ctx=_FakeCtx(llm), envelope=_envelope()))

    assert out["status"] == "none"
    assert out["facts"] == []


def test_extract_schema_mismatch_dead_letters_without_secret_leakage():
    extractor = _load_extractor_module()

    llm = _FakeLLM(lambda **_kwargs: _FakeStructuredResult(parsed={"schema_name": "bad", "facts": []}))
    out = _run(extractor.extract_candidates(ctx=_FakeCtx(llm), envelope=_envelope()))

    assert out["status"] == "dead_letter"
    assert out["dead_letter"]["reason_code"] == "schema_mismatch"
    assert "validation failed" in out["dead_letter"]["last_error"]
    assert "conversation_history" not in out["dead_letter"]["last_error"]


def test_extract_malformed_parsed_payload_is_conservative_dead_letter():
    extractor = _load_extractor_module()

    llm = _FakeLLM(lambda **_kwargs: _FakeStructuredResult(parsed="not-json-object"))
    out = _run(extractor.extract_candidates(ctx=_FakeCtx(llm), envelope=_envelope()))

    assert out["status"] == "dead_letter"
    assert out["dead_letter"]["reason_code"] == "schema_mismatch"


def test_extract_timeout_is_retry_with_jitter_delay():
    extractor = _load_extractor_module()

    def _slow(**_kwargs):
        time.sleep(0.05)
        return _FakeStructuredResult(parsed={"schema_name": "truth-ledger.fact-candidates.v1", "facts": []})

    llm = _FakeLLM(_slow)
    settings = extractor.ExtractorSettings(timeout_seconds=0.01, base_delay_ms=100, max_delay_ms=1000)

    out = _run(extractor.extract_candidates(ctx=_FakeCtx(llm), envelope=_envelope(), settings=settings))

    assert out["status"] == "retry"
    assert out["reason"] == "timeout"
    assert 0 <= out["retry_delay_ms"] <= 100


def test_extract_timeout_dead_letters_when_persisted_attempt_budget_is_exhausted():
    extractor = _load_extractor_module()

    def _slow(**_kwargs):
        time.sleep(0.05)
        return _FakeStructuredResult(parsed={"schema_name": "truth-ledger.fact-candidates.v1", "facts": []})

    llm = _FakeLLM(_slow)
    settings = extractor.ExtractorSettings(timeout_seconds=0.01, max_attempts=6)
    out = _run(
        extractor.extract_candidates(
            ctx=_FakeCtx(llm),
            envelope=_envelope(attempt_count=6),
            settings=settings,
        )
    )

    assert out["status"] == "dead_letter"
    assert out["reason"] == "permanent_failure"
    assert out["dead_letter"]["reason_code"] == "extraction_failed"


def test_extract_http_5xx_is_retry_then_dead_letter_when_attempts_exhausted():
    extractor = _load_extractor_module()

    class _ServerError(RuntimeError):
        status_code = 503

    llm = _FakeLLM(lambda **_kwargs: (_ for _ in ()).throw(_ServerError("upstream 503")))

    retry = _run(
        extractor.extract_candidates(
            ctx=_FakeCtx(llm),
            envelope=_envelope(attempt_count=1),
            settings=extractor.ExtractorSettings(max_attempts=6, base_delay_ms=50),
        )
    )
    assert retry["status"] == "retry"
    assert retry["reason"] == "upstream_5xx"

    dead = _run(
        extractor.extract_candidates(
            ctx=_FakeCtx(llm),
            envelope=_envelope(attempt_count=6),
            settings=extractor.ExtractorSettings(max_attempts=6, base_delay_ms=50),
        )
    )
    assert dead["status"] == "dead_letter"
    assert dead["dead_letter"]["reason_code"] == "extraction_failed"


def test_extract_non_retryable_failure_is_permanent_dead_letter_with_redaction():
    extractor = _load_extractor_module()

    class _BadRequest(RuntimeError):
        status_code = 400

    llm = _FakeLLM(
        lambda **_kwargs: (_ for _ in ()).throw(
            _BadRequest("invalid payload token=sk-live-secret conversation_history=[...]"),
        )
    )

    out = _run(extractor.extract_candidates(ctx=_FakeCtx(llm), envelope=_envelope()))

    assert out["status"] == "dead_letter"
    assert out["dead_letter"]["reason_code"] == "extraction_failed"
    assert "sk-live-secret" not in out["dead_letter"]["last_error"]
    assert "conversation_history" not in out["dead_letter"]["last_error"]


def test_explicit_override_mode_passes_provider_and_model():
    extractor = _load_extractor_module()

    llm = _FakeLLM(
        lambda **_kwargs: _FakeStructuredResult(
            parsed={"schema_name": "truth-ledger.fact-candidates.v1", "facts": [_candidate()]},
            provider="anthropic",
            model="claude-sonnet-4",
        )
    )
    settings = extractor.ExtractorSettings(
        override_mode="explicit",
        provider_override="anthropic",
        model_override="claude-sonnet-4",
    )

    out = _run(extractor.extract_candidates(ctx=_FakeCtx(llm), envelope=_envelope(), settings=settings))

    assert out["status"] == "ok"
    call = llm.calls[0]
    assert call["provider"] == "anthropic"
    assert call["model"] == "claude-sonnet-4"
