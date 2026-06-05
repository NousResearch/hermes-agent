import json
from pathlib import Path

import pytest

from gateway.nutanix_answer_shadow import (
    build_evidence_fallback,
    build_revision_request_prompt,
    decide_delivery_action,
    enforcement_enabled_from_config,
    extract_rag_evidence,
    maybe_run_shadow_verification,
)


def test_extract_rag_evidence_from_mcp_tool_messages():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "hermes_master_search",
                        "arguments": '{"query":"Security Advisory 0048"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "[EXACT-ID] [OFFICIAL_PORTAL] Security Advisory 0048 covers CVE-2026-43500.",
        },
    ]

    evidence = extract_rag_evidence(messages)

    assert evidence == [
        {
            "source": "mcp_tool:hermes_master_search",
            "tool_name": "hermes_master_search",
            "query": "Security Advisory 0048",
            "text": "[EXACT-ID] [OFFICIAL_PORTAL] Security Advisory 0048 covers CVE-2026-43500.",
            "source_authority": "official_nutanix_portal",
        }
    ]


def test_extract_rag_evidence_from_gateway_registered_mcp_tool_name():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "mcp_nutanix_rag_search_hermes_master_search",
                        "arguments": '{"query":"Prism Central VM HA migrate node failure"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": (
                "[OFFICIAL_PORTAL] Prism Central supports high availability by "
                "migrating and restarting the PCVM from a failed node to another node."
            ),
        },
    ]

    evidence = extract_rag_evidence(messages)

    assert len(evidence) == 1
    assert evidence[0]["tool_name"] == "mcp_nutanix_rag_search_hermes_master_search"
    assert evidence[0]["query"] == "Prism Central VM HA migrate node failure"
    assert evidence[0]["source_authority"] == "official_nutanix_portal"


def test_extract_rag_evidence_from_gateway_tool_name_without_assistant_pairing():
    messages = [
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "tool_name": "mcp_nutanix_rag_search_hermes_master_search",
            "content": "[OFFICIAL_PORTAL] Prism Central scale-out requires a deployment container mounted on hypervisor hosts.",
        }
    ]

    evidence = extract_rag_evidence(messages)

    assert len(evidence) == 1
    assert evidence[0]["tool_name"] == "mcp_nutanix_rag_search_hermes_master_search"
    assert evidence[0]["query"] == ""
    assert evidence[0]["source_authority"] == "official_nutanix_portal"


def test_extract_rag_evidence_from_json_string_tool_calls():
    messages = [
        {
            "role": "assistant",
            "tool_calls": json.dumps(
                [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "mcp_nutanix_rag_search_hermes_master_search",
                            "arguments": '{"query":"Prism Central scale-out single cluster"}',
                        },
                    }
                ]
            ),
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "tool_name": "mcp_nutanix_rag_search_hermes_master_search",
            "content": "[OFFICIAL_PORTAL] The container used for deployment is mounted on hypervisor hosts.",
        },
    ]

    evidence = extract_rag_evidence(messages)

    assert len(evidence) == 1
    assert evidence[0]["query"] == "Prism Central scale-out single cluster"


def test_shadow_verification_writes_report_without_changing_answer(tmp_path):
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "hermes_master_search",
                        "arguments": '{"query":"Security Advisory 0048"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "Security Advisory 0048 covers CVE-2026-43500.",
        },
    ]

    def fake_verifier(query, answer, evidence, identity):
        assert query == "Security Advisory 0048"
        assert answer == "Security Advisory 0048 covers CVE-2026-43500."
        assert evidence[0]["text"] == "Security Advisory 0048 covers CVE-2026-43500."
        assert identity == "nx_shield"
        return {
            "verdict": "PASS",
            "query_class": "security_advisory",
            "claim_results": [],
            "issues": [],
            "warnings": [],
        }

    report_path = maybe_run_shadow_verification(
        enabled=True,
        query="Security Advisory 0048",
        answer="Security Advisory 0048 covers CVE-2026-43500.",
        messages=messages,
        identity="nx_shield",
        audit_dir=tmp_path,
        session_id="sid-1",
        platform="telegram",
        chat_id="chat-1",
        verifier=fake_verifier,
    )

    assert report_path is not None
    report = json.loads(Path(report_path).read_text())
    assert report["mode"] == "shadow"
    assert report["verdict"] == "PASS"
    assert report["identity"] == "nx_shield"
    assert report["delivery_action"] == "unchanged"
    assert report["evidence_count"] == 1
    assert report["evidence"][0]["source"] == "mcp_tool:hermes_master_search"


def test_shadow_verification_skips_when_no_rag_evidence(tmp_path):
    report_path = maybe_run_shadow_verification(
        enabled=True,
        query="hello",
        answer="hello back",
        messages=[{"role": "assistant", "content": "hello back"}],
        identity="hermes",
        audit_dir=tmp_path,
        session_id="sid-1",
        platform="telegram",
        chat_id="chat-1",
        verifier=lambda **_: {"verdict": "PASS"},
    )

    assert report_path is None
    assert not list(tmp_path.glob("*.json"))

def test_enforcement_gate_defaults_to_disabled(monkeypatch):
    monkeypatch.delenv("NUTANIX_ANSWER_VERIFIER_ENFORCE", raising=False)
    assert enforcement_enabled_from_config({}) is False
    assert enforcement_enabled_from_config({"rag": {"answer_verification": {"enforce_enabled": True}}}) is True


def test_enforcement_gate_env_overrides_config(monkeypatch):
    monkeypatch.setenv("NUTANIX_ANSWER_VERIFIER_ENFORCE", "false")
    assert enforcement_enabled_from_config({"rag": {"answer_verification": {"enforce_enabled": True}}}) is False
    monkeypatch.setenv("NUTANIX_ANSWER_VERIFIER_ENFORCE", "true")
    assert enforcement_enabled_from_config({}) is True


def test_decide_delivery_action_keeps_shadow_only_unchanged():
    decision = decide_delivery_action(
        enforce_enabled=False,
        verification={"verdict": "FAIL_CLOSED"},
        query="Security Advisory 0048",
        evidence=[{"text": "evidence"}],
    )
    assert decision == {"action": "unchanged", "reason": "enforcement_disabled"}


def test_decide_delivery_action_sends_original_for_pass():
    decision = decide_delivery_action(
        enforce_enabled=True,
        verification={"verdict": "PASS"},
        query="Security Advisory 0048",
        evidence=[{"text": "evidence"}],
    )
    assert decision["action"] == "send_original"


def test_decide_delivery_action_requests_revision_once_for_rewrite_required():
    decision = decide_delivery_action(
        enforce_enabled=True,
        verification={"verdict": "REWRITE_REQUIRED", "issues": ["unsupported claim"]},
        query="Security Advisory 0048",
        evidence=[{"text": "evidence"}],
        revision_attempted=False,
        draft_answer="Unsupported draft.",
    )
    assert decision["action"] == "request_revision"
    assert decision["issues"] == ["unsupported claim"]
    assert "revision_prompt" in decision
    assert "Unsupported draft." in decision["revision_prompt"]
    assert "unsupported claim" in decision["revision_prompt"]


def test_build_revision_request_prompt_limits_evidence_and_instructs_reverify_safe_revision():
    evidence = [{"source": f"source-{i}", "text": "row text " * 80} for i in range(5)]
    prompt = build_revision_request_prompt(
        query="Is 25Gb always required?",
        draft_answer="25Gb is always required and 10Gb is never acceptable.",
        evidence=evidence,
        verification={"verdict": "REWRITE_REQUIRED", "issues": ["absolute claim lacks direct support"], "warnings": ["thin evidence"]},
    )

    assert "using only the retrieved evidence" in prompt
    assert "Return only the revised user-facing answer" in prompt
    assert "25Gb is always required" in prompt
    assert "absolute claim lacks direct support" in prompt
    assert "thin evidence" in prompt
    assert "source-0" in prompt
    assert "source-2" in prompt
    assert "source-3" not in prompt
    assert len(prompt) < 3500


def test_decide_delivery_action_falls_back_after_revision_or_fail_closed():
    evidence = [{"source": "KB-021613", "text": "KB-21613 says AssignIp task fails on unmanaged subnets."}]
    decision = decide_delivery_action(
        enforce_enabled=True,
        verification={"verdict": "FAIL_CLOSED", "issues": ["restricted evidence"]},
        query="KB-21613",
        evidence=evidence,
        revision_attempted=True,
    )
    assert decision["action"] == "send_evidence_fallback"
    assert "KB-21613" in decision["fallback_answer"]
    assert "restricted evidence" in decision["fallback_answer"]


def test_build_evidence_fallback_limits_to_three_evidence_rows():
    evidence = [{"source": f"source-{i}", "text": "text"} for i in range(5)]
    fallback = build_evidence_fallback("query", evidence)
    assert "source-0" in fallback
    assert "source-2" in fallback
    assert "source-3" not in fallback



def test_gateway_shadow_gate_invokes_shadow_runner_when_enabled(monkeypatch, tmp_path):
    import gateway.run as gateway_run

    calls = []

    def fake_enabled(config):
        assert config == {"rag": {"answer_verification": {"shadow_enabled": True}}}
        return True

    def fake_enforcement(config):
        return False

    def fake_shadow_runner(**kwargs):
        calls.append(kwargs)
        return str(tmp_path / "shadow.json")

    monkeypatch.setattr(gateway_run, "_nutanix_shadow_enabled_from_config", fake_enabled)
    monkeypatch.setattr(gateway_run, "_nutanix_enforcement_enabled_from_config", fake_enforcement)
    monkeypatch.setattr(gateway_run, "_nutanix_shadow_maybe_run", fake_shadow_runner)

    source = type("Source", (), {"platform": "discord", "chat_id": "chat-1"})()
    result = gateway_run._run_nutanix_answer_verifier_gate(
        config={"rag": {"answer_verification": {"shadow_enabled": True}}},
        query="Security Advisory 0048",
        answer="Security Advisory 0048 covers CVE-2026-43500.",
        messages=[{"role": "assistant", "content": "x"}],
        source=source,
        session_id="sid-1",
    )

    assert result["delivery_response"] == "Security Advisory 0048 covers CVE-2026-43500."
    assert result["shadow_report_path"].endswith("shadow.json")
    assert calls[0]["enabled"] is True
    assert calls[0]["identity"] == "hermes"
    assert calls[0]["platform"] == "discord"
    assert calls[0]["chat_id"] == "chat-1"


def test_gateway_enforcement_gate_fallback_only_replaces_response(monkeypatch):
    import gateway.run as gateway_run

    def fake_enabled(config):
        return True

    def fake_enforcement(config):
        return True

    def fake_shadow_runner(**kwargs):
        path = "/tmp/verifier-report.json"
        return path

    def fake_decide(**kwargs):
        assert kwargs["enforce_enabled"] is True
        assert kwargs["revision_attempted"] is True
        return {
            "action": "send_evidence_fallback",
            "reason": "fail_closed_or_revision_exhausted",
            "fallback_answer": "safe fallback",
        }

    monkeypatch.setattr(gateway_run, "_nutanix_shadow_enabled_from_config", fake_enabled)
    monkeypatch.setattr(gateway_run, "_nutanix_enforcement_enabled_from_config", fake_enforcement)
    monkeypatch.setattr(gateway_run, "_nutanix_shadow_maybe_run", fake_shadow_runner)
    monkeypatch.setattr(gateway_run, "_nutanix_decide_delivery_action", fake_decide)
    monkeypatch.setattr(gateway_run, "_read_nutanix_shadow_report", lambda _path: {"verification": {"verdict": "REWRITE_REQUIRED"}, "evidence": [{"text": "evidence"}]})

    source = type("Source", (), {"platform": "discord", "chat_id": "chat-1"})()
    result = gateway_run._run_nutanix_answer_verifier_gate(
        config={"rag": {"answer_verification": {"shadow_enabled": True, "enforce_enabled": True}}},
        query="Networking query",
        answer="25Gb is always required and 10Gb is never acceptable.",
        messages=[{"role": "assistant", "content": "x"}],
        source=source,
        session_id="sid-1",
    )

    assert result["delivery_response"] == "safe fallback"
    assert result["delivery_action"] == "send_evidence_fallback"


def test_gateway_enforcement_gate_requests_revision_when_max_regeneration_enabled(monkeypatch):
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_nutanix_shadow_enabled_from_config", lambda _config: True)
    monkeypatch.setattr(gateway_run, "_nutanix_enforcement_enabled_from_config", lambda _config: True)
    monkeypatch.setattr(gateway_run, "_nutanix_shadow_maybe_run", lambda **_kwargs: "/tmp/verifier-report.json")
    monkeypatch.setattr(gateway_run, "_read_nutanix_shadow_report", lambda _path: {"verification": {"verdict": "REWRITE_REQUIRED"}, "evidence": [{"text": "evidence"}]})

    def fake_decide(**kwargs):
        assert kwargs["revision_attempted"] is False
        assert kwargs["draft_answer"] == "unsupported draft"
        return {"action": "request_revision", "reason": "rewrite_required_before_delivery", "revision_prompt": "revise now"}

    monkeypatch.setattr(gateway_run, "_nutanix_decide_delivery_action", fake_decide)

    source = type("Source", (), {"platform": "discord", "chat_id": "chat-1"})()
    result = gateway_run._run_nutanix_answer_verifier_gate(
        config={"rag": {"answer_verification": {"shadow_enabled": True, "enforce_enabled": True, "max_regenerations": 1}}},
        query="query",
        answer="unsupported draft",
        messages=[{"role": "assistant", "content": "x"}],
        source=source,
        session_id="sid-1",
    )

    assert result["delivery_action"] == "request_revision"
    assert result["revision_prompt"] == "revise now"


@pytest.mark.asyncio
async def test_gateway_revision_wrapper_sends_revised_answer_after_reverify_pass(monkeypatch):
    import gateway.run as gateway_run

    calls = {"shadow": 0, "runner": 0}

    monkeypatch.setattr(gateway_run, "_nutanix_shadow_enabled_from_config", lambda _config: True)
    monkeypatch.setattr(gateway_run, "_nutanix_enforcement_enabled_from_config", lambda _config: True)

    def fake_shadow_runner(**kwargs):
        calls["shadow"] += 1
        assert kwargs["session_id"] in {"sid-1", "sid-1_revision"}
        return f"/tmp/verifier-report-{calls['shadow']}.json"

    def fake_read(path):
        if str(path).endswith("-1.json"):
            return {"verification": {"verdict": "REWRITE_REQUIRED"}, "evidence": [{"text": "evidence"}]}
        return {"verification": {"verdict": "PASS"}, "evidence": [{"text": "evidence"}]}

    def fake_decide(**kwargs):
        verdict = kwargs["verification"]["verdict"]
        if verdict == "REWRITE_REQUIRED":
            assert kwargs["revision_attempted"] is False
            return {"action": "request_revision", "reason": "rewrite_required_before_delivery", "revision_prompt": "revise now"}
        assert kwargs["revision_attempted"] is True
        return {"action": "send_original", "reason": "verdict_pass", "verdict": "PASS"}

    async def fake_revision_runner(prompt):
        calls["runner"] += 1
        assert prompt == "revise now"
        return {"final_response": "revised safe answer"}

    monkeypatch.setattr(gateway_run, "_nutanix_shadow_maybe_run", fake_shadow_runner)
    monkeypatch.setattr(gateway_run, "_read_nutanix_shadow_report", fake_read)
    monkeypatch.setattr(gateway_run, "_nutanix_decide_delivery_action", fake_decide)

    source = type("Source", (), {"platform": "discord", "chat_id": "chat-1"})()
    result = await gateway_run._run_nutanix_answer_verifier_gate_with_revision(
        config={"rag": {"answer_verification": {"shadow_enabled": True, "enforce_enabled": True, "max_regenerations": 1}}},
        query="query",
        answer="unsupported draft",
        messages=[{"role": "assistant", "content": "x"}],
        source=source,
        session_id="sid-1",
        revision_runner=fake_revision_runner,
    )

    assert calls == {"shadow": 2, "runner": 1}
    assert result["delivery_action"] == "send_original"
    assert result["delivery_response"] == "revised safe answer"
    assert result["revision_report_path"] == "/tmp/verifier-report-2.json"


@pytest.mark.asyncio
async def test_gateway_revision_wrapper_falls_back_when_revised_answer_fails(monkeypatch):
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_nutanix_shadow_enabled_from_config", lambda _config: True)
    monkeypatch.setattr(gateway_run, "_nutanix_enforcement_enabled_from_config", lambda _config: True)
    monkeypatch.setattr(gateway_run, "_nutanix_shadow_maybe_run", lambda **_kwargs: "/tmp/verifier-report.json")
    monkeypatch.setattr(gateway_run, "_read_nutanix_shadow_report", lambda _path: {"verification": {"verdict": "REWRITE_REQUIRED"}, "evidence": [{"text": "evidence"}]})

    def fake_decide(**kwargs):
        if kwargs["revision_attempted"] is False:
            return {"action": "request_revision", "reason": "rewrite_required_before_delivery", "revision_prompt": "revise now"}
        return {"action": "send_evidence_fallback", "reason": "fail_closed_or_revision_exhausted", "fallback_answer": "safe fallback"}

    async def fake_revision_runner(_prompt):
        return "still unsupported"

    monkeypatch.setattr(gateway_run, "_nutanix_decide_delivery_action", fake_decide)

    source = type("Source", (), {"platform": "discord", "chat_id": "chat-1"})()
    result = await gateway_run._run_nutanix_answer_verifier_gate_with_revision(
        config={"rag": {"answer_verification": {"shadow_enabled": True, "enforce_enabled": True, "max_regenerations": 1}}},
        query="query",
        answer="unsupported draft",
        messages=[{"role": "assistant", "content": "x"}],
        source=source,
        session_id="sid-1",
        revision_runner=fake_revision_runner,
    )

    assert result["delivery_action"] == "send_evidence_fallback"
    assert result["delivery_response"] == "safe fallback"


def test_extract_rag_evidence_strips_query_echo_from_tool_output():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "hermes_master_search",
                        "arguments": '{"query":"Is 25Gb always required and is 10Gb never acceptable?"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "  [v4] backend=v4_hybrid_rrf identity=nx_shield\n\nQuery: Is 25Gb always required and is 10Gb never acceptable?\n\n[SOURCE: SEMANTIC CONTEXT]\n[1] Official guide says use appropriate network design.",
        },
    ]

    evidence = extract_rag_evidence(messages)

    assert "Query: Is 25Gb always required" not in evidence[0]["text"]
    assert "[1] Official guide says use appropriate network design." in evidence[0]["text"]


def test_gateway_enforcement_gate_leaves_non_rag_turns_unchanged(monkeypatch):
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_nutanix_shadow_enabled_from_config", lambda _config: True)
    monkeypatch.setattr(gateway_run, "_nutanix_enforcement_enabled_from_config", lambda _config: True)
    monkeypatch.setattr(gateway_run, "_nutanix_shadow_maybe_run", lambda **_kwargs: None)

    source = type("Source", (), {"platform": "discord", "chat_id": "chat-1"})()
    result = gateway_run._run_nutanix_answer_verifier_gate(
        config={"rag": {"answer_verification": {"shadow_enabled": True, "enforce_enabled": True}}},
        query="hello",
        answer="hello back",
        messages=[{"role": "assistant", "content": "hello back"}],
        source=source,
        session_id="sid-1",
    )

    assert result["delivery_response"] == "hello back"
    assert result["delivery_action"] == "unchanged"
    assert result["delivery_reason"] == "no_rag_evidence_report"
