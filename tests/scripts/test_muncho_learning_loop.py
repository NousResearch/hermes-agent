import json

from scripts.muncho_learning_loop import (
    SCHEMA_VERSION,
    build_digest,
    build_schema,
    packet_from_case,
    render_public_digest,
    validate_case_draft,
    validate_packet,
)


def _case(**overrides):
    data = {
        "case_id": "case:synthetic:backend-routing",
        "source_refs": [
            {
                "type": "discord_thread",
                "thread_id": "thread-synthetic",
                "message_id": "msg-start",
                "description": "synthetic case start",
            }
        ],
        "requester": "support-operator",
        "involved_people": ["support-operator", "backend-owner"],
        "business_area": "support-ops",
        "knowledge_classes": ["operational_process", "team_routing_knowledge"],
        "problem": "A support case needed a direct owner handoff.",
        "expected_action": "Prepare useful next action and route with exact evidence.",
        "actual_muncho_action": "Drafted an incomplete handoff.",
        "what_went_wrong": "The case needed follow-through, not courier-only behavior.",
        "what_worked": "The later owner response carried exact evidence.",
        "final_status": "resolved",
        "evidence_refs": [
            {
                "type": "discord_message",
                "thread_id": "thread-synthetic",
                "message_id": "msg-resolution",
                "description": "synthetic resolution evidence",
            }
        ],
        "lesson_candidate": "When a support owner answers, propose the next operator-facing action.",
        "promotion_recommendation": "pattern_candidate",
        "confidence": "high",
        "missing_evidence": [],
    }
    data.update(overrides)
    return data


def test_packet_preserves_report_only_and_hermes_self_improvement_boundaries():
    packet = packet_from_case(_case())

    validate_packet(packet)
    assert packet["schema_version"] == SCHEMA_VERSION
    assert packet["status"] == "draft_report_only"
    assert packet["safety"]["report_only"] is True
    assert packet["safety"]["runtime_behavior_change"] is False
    assert packet["safety"]["durable_promotion_performed"] is False
    assert packet["safety"]["standard_hermes_self_improvement_preserved"] is True
    assert packet["safety"]["keyword_router_authority"] is False
    assert packet["promotion"]["requires_explicit_owner_approval"] is True
    assert packet["promotion"]["performed"] is False


def test_case_draft_requires_exact_source_refs():
    bad = _case(source_refs=[{"type": "discord_thread", "description": "no ids"}])

    try:
        validate_case_draft(bad)
    except ValueError as exc:
        assert "source_refs[0] requires" in str(exc)
    else:
        raise AssertionError("invalid source refs should fail")


def test_unknown_knowledge_class_is_rejected():
    bad = _case(knowledge_classes=["keyword_router_authority"])

    try:
        validate_case_draft(bad)
    except ValueError as exc:
        assert "unknown knowledge_classes" in str(exc)
    else:
        raise AssertionError("unknown class should fail")


def test_secret_like_payload_is_rejected():
    bad = _case(problem="Bearer abc123 should not be stored")

    try:
        validate_case_draft(bad)
    except ValueError as exc:
        assert "secret-like" in str(exc)
        assert "$.problem" in str(exc)
    else:
        raise AssertionError("secret-like value should fail")


def test_public_digest_omits_private_case_text():
    packet = packet_from_case(
        _case(
            case_id="case:synthetic:PRIVATECASE123",
            problem="Sensitive customer voucher PRIVATECASE123 failed in a private flow.",
        )
    )
    digest = build_digest([packet])

    public = render_public_digest(digest)

    assert "PRIVATECASE123" not in public
    assert "Sensitive customer" not in public
    assert "No raw case transcripts included" in public
    assert "Standard Hermes self-improvement remains untouched" in public


def test_schema_encodes_safety_constants():
    schema = build_schema()

    assert schema["properties"]["schema_version"]["const"] == SCHEMA_VERSION
    safety = schema["properties"]["safety"]["properties"]
    assert safety["report_only"]["const"] is True
    assert safety["runtime_behavior_change"]["const"] is False
    assert safety["durable_promotion_performed"]["const"] is False
    assert safety["standard_hermes_self_improvement_preserved"]["const"] is True
    assert safety["keyword_router_authority"]["const"] is False


def test_cli_generate_creates_packets_and_digests(tmp_path):
    from scripts import muncho_learning_loop

    input_path = tmp_path / "cases.json"
    output_dir = tmp_path / "out"
    input_path.write_text(json.dumps([_case()], ensure_ascii=False), encoding="utf-8")

    rc = muncho_learning_loop.main(
        [
            "generate",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert rc == 0
    packet_files = [path for path in output_dir.glob("*.json") if not path.name.startswith("digest-")]
    assert len(packet_files) == 1
    assert (output_dir / "digest-private.json").exists()
    assert (output_dir / "digest-private.md").exists()
    assert (output_dir / "digest-public-safe.md").exists()

    packet = json.loads(packet_files[0].read_text(encoding="utf-8"))
    validate_packet(packet)
