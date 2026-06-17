from agent.memory_forgetting import (
    OVERLONG_TEXT_CHARS,
    STALE_AFTER_SECONDS,
    audit_memory_record_sets,
    audit_memory_records,
    finding_to_dict,
    report_to_dict,
)


def _record(text, **overrides):
    record = {
        "id": text[:8] or "record",
        "target": "memory",
        "kind": "semantic_fact",
        "text": text,
        "salience": 0.6,
        "confidence": 0.75,
        "source": "test",
        "created_at": 1_000,
        "updated_at": 1_000,
    }
    record.update(overrides)
    return record


def _finding_for(report, signal):
    return next(f for f in report.findings if signal in f.signals)


def test_procedural_candidate_suggests_skill_promotion():
    report = audit_memory_records(
        [
            _record(
                "When deploying, first run pytest, then check logs.",
                kind="procedural_candidate",
                consolidation_action="procedural_skill_candidate",
                salience=0.7,
            )
        ],
        target="memory",
        now=2_000,
    )

    finding = report.findings[0]
    assert finding.suggested_action == "promote_to_skill"
    assert finding.severity == "medium"
    assert finding.signals == ("procedural_candidate",)
    assert report.summary["action:promote_to_skill"] == 1


def test_episodic_low_salience_candidate_is_reported_not_deleted():
    record = _record(
        "Fixed PR #123 today.",
        kind="episodic_note",
        consolidation_action="episodic_only",
        salience=0.2,
        confidence=0.8,
    )
    report = audit_memory_records([record], target="memory", now=2_000)

    finding = report.findings[0]
    assert finding.suggested_action == "remove"
    assert finding.severity == "medium"
    assert finding.text == "Fixed PR #123 today."
    assert record["text"] == "Fixed PR #123 today."


def test_low_confidence_record_suggests_review():
    report = audit_memory_records(
        [_record("Possibly uses an old deployment target.", confidence=0.4)],
        target="memory",
        now=2_000,
    )

    finding = _finding_for(report, "low_confidence")
    assert finding.suggested_action == "review"
    assert finding.severity == "low"


def test_overlong_semantic_record_suggests_compression():
    long_text = "Stable project detail. " + "x" * OVERLONG_TEXT_CHARS
    report = audit_memory_records([_record(long_text)], target="memory", now=2_000)

    finding = _finding_for(report, "overlong")
    assert finding.suggested_action == "compress"
    assert finding.severity == "low"


def test_old_low_salience_record_suggests_review_without_age_only_removal():
    now = 10_000 + STALE_AFTER_SECONDS + 1
    report = audit_memory_records(
        [_record("Minor project convention.", salience=0.4, updated_at=10_000)],
        target="memory",
        now=now,
    )

    finding = _finding_for(report, "old_low_salience")
    assert finding.suggested_action == "review"
    assert finding.severity == "low"


def test_exact_duplicate_normalized_text_is_flagged_for_later_entries_only():
    report = audit_memory_records(
        [
            _record("Project uses pytest.", id="one"),
            _record("  project   uses PYTEST. ", id="two"),
        ],
        target="memory",
        now=2_000,
    )

    duplicate_findings = [f for f in report.findings if "exact_duplicate" in f.signals]
    assert len(duplicate_findings) == 1
    assert duplicate_findings[0].record_id == "two"
    assert duplicate_findings[0].suggested_action == "replace"


def test_high_salience_user_preference_gets_no_cleanup_finding():
    report = audit_memory_records(
        [
            _record(
                "User prefers concise direct answers.",
                target="user",
                kind="user_profile_fact",
                salience=0.9,
                confidence=0.9,
            )
        ],
        target="user",
        now=2_000,
    )

    assert report.findings == ()
    assert report.summary == {"total_records": 1, "total_findings": 0}


def test_stale_marker_language_is_review_only():
    report = audit_memory_records(
        [_record("Project previously used Docker Compose.")],
        target="memory",
        now=2_000,
    )

    finding = _finding_for(report, "stale_marker")
    assert finding.suggested_action == "review"
    assert finding.severity == "low"


def test_audit_record_sets_combines_memory_and_user_reports():
    report = audit_memory_record_sets(
        {
            "memory": [_record("Fixed issue #22 today.", kind="episodic_note", salience=0.2)],
            "user": [_record("User prefers concise replies.", target="user", kind="user_profile_fact", salience=0.9)],
        },
        now=2_000,
    )

    assert report.target == "all"
    assert report.summary["total_records"] == 2
    assert report.summary["total_findings"] == 1
    assert report.findings[0].target == "memory"


def test_report_and_finding_dicts_are_json_ready():
    report = audit_memory_records(
        [_record("When debugging, first reproduce the error.", kind="procedural_candidate")],
        target="memory",
        now=2_000,
    )

    finding_dict = finding_to_dict(report.findings[0])
    report_dict = report_to_dict(report)

    assert isinstance(finding_dict["signals"], list)
    assert isinstance(report_dict["findings"], list)
    assert report_dict["target"] == "memory"
