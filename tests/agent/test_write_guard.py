from agent.write_guard import (
    Bucket,
    Confidence,
    ScopedContent,
    Target,
    WriteRequest,
    evaluate_write_guard,
)


def decision_for(source_bucket, target_bucket, **kwargs):
    return evaluate_write_guard(
        WriteRequest(
            sources=[ScopedContent(content_ref="src", bucket=source_bucket)],
            target=Target(content_ref="dst", bucket=target_bucket),
            **kwargs,
        )
    )


def test_same_bucket_allows_without_over_confirmation():
    decision = decision_for(Bucket.TEAMSPACE, Bucket.TEAMSPACE)
    assert decision.verdict == "allow"
    assert decision.triggered_rules == []


def test_private_to_workspace_requires_confirmation():
    decision = decision_for(Bucket.PRIVATE, Bucket.WORKSPACE)
    assert decision.verdict == "confirm"
    assert "broadening:private->workspace" in decision.triggered_rules


def test_restricted_to_public_blocks():
    decision = decision_for(Bucket.RESTRICTED, Bucket.PUBLIC)
    assert decision.verdict == "block"
    assert "restricted->public" in decision.triggered_rules


def test_injection_signal_escalates_allow_to_confirm():
    decision = decision_for(
        Bucket.TEAMSPACE,
        Bucket.TEAMSPACE,
        injection_signals=["skip_confirmation_directive"],
    )
    assert decision.verdict == "confirm"
    assert "injection_signal:skip_confirmation_directive" in decision.triggered_rules


def test_tentative_memory_to_shared_destination_blocks():
    decision = decision_for(
        Bucket.PRIVATE,
        Bucket.TEAMSPACE,
        memory_confidences=[Confidence.TENTATIVE],
    )
    assert decision.verdict == "block"
    assert "tentative_memory_to_shared" in decision.triggered_rules


def test_unknown_bucket_confidence_fails_closed_as_restricted():
    decision = evaluate_write_guard(
        WriteRequest(
            sources=[ScopedContent(content_ref="src", bucket=Bucket.WORKSPACE, bucket_confidence="unknown")],
            target=Target(content_ref="dst", bucket=Bucket.WORKSPACE),
        )
    )
    assert decision.source_min_bucket == Bucket.RESTRICTED
    assert decision.verdict == "confirm"
    assert "broadening:restricted->workspace" in decision.triggered_rules


def test_multiple_sources_use_narrowest_source():
    decision = evaluate_write_guard(
        WriteRequest(
            sources=[
                ScopedContent(content_ref="a", bucket=Bucket.WORKSPACE),
                ScopedContent(content_ref="b", bucket=Bucket.PRIVATE),
            ],
            target=Target(content_ref="dst", bucket=Bucket.TEAMSPACE),
        )
    )
    assert decision.source_min_bucket == Bucket.PRIVATE
    assert decision.verdict == "confirm"
    assert "broadening:private->teamspace" in decision.triggered_rules
