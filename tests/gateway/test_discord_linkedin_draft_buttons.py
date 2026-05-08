"""Discord LinkedIn draft button workflow tests.

These tests pin the user-facing contract that workflow approval copies the
exact approved draft forward without regenerating or summarizing it.
"""

from gateway.platforms.discord import (
    LINKEDIN_DRAFT_CUSTOM_ID_PREFIX,
    _build_linkedin_approved_stage_button_components,
    _build_linkedin_draft_button_components,
    _extract_linkedin_api_post_text,
    _format_approved_linkedin_draft,
    _format_linkedin_publish_confirmation_message,
    _mark_linkedin_draft_archived,
    _mark_linkedin_draft_published,
    _mark_linkedin_publish_requested,
    _parse_linkedin_draft_custom_id,
)


def test_linkedin_approved_message_preserves_exact_draft_body():
    draft_body = """Topic summary
Confirmed fact: Buttons are being trialed.

Draft post
Testing a small but important workflow today...

Three alternative hooks
1. Hook A
2. Hook B
3. Hook C

Suggested hashtags
#AI #Workflow"""
    record = {
        "draft_id": "draft_123",
        "source_channel_id": "1500764401446289549",
        "source_message_id": "1501939717489557737",
        "draft_body": draft_body,
    }

    approved = _format_approved_linkedin_draft(record, approver_name="Roger")

    assert "Approved for workflow stage only — not published to LinkedIn" in approved
    assert "Exact approved draft follows:" in approved
    assert approved.endswith(draft_body)
    assert draft_body in approved
    before_exact_draft = approved.split("Exact approved draft follows:", 1)[0]
    assert "summary" not in before_exact_draft.lower()


def test_linkedin_button_components_have_expected_custom_ids():
    components = _build_linkedin_draft_button_components("draft_123")

    custom_ids = [button["custom_id"] for row in components for button in row["components"]]

    assert f"{LINKEDIN_DRAFT_CUSTOM_ID_PREFIX}:approve:draft_123" in custom_ids
    assert f"{LINKEDIN_DRAFT_CUSTOM_ID_PREFIX}:changes:draft_123" in custom_ids
    assert f"{LINKEDIN_DRAFT_CUSTOM_ID_PREFIX}:preview:draft_123" in custom_ids
    assert f"{LINKEDIN_DRAFT_CUSTOM_ID_PREFIX}:publish:draft_123" not in custom_ids


def test_linkedin_publish_button_is_opt_in():
    components = _build_linkedin_draft_button_components("draft_123", include_publish=True)

    custom_ids = [button["custom_id"] for row in components for button in row["components"]]

    assert f"{LINKEDIN_DRAFT_CUSTOM_ID_PREFIX}:publish:draft_123" in custom_ids


def test_approved_stage_components_expose_publish_but_not_direct_confirm():
    components = _build_linkedin_approved_stage_button_components("draft_123")

    buttons = [button for row in components for button in row["components"]]
    labels = [button["label"] for button in buttons]
    custom_ids = [button["custom_id"] for button in buttons]

    assert labels == [
        "Publish to LinkedIn",
        "Request final changes",
        "Preview API-ready post",
        "Archive / cancel",
    ]
    assert f"{LINKEDIN_DRAFT_CUSTOM_ID_PREFIX}:publish:draft_123" in custom_ids
    assert f"{LINKEDIN_DRAFT_CUSTOM_ID_PREFIX}:confirm_publish:draft_123" not in custom_ids


def test_publish_confirmation_message_requires_guarded_phrase():
    record = {
        "draft_id": "draft_123",
        "approved_message_id": "1502058378715861174",
        "draft_body": "Exact post body",
    }

    confirmation = _format_linkedin_publish_confirmation_message(record, requester_name="Roger")

    assert "APPROVED FOR LINKEDIN PERSONAL POST" in confirmation
    assert "Exact post body" in confirmation
    assert "not been published" in confirmation
    assert "Roger" in confirmation


def test_publish_and_archive_helpers_transition_stage_without_dropping_body():
    record = {
        "draft_id": "draft_123",
        "status": "approved_for_workflow_stage",
        "draft_body": "Exact",
    }

    publish_requested = _mark_linkedin_publish_requested(
        dict(record),
        requester_name="Roger",
        requested_at=1778175000,
    )
    assert publish_requested["status"] == "publish_confirmation_requested"
    assert publish_requested["publish_requested_by"] == "Roger"
    assert publish_requested["publish_requested_at"] == 1778175000
    assert publish_requested["draft_body"] == "Exact"

    archived = _mark_linkedin_draft_archived(
        dict(publish_requested),
        actor_name="Roger",
        archived_at=1778175100,
    )
    assert archived["status"] == "cancelled"
    assert archived["archived_by"] == "Roger"
    assert archived["archived_at"] == 1778175100
    assert archived["previous_status"] == "publish_confirmation_requested"
    assert archived["draft_body"] == "Exact"


def test_publish_extracts_only_api_ready_draft_post_section():
    draft_body = """## LinkedIn workflow live publish test

**Topic summary**
Internal workflow validation.

**Draft post**
This is the text that should reach LinkedIn.

It has multiple paragraphs.

**Three alternative hooks**
1. Not this
2. Not this either

**Suggested hashtags**
#Nope"""

    assert _extract_linkedin_api_post_text(draft_body) == (
        "This is the text that should reach LinkedIn.\n\n"
        "It has multiple paragraphs."
    )


def test_plain_publish_text_is_left_unchanged():
    assert _extract_linkedin_api_post_text("Just the final post") == "Just the final post"


def test_published_helper_records_safe_linkedin_result_without_dropping_body():
    record = {
        "draft_id": "draft_123",
        "status": "publish_confirmation_requested",
        "draft_body": "Exact",
    }

    published = _mark_linkedin_draft_published(
        dict(record),
        publisher_name="Roger",
        publish_result={
            "status_code": 201,
            "linkedin_id": "urn:li:share:123",
            "response_text": "{}",
        },
        published_at=1778175200,
    )

    assert published["status"] == "published"
    assert published["previous_status"] == "publish_confirmation_requested"
    assert published["published_by"] == "Roger"
    assert published["published_at"] == 1778175200
    assert published["draft_body"] == "Exact"
    assert published["linkedin_publish_result"] == {
        "status_code": 201,
        "linkedin_id": "urn:li:share:123",
        "response_text": "{}",
    }


def test_parse_linkedin_draft_custom_id():
    parsed = _parse_linkedin_draft_custom_id(
        f"{LINKEDIN_DRAFT_CUSTOM_ID_PREFIX}:approve:draft_123"
    )

    assert parsed == ("approve", "draft_123")
    assert _parse_linkedin_draft_custom_id(
        f"{LINKEDIN_DRAFT_CUSTOM_ID_PREFIX}:confirm_publish:draft_123"
    ) == ("confirm_publish", "draft_123")
    assert _parse_linkedin_draft_custom_id("not-ours:approve:draft_123") is None
    assert _parse_linkedin_draft_custom_id(
        f"{LINKEDIN_DRAFT_CUSTOM_ID_PREFIX}:unknown:draft_123"
    ) is None
