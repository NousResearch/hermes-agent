"""Tests for Teams card builders (FileConsent / FileInfo / FileDownload)."""
from __future__ import annotations

from plugins.platforms.teams.cards import (
    FILE_CONSENT_CONTENT_TYPE,
    FILE_DOWNLOAD_INFO_CONTENT_TYPE,
    FILE_INFO_CONTENT_TYPE,
    build_file_consent_card,
    build_file_download_card,
    build_file_info_card,
)


# ---------------------------------------------------------------------------
# FileConsentCard
# ---------------------------------------------------------------------------

def test_file_consent_card_has_correct_content_type():
    card = build_file_consent_card(
        "foo.pdf",
        size_bytes=1234,
        accept_context={"upload_id": "u1"},
        description="hello",
    )
    assert card["contentType"] == FILE_CONSENT_CONTENT_TYPE
    assert card["contentType"] == "application/vnd.microsoft.teams.card.file.consent"
    assert card["name"] == "foo.pdf"
    assert card["content"]["sizeInBytes"] == 1234
    assert card["content"]["description"] == "hello"
    # acceptContext is echoed back so the bot can correlate the invoke
    assert card["content"]["acceptContext"] == {"upload_id": "u1"}


def test_file_consent_card_seeds_upload_id_when_missing():
    card = build_file_consent_card("foo.pdf", size_bytes=1, accept_context={})
    ctx = card["content"]["acceptContext"]
    assert "upload_id" in ctx
    assert isinstance(ctx["upload_id"], str)
    assert ctx["upload_id"]  # non-empty


def test_file_consent_card_seeds_upload_id_when_context_is_none():
    card = build_file_consent_card("foo.pdf", size_bytes=1)
    ctx = card["content"]["acceptContext"]
    assert "upload_id" in ctx
    assert ctx["upload_id"]


def test_file_consent_card_preserves_caller_upload_id():
    card = build_file_consent_card(
        "foo.pdf", size_bytes=1, accept_context={"upload_id": "fixed-id"}
    )
    assert card["content"]["acceptContext"]["upload_id"] == "fixed-id"


def test_file_consent_card_decline_context_matches_accept():
    card = build_file_consent_card(
        "foo.pdf", size_bytes=1, accept_context={"upload_id": "u1", "extra": 7}
    )
    # They should be value-equal so either user response correlates back identically...
    assert card["content"]["acceptContext"] == card["content"]["declineContext"]
    # ...but NOT the same object — mutation of one must not bleed into the other.
    assert card["content"]["acceptContext"] is not card["content"]["declineContext"]
    # Prove independence: mutating accept must leave decline untouched.
    card["content"]["acceptContext"]["status"] = "x"
    assert "status" not in card["content"]["declineContext"]


def test_file_consent_card_does_not_mutate_caller_context():
    caller = {}
    build_file_consent_card("f.pdf", 1, accept_context=caller)
    # upload_id was added to the COPY, not the caller's dict.
    assert caller == {}


def test_file_consent_card_size_is_coerced_to_int():
    card = build_file_consent_card("foo.pdf", size_bytes="2048")
    assert card["content"]["sizeInBytes"] == 2048
    assert isinstance(card["content"]["sizeInBytes"], int)


def test_file_consent_card_description_defaults_to_empty():
    card = build_file_consent_card("foo.pdf", size_bytes=1)
    assert card["content"]["description"] == ""


# ---------------------------------------------------------------------------
# FileInfoCard
# ---------------------------------------------------------------------------

def test_file_info_card_uses_correct_content_type():
    card = build_file_info_card(
        "foo.pdf",
        content_url="https://example/foo.pdf",
        unique_id="drive-item-123",
        file_type="pdf",
    )
    assert card["contentType"] == FILE_INFO_CONTENT_TYPE
    assert card["contentType"] == "application/vnd.microsoft.teams.card.file.info"
    assert card["contentUrl"] == "https://example/foo.pdf"
    assert card["name"] == "foo.pdf"
    assert card["content"]["fileType"] == "pdf"
    # When unique_id is supplied (the Graph drive-item id) it is echoed
    # through unchanged so Teams clients can resolve the attachment.
    assert card["content"]["uniqueId"] == "drive-item-123"


def test_file_info_card_uniqueId_falls_back_to_uuid_when_unknown():
    # Two cards built without unique_id should each get a fresh uuid
    # (preserves the old behaviour for callers that don't yet have the
    # Graph drive-item id).
    a = build_file_info_card("a.pdf", content_url="https://x/a")
    b = build_file_info_card("a.pdf", content_url="https://x/a")
    assert a["content"]["uniqueId"] != b["content"]["uniqueId"]


def test_file_info_card_passes_unique_id_through():
    card = build_file_info_card(
        "report.docx",
        content_url="https://x/y",
        unique_id="graph-item-xyz",
    )
    assert card["content"]["uniqueId"] == "graph-item-xyz"
    assert card["name"] == "report.docx"


def test_file_info_card_infers_file_type_from_filename():
    card = build_file_info_card("diagram.PNG", content_url="https://x/y")
    assert card["content"]["fileType"] == "png"
    # No extension → "file" fallback.
    card2 = build_file_info_card("LICENSE", content_url="https://x/y")
    assert card2["content"]["fileType"] == "file"


# ---------------------------------------------------------------------------
# FileDownloadCard (file.download.info)
# ---------------------------------------------------------------------------

def test_file_download_card_uses_download_info_content_type():
    card = build_file_download_card(
        filename="foo.pdf",
        content_url="https://contoso.sharepoint.com/sites/foo/Shared%20Documents/foo.pdf",
        unique_id="aaaa-bbbb",
    )
    assert card["contentType"] == FILE_DOWNLOAD_INFO_CONTENT_TYPE
    assert card["contentType"] == "application/vnd.microsoft.teams.file.download.info"
    assert card["contentUrl"].startswith("https://contoso.sharepoint.com/")
    assert card["content"]["uniqueId"] == "aaaa-bbbb"
    assert card["content"]["fileType"] == "pdf"
    assert card["name"] == "foo.pdf"


def test_file_download_card_passes_unique_id_through():
    card = build_file_download_card(
        filename="report.docx",
        content_url="https://x/y",
        unique_id="xyz-123",
    )
    assert card["content"]["uniqueId"] == "xyz-123"
    # The card's `name` is the filename (Teams uses this as the on-screen label).
    assert card["name"] == "report.docx"


def test_file_download_card_infers_file_type_from_filename():
    card = build_file_download_card(filename="diagram.PNG", content_url="https://x/y")
    assert card["content"]["fileType"] == "png"
    # No extension → "file" fallback.
    card2 = build_file_download_card(filename="LICENSE", content_url="https://x/y")
    assert card2["content"]["fileType"] == "file"
    # Empty after dot → also "file".
    card3 = build_file_download_card(filename="weird.", content_url="https://x/y")
    assert card3["content"]["fileType"] == "file"


def test_file_download_card_omits_unique_id_when_unknown():
    card = build_file_download_card(filename="x.txt", content_url="https://x/y")
    assert "uniqueId" not in card["content"]
    assert card["content"]["downloadUrl"] == "https://x/y"
