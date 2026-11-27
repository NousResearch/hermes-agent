"""Contract tests for the optional SimplePost skill's safety workflow."""

from pathlib import Path


SKILL = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "social-media"
    / "simplepost"
    / "SKILL.md"
)
REFERENCE = SKILL.parent / "references" / "mcp.md"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_skill_keeps_account_resolution_and_write_safety_explicit():
    text = _text(SKILL)

    assert "Call `list_accounts` before a write" in text
    assert "never invent account IDs" in text
    assert "one unique `idempotencyKey`" in text
    assert "Never use a new key for an uncertain write" in text


def test_skill_preserves_exact_content_and_reports_partial_failures():
    text = _text(SKILL)

    assert "Preserve user-supplied text exactly" in text
    assert "per-account or per-thread failures" in text
    assert "Clearly distinguish “draft saved”" in text


def test_reference_documents_endpoint_and_oauth_recovery():
    text = _text(REFERENCE)

    assert "https://app.simplepost.social/mcp" in text
    assert "dynamic client registration" in text
    assert "hermes mcp login simplepost" in text
    assert "retry only with the same idempotency key" in text


def test_preflight_instructions_disclose_media_storage_side_effects():
    text = _text(SKILL)
    for tool in ("validate_post", "preview_post"):
        row = next(line for line in text.splitlines() if f"| `{tool}` |" in line)
        assert "May import media" in row
    for document in (text, _text(REFERENCE)):
        assert "without saving a post" in document
        assert "managed storage" in document
    assert "text-only test content without media" in text


def test_oauth_expiry_requires_login_instead_of_promising_refresh():
    text = _text(REFERENCE)
    assert "does not issue refresh tokens" in text
    assert "hermes mcp login simplepost" in text
    assert "token exchange, and refresh" not in text
