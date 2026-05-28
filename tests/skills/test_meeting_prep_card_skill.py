import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "optional-skills" / "productivity" / "meeting-prep-card" / "scripts" / "meeting_prep_card.py"
SKILL = ROOT / "optional-skills" / "productivity" / "meeting-prep-card" / "SKILL.md"
SAMPLE = ROOT / "optional-skills" / "productivity" / "meeting-prep-card" / "templates" / "sample_fixture.json"


def load_module():
    spec = importlib.util.spec_from_file_location("meeting_prep_card", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["meeting_prep_card"] = module
    spec.loader.exec_module(module)
    return module


def test_skill_frontmatter_is_discreet_and_valid():
    text = SKILL.read_text()
    assert text.startswith("---\n")
    assert "name: meeting-prep-card" in text
    assert "description: Use when preparing privacy-safe meeting context cards." in text
    for forbidden in ["private.example", "personal name", "internal company"]:
        assert forbidden not in text


def test_sanitizer_redacts_public_output_risks():
    module = load_module()
    raw = "Email me at person@example.com, call +971 50 123 4567, visit https://secret.example/path?token=abc, jid 971501234567@s.whatsapp.net, Bearer abcdefghijklmnop"
    clean = module.sanitize_text(raw, limit=500)
    assert "person@example.com" not in clean
    assert "+971" not in clean
    assert "https://" not in clean
    assert "s.whatsapp.net" not in clean
    assert "abcdefghijklmnop" not in clean
    assert "[redacted-email]" in clean
    assert "[redacted-number]" in clean
    assert "[redacted-link]" in clean
    assert "[redacted-id]" in clean
    assert "[redacted-secret]" in clean


def test_markdown_and_json_are_public_safe(tmp_path):
    md = subprocess.run(
        [sys.executable, str(SCRIPT), "--fixture", str(SAMPLE), "--event-id", "evt_acme", "--format", "markdown", "--strict"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    assert "[meeting link available — hidden]" in md
    assert "https://" not in md
    assert "@" not in md
    assert "+971" not in md
    assert "raw_" not in md
    assert "No messages sent" in md
    assert len(md) <= 1500

    js = subprocess.run(
        [sys.executable, str(SCRIPT), "--fixture", str(SAMPLE), "--event-id", "evt_acme", "--format", "json", "--strict"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    data = json.loads(js)
    dumped = json.dumps(data)
    assert "source_id" not in dumped
    assert "safe_ref" in dumped
    assert "evt_acme" not in dumped
    assert data[0]["event"]["attendees"] == ["Sara Founder"]
    assert "https://" not in dumped
    assert "@" not in dumped
    assert "+971" not in dumped


def test_metadata_fields_are_sanitized_before_output(tmp_path):
    fixture = json.loads(SAMPLE.read_text())
    fixture["events"][0]["id"] = "evt_PRIVATE_HANDLE_123"
    fixture["evidence"][0]["source"] = "<!channel>"
    fixture["evidence"][0]["kind"] = "risk<script>"
    fixture["evidence"][0]["confidence"] = "Bearer abcdefghijklmnop"
    fixture["evidence"][0]["source_id"] = "RAW_EVENT_OR_SOURCE_HANDLE"
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps(fixture))

    for fmt in ["markdown", "json"]:
        output = subprocess.run(
            [sys.executable, str(SCRIPT), "--fixture", str(fixture_path), "--event-id", "evt_PRIVATE_HANDLE_123", "--format", fmt, "--strict"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout
        assert "<!channel>" not in output
        assert "Bearer" not in output
        assert "RAW_EVENT_OR_SOURCE_HANDLE" not in output
        assert "evt_PRIVATE_HANDLE_123" not in output
        assert "safe_ref" in output if fmt == "json" else True


def test_internal_meetings_skip_by_default():
    module = load_module()
    fixture = json.loads(SAMPLE.read_text())
    cards = module.build_cards_from_fixture(fixture)
    assert [card.event.id for card in cards] == ["evt_acme"]


def test_custom_internal_domains_do_not_render_internal_attendees():
    module = load_module()
    fixture = json.loads(SAMPLE.read_text())
    fixture["events"][0]["attendees"] = [
        {"name": "Internal Host", "email": "host@myco.example"},
        {"name": "External Guest", "email": "guest@client.example"},
    ]
    cards = module.build_cards_from_fixture(fixture, event_id="evt_acme", internal_domains={"myco.example"})
    assert len(cards) == 1

    markdown = module.render_markdown(cards[0])
    public_json = json.dumps(module.card_to_public_json(cards[0]))

    assert "Internal Host" not in markdown
    assert "Internal Host" not in public_json
    assert "External Guest" in markdown
    assert module.card_to_public_json(cards[0])["event"]["attendees"] == ["External Guest"]
