from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REFS = ROOT / "team-shortcuts/payload/skills/prompt-shortcuts/references"


def test_new_chat_rechecks_branch_for_every_writable_task():
    text = (REFS / "use-new-chat.md").read_text(encoding="utf-8")

    assert 'version: "2.6"' in text
    assert "ด่านก่อนเขียนทุกงาน" in text
    assert "NEW_WRITABLE_TASK" in text
    assert "Write Permit" in text
    assert "คำสั่งผู้ใช้ก้อนใหม่" in text
    assert "ต้องเสนอ branch ใหม่" in text


def test_relay_and_continue_require_task_scoped_write_permit():
    relay = (REFS / "use-ai-relay.md").read_text(encoding="utf-8")
    continuation = (REFS / "use-continue.md").read_text(encoding="utf-8")

    assert 'version: "2.16"' in relay
    assert "Write Permit" in relay
    assert "สิทธิ์หนึ่งชุดใช้ได้กับงานเดียว" in relay
    assert "Write Permit จาก Use New Chat" in continuation
    assert "NEW_WRITABLE_TASK" in continuation


def test_same_reviewer_method_stops_after_two_failures():
    relay = (REFS / "use-ai-relay.md").read_text(encoding="utf-8")
    new_chat = (REFS / "use-new-chat.md").read_text(encoding="utf-8")
    continuation = (REFS / "use-continue.md").read_text(encoding="utf-8")

    assert "สูงสุด 2 รอบต่อปัญหา" in relay
    assert "ห้ามยิงรอบที่ 3" in relay
    assert "เปลี่ยนเป็น test/lint/build/gate-run" in relay
    assert "ห้ามเรียกรอบที่ 3" in new_chat
    assert 'version: "4.4"' in continuation
    assert "ผู้ตรวจคนละค่าย" in continuation


def test_conditional_details_are_not_loaded_for_every_chat():
    main = (REFS / "use-new-chat.md").read_text(encoding="utf-8")
    detail = (REFS / "use-new-chat-conditional-gates.md").read_text(encoding="utf-8")

    assert "ห้ามโหลดไฟล์นั้นเมื่อไม่เข้าเงื่อนไข" in main
    assert "อ่านไฟล์นี้เฉพาะเมื่อ" in detail
    assert "Team Claim Gate" in detail
