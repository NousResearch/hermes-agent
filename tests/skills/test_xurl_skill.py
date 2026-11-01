"""Contracts for the shipped xurl instructions, not live X API availability."""

import re
import shlex
from pathlib import Path


SKILL = Path(__file__).resolve().parents[2] / "skills/social-media/xurl/SKILL.md"


def test_message_read_examples_default_to_chat_without_receipts():
    text = SKILL.read_text(encoding="utf-8")
    rows = re.findall(r"^\| ([^|]+) \| `([^`]+)` \|$", text, re.M)
    message_reads = [(label, shlex.split(command)) for label, command in rows
                     if command.startswith(("xurl dms", "xurl chat read"))]
    assert message_reads
    defaults = [command for label, command in message_reads if "default" in label.lower()]
    assert defaults, "The documented default must read encrypted messages"
    assert all(command[:3] == ["xurl", "chat", "read"] for command in defaults)
    for label, command in message_reads:
        if command[1] == "dms":
            assert "legacy" in label.lower()

    examples = re.findall(r"\bxurl chat (?:read|listen) [^`\n]+", text)
    assert examples, "Encrypted-message reads must not fall back to the legacy DM endpoint"
    for example in examples:
        args = shlex.split(example)
        assert {"--json", "--no-mark-read"} <= set(args), example


def test_key_recovery_examples_use_hidden_prompts():
    text = SKILL.read_text(encoding="utf-8")
    key_commands = re.findall(r"`(xurl chat keys (?:restore|import)[^`]*)`", text)
    assert key_commands, "Missing safe one-time recovery instructions"
    for command in key_commands:
        assert len(shlex.split(command)) == 4, "PINs and key blobs must not be arguments"
