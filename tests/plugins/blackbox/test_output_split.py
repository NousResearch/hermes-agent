from __future__ import annotations

import json
import re
from pathlib import Path

from plugins.blackbox import _comp_calls_json
from plugins.blackbox.record import _normalize_call, turn_output_split


OLD_COMPOSITION_14_KEYS = {
    "sys_tokens": 1000,
    "identity_tokens": 400,
    "skills_tokens": 600,
    "skills_count": 58,
    "tool_schema_tokens": 2000,
    "history_tokens": 300,
    "history_message_count": 12,
    "tool_result_tokens": 1400,
    "tool_arg_tokens": 200,
    "tool_result_count": 4,
    "framing_tokens": 720,
    "fixed_tokens": 3000,
    "nonfixed_tokens": 2620,
    "total_tokens": 5620,
}


def _new_call(composition: dict | None, output: int, reasoning: int = 0) -> dict:
    return {
        "composition": composition,
        "output_tokens": output,
        "reasoning_tokens": reasoning,
    }


def test_comp_calls_json_round_trips_new_shape_with_multiple_calls():
    calls = [
        _new_call({**OLD_COMPOSITION_14_KEYS, "total_tokens": 5600}, 13, 2),
        _new_call({**OLD_COMPOSITION_14_KEYS, "total_tokens": 5900}, 21, 3),
    ]

    blob = _comp_calls_json({"composition_calls": calls})
    assert blob is not None

    assert json.loads(blob) == calls
    assert turn_output_split(json.loads(blob), output_billed=34, turn_id="turn_multi") == (21, 13)


def test_old_shape_and_null_blob_have_unknown_split():
    assert turn_output_split([OLD_COMPOSITION_14_KEYS], output_billed=99, turn_id="turn_old") == (None, None)
    assert turn_output_split(None, output_billed=99, turn_id="turn_null") == (None, None)


def test_new_entry_with_zero_output_tokens_is_classified_new():
    call = _new_call(OLD_COMPOSITION_14_KEYS, output=0, reasoning=7)

    assert _normalize_call(call) == (OLD_COMPOSITION_14_KEYS, 0, 7)
    assert turn_output_split([call], output_billed=8, turn_id="turn_zero") == (0, 8)


def test_mixed_blob_with_last_new_but_earlier_unknown_is_unknown():
    last_new = _new_call(OLD_COMPOSITION_14_KEYS, output=5)

    assert turn_output_split([None, last_new], output_billed=20, turn_id="turn_mixed") == (None, None)
    assert turn_output_split([OLD_COMPOSITION_14_KEYS, last_new], output_billed=20, turn_id="turn_mixed_old") == (None, None)


def test_negative_split_clamps_and_logs_turn_context(caplog):
    call = _new_call(OLD_COMPOSITION_14_KEYS, output=50)

    with caplog.at_level("WARNING", logger="plugins.blackbox.record"):
        assert turn_output_split([call], output_billed=20, turn_id="turn_bad") == (50, 0)

    text = caplog.text
    assert "turn_bad" in text
    assert "output_billed=20" in text
    assert "finished=50" in text


def test_tool_using_turn_uses_last_completion_as_finished_output():
    # A tool-using turn has billed completions around the tool call:
    # completion asks for tool -> tool runs (not in comp_calls) -> completion emits answer.
    comp_calls = [
        _new_call({**OLD_COMPOSITION_14_KEYS, "total_tokens": 4100}, output=75),
        _new_call({**OLD_COMPOSITION_14_KEYS, "total_tokens": 4800}, output=625),
    ]

    assert turn_output_split(comp_calls, output_billed=700, turn_id="turn_tool") == (625, 75)


_RAW_BLOB_READ_RE = re.compile(
    r"\bentry\s*(?:\[\s*['\"]fixed_tokens['\"]\s*\]|\.get\(\s*['\"]fixed_tokens['\"]\s*\))"
)


def _reader_audit_violations(paths: list[Path]) -> list[str]:
    violations: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if _RAW_BLOB_READ_RE.search(text):
            violations.append(str(path))
    return violations


def test_reader_audit_gate_fails_for_synthetic_raw_blob_read_and_real_tree_is_clean(tmp_path):
    quote = chr(34)
    bad = tmp_path / "bad_reader.py"
    bad.write_text(
        "def read_blob_element(entry):\n"
        f"    return entry[{quote}fixed_tokens{quote}]\n",
        encoding="utf-8",
    )
    assert _reader_audit_violations([bad]) == [str(bad)]

    root = Path(__file__).resolve().parents[3]
    runtime_paths = [root / "agent" / "conversation_loop.py"]
    runtime_paths.extend((root / "plugins" / "blackbox").glob("*.py"))
    assert _reader_audit_violations(runtime_paths) == []
