"""Real dispatcher/file backend regression checks, isolated before imports.

Each level/scenario gets a fresh interpreter because redaction policy is frozen
at import time. No registry, dispatcher, persistence, or file backend is mocked.
Run with the supported Python and pytest; child artifacts stay in the verification
scratch tree (override HERMES_REDACTION_TEST_ROOT for other development hosts).
"""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest


SCENARIOS = (
    "write", "replace", "v4a", "replay", "display",
    "read_edit_write", "redacted_copy", "stale_overwrite",
)
@pytest.mark.parametrize("level", ("basic", "standard", "strict"))
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_real_file_roundtrip(level, scenario, tmp_path):
    scratch_override = os.environ.get("HERMES_REDACTION_TEST_ROOT")
    if scratch_override:
        scratch = Path(scratch_override)
        scratch.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix=f"filetools-{level}-{scenario}-", dir=scratch))
    else:
        root = tmp_path / "roundtrip"
        root.mkdir()
    home, tmp = root / "home", root / "tmp"
    home.mkdir()
    tmp.mkdir()
    env = os.environ.copy()
    # Avoid inherited gateway/plugin/session scope; the child has only an empty home.
    for key in list(env):
        if key.startswith(("HERMES_SESSION_", "HERMES_KANBAN_")):
            env.pop(key)
    env.update(
        HERMES_HOME=str(home), TMPDIR=str(tmp),
        HERMES_DISABLE_LAZY_INSTALLS="1", HERMES_REDACT_SECRETS="true",
        HERMES_REDACT_LEVEL=level, HERMES_TEST_ISOLATION=str(home),
        TERMINAL_ENV="local", TERMINAL_CWD=str(root),
        # Scope the child-only write guard to its dedicated synthetic fixture tree.
        HERMES_WRITE_SAFE_ROOT=str(root),
        PYTHONPATH=str(Path(__file__).resolve().parents[2]),
        PYTHONDONTWRITEBYTECODE="1",
    )
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), scenario, level, str(root)],
        cwd=Path(__file__).resolve().parents[2], env=env,
        capture_output=True, text=True, timeout=120, check=False,
    )
    (root / "stdout.log").write_text(result.stdout)
    (root / "stderr.log").write_text(result.stderr)
    assert result.returncode == 0, (
        f"Child failed; artifacts: {root}\n{result.stdout}\n{result.stderr}"
    )
    assert f"PASS {scenario} {level}" in result.stdout


def _exercise(scenario, level, root):
    from model_tools import handle_function_call
    from agent.display import redact_tool_args_for_display
    from agent.redact import _REDACT_LEVEL, redact_sensitive_text
    from hermes_state import SessionDB

    assert _REDACT_LEVEL == level
    # Deliberately synthetic, but recognized by the built-in GitHub PAT pattern.
    secret = "ghp_SYNTHETICroundtripVerification1234567890"
    second = "ghp_SYNTHETICreplacementVerification0987654321"
    assert secret not in redact_sensitive_text(secret)
    assert second not in redact_sensitive_text(second)
    target = root / "sample.txt"
    original = f"credential={secret}\nstatus=before\n"
    changed = original.replace(secret, second)
    task = f"filetools-{scenario}-{level}"
    events = []

    def dispatch(name, args, *, ok=True, task_id=task):
        before = copy.deepcopy(args)
        raw = handle_function_call(name, args, task_id=task_id)
        assert args == before, (name, "mutated live arguments")
        data = json.loads(raw)
        events.append({"name": name, "args": before, "result": data})
        if ok:
            assert not data.get("error"), data
        else:
            assert data.get("error"), data
        return data

    def write(content=original):
        return dispatch("write_file", {"path": str(target), "content": content})

    def assert_bytes(expected):
        assert target.read_bytes() == expected.encode("utf-8")

    if scenario == "write":
        write()
        assert_bytes(original)
    elif scenario in ("replace", "v4a"):
        write()
        if scenario == "replace":
            args = {"mode": "replace", "path": str(target),
                    "old_string": secret, "new_string": second}
        else:
            args = {"mode": "patch", "patch": (
                f"*** Begin Patch\n*** Update File: {target}\n@@\n"
                f"-credential={secret}\n+credential={second}\n status=before\n*** End Patch\n"
            )}
        display = redact_tool_args_for_display("patch", args)
        assert secret not in json.dumps(display)
        assert second not in json.dumps(display)
        dispatch("patch", args)
        assert_bytes(changed)
    elif scenario == "replay":
        calls = [
            {"id": "write-1", "type": "function", "function": {"name": "write_file", "arguments": json.dumps({"path": str(target), "content": original})}},
            {"id": "replace-1", "type": "function", "function": {"name": "patch", "arguments": json.dumps({"mode": "replace", "path": str(target), "old_string": secret, "new_string": second})}},
            {"id": "v4a-1", "type": "function", "function": {"name": "patch", "arguments": json.dumps({"mode": "patch", "patch": f"*** Begin Patch\n*** Update File: {target}\n@@\n-credential={second}\n+credential={secret}\n status=before\n*** End Patch\n"})}},
        ]
        db_path = root / "home" / "replay.db"
        db = SessionDB(db_path)
        try:
            db.create_session("roundtrip", source="cli")
            db.append_message("roundtrip", role="assistant", tool_calls=calls)
        finally:
            db.close()
        db = SessionDB(db_path)
        try:
            messages = db.get_messages("roundtrip")
        finally:
            db.close()
        stored = messages[0]["tool_calls"]
        if isinstance(stored, str):
            stored = json.loads(stored)
        assert stored == calls
        for index, call in enumerate(stored):
            function = call["function"]
            dispatch(function["name"], json.loads(function["arguments"]))
            assert_bytes(changed if index == 1 else original)
    elif scenario == "display":
        args = {"path": str(target), "content": original,
                "nested": {"values": [secret, {"token": second}]},
                "tuple": (secret, second)}
        before = copy.deepcopy(args)
        projected = redact_tool_args_for_display("write_file", args)
        assert projected is not args
        assert secret not in json.dumps(projected)
        assert second not in json.dumps(projected)
        projected["nested"]["values"].append("display-only")
        assert args == before
        dispatch("write_file", {"path": args["path"], "content": args["content"]})
        assert_bytes(original)
    elif scenario == "read_edit_write":
        # Ordinary non-secret content may round-trip after stripping read numbering.
        plain = "status=before\nkeep=this\n"
        target.write_text(plain)
        read = dispatch("read_file", {"path": str(target)})
        content = "\n".join(line.split("|", 1)[1] for line in read["content"].splitlines()) + "\n"
        assert content == plain
        dispatch("write_file", {"path": str(target), "content": content.replace("before", "after")})
        assert_bytes(plain.replace("before", "after"))
    elif scenario == "redacted_copy":
        # A redacted read is NOT a whole-file baseline for a new editor task.
        target.write_text(original)
        read = dispatch("read_file", {"path": str(target)})
        assert secret not in read["content"]
        assert "redacted" in read["content"]
        content = "\n".join(line.split("|", 1)[1] for line in read["content"].splitlines()) + "\n"
        refused = dispatch("write_file", {"path": str(target), "content": content.replace("before", "after")}, ok=False)
        assert refused.get("stale_write_blocked") is True
        assert_bytes(original)
        # Redaction is lossy, not a reversible transport. A NEW destination has
        # no overwrite baseline to protect: copying displayed text copies masks.
        destination = root / "redacted-copy.txt"
        dispatch("write_file", {"path": str(destination), "content": content})
        assert destination.read_bytes() == content.encode("utf-8")
        assert secret.encode() not in destination.read_bytes()
        assert "redacted" in destination.read_text()
        # Targeted edit of a non-secret line remains safe and preserves unknown bytes.
        dispatch("patch", {"mode": "replace", "path": str(target), "old_string": "status=before", "new_string": "status=after"})
        assert_bytes(original.replace("before", "after"))
    elif scenario == "stale_overwrite":
        target.write_text("status=before\n")
        dispatch("read_file", {"path": str(target)})
        target.write_text(original)  # simulate another writer after the read
        refused = dispatch("write_file", {"path": str(target), "content": "status=after\n"}, ok=False)
        assert refused.get("stale_write_blocked") is True
        assert "modified since" in refused["error"]
        assert_bytes(original)
    else:
        raise AssertionError(scenario)
    (root / "events.json").write_text(json.dumps(events, indent=2))
    print(f"PASS {scenario} {level}: {len(events)} real dispatches; artifacts={root}")


if __name__ == "__main__":
    _exercise(sys.argv[1], sys.argv[2], Path(sys.argv[3]))
