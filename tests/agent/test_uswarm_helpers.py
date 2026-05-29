import pytest

from agent.uswarm_helpers import (
    build_child_run_sidecar,
    build_context_pack,
    evaluate_rewrite_smell,
    is_uswarm_child_sidecar_enabled,
    is_uswarm_context_pack_enabled,
    is_uswarm_rewrite_smell_enabled,
)


def test_uswarm_helper_flags_default_off(monkeypatch):
    monkeypatch.delenv("HERMES_USWARM_CONTEXT_PACK", raising=False)
    monkeypatch.delenv("HERMES_USWARM_CHILD_SIDECAR", raising=False)
    monkeypatch.delenv("HERMES_USWARM_REWRITE_SMELL", raising=False)

    assert is_uswarm_context_pack_enabled({}) is False
    assert is_uswarm_child_sidecar_enabled({}) is False
    assert is_uswarm_rewrite_smell_enabled({}) is False


def test_uswarm_helper_flags_accept_config_and_env(monkeypatch):
    cfg = {
        "uswarm_helpers": {
            "context_pack": {"enabled": True},
            "child_run_sidecar": {"enabled": "yes"},
            "rewrite_smell": {"enabled": 1},
        }
    }

    assert is_uswarm_context_pack_enabled(cfg) is True
    assert is_uswarm_child_sidecar_enabled(cfg) is True
    assert is_uswarm_rewrite_smell_enabled(cfg) is True

    monkeypatch.setenv("HERMES_USWARM_CONTEXT_PACK", "0")
    assert is_uswarm_context_pack_enabled(cfg) is False


def test_context_pack_keeps_budget_and_structured_metadata():
    pack = build_context_pack(
        [
            {"path": "a.py", "content": "alpha\n" * 100, "kind": "file"},
            {"path": "b.py", "content": "beta\n" * 100, "kind": "file"},
        ],
        token_budget=40,
    )

    assert pack["schema_version"] == "uswarm.context_pack.v1"
    assert pack["token_budget"] == 40
    assert pack["estimated_tokens"] <= 40
    assert pack["entries"]
    assert all("path" in entry and "estimated_tokens" in entry for entry in pack["entries"])
    assert pack["truncated"] is True


def test_context_pack_rejects_unsafe_paths(tmp_path):
    safe_file = tmp_path / "safe.md"
    safe_file.write_text("ok", encoding="utf-8")

    pack = build_context_pack(
        [
            {"path": "notes/safe.md", "content": "safe relative"},
            {"path": "../escape.md", "content": "escape relative"},
            {"path": "..\\escape.md", "content": "escape windows relative"},
            {"path": "C:\\Windows\\System32\\drivers\\etc\\hosts", "content": "escape windows absolute"},
            {"path": "\\Windows\\System32\\drivers\\etc\\hosts", "content": "escape windows rooted"},
            {"path": "\\\\server\\share\\secret.txt", "content": "escape unc"},
            {"path": "/etc/passwd", "content": "escape absolute"},
            {"path": str(safe_file), "content": "safe absolute"},
        ],
        token_budget=200,
        allowed_base=str(tmp_path),
    )

    paths = [entry["path"] for entry in pack["entries"]]
    assert "notes/safe.md" in paths
    assert str(safe_file.resolve()) in paths
    assert "../escape.md" not in paths
    assert "/etc/passwd" not in paths
    assert {item["reason"] for item in pack["omitted"]} == {"unsafe_path"}


def test_context_pack_rejects_absolute_and_parent_paths_without_base():
    pack = build_context_pack(
        [
            {"path": "safe.md", "content": "ok"},
            {"path": "../escape.md", "content": "bad"},
            {"path": "..\\escape.md", "content": "bad"},
            {"path": "C:\\tmp\\escape.md", "content": "bad"},
            {"path": "\\tmp\\escape.md", "content": "bad"},
            {"path": "\\\\server\\share\\escape.md", "content": "bad"},
            {"path": "/tmp/escape.md", "content": "bad"},
        ],
        token_budget=100,
    )

    assert [entry["path"] for entry in pack["entries"]] == ["safe.md"]
    assert len(pack["omitted"]) == 6


def test_child_run_sidecar_has_stable_minimal_wire_shape():
    sidecar = build_child_run_sidecar(
        task_id="parent-task",
        child_id="child-1",
        status="completed",
        goal="inspect repo",
        summary="ok",
        allowed_tools=["read_file"],
        evidence=[{"kind": "test", "ref": "pytest tests/foo.py -q"}],
    )

    assert sidecar["schema_version"] == "uswarm.child_run_sidecar.v1"
    assert sidecar["task_id"] == "parent-task"
    assert sidecar["child_id"] == "child-1"
    assert sidecar["status"] == "completed"
    assert sidecar["evidence"] == [{"kind": "test", "ref": "pytest tests/foo.py -q"}]


@pytest.mark.parametrize(
    "old,new,expected_level,expected_smell",
    [
        ("x\n" * 1500, ("x\n" * 1499) + "y\n", "warn", True),
        ("x\n" * 20, ("x\n" * 19) + "y\n", "clean", False),
        ("x\n" * 1500, "y\n" * 1500, "clean", False),
    ],
)
def test_rewrite_smell_detects_whole_file_rewrite_when_patch_would_be_safer(old, new, expected_level, expected_smell):
    decision = evaluate_rewrite_smell("big.py", old, new, patch_tool_available=True)

    assert decision["smell_id"] == "gond-smell-011-whole-file-rewrite"
    assert decision["level"] == expected_level
    assert decision["smell"] is expected_smell
