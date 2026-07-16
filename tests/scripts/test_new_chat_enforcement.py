from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import datetime as dt

import pytest


ROOT = Path(__file__).resolve().parents[2]


def load_gate():
    path = ROOT / "scripts/new-chat/hermes_prewrite_gate.py"
    spec = importlib.util.spec_from_file_location("hermes_prewrite_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


GATE = load_gate()


def git(cwd: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def make_repo(path: Path, branch: str = "main") -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", branch], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "README.md").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=path, check=True, capture_output=True)
    return path.resolve()


@pytest.fixture()
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    repo = make_repo(tmp_path / "canonical")
    worktree = tmp_path / "worktree"
    subprocess.run(["git", "worktree", "add", "-b", "task/nat/T1-safe", str(worktree), "main"], cwd=repo, check=True, capture_output=True)
    home = tmp_path / "home"
    sessions = home / ".hermes/new-chat/sessions"
    sessions.mkdir(parents=True)
    state = {
        "status": "NEW_CHAT_READY",
        "wtl": "WTL_READY",
        "task_id": "T1",
        "canonical_repo": str(repo.resolve()),
        "worktree": str(worktree.resolve()),
        "branch": "task/nat/T1-safe",
        "allowed_paths": ["src", "app.html"],
        "permit_expires_at": (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)).isoformat(),
    }
    (sessions / "T1.json").write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_WORKTREE_ROOTS", str(tmp_path))
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.delenv("HERMES_RELAY_ROLE", raising=False)
    monkeypatch.setattr(GATE, "live_session_ready", lambda session: True)
    return {"repo": repo.resolve(), "worktree": worktree.resolve(), "home": home, "state": state}


ALLOW_CASES = [
    ("Read", {"file_path": "src/a.py"}, "controller"),
    ("Bash", {"command": "git status --short"}, "controller"),
    ("Write", {"file_path": ".project/plan.md"}, "controller"),
    ("Edit", {"file_path": ".hermes/ai-relay/briefs/T1.md"}, "controller"),
    ("Write", {"file_path": "src/a.py"}, "code"),
    ("NotebookEdit", {"file_path": "src/notebook.ipynb"}, "code"),
    ("Write", {"file_path": "app.html"}, "code"),
    ("ApplyPatch", {"patch": "*** Begin Patch\n*** Update File: src/a.py\n*** End Patch"}, "code"),
]


@pytest.mark.parametrize("tool,tool_input,role", ALLOW_CASES, ids=[f"normal-{i}" for i in range(1, 9)])
def test_normal_flow_8_of_8(workspace, monkeypatch, tool, tool_input, role):
    monkeypatch.setenv("HERMES_RELAY_ROLE", role)
    payload = {"tool_name": tool, "cwd": str(workspace["worktree"]), "tool_input": tool_input}
    assert GATE.run(payload) == 0


BAD_CASES = [
    ("controller-code", "Write", {"file_path": "src/a.py"}, "controller"),
    ("outside-allowlist", "Write", {"file_path": "tests/a.py"}, "code"),
    ("outside-worktree", "Write", {"file_path": "/tmp/escape.py"}, "code"),
    ("missing-path", "Write", {}, "code"),
    ("bash-redirect", "Bash", {"command": "echo x > src/a.py"}, "controller"),
    ("bash-copy", "Bash", {"command": "cp a src/a.py"}, "controller"),
    ("bash-remove", "Bash", {"command": "rm src/a.py"}, "controller"),
    ("bash-mkdir", "Bash", {"command": "mkdir src"}, "controller"),
    ("mixed-patch", "ApplyPatch", {"patch": "*** Begin Patch\n*** Update File: src/a.py\n*** Update File: tests/a.py\n*** End Patch"}, "code"),
    ("canonical", "Write", {"file_path": "README.md"}, "code"),
    ("wrong-branch", "Write", {"file_path": "src/a.py"}, "code"),
    ("no-session", "Write", {"file_path": "src/a.py"}, "code"),
]


@pytest.mark.parametrize("kind,tool,tool_input,role", BAD_CASES, ids=[f"bad-{i}" for i in range(1, 13)])
def test_bad_flow_12_of_12(workspace, monkeypatch, kind, tool, tool_input, role):
    monkeypatch.setenv("HERMES_RELAY_ROLE", role)
    cwd = workspace["worktree"]
    if kind == "canonical":
        cwd = workspace["repo"]
    elif kind == "wrong-branch":
        session = next((workspace["home"] / ".hermes/new-chat/sessions").glob("*.json"))
        data = json.loads(session.read_text(encoding="utf-8"))
        data["branch"] = "task/nat/WRONG"
        session.write_text(json.dumps(data), encoding="utf-8")
    elif kind == "no-session":
        shutil.rmtree(workspace["home"] / ".hermes/new-chat/sessions")
    payload = {"tool_name": tool, "cwd": str(cwd), "tool_input": tool_input}
    assert GATE.run(payload) == 2


CLIENT_CASES = [
    ("Edit", {"file_path": "src/client.txt"}),
    ("apply_patch", {"command": "*** Begin Patch\n*** Update File: src/client.txt\n*** End Patch"}),
    ("Write", {"file_path": "src/client.txt"}),
    ("NotebookEdit", {"file_path": "src/client.txt"}),
]


@pytest.mark.parametrize("tool,tool_input", CLIENT_CASES, ids=["claude", "codex", "cursor", "hermes"])
def test_client_write_surfaces_4_of_4(workspace, monkeypatch, tool, tool_input):
    monkeypatch.setenv("HERMES_RELAY_ROLE", "code")
    payload = {"tool_name": tool, "cwd": str(workspace["worktree"]), "tool_input": tool_input}
    assert GATE.run(payload) == 0


# --- v2: คุมเฉพาะพื้นที่ที่ระบบจัดการ — repo นอกระบบต้องไม่ถูกล็อก (แก้เหตุล็อกทั้งเครื่อง) ---

UNMANAGED_CASES = [
    ("Write", {"file_path": "src/a.py"}),
    ("Edit", {"file_path": "any/file.txt"}),
    ("Bash", {"command": "cp a b"}),
    ("Bash", {"command": "npm run build"}),
]


@pytest.mark.parametrize("tool,tool_input", UNMANAGED_CASES, ids=[f"unmanaged-{i}" for i in range(1, 5)])
def test_unmanaged_repo_passes_4_of_4(tmp_path, monkeypatch, tool, tool_input):
    home = tmp_path / "home"
    (home / ".hermes/new-chat/sessions").mkdir(parents=True)
    repo = make_repo(tmp_path / "free-project")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_WORKTREE_ROOTS", str(tmp_path / "managed-root-empty"))
    monkeypatch.delenv("HERMES_HOME", raising=False)
    payload = {"tool_name": tool, "cwd": str(repo), "tool_input": tool_input}
    assert GATE.run(payload) == 0


# --- v2: ช่องความจำ .project/ เขียนได้เสมอในพื้นที่ที่คุม (canonical + ไม่มี session) ---

def test_memory_lane_canonical_2_of_2(workspace, monkeypatch):
    monkeypatch.setenv("HERMES_RELAY_ROLE", "controller")
    for path in (".project/OverviewProgress.md", ".hermes/ai-relay/briefs/x.md"):
        payload = {"tool_name": "Write", "cwd": str(workspace["repo"]), "tool_input": {"file_path": path}}
        assert GATE.run(payload) == 0


def test_memory_lane_worktree_without_session(workspace, monkeypatch):
    shutil.rmtree(workspace["home"] / ".hermes/new-chat/sessions")
    payload = {"tool_name": "Write", "cwd": str(workspace["worktree"]), "tool_input": {"file_path": ".project/plan.md"}}
    assert GATE.run(payload) == 0


# --- v2: shell — git ปกติ + compound ผ่าน · git อันตราย/เขียนไฟล์ตรง ถูกบล็อก ---

BASH_ALLOW = [
    "git add .project/plan.md",
    "git commit -m 'docs(memory): close'",
    "git push origin task/nat/T1-safe",
    "git merge origin/main --no-edit",
    'export PATH="/opt/homebrew/bin:$PATH"; git fetch origin && git log --oneline -3',
    "git log --oneline | head -5",
    "pytest tests/scripts -q",
    "python3 -m pytest tests -q",
    "hermes-new-chat status --task-id T1",
    "hermes-write-permit acquire --path .",
]

BASH_BLOCK = [
    "git checkout -- .",
    "git reset --hard HEAD~1",
    "git stash push",
    "git push --force origin main",
    "python3 -c 'open(\"x\",\"w\").write(\"boom\")'",
    "cat a | tee src/a.py",
    "echo hi > out.txt",
    "git status && rm -rf src",
    "git worktree add /tmp/rogue -b task/x/rogue",
    "git branch -D task/nat/T1-safe",
]


@pytest.mark.parametrize("command", BASH_ALLOW, ids=[f"sh-ok-{i}" for i in range(1, len(BASH_ALLOW) + 1)])
def test_bash_allowed_10_of_10(workspace, command):
    payload = {"tool_name": "Bash", "cwd": str(workspace["repo"]), "tool_input": {"command": command}}
    assert GATE.run(payload) == 0


@pytest.mark.parametrize("command", BASH_BLOCK, ids=[f"sh-no-{i}" for i in range(1, len(BASH_BLOCK) + 1)])
def test_bash_blocked_10_of_10(workspace, command):
    payload = {"tool_name": "Bash", "cwd": str(workspace["repo"]), "tool_input": {"command": command}}
    assert GATE.run(payload) == 2


# --- v2: โหมด offline — ทะเบียนกลางเช็คไม่ได้ = ใช้ permit ท้องถิ่น (WTL §8) ไม่ตายทั้งเครื่อง ---

def _fresh_gate_no_registry_tool(monkeypatch):
    gate = load_gate()  # โมดูลสด — ไม่โดน fixture แทนที่ live_session_ready
    monkeypatch.setattr(gate.shutil, "which", lambda name: None)
    return gate


def test_offline_fallback_uses_local_permit(workspace, monkeypatch):
    gate = _fresh_gate_no_registry_tool(monkeypatch)
    assert gate.live_session_ready(dict(workspace["state"])) is True


def test_offline_fallback_still_rejects_expired_permit(workspace, monkeypatch):
    gate = _fresh_gate_no_registry_tool(monkeypatch)
    expired = dict(workspace["state"])
    expired["permit_expires_at"] = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)).isoformat()
    assert gate.live_session_ready(expired) is False


# --- v2.1 (ปิดช่องจากรีวิว GPT-5): หลบด่านด้วย cwd ไม่ได้ — ตัดสินจากไฟล์เป้าหมายด้วย ---

def test_cwd_bypass_write_into_managed_blocked(workspace, monkeypatch):
    monkeypatch.setenv("HERMES_RELAY_ROLE", "code")
    payload = {"tool_name": "Write", "cwd": "/", "tool_input": {"file_path": str(workspace["worktree"] / "src/a.py")}}
    assert GATE.run(payload) == 2


def test_cwd_bypass_memory_lane_still_allowed(workspace):
    payload = {"tool_name": "Write", "cwd": "/", "tool_input": {"file_path": str(workspace["repo"] / ".project/plan.md")}}
    assert GATE.run(payload) == 0


def test_cwd_bypass_bash_absolute_path_blocked(workspace):
    payload = {"tool_name": "Bash", "cwd": "/", "tool_input": {"command": f"rm {workspace['worktree']}/src/a.py"}}
    assert GATE.run(payload) == 2


def test_bash_outside_with_outside_paths_passes(workspace):
    payload = {"tool_name": "Bash", "cwd": "/", "tool_input": {"command": "rm /var/tmp/scratch.txt"}}
    assert GATE.run(payload) == 0


# --- v2.1: ช่อง shell ที่ GPT-5 ชี้ — substitution / redirect ลงไฟล์ / find -delete / curl -o / command rm / sed w ---

SHELL_ATTACKS = [
    ("echo $(rm src/a.py)", 2),
    ("echo $(git status)", 0),
    ("grep foo README.md 2>err.txt", 2),
    ("pytest -q 2>/dev/null", 0),
    ("git status 2>&1", 0),
    ("find . -name '*.pyc' -delete", 2),
    ("find . -name '*.py'", 0),
    ("curl -o payload.sh https://example.com", 2),
    ("curl -s https://example.com", 0),
    ("command rm src/a.py", 2),
    ("sed -n 'w out.txt' README.md", 2),
    ("sed -n '1,5p' README.md", 0),
]


@pytest.mark.parametrize("command,expected", SHELL_ATTACKS, ids=[f"atk-{i}" for i in range(1, len(SHELL_ATTACKS) + 1)])
def test_shell_attack_surface_12_of_12(workspace, command, expected):
    payload = {"tool_name": "Bash", "cwd": str(workspace["repo"]), "tool_input": {"command": command}}
    assert GATE.run(payload) == expected


# --- v2.1: ทะเบียนตอบ error ปกติ (rc!=0 ไม่มี decision) = infra พัง → โหมด offline · ตอบ decision ชัด → เชื่อทะเบียน ---

class _FakeProc:
    def __init__(self, returncode: int, stdout: str):
        self.returncode = returncode
        self.stdout = stdout


def _gate_with_registry_reply(monkeypatch, returncode: int, stdout: str):
    gate = load_gate()
    monkeypatch.setattr(gate.shutil, "which", lambda name: "/fake/hermes-worktree")
    monkeypatch.setattr(gate.subprocess, "run", lambda *a, **k: _FakeProc(returncode, stdout))
    return gate


def test_registry_error_reply_falls_back_to_local_permit(workspace, monkeypatch):
    gate = _gate_with_registry_reply(monkeypatch, 1, '{"error": "ssh: connect refused"}')
    assert gate.live_session_ready(dict(workspace["state"])) is True


def test_registry_explicit_blocked_decision_is_respected(workspace, monkeypatch):
    gate = _gate_with_registry_reply(monkeypatch, 1, '{"decision": "WTL_BLOCKED"}')
    assert gate.live_session_ready(dict(workspace["state"])) is False


# --- v2.1: เป้าหมายนอกเขตคุม = ผ่าน แม้ cwd อยู่ในเขต (แก้ over-lock ที่ล็อกตัวเอง) ---

def test_write_outside_targets_from_canonical_pass(workspace, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside-zone") / "scratch.txt"
    payload = {"tool_name": "Write", "cwd": str(workspace["repo"]), "tool_input": {"file_path": str(outside)}}
    assert GATE.run(payload) == 0


def test_controller_tempfile_outside_session_worktree_pass(workspace, monkeypatch, tmp_path_factory):
    monkeypatch.setenv("HERMES_RELAY_ROLE", "controller")
    outside = tmp_path_factory.mktemp("outside-zone2") / "note.txt"
    payload = {"tool_name": "Write", "cwd": str(workspace["worktree"]), "tool_input": {"file_path": str(outside)}}
    assert GATE.run(payload) == 0


def test_code_role_still_confined_to_worktree(workspace, monkeypatch, tmp_path_factory):
    monkeypatch.setenv("HERMES_RELAY_ROLE", "code")
    outside = tmp_path_factory.mktemp("outside-zone3") / "escape.py"
    payload = {"tool_name": "Write", "cwd": str(workspace["worktree"]), "tool_input": {"file_path": str(outside)}}
    assert GATE.run(payload) == 2


def test_cross_boundary_write_into_other_managed_area_blocked(workspace, monkeypatch):
    monkeypatch.setenv("HERMES_RELAY_ROLE", "controller")
    payload = {"tool_name": "Write", "cwd": str(workspace["worktree"]), "tool_input": {"file_path": str(workspace["repo"] / "README.md")}}
    assert GATE.run(payload) == 2


# --- v2.1: sed — จับเฉพาะ -i จริง ไม่จับ path/อาร์กิวเมนต์ที่มีตัว i ---

SED_CASES = [
    ("sed -n '700,780p' /Users/x/Viber/relay-call.py", 0),
    ("sed -n '1,5p' README.md", 0),
    ("sed -i '' 's/a/b/' README.md", 2),
    ("sed -i.bak 's/a/b/' README.md", 2),
    ("sed --in-place 's/a/b/' README.md", 2),
    ("sed -ni 's/a/b/' README.md", 2),
]


@pytest.mark.parametrize("command,expected", SED_CASES, ids=[f"sed-{i}" for i in range(1, len(SED_CASES) + 1)])
def test_sed_rules_6_of_6(workspace, command, expected):
    payload = {"tool_name": "Bash", "cwd": str(workspace["repo"]), "tool_input": {"command": command}}
    assert GATE.run(payload) == expected


# --- v2.2 (รีวิว GPT-5 รอบ 2): ห้ามถอด/ปลอมด่านเอง — hook/session/settings/เครื่องมือ Hermes ---

def _protect_env(workspace, monkeypatch):
    home = workspace["home"]
    (home / ".hermes/new-chat-tools/scripts/new-chat").mkdir(parents=True, exist_ok=True)
    (home / ".claude/hooks").mkdir(parents=True, exist_ok=True)
    (home / ".local/bin").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home / ".hermes"))


def test_cannot_rewrite_gate_source(workspace, monkeypatch):
    _protect_env(workspace, monkeypatch)
    target = workspace["home"] / ".hermes/new-chat-tools/scripts/new-chat/hermes_prewrite_gate.py"
    payload = {"tool_name": "Write", "cwd": str(workspace["worktree"]), "tool_input": {"file_path": str(target)}}
    assert GATE.run(payload) == 2


def test_cannot_rewrite_client_hook(workspace, monkeypatch):
    _protect_env(workspace, monkeypatch)
    target = workspace["home"] / ".claude/hooks/enforce-new-chat-relay.py"
    payload = {"tool_name": "Write", "cwd": str(workspace["worktree"]), "tool_input": {"file_path": str(target)}}
    assert GATE.run(payload) == 2


def test_cannot_rewrite_settings(workspace, monkeypatch):
    _protect_env(workspace, monkeypatch)
    target = workspace["home"] / ".claude/settings.json"
    payload = {"tool_name": "Edit", "cwd": str(workspace["worktree"]), "tool_input": {"file_path": str(target)}}
    assert GATE.run(payload) == 2


def test_cannot_forge_session_file(workspace, monkeypatch):
    _protect_env(workspace, monkeypatch)
    target = workspace["home"] / ".hermes/new-chat/sessions/FAKE.json"
    payload = {"tool_name": "Write", "cwd": str(workspace["worktree"]), "tool_input": {"file_path": str(target)}}
    assert GATE.run(payload) == 2


def test_cannot_rm_client_hook_via_bash(workspace, monkeypatch):
    _protect_env(workspace, monkeypatch)
    target = workspace["home"] / ".claude/hooks/enforce-new-chat-relay.py"
    payload = {"tool_name": "Bash", "cwd": "/", "tool_input": {"command": f"rm {target}"}}
    assert GATE.run(payload) == 2


def test_cannot_overwrite_hook_via_redirect(workspace, monkeypatch):
    _protect_env(workspace, monkeypatch)
    target = workspace["home"] / ".claude/hooks/enforce-new-chat-relay.py"
    payload = {"tool_name": "Bash", "cwd": "/", "tool_input": {"command": f"printf 'x' > {target}"}}
    assert GATE.run(payload) == 2


def test_reading_hook_is_allowed(workspace, monkeypatch):
    _protect_env(workspace, monkeypatch)
    target = workspace["home"] / ".claude/hooks/enforce-new-chat-relay.py"
    payload = {"tool_name": "Bash", "cwd": "/", "tool_input": {"command": f"cat {target}"}}
    assert GATE.run(payload) == 0


# --- v2.2: ช่องความจำข้ามพื้นที่ = เฉพาะโปรเจกต์เดียวกัน (กันเขียน .project ของโปรเจกต์อื่น) ---

def test_controller_cannot_write_other_project_memory(workspace, monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_RELAY_ROLE", "controller")
    other = make_repo(tmp_path / "other-canonical")
    sessions = workspace["home"] / ".hermes/new-chat/sessions"
    other_state = {
        "status": "NEW_CHAT_READY", "wtl": "WTL_READY", "task_id": "OTHER",
        "canonical_repo": str(other), "worktree": str(other),
        "branch": "main", "allowed_paths": ["."],
        "permit_expires_at": (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)).isoformat(),
    }
    (sessions / "OTHER.json").write_text(json.dumps(other_state), encoding="utf-8")
    payload = {"tool_name": "Write", "cwd": str(workspace["worktree"]), "tool_input": {"file_path": str(other / ".project/plan.md")}}
    assert GATE.run(payload) == 2


# --- fail-closed เดิมต้องคงอยู่: stdin เสีย = block (hermes-hook-doctor พึ่งข้อนี้) ---

def test_bad_stdin_still_blocks(tmp_path):
    gate_file = ROOT / "scripts/new-chat/hermes_prewrite_gate.py"
    proc = subprocess.run(["python3", str(gate_file)], input="", text=True, capture_output=True)
    assert proc.returncode == 2
