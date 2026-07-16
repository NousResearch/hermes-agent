#!/usr/bin/env python3
"""Fail-closed write gate for New Chat + Worktree + AI Relay.

v2 (2026-07-16) — แก้ "ล็อกตัวเอง" 4 จุดตามคำสั่งเจ้าของ + ปิดช่องที่ GPT-5 ตรวจเจอ:
1. คุมเฉพาะพื้นที่ที่ระบบจัดการ (registered worktree roots + canonical repo
   ที่มี session อ้างถึง) — ไม่ล็อกทุก git repo ทั้งเครื่อง
   · ตัดสินจากทั้ง cwd และ "ไฟล์เป้าหมาย" — ย้าย cwd ออกนอกเขตแล้วหลบไม่ได้
2. ช่องความจำ `.project/` + ใบงาน relay เขียนได้เสมอในพื้นที่ที่คุม
   (กติกา Use New Chat: ตัวควบคุมเขียนตรงได้เฉพาะสองที่นี้)
3. shell อนุญาต git ปกติ (add/commit/push/merge) + คำสั่งอ่าน + คำสั่ง
   lifecycle — ห้าม git อันตราย, การเขียนไฟล์ตรง, redirect ลงไฟล์,
   command substitution ที่ซ่อนคำสั่งเขียน, find -delete, curl -o
4. ทะเบียนกลาง (VPS) ติดต่อไม่ได้/ตอบผิดรูป = ใช้ permit ท้องถิ่นที่ยังไม่
   หมดอายุแทน (WTL Contract §8 โหมด offline) — ไม่ตายทั้งเครื่องตอน VPS ล่ม
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


WRITE_TOOLS = {"edit", "write", "multiedit", "notebookedit", "applypatch", "apply_patch"}
PATCH_PATH = re.compile(r"^\*\*\* (?:Add|Update|Delete) File: (.+)$", re.MULTILINE)
CONTROL_PATHS = (".project/", ".hermes/ai-relay/briefs/")
READ_ONLY_BINS = {
    "rg", "grep", "find", "ls", "pwd", "test", "head", "tail", "cat", "sed", "awk",
    "wc", "stat", "du", "df", "ps", "which", "diff", "sort", "uniq", "date",
    "basename", "dirname", "realpath", "file", "sleep", "gh", "jq", "curl",
    "pytest", "ruff", "mypy", "eslint", "tsc",
}
NEUTRAL_BINS = {"export", "cd", "echo", "printf", "true", ":", "set", "unset"}
LIFECYCLE_BINS = {
    "relay-call", "gate-run", "hermes-new-chat", "hermes-worktree",
    "hermes-write-permit", "hermes-hook-doctor", "hermes-prewrite-gate",
}
GIT_BLOCKED_SUBCOMMANDS = {
    "apply", "checkout", "switch", "reset", "clean", "stash",
    "filter-branch", "update-ref", "reflog",
}
FIND_WRITE_FLAGS = {"-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprintf", "-fls"}
CURL_WRITE_FLAGS = {"-o", "-O", "--output", "--remote-name", "--output-dir", "-J", "--remote-header-name"}
SED_WRITE_CMD = re.compile(r"(?:^|;)\s*[wW]\s")
FD_DUP = re.compile(r"[0-9]*>&[0-9]+")
REDIRECT_TARGET = re.compile(r"[0-9]*>{1,2}\s*([^\s;|&]+)")
REDIRECT_SAFE_TARGETS = {"/dev/null", "/dev/stdout", "/dev/stderr"}
SUBSTITUTION = re.compile(r"\$\(([^()]*)\)|`([^`]*)`")
SEGMENT_SPLIT = re.compile(r"&&|\|\||;|\n|\|")
ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
ABS_PATH_TOKEN = re.compile(r"(?:~/|/)[^\s'\";|&<>()]+")
DEFAULT_WORKTREE_ROOTS = (
    Path.home() / "Documents" / "Worktrees",
    Path("/home/linux-nat/.worktree"),
)


def block(reason: str) -> int:
    print(f"[Hermes New Chat Gate] BLOCKED: {reason}", file=sys.stderr)
    return 2


def warn(message: str) -> None:
    print(f"[Hermes New Chat Gate] WARN: {message}", file=sys.stderr)


def git_root(cwd: Path) -> Path | None:
    if not cwd.is_dir():
        return None
    proc = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=cwd, text=True, capture_output=True)
    return Path(proc.stdout.strip()).resolve() if proc.returncode == 0 else None


def sessions_dir() -> Path:
    return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")) / "new-chat" / "sessions"


def iter_sessions():
    folder = sessions_dir()
    if not folder.is_dir():
        return
    for path in sorted(folder.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            yield data


def worktree_roots() -> list[Path]:
    env = os.environ.get("HERMES_WORKTREE_ROOTS")
    if env:
        return [Path(item).expanduser() for item in env.split(os.pathsep) if item.strip()]
    return list(DEFAULT_WORKTREE_ROOTS)


def resolve_loose(path: Path) -> Path:
    """resolve แบบไม่บังคับว่าไฟล์ต้องมีอยู่จริง (ไฟล์ใหม่ยังไม่ถูกสร้าง)"""
    try:
        return path.expanduser().resolve()
    except OSError:
        return path.expanduser().absolute()


def under_worktree_root(path: Path) -> bool:
    for base in worktree_roots():
        try:
            path.relative_to(resolve_loose(base))
            return True
        except (ValueError, OSError):
            continue
    return False


def session_for(root: Path) -> dict | None:
    for data in iter_sessions():
        if resolve_loose(Path(data.get("worktree", "/__missing__"))) == root:
            return data
    return None


def is_known_canonical(root: Path) -> bool:
    for data in iter_sessions():
        if resolve_loose(Path(data.get("canonical_repo", "/__missing__"))) == root:
            return True
    return False


def in_managed_scope(root: Path) -> bool:
    """คุมเฉพาะพื้นที่ที่ระบบจัดการ — ที่อื่นปล่อยผ่าน (ด่านอื่นดูแลของมันเอง)"""
    return under_worktree_root(root) or is_known_canonical(root) or session_for(root) is not None


def protected_paths() -> list[Path]:
    """พื้นที่โครงสร้างของด่านเอง — ห้ามเครื่องมือเขียนเด็ดขาด (กัน AI ถอดปลั๊ก/ปลอม session)"""
    home = Path.home()
    hermes = Path(os.environ.get("HERMES_HOME", home / ".hermes"))
    return [
        hermes / "new-chat",
        hermes / "new-chat-tools",
        home / ".claude" / "hooks",
        home / ".claude" / "settings.json",
        home / ".claude" / "settings.local.json",
    ]


def protected_target(path: Path) -> bool:
    resolved = resolve_loose(path)
    for base in protected_paths():
        base_resolved = resolve_loose(base)
        if resolved == base_resolved or base_resolved in resolved.parents:
            return True
    local_bin = resolve_loose(Path.home() / ".local" / "bin")
    if resolved.parent == local_bin and (
        resolved.name.startswith("hermes-") or resolved.name in {"relay-call", "gate-run", "relay-portal"}
    ):
        return True
    return False


def project_key(root: Path) -> Path | None:
    """canonical ประจำพื้นที่ — ใช้เทียบว่า 2 พื้นที่เป็นโปรเจกต์เดียวกันไหม"""
    data = session_for(root)
    if data:
        return resolve_loose(Path(data.get("canonical_repo", "/__missing__")))
    if is_known_canonical(root):
        return root
    for base in worktree_roots():
        base_resolved = resolve_loose(base)
        try:
            rel = root.relative_to(base_resolved)
        except ValueError:
            continue
        return base_resolved / rel.parts[0] if rel.parts else None
    return None


def managed_root_for_path(target: Path) -> Path | None:
    """หา root ของเขตคุมที่ครอบ path นี้ — เทียบจากตัวไฟล์เป้าหมาย ไม่ใช่ cwd (กันหลบด่านด้วย cwd)"""
    resolved = resolve_loose(target)
    best: Path | None = None
    for data in iter_sessions():
        for key in ("worktree", "canonical_repo"):
            candidate = resolve_loose(Path(data.get(key, "/__missing__")))
            if resolved == candidate or candidate in resolved.parents:
                if best is None or len(candidate.parts) > len(best.parts):
                    best = candidate
    if best is not None:
        return best
    for base in worktree_roots():
        base_resolved = resolve_loose(base)
        try:
            rel = resolved.relative_to(base_resolved)
        except ValueError:
            continue
        # ใต้ registered root: ถาม git จริงจาก ancestor ที่มีอยู่ (แม่นสุด)
        probe = resolved
        while not probe.is_dir() and probe != base_resolved and probe.parent != probe:
            probe = probe.parent
        found = git_root(probe)
        if found is not None:
            try:
                found.relative_to(base_resolved)
                return found
            except ValueError:
                pass
        # สำรอง: โครงมาตรฐาน <root>/<project>/<staff>/<task>/... — เขตคุม = ระดับ task
        depth = min(len(rel.parts) - 1, 3) if rel.parts else 0
        return base_resolved.joinpath(*rel.parts[:depth]) if depth else base_resolved
    return None


def permit_locally_valid(session: dict) -> bool:
    if session.get("status") != "NEW_CHAT_READY" or session.get("wtl") != "WTL_READY":
        return False
    try:
        expires = dt.datetime.fromisoformat(str(session["permit_expires_at"]))
        return expires > dt.datetime.now(dt.timezone.utc)
    except (KeyError, TypeError, ValueError):
        return False


def live_session_ready(session: dict) -> bool:
    if not permit_locally_valid(session):
        return False
    tool = shutil.which("hermes-worktree")
    if not tool:
        # WTL §8: ทะเบียนกลางเช็คไม่ได้ = ทำต่อได้เฉพาะ task เดิมที่ permit ท้องถิ่นยังไม่หมดอายุ
        warn("ไม่พบ hermes-worktree — ใช้ permit ท้องถิ่น (โหมด offline · ห้ามเปิด/โอน/ปิดงานจนกว่าทะเบียนกลับมา)")
        return True
    command = [tool, "status", "--task-id", str(session.get("task_id") or ""), "--json"]
    if session.get("registry"):
        command += ["--registry", str(session["registry"])]
    try:
        proc = subprocess.run(command, text=True, capture_output=True, timeout=20)
        result = json.loads(proc.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        warn("ทะเบียนกลางตอบไม่ได้ — ใช้ permit ท้องถิ่น (โหมด offline · WTL §8)")
        return True
    if isinstance(result, dict) and "decision" in result:
        # ทะเบียนตอบชัด: เชื่อคำตัดสินทะเบียน (READY เท่านั้นที่ผ่าน)
        return result.get("decision") == "WTL_READY"
    # ทะเบียนตอบผิดรูป (เช่น error ฝั่ง ssh) = infra พัง ไม่ใช่งานไม่พร้อม → โหมด offline
    warn("ทะเบียนกลางตอบผิดรูป — ใช้ permit ท้องถิ่น (โหมด offline · WTL §8)")
    return True


def is_control_path(path: Path, root: Path) -> bool:
    try:
        rel = resolve_loose(path).relative_to(root).as_posix()
    except ValueError:
        return False
    return any(rel == prefix.rstrip("/") or rel.startswith(prefix) for prefix in CONTROL_PATHS)


def path_allowed(path: Path, root: Path, session: dict, role: str) -> bool:
    try:
        rel = resolve_loose(path).relative_to(root).as_posix()
    except ValueError:
        return False
    if role == "code":
        allowed = session.get("allowed_paths") or ["."]
        return any(item == "." or rel == item.rstrip("/") or rel.startswith(item.rstrip("/") + "/") for item in allowed)
    return any(rel.startswith(prefix) for prefix in CONTROL_PATHS)


def lifecycle_segment_ok(tokens: list[str]) -> bool:
    name = Path(tokens[0]).name
    if name not in LIFECYCLE_BINS:
        return False
    if name == "relay-call":
        return "--role" in tokens and "--cwd" in tokens
    if name == "hermes-new-chat":
        return len(tokens) > 1 and tokens[1] in {"open", "status"}
    if name == "hermes-worktree":
        return len(tokens) > 1 and tokens[1] in {"status", "list", "doctor", "enter", "report", "scan"}
    if name == "hermes-write-permit":
        return len(tokens) > 1 and tokens[1] in {"status", "check", "acquire", "release"}
    return True  # gate-run / hermes-hook-doctor / hermes-prewrite-gate


def git_segment_ok(tokens: list[str]) -> bool:
    args = tokens[1:]
    sub = next((t for t in args if not t.startswith("-")), "")
    if sub in GIT_BLOCKED_SUBCOMMANDS:
        return False
    if sub == "push" and any(t in {"--force", "-f", "--force-with-lease"} for t in args):
        return False
    if sub == "worktree" and any(t in {"add", "remove", "move", "prune", "lock", "unlock"} for t in args):
        return False  # สร้าง/ลบ worktree ต้องผ่าน Worktree Manager เท่านั้น (WTL §12)
    if sub == "branch" and any(t in {"-D", "-d", "--delete", "-f", "--force", "-m", "--move"} for t in args):
        return False
    return bool(sub)


def redirects_safe(segment: str) -> bool:
    cleaned = FD_DUP.sub("", segment)  # 2>&1 = ปลอดภัย (ไม่แตะไฟล์)
    if ">" not in cleaned:
        return True
    targets = REDIRECT_TARGET.findall(cleaned)
    stripped = REDIRECT_TARGET.sub("", cleaned)
    if ">" in stripped:
        return False  # มี > ที่จับ target ไม่ได้ = ไม่รู้ปลายทาง → fail-closed
    return bool(targets) and all(t in REDIRECT_SAFE_TARGETS for t in targets)


def segment_ok(segment: str) -> bool:
    segment = segment.strip()
    if not segment:
        return True
    # command substitution: เนื้อในต้องผ่านด่านเองด้วย (กัน echo $(rm ...))
    def _check_inner(match: re.Match) -> str:
        inner = match.group(1) or match.group(2) or ""
        return "__SUBST_OK__" if bash_allowed(inner) else "__SUBST_BAD__"

    flattened = SUBSTITUTION.sub(_check_inner, segment)
    if "__SUBST_BAD__" in flattened or "$(" in flattened or "`" in flattened:
        return False  # ซ้อนหลายชั้น/จับไม่หมด = fail-closed
    if not redirects_safe(flattened):
        return False
    try:
        tokens = shlex.split(flattened)
    except ValueError:
        return False
    while tokens and ENV_ASSIGN.match(tokens[0]):
        tokens = tokens[1:]
    while tokens and Path(tokens[0]).name in {"command", "nice", "time", "nohup"}:
        tokens = tokens[1:]  # ตัวห่อคำสั่ง: ตรวจคำสั่งจริงข้างใน
    if not tokens:
        return True
    first = Path(tokens[0]).name
    if first in NEUTRAL_BINS:
        return True
    if first == "git":
        return git_segment_ok(tokens)
    if first.startswith("python"):
        return "-m" in tokens and ("pytest" in tokens or "unittest" in tokens) and "-c" not in tokens
    if first == "find":
        return not any(t in FIND_WRITE_FLAGS for t in tokens)
    if first == "curl":
        return not any(t in CURL_WRITE_FLAGS for t in tokens)
    if first == "sed":
        for t in tokens[1:]:
            if t.startswith("--in-place"):
                return False
            if t.startswith("-") and t != "-" and not t.startswith("--") and "i" in t[1:]:
                return False  # -i / -i.bak / -ni = แก้ไฟล์ตรง
            if SED_WRITE_CMD.search(t):
                return False
        return True
    if first in READ_ONLY_BINS:
        return True
    return lifecycle_segment_ok(tokens)


def bash_allowed(command: str) -> bool:
    return all(segment_ok(part) for part in SEGMENT_SPLIT.split(command))


def bash_touches_managed(command: str, cwd: Path) -> bool:
    """เขต bash: จาก cwd หรือ path เต็มในคำสั่งที่ชี้เข้าเขตคุม (กันตั้ง cwd นอกเขตแล้วยิงกลับเข้า)"""
    root = git_root(cwd)
    if root is not None and in_managed_scope(root):
        return True
    for token in ABS_PATH_TOKEN.findall(command):
        p = Path(token)
        if managed_root_for_path(p) is not None or protected_target(p):
            return True
    return False


def bash_hits_protected(command: str) -> bool:
    """คำสั่ง shell แตะพื้นที่โครงสร้างด่านเอง (hook/session/settings/เครื่องมือ) = บล็อกเสมอ"""
    return any(protected_target(Path(token)) for token in ABS_PATH_TOKEN.findall(command))


def run(payload: dict) -> int:
    raw_tool = str(payload.get("tool_name") or "")
    tool = re.split(r"[.:/]", raw_tool)[-1].lower()
    if tool in {"exec_command", "run_shell_command", "terminal", "shell"}:
        tool = "bash"
    if tool not in WRITE_TOOLS | {"bash"}:
        return 0
    raw = payload.get("tool_input")
    tool_input = raw if isinstance(raw, dict) else {}
    cwd = Path(str(payload.get("cwd") or Path.cwd())).expanduser().resolve()
    if tool == "bash":
        command = str(tool_input.get("command") or tool_input.get("cmd") or "")
        if bash_hits_protected(command) and not bash_allowed(command):
            return block("คำสั่ง shell แตะพื้นที่โครงสร้างของด่าน (hook/session/settings/เครื่องมือ Hermes) — ห้ามถอดหรือแก้ด่านผ่าน shell")
        if not bash_touches_managed(command, cwd):
            return 0
        if bash_allowed(command):
            return 0
        return block(
            "คำสั่ง shell เขียนไฟล์/คำสั่งเสี่ยงในพื้นที่ที่ระบบคุม — ใช้ได้เฉพาะคำสั่งอ่าน, git ปกติ, "
            "และคำสั่ง lifecycle (relay-call/gate-run/hermes-*) · งานเขียนโค้ดต้องผ่าน relay-call --role code"
        )
    values = []
    file_value = tool_input.get("file_path") or tool_input.get("path")
    if isinstance(file_value, str) and file_value.strip():
        values.append(file_value)
    if tool in {"applypatch", "apply_patch"}:
        # Claude/Cursor ส่ง patch ใน `patch` หรือ `input`; Codex Hook ส่งใน `command`.
        patch_text = str(
            tool_input.get("patch")
            or tool_input.get("input")
            or tool_input.get("command")
            or ""
        )
        values.extend(PATCH_PATH.findall(patch_text))
    root = git_root(cwd)
    cwd_managed = root is not None and in_managed_scope(root)
    if not values:
        return block("คำสั่งเขียนไม่มี path ให้ตรวจ") if cwd_managed else 0
    targets = []
    for value in values:
        target = Path(value).expanduser()
        if not target.is_absolute():
            target = cwd / target
        targets.append(target)
    # โครงสร้างด่านเอง (hook/session/settings/เครื่องมือ Hermes) = ห้ามเขียนเด็ดขาด ทุกกรณี
    for target in targets:
        if protected_target(target):
            return block("ห้ามเขียนทับพื้นที่โครงสร้างของด่าน (hook/session/settings/เครื่องมือ Hermes) — จะถอด/ปลอมด่านไม่ได้")
    session = session_for(root) if cwd_managed else None
    if cwd_managed and session:
        if resolve_loose(Path(session.get("canonical_repo", "/__missing__"))) == root:
            return block("ห้ามเขียนโค้ดใน canonical repo ที่ใช้ร่วมกัน — เปิดงานผ่าน `hermes-new-chat open`")
        role = os.environ.get("HERMES_RELAY_ROLE", "controller")
        this_project = project_key(root)
        inside, outside = [], []
        for target in targets:
            try:
                resolve_loose(target).relative_to(root)
                inside.append(target)
            except ValueError:
                outside.append(target)
        if outside:
            if role == "code":
                return block("บทบาท code ห้ามเขียนนอก worktree ของงาน — เขียนได้เฉพาะ allowed paths ใน task")
            for target in outside:
                governed = managed_root_for_path(target)
                if governed is None:
                    continue  # เป้าหมายนอกเขตคุมทั้งหมด (scratchpad/repo อิสระ) = ผ่าน
                # ช่องความจำข้ามพื้นที่ = ผ่านเฉพาะโปรเจกต์เดียวกัน (กัน controller เขียน .project ของโปรเจกต์อื่น)
                if is_control_path(target, governed) and project_key(governed) == this_project:
                    continue
                return block("เขียนไฟล์เข้าเขต Worktree/canonical อื่นที่ระบบคุม — เข้าไปทำงานในพื้นที่นั้นผ่าน `hermes-new-chat open`")
        if inside:
            # ช่องความจำ: .project/ + ใบงาน relay เขียนได้เสมอ (กติกา Use New Chat)
            if not all(is_control_path(target, root) for target in inside):
                branch = subprocess.run(["git", "branch", "--show-current"], cwd=root, text=True, capture_output=True).stdout.strip()
                if branch != session.get("branch") or not live_session_ready(session):
                    return block("branch, lease, permit หรือ WTL state ปัจจุบันไม่พร้อม — เช็คด้วย `hermes-new-chat status --task-id <task>`")
                for target in inside:
                    if not path_allowed(target, root, session, role):
                        return block("บทบาทนี้ไม่มีสิทธิ์เขียน path ดังกล่าว; งานโปรเจกต์ต้องผ่าน relay-call --role code")
        return 0
    # cwd นอกเขต หรือเขตคุมแบบไม่มี session (เช่น canonical repo): ตัดสินรายไฟล์เป้าหมาย
    for target in targets:
        governed = managed_root_for_path(target)
        if governed is None:
            continue  # เป้าหมายนอกเขตคุมทั้งหมด (scratchpad, ~/.claude, โปรเจกต์อิสระ) = ผ่าน
        if is_control_path(target, governed):
            continue  # ช่องความจำ .project/ + ใบงาน relay = ผ่านเสมอ
        if is_known_canonical(governed):
            return block("ห้ามเขียนโค้ดใน canonical repo ที่ใช้ร่วมกัน — เปิดงานผ่าน `hermes-new-chat open` (เขียนตรงได้เฉพาะ .project/ และใบงาน relay)")
        return block("ไม่พบ NEW_CHAT_READY สำหรับ Worktree นี้ — เปิดงานผ่าน `hermes-new-chat open` ก่อน")
    return 0


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeError) as exc:
        return block(f"อ่านข้อมูล Hook ไม่ได้: {exc}")
    return run(payload) if isinstance(payload, dict) else block("ข้อมูล Hook ต้องเป็น JSON object")


if __name__ == "__main__":
    raise SystemExit(main())
