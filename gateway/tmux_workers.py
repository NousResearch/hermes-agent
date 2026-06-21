"""tmux-backed Claude/Codex worker helpers for gateway commands.

This module deliberately keeps shell-sensitive user text out of command-line
arguments. User prompts are written to request files, loaded into tmux buffers,
and pasted into the target pane.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home


DEFAULT_TMUX_SESSION = "hermes"

_TOOL_DEFAULTS = {
    "claude": {"model": "opus", "effort": "xhigh"},
    "codex": {"model": "gpt-5.5", "effort": "xhigh"},
}

_CLAUDE_EFFORTS = {"low", "medium", "high", "xhigh", "max"}
_CODEX_EFFORTS = {"minimal", "low", "medium", "high", "xhigh"}
_EFFORT_FLAGS = {f"--{name}": name for name in sorted(_CLAUDE_EFFORTS | _CODEX_EFFORTS)}
_CLAUDE_MODEL_FLAGS = {
    "--opus": "opus",
    "--sonnet": "sonnet",
    "--fable": "fable",
}
_CODEX_MODEL_FLAGS = {
    "--gpt55": "gpt-5.5",
    "--gpt-55": "gpt-5.5",
    "--gpt5.5": "gpt-5.5",
    "--gpt-5.5": "gpt-5.5",
    "--5.5": "gpt-5.5",
}

_ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_REPORT_BLOCK_RE = re.compile(
    r"<HERMES_REPORT>\s*(.*?)\s*</HERMES_REPORT>",
    re.IGNORECASE | re.DOTALL,
)

WORKER_REPORT_CONTRACT = """You are running inside a Hermes-managed Discord tmux worker.
The Hermes gateway already knows the Discord thread/channel metadata, so do not say that replying is impossible because you cannot see Discord metadata.
For every meaningful progress update, question, blocker, or final answer, end with exactly one report block in Korean:

<HERMES_REPORT>
상태: running | needs_input | done | blocked
결론:
근거:
다음:
</HERMES_REPORT>

Keep the report block concise. Do not include hidden chain-of-thought, credentials, tokens, cookies, or raw secrets."""


@dataclass(frozen=True)
class WorkerLaunchSpec:
    tool: str
    model: str
    effort: str
    task: str
    ultracode: bool = False


@dataclass(frozen=True)
class WorkerRelayUpdate:
    worker: "WorkerRecord"
    text: str
    next_offset: int
    digest: str


@dataclass
class WorkerRecord:
    id: str
    tool: str
    mode: str
    model: str
    effort: str
    task: str
    tmux_session: str
    tmux_window: str
    pane_id: str
    platform: str
    thread_id: str | None
    chat_id: str | None
    chat_name: str | None
    user_id: str | None
    status: str
    request_path: Path
    log_path: Path
    created_at: str = ""
    updated_at: str = ""
    relay_offset: int = 0
    relay_seen_size: int = 0
    relay_seen_at: float = 0.0
    relay_last_hash: str = ""
    relay_last_sent_at: str = ""
    initial_sent: bool = True

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["request_path"] = str(self.request_path)
        data["log_path"] = str(self.log_path)
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "WorkerRecord":
        copied = dict(data)
        copied["request_path"] = Path(copied["request_path"])
        copied["log_path"] = Path(copied["log_path"])
        copied.setdefault("initial_sent", True)
        return cls(**copied)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(text: str, *, fallback: str = "task", max_len: int = 36) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z가-힣._-]+", "-", text.strip())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-._")
    if not cleaned:
        cleaned = fallback
    return cleaned[:max_len].strip("-._") or fallback


def extract_worker_report_block(text: str) -> str:
    """Return the last complete Hermes report block from worker output."""
    matches = [match.group(1).strip() for match in _REPORT_BLOCK_RE.finditer(text or "")]
    matches = [match for match in matches if match]
    if not matches:
        return ""
    report = matches[-1]
    report = _ANSI_ESCAPE_RE.sub("", report)
    report = _CONTROL_RE.sub("", report.replace("\r", "\n"))
    lines = [line.strip() for line in report.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def build_worker_prompt(text: str, *, kind: str = "initial") -> str:
    """Wrap user text with Discord-worker routing context and report contract."""
    text = (text or "").strip()
    label = "Initial Discord tmux worker request" if kind == "initial" else "Discord thread follow-up"
    return f"""{label}

User message:
{text}

{WORKER_REPORT_CONTRACT}""".strip()


def parse_worker_args(tool: str, raw_args: str) -> WorkerLaunchSpec:
    """Parse model/effort shortcuts and return a launch spec."""
    tool = tool.lower().strip()
    if tool not in _TOOL_DEFAULTS:
        raise ValueError(f"Unsupported worker tool: {tool}")

    defaults = _TOOL_DEFAULTS[tool]
    model = defaults["model"]
    effort = defaults["effort"]
    ultracode = False
    task_parts: list[str] = []

    try:
        tokens = shlex.split(raw_args or "")
    except ValueError as exc:
        raise ValueError(f"Could not parse arguments: {exc}") from exc

    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token == "--model":
            idx += 1
            if idx >= len(tokens):
                raise ValueError("--model requires a value")
            model = tokens[idx]
        elif token.startswith("--model="):
            model = token.split("=", 1)[1].strip()
            if not model:
                raise ValueError("--model requires a value")
        elif token == "--effort":
            idx += 1
            if idx >= len(tokens):
                raise ValueError("--effort requires a value")
            effort = tokens[idx]
        elif token.startswith("--effort="):
            effort = token.split("=", 1)[1].strip()
            if not effort:
                raise ValueError("--effort requires a value")
        elif token in _EFFORT_FLAGS:
            effort = _EFFORT_FLAGS[token]
        elif tool == "claude" and token == "--ultracode":
            ultracode = True
            effort = "xhigh"
        elif tool == "claude" and token in _CLAUDE_MODEL_FLAGS:
            model = _CLAUDE_MODEL_FLAGS[token]
        elif tool == "codex" and token in _CODEX_MODEL_FLAGS:
            model = _CODEX_MODEL_FLAGS[token]
        elif token.startswith("--"):
            raise ValueError(f"Unknown option: {token}")
        else:
            task_parts.extend(tokens[idx:])
            break
        idx += 1

    task = " ".join(task_parts).strip()
    allowed_efforts = _CLAUDE_EFFORTS if tool == "claude" else _CODEX_EFFORTS
    if effort not in allowed_efforts:
        raise ValueError(
            f"Unsupported {tool} effort '{effort}'. Allowed: {', '.join(sorted(allowed_efforts))}"
        )

    return WorkerLaunchSpec(tool=tool, model=model, effort=effort, task=task, ultracode=ultracode)


def public_log_label(path: Path | str) -> str:
    """Return a user-safe log label without exposing host paths."""
    try:
        return Path(path).name or "local tmux worker log"
    except Exception:
        return "local tmux worker log"


def clean_relay_text(raw: str) -> str:
    """Strip TUI/ANSI noise from tmux pipe-pane output before summarization."""
    text = _ANSI_ESCAPE_RE.sub("", raw or "")
    text = re.sub(r"\[\?\d+[A-Za-z]", "", text)
    text = _CONTROL_RE.sub("", text.replace("\r", "\n"))
    lines: list[str] = []
    previous = ""
    for line in text.splitlines():
        cleaned = line.strip()
        cleaned = re.sub(r"\d*;[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏].*$", "", cleaned).strip()
        cleaned = re.sub(r"\d*;[0-9a-f]{8}-[0-9a-f-]{20,}.*$", "", cleaned, flags=re.IGNORECASE).strip()
        if not cleaned:
            continue
        # Drop common full-screen TUI border fragments and pure spinner/status art.
        if re.fullmatch(r"[╭╮╰╯│─┌┐└┘├┤┬┴┼━┃\s]+", cleaned):
            continue
        if cleaned in {"›", "❯", ">", "│"}:
            continue
        lower = cleaned.lower()
        if (
            "openai codex" in lower
            or "claude code" in lower
            or lower.startswith("tip:")
            or "esc to interrupt" in lower
            or lower.startswith("model:")
            or lower.startswith("directory:")
            or "find and fix a bug in @filename" in lower
            or lower.startswith("starting mcp servers")
        ):
            continue
        if cleaned == previous:
            continue
        lines.append(cleaned)
        previous = cleaned
    return "\n".join(lines).strip()


class TmuxWorkerManager:
    """Profile-scoped tmux worker state and process launcher."""

    def __init__(self, home: Path | None = None, tmux_session: str = DEFAULT_TMUX_SESSION):
        self.home = Path(home) if home is not None else get_hermes_home() / "tmux-agents"
        self.tmux_session = tmux_session
        self.requests_dir = self.home / "requests"
        self.logs_dir = self.home / "logs"
        self.state_dir = self.home / "state"
        for directory in (self.requests_dir, self.logs_dir, self.state_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def _run(self, cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if check and result.returncode != 0:
            raise RuntimeError(
                f"Command failed ({result.returncode}): {shlex.join(cmd)}\n{result.stderr.strip()}"
            )
        return result

    def _state_path(self, worker_id: str) -> Path:
        return self.state_dir / f"{worker_id}.json"

    def _next_id(self, tool: str) -> str:
        prefix = "c" if tool == "codex" else "h"
        existing = []
        for path in self.state_dir.glob(f"{prefix}[0-9]*.json"):
            match = re.match(rf"{prefix}(\d+)$", path.stem)
            if match:
                existing.append(int(match.group(1)))
        return f"{prefix}{(max(existing) + 1) if existing else 1}"

    def _ensure_tmux_session(self) -> None:
        result = self._run(["tmux", "has-session", "-t", self.tmux_session], check=False)
        if result.returncode != 0:
            self._run(["tmux", "new-session", "-d", "-s", self.tmux_session, "-n", "main"])

    def _worker_command(self, tool: str, spec: WorkerLaunchSpec) -> list[str]:
        if tool == "claude":
            parts = ["ccd", "--model", spec.model, "--effort", spec.effort]
            if spec.ultracode:
                parts.append("--ultracode")
            # ccd is a user shell function; run through an interactive shell.
            return ["bash", "-ic", shlex.join(parts)]
        if tool == "codex":
            parts = [
                "codex",
                "--model", spec.model,
                "--config", f"model_reasoning_effort={spec.effort}",
            ]
            return ["bash", "-ic", shlex.join(parts)]
        raise ValueError(f"Unsupported worker tool: {tool}")

    def _write_text_file(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _capture_pane(self, pane_id: str, lines: int = 80) -> str:
        result = self._run(
            ["tmux", "capture-pane", "-t", pane_id, "-p", "-S", f"-{max(1, int(lines))}"],
            check=False,
        )
        return (result.stdout or "") if result.returncode == 0 else ""

    def _wait_for_prompt_ready(self, pane_id: str, *, tool: str, timeout: float | None = None) -> bool:
        """Best-effort wait until an interactive CLI is ready to receive text.

        Claude/Codex TUIs can take several seconds to initialize. Sending the
        first prompt during that startup window may silently disappear, which
        looks like "the Discord thread title was created but no task entered".
        """
        timeout = 20.0 if timeout is None and tool == "claude" else (10.0 if timeout is None else timeout)
        deadline = time.time() + timeout
        markers = (
            ("❯", "Claude Code", "Welcome back"),
            ("›", "Codex"),
            ("What can I help",),
        )
        while time.time() < deadline:
            text = self._capture_pane(pane_id, lines=80)
            if any(all(marker in text for marker in marker_group) for marker_group in markers):
                return True
            time.sleep(0.25)
        return False

    @staticmethod
    def _submit_delay_seconds(text: str) -> float:
        """Delay between paste end and Enter submit for interactive TUIs."""
        length = len(text or "")
        if length >= 1000:
            return 2.0
        if length >= 200:
            return 1.0
        return 0.35

    def save(self, worker: WorkerRecord) -> None:
        worker.updated_at = _now()
        if not worker.created_at:
            worker.created_at = worker.updated_at
        path = self._state_path(worker.id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(worker.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def load(self, worker_id: str) -> WorkerRecord | None:
        path = self._state_path(worker_id)
        if not path.exists():
            return None
        return WorkerRecord.from_json(json.loads(path.read_text(encoding="utf-8")))

    def iter_workers(self) -> list[WorkerRecord]:
        workers: list[WorkerRecord] = []
        for path in sorted(self.state_dir.glob("*.json")):
            try:
                workers.append(WorkerRecord.from_json(json.loads(path.read_text(encoding="utf-8"))))
            except Exception:
                continue
        return workers

    def collect_relay_updates(
        self,
        *,
        idle_seconds: float = 6.0,
        max_chars: int = 12000,
        min_chars: int = 8,
    ) -> list[WorkerRelayUpdate]:
        """Return stable tmux log deltas that should be summarized to Discord.

        The pipe-pane log may grow character-by-character while Claude/Codex is
        still streaming. To avoid sending half-answers, the first pass only
        records the observed size/time. A later pass emits a delta only after the
        size has stayed unchanged for ``idle_seconds``.
        """
        now = time.time()
        updates: list[WorkerRelayUpdate] = []
        for worker in self.iter_workers():
            if worker.status != "running" or worker.platform != "discord" or not worker.thread_id:
                continue
            try:
                size = worker.log_path.stat().st_size
            except OSError:
                continue

            if size <= worker.relay_offset:
                if worker.relay_seen_size != size:
                    worker.relay_seen_size = size
                    worker.relay_seen_at = now
                    self.save(worker)
                continue

            if size != worker.relay_seen_size:
                worker.relay_seen_size = size
                worker.relay_seen_at = now
                self.save(worker)
                continue

            if worker.relay_seen_at and (now - worker.relay_seen_at) < idle_seconds:
                continue

            try:
                with worker.log_path.open("rb") as fh:
                    fh.seek(max(0, worker.relay_offset))
                    raw_bytes = fh.read(max(0, size - worker.relay_offset))
            except OSError:
                continue

            raw_text = raw_bytes.decode("utf-8", errors="replace")
            cleaned = clean_relay_text(raw_text)
            if len(cleaned) < min_chars:
                worker.relay_offset = size
                worker.relay_seen_size = size
                worker.relay_seen_at = now
                self.save(worker)
                continue

            if len(cleaned) > max_chars:
                cleaned = cleaned[-max_chars:]
            digest = hashlib.sha256(cleaned.encode("utf-8", errors="replace")).hexdigest()
            if digest == worker.relay_last_hash:
                worker.relay_offset = size
                worker.relay_seen_size = size
                worker.relay_seen_at = now
                self.save(worker)
                continue

            updates.append(WorkerRelayUpdate(worker=worker, text=cleaned, next_offset=size, digest=digest))
        return updates

    def mark_relay_sent(self, update: WorkerRelayUpdate) -> None:
        """Persist relay offset after Discord accepted the summarized update."""
        worker = self.load(update.worker.id) or update.worker
        worker.relay_offset = max(worker.relay_offset, update.next_offset)
        worker.relay_seen_size = worker.relay_offset
        worker.relay_seen_at = time.time()
        worker.relay_last_hash = update.digest
        worker.relay_last_sent_at = _now()
        self.save(worker)

    def find_by_thread(self, platform: str, thread_id: str | None) -> WorkerRecord | None:
        if not thread_id:
            return None
        candidates = [
            worker for worker in self.iter_workers()
            if worker.platform == platform and worker.thread_id == thread_id and worker.status == "running"
        ]
        if not candidates:
            return None
        return sorted(candidates, key=lambda worker: worker.updated_at or worker.created_at)[-1]

    def start_persistent(
        self,
        *,
        tool: str,
        spec: WorkerLaunchSpec,
        platform: str,
        thread_id: str | None,
        chat_id: str | None,
        chat_name: str | None,
        user_id: str | None,
        send_initial: bool = True,
    ) -> WorkerRecord:
        if not spec.task:
            raise ValueError(f"/{'tmux_' + tool} requires a task")
        self._ensure_tmux_session()
        worker_id = self._next_id(tool)
        slug = _slug(spec.task)
        window = f"{tool}-{worker_id}-{slug}"[:80]
        request_path = self.requests_dir / f"{worker_id}.md"
        log_path = self.logs_dir / f"{worker_id}.log"
        self._write_text_file(request_path, spec.task)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.touch(exist_ok=True)

        command = shlex.join(self._worker_command(tool, spec))
        pane_result = self._run([
            "tmux", "new-window", "-t", self.tmux_session,
            "-n", window,
            "-P", "-F", "#{pane_id}",
            command,
        ])
        pane_id = pane_result.stdout.strip() or f"{self.tmux_session}:{window}"
        self._run(["tmux", "pipe-pane", "-o", "-t", pane_id, f"cat >> {shlex.quote(str(log_path))}"])

        worker = WorkerRecord(
            id=worker_id,
            tool=tool,
            mode="persistent",
            model=spec.model,
            effort=spec.effort,
            task=spec.task,
            tmux_session=self.tmux_session,
            tmux_window=window,
            pane_id=pane_id,
            platform=platform,
            thread_id=thread_id,
            chat_id=chat_id,
            chat_name=chat_name,
            user_id=user_id,
            status="running",
            request_path=request_path,
            log_path=log_path,
            initial_sent=send_initial,
        )
        self.save(worker)
        if send_initial:
            # Wait for the interactive TUI before pasting the first prompt. A fixed
            # sleep is not reliable for Claude Code startup; sending too early can
            # create the Discord thread/title while the actual task never enters the
            # pane.
            self._wait_for_prompt_ready(pane_id, tool=tool)
            self.send_followup(worker, build_worker_prompt(spec.task, kind="initial"), wait_ready=False)
        return worker

    def send_followup(self, worker: WorkerRecord, text: str, *, wait_ready: bool = True) -> str:
        text = (text or "").strip()
        if not text:
            return "보낼 내용이 없습니다."
        if worker.status != "running":
            return f"작업방 `{worker.id}` 상태가 `{worker.status}`라서 전달하지 않았습니다."
        input_path = self.requests_dir / f"{worker.id}-{int(time.time() * 1000)}.input"
        self._write_text_file(input_path, text)
        buffer_name = f"hermes-{worker.id}-{int(time.time() * 1000)}"
        # Load prompt text from a request file so user content does not appear in
        # shell/process arguments, then paste with -p and delay before Enter.
        # The -p bracketed paste path is important for prompt_toolkit/Claude/
        # Codex TUIs; sending Enter immediately after a raw paste can be
        # swallowed as paste content.
        if wait_ready:
            self._wait_for_prompt_ready(worker.pane_id, tool=worker.tool)
        self._run(["tmux", "load-buffer", "-b", buffer_name, str(input_path)])
        self._run(["tmux", "paste-buffer", "-p", "-d", "-b", buffer_name, "-t", worker.pane_id])
        time.sleep(self._submit_delay_seconds(text))
        self._run(["tmux", "send-keys", "-t", worker.pane_id, "Enter"])
        worker.initial_sent = True
        worker.updated_at = _now()
        self.save(worker)
        return f"전달했습니다 → `{worker.id}` · `{worker.tmux_window}`"

    def tail(self, worker: WorkerRecord, lines: int = 80) -> str:
        if not worker.log_path.exists():
            return f"로그가 아직 없습니다: `{worker.log_path}`"
        content = worker.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = "\n".join(content[-lines:])
        return tail or "로그가 아직 비어 있습니다."

    def stop(self, worker: WorkerRecord) -> str:
        result = self._run(["tmux", "kill-pane", "-t", worker.pane_id], check=False)
        worker.status = "stopped" if result.returncode == 0 else "stop_failed"
        self.save(worker)
        if result.returncode == 0:
            return f"중지했습니다 → `{worker.id}` · `{worker.tmux_window}`"
        return f"중지 실패: `{result.stderr.strip() or result.stdout.strip()}`"

    def status_text(self, *, tool: str | None = None) -> str:
        workers = self.iter_workers()
        if tool:
            workers = [worker for worker in workers if worker.tool == tool]
        if not workers:
            return "tmux 작업방이 없습니다."
        lines = ["tmux 작업방"]
        for worker in workers[-20:]:
            thread = f" · thread `{worker.thread_id}`" if worker.thread_id else ""
            lines.append(
                f"- `{worker.id}` · {worker.tool} · {worker.status} · {worker.model}/{worker.effort} · `{worker.tmux_window}`{thread}"
            )
        return "\n".join(lines)

    def run_once(self, *, tool: str, spec: WorkerLaunchSpec, timeout: int = 900) -> str:
        if not spec.task:
            raise ValueError(f"/{tool} requires a task")
        worker_id = self._next_id(tool)
        request_path = self.requests_dir / f"{worker_id}-once.md"
        log_path = self.logs_dir / f"{worker_id}-once.log"
        self._write_text_file(request_path, spec.task)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if tool == "claude":
            parts = ["ccd", "--model", spec.model, "--effort", spec.effort]
            if spec.ultracode:
                parts.append("--ultracode")
            command = f"{shlex.join(parts)} -p \"$(cat {shlex.quote(str(request_path))})\""
            cmd = ["bash", "-ic", command]
        elif tool == "codex":
            command = (
                "codex "
                f"--model {shlex.quote(spec.model)} "
                f"--config {shlex.quote(f'model_reasoning_effort={spec.effort}')} "
                "--ask-for-approval never "
                "--sandbox read-only "
                f"exec \"$(cat {shlex.quote(str(request_path))})\""
            )
            cmd = ["bash", "-ic", command]
        else:
            raise ValueError(f"Unsupported worker tool: {tool}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
            output = (result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")
            log_path.write_text(output, encoding="utf-8")
            if result.returncode != 0:
                return f"`/{tool}` 실패(exit {result.returncode}). local log: `{public_log_label(log_path)}`\n\n{output[-3000:]}"
            return output[-3800:] or f"완료됐지만 출력이 비어 있습니다. local log: `{public_log_label(log_path)}`"
        except subprocess.TimeoutExpired:
            return f"`/{tool}`가 {timeout}초 안에 끝나지 않았습니다. local log: `{public_log_label(log_path)}`"


def worker_usage(command: str) -> str:
    if command in {"claude", "codex"}:
        return f"사용법: `/{command} [--xhigh|--high|--medium|--low] [--model MODEL] <질문>`"
    efforts = "--max|--xhigh|--high|--medium|--low" if command.endswith("claude") else "--xhigh|--high|--medium|--low"
    return f"사용법: `/{command} [status|tail|stop]` 또는 `/{command} [{efforts}] [--model MODEL] <제목>` — 첫 지시는 생성된 Discord thread에 일반 메시지로 쓰세요."
