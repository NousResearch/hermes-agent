"""Gateway /devrun mobile development loop helpers.

This module keeps the first mobile-dev loop intentionally small: parse a
Telegram/gateway command, classify risk, persist a job record, and build the
bounded prompt that the existing background-agent runner can execute.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


JOB_STATUSES = {
    "queued",
    "reviewing",
    "awaiting_approval",
    "running",
    "testing",
    "done",
    "failed",
    "blocked",
}

LOW_RISK = "low"
MEDIUM_RISK = "medium"
HIGH_RISK = "high"

_DEFAULT_CANCELLED_STATUSES = {"done", "failed", "blocked"}

_DANGEROUS_RE = re.compile(
    r"\b("
    r"restart|reload|launchctl|pkill|killall|rm\s+-rf|delete|remove|drop\s+table|"
    r"migration|migrate|database|db|credential|secret|token|api[_ -]?key|"
    r"config\.ya?ml|\.env|push|commit|rebase|reset\s+--hard|chmod|chown|sudo|"
    r"hermes|openclaw|kanban|gateway|production|deploy"
    r")\b",
    re.IGNORECASE,
)

_MEDIUM_RE = re.compile(
    r"("
    r"\b(?:refactor|architecture|multi[- ]?file|many files|auth|login|sign[- ]?in|"
    r"sign[- ]?up|permission|payment|security|privacy|release|ci|workflow|"
    r"integration|api|schema)\b|"
    r"登录|登入|注册|鉴权|认证|权限|账号|账户|密码|用户系统|用户中心|找回密码"
    r")",
    re.IGNORECASE,
)

_READ_ONLY_RE = re.compile(
    r"("
    r"\bread[- ]?only\b|\bplan[- ]?only\b|\bno[- ]?edit(?:s|ing)?\b|"
    r"\bdo not (?:edit|modify|write|change)\b|\bwithout (?:editing|modifying|writing)\b|"
    r"只读|不要修改|不要改|不修改|不改动|不写入|禁止修改|仅评估|只评估|只检查|只分析|只总结|只出方案|方案-only"
    r")",
    re.IGNORECASE,
)

_REVIEW_FALLBACK_ERROR_RE = re.compile(
    r"("
    r"\b(?:quota|insufficient(?: account)? balance|insufficient_user_quota|"
    r"payment required|billing|credit|permissiondenied|permission denied|"
    r"unauthorized|authentication|auth error|http 40[123])\b|"
    r"额度不足|余额不足|剩余额度|权限不足|鉴权失败|认证失败"
    r")",
    re.IGNORECASE,
)


@dataclass
class DevRunRequest:
    task: str
    repo: str | None = None
    board: str | None = None


@dataclass
class DevRunJob:
    job_id: str
    raw_task: str
    task: str
    repo: str
    risk: str
    status: str
    board: str | None = None
    review_verdict: str = ""
    review_summary: str = ""
    commands: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    test_results: str = ""
    telegram_messages: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    error: str = ""
    cancelled: bool = False


@dataclass(frozen=True)
class DevFlowBudgetEstimate:
    route: str
    agent_count_min: int
    agent_count_max: int
    minimax_calls_min: int
    minimax_calls_max: int
    notes: list[str]


DEVREVIEW_WORKERS = (
    ("product", "需求审查"),
    ("architecture", "架构审查"),
    ("testplan", "测试审查"),
    ("review", "安全与最终验收审查"),
)

_DEVREVIEW_CONTEXT_FILES = {
    "README.md",
    "README",
    "package.json",
    "pyproject.toml",
    "requirements.txt",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "Dockerfile",
    "docker-compose.yml",
    "Makefile",
}

_DEVREVIEW_EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}

_SECRET_LINE_RE = re.compile(
    r"(secret|token|api[_-]?key|password|passwd|private[_-]?key|credential|cookie)\s*[:=]",
    re.IGNORECASE,
)

_DEVFLOW_KANBAN_RE = re.compile(
    r"("
    r"\b(?:large|major|complex|multi[- ]?agent|swarm|kanban|milestone|project|"
    r"decompose|break down|roadmap|many tasks)\b|"
    r"大项目|复杂|军团|多智能体|拆解|里程碑|路线图|长期|完整项目"
    r")",
    re.IGNORECASE,
)


def is_read_only_task(task: str) -> bool:
    return bool(_READ_ONLY_RE.search(task or ""))


def task_mode(job: DevRunJob) -> str:
    return "read_only" if is_read_only_task(job.task) else "editable"


def devrun_state_dir() -> Path:
    path = get_hermes_home() / "devrun"
    _ensure_private_dir(path)
    return path


def _jobs_dir() -> Path:
    path = devrun_state_dir() / "jobs"
    _ensure_private_dir(path)
    return path


def _job_path(job_id: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", job_id)
    return _jobs_dir() / f"{safe}.json"


def _chmod_best_effort(path: Path, mode: int) -> None:
    try:
        path.chmod(mode)
    except OSError:
        pass


def _ensure_private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _chmod_best_effort(path, 0o700)


def parse_devrun_args(raw_args: str) -> DevRunRequest:
    """Parse `/devrun` args.

    Supported forms:
    - `/devrun fix the failing test`
    - `/devrun repo=/path/to/project task=fix the failing test`
    - `/devrun board=project-a repo=/path/to/project task=fix the failing test`
    """
    raw_args = (raw_args or "").strip()
    if not raw_args:
        raise ValueError("Usage: /devrun [repo=/path/to/project] task=<task>")

    repo: str | None = None
    board: str | None = None
    task = raw_args

    try:
        parts = shlex.split(raw_args)
    except ValueError:
        parts = raw_args.split()

    remaining: list[str] = []
    task_seen = False
    for part in parts:
        if part.startswith("repo=") and not task_seen:
            repo = part.split("=", 1)[1].strip() or None
            continue
        if part.startswith("board=") and not task_seen:
            board = part.split("=", 1)[1].strip() or None
            continue
        if part.startswith("task="):
            task_seen = True
            value = part.split("=", 1)[1].strip()
            if value:
                remaining.append(value)
            continue
        remaining.append(part)

    if repo or board or task_seen:
        task = " ".join(remaining).strip()

    if not task:
        raise ValueError("Usage: /devrun [repo=/path/to/project] task=<task>")

    return DevRunRequest(task=task, repo=repo, board=board)


def resolve_kanban_board(board: str | None) -> str | None:
    """Validate an optional Kanban board slug for DevFlow routing."""
    raw = (board or "").strip()
    if not raw:
        return None
    from hermes_cli import kanban_db as kb

    try:
        normed = kb._normalize_board_slug(raw)
    except ValueError as exc:
        raise ValueError(f"invalid Kanban board slug: {exc}") from exc
    if not normed:
        return None
    if normed != kb.DEFAULT_BOARD and not kb.board_exists(normed):
        raise ValueError(
            f"Kanban board does not exist: {normed}. "
            f"Create it first with /kanban boards create {normed}."
        )
    return normed


def resolve_repo_path(repo: str | None) -> Path:
    raw = (repo or "").strip()
    if not raw:
        raw = os.environ.get("TERMINAL_CWD", "").strip()
    if not raw or raw.lower() in {".", "./", "auto", "cwd"}:
        raw = str(Path.cwd())
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists() or not path.is_dir():
        raise ValueError(f"repo path does not exist or is not a directory: {path}")
    resolved = path.resolve()
    sensitive_reason = _sensitive_repo_reason(resolved)
    if sensitive_reason:
        raise ValueError(f"repo path is not allowed for DevRun: {resolved} ({sensitive_reason})")
    allowed_roots = allowed_repo_roots()
    if not any(_is_path_within(resolved, root) for root in allowed_roots):
        roots = ", ".join(str(root) for root in allowed_roots[:6]) or "(none)"
        raise ValueError(
            "repo path is outside DevRun allowed roots: "
            f"{resolved}. Set HERMES_DEVRUN_ALLOWED_ROOTS to allow it. Current roots: {roots}"
        )
    return resolved


def infer_recent_repo_from_jobs(limit: int = 30) -> Path | None:
    """Return the most recent valid DevRun/DevFlow repo, if one is known.

    This is intentionally conservative: a repo is only accepted if it still
    exists and passes the same allowlist/sensitive-path checks as an explicit
    repo argument.
    """
    for job in list_jobs(limit=limit):
        raw_repo = (job.repo or "").strip()
        if not raw_repo:
            continue
        try:
            return resolve_repo_path(raw_repo)
        except ValueError:
            continue
    return None


def allowed_repo_roots() -> list[Path]:
    roots: list[Path] = []

    for var_name in ("HERMES_DEVRUN_ALLOWED_ROOTS", "HERMES_DEVRUN_ROOTS"):
        raw = os.environ.get(var_name, "").strip()
        if raw:
            for item in raw.split(os.pathsep):
                _append_existing_root(roots, item)

    cwd = Path.cwd()
    if cwd != Path.home():
        _append_existing_root(roots, str(cwd))

    home = Path.home()
    for name in ("hermes-agent", "projects", "project", "dev", "code", "work", "src", "codex001"):
        _append_existing_root(roots, str(home / name))

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen and not _sensitive_repo_reason(root):
            unique.append(root)
            seen.add(key)
    return unique


def _append_existing_root(roots: list[Path], raw: str) -> None:
    raw = (raw or "").strip()
    if not raw:
        return
    try:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if path.exists() and path.is_dir():
            roots.append(path.resolve())
    except OSError:
        return


def _is_path_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _sensitive_repo_reason(path: Path) -> str:
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    try:
        if resolved == Path(resolved.anchor).resolve():
            return f"filesystem root: {resolved}"
    except OSError:
        pass
    home = Path.home().resolve()
    if resolved == home:
        return "home directory itself is too broad"

    sensitive_home_dirs = {
        ".ssh",
        ".gnupg",
        ".aws",
        ".azure",
        ".kube",
        ".docker",
        ".config",
        ".hermes",
        ".opensquilla",
    }
    try:
        rel_parts = {part.lower() for part in resolved.relative_to(home).parts}
    except ValueError:
        rel_parts = set()
    hit = rel_parts.intersection(sensitive_home_dirs)
    if hit:
        return f"sensitive home directory: {sorted(hit)[0]}"

    system_roots = [
        Path("/etc"),
        Path("/private/etc"),
        Path("/System"),
        Path("/Library"),
        Path("/Applications"),
    ]
    for root in system_roots:
        try:
            root_resolved = root.resolve()
        except OSError:
            root_resolved = root
        if resolved == root_resolved or _is_path_within(resolved, root_resolved):
            return f"system directory: {root_resolved}"
    return ""


def classify_risk(task: str, repo: str | Path | None = None) -> str:
    text = f"{task or ''} {repo or ''}"
    if _DANGEROUS_RE.search(text):
        return HIGH_RISK
    if _MEDIUM_RE.search(text):
        return MEDIUM_RISK
    return LOW_RISK


def classify_devflow_route(task: str, risk: str) -> str:
    if risk == LOW_RISK and not _DEVFLOW_KANBAN_RE.search(task or ""):
        return "devrun"
    return "kanban"


def estimate_devflow_budget(task: str, risk: str, route: str, *, sidecar_available: bool = True) -> DevFlowBudgetEstimate:
    text = task or ""
    if route == "devrun":
        return DevFlowBudgetEstimate(
            route=route,
            agent_count_min=1,
            agent_count_max=1,
            minimax_calls_min=1,
            minimax_calls_max=4,
            notes=[
                "Low-risk scoped task routed to one DevRun executor.",
                "No Kanban army is expected.",
            ],
        )

    agent_min = 2 if sidecar_available else 1
    agent_max = 6
    calls_min = 3 if sidecar_available else 1
    calls_max = 18
    notes = [
        "OpenSquilla preflight is one independent review worker when available.",
        "Kanban triage may spawn orchestrator plus several role workers after the card is created.",
    ]
    if risk == HIGH_RISK:
        agent_max += 2
        calls_max += 8
        notes.append("High-risk wording can trigger extra review or approval loops.")
    if re.search(r"\b(full|complete|end[- ]?to[- ]?end|production|release|migration)\b|完整|全链路|生产|迁移", text, re.IGNORECASE):
        agent_max += 3
        calls_max += 12
        notes.append("Full/end-to-end wording usually expands into more Kanban subtasks.")
    if not sidecar_available:
        notes.append("OpenSquilla is unavailable, so the estimate excludes sidecar review calls.")

    return DevFlowBudgetEstimate(
        route=route,
        agent_count_min=agent_min,
        agent_count_max=agent_max,
        minimax_calls_min=calls_min,
        minimax_calls_max=calls_max,
        notes=notes,
    )


def format_devflow_budget_estimate(estimate: DevFlowBudgetEstimate) -> str:
    return "\n".join(
        [
            f"Estimated agents: {estimate.agent_count_min}-{estimate.agent_count_max}",
            f"Estimated MiniMax-M3 calls: {estimate.minimax_calls_min}-{estimate.minimax_calls_max}",
            "Notes:",
            *[f"- {note}" for note in estimate.notes],
        ]
    )


def format_devflow_budget_audit(estimate: DevFlowBudgetEstimate) -> str:
    """Render a stable, reviewable budget evidence block for DevFlow jobs.

    The Telegram card is intentionally concise. This block is for the persisted
    job summary and Kanban card, so later reviewers can distinguish a real
    budget gate estimate from a vague progress message.
    """
    notes = "\n".join(f"- {note}" for note in estimate.notes)
    return "\n".join(
        [
            "## DevFlow Budget Gate Evidence",
            f"- Route: {estimate.route}",
            f"- Estimated agents: {estimate.agent_count_min}-{estimate.agent_count_max}",
            f"- Estimated MiniMax-M3 calls: {estimate.minimax_calls_min}-{estimate.minimax_calls_max}",
            "- Gate type: pre-spend estimate and mobile approval gate",
            "- Hard stop: cancel/decline spends no OpenSquilla or Kanban budget",
            "- Accounting note: actual provider token/cost usage may be unavailable when receipts omit usage fields",
            "- Notes:",
            notes,
        ]
    )


def _zh_risk_label(risk: str) -> str:
    return {
        LOW_RISK: "低风险",
        MEDIUM_RISK: "中风险",
        HIGH_RISK: "高风险",
    }.get(risk, risk or "未知")


def _zh_status_label(status: str) -> str:
    return {
        "queued": "等待启动",
        "reviewing": "预审中",
        "awaiting_approval": "等待手机确认",
        "running": "执行中",
        "testing": "验证中",
        "done": "已完成路由",
        "failed": "失败",
        "blocked": "已停止",
    }.get(status, status or "未知")


def _zh_repo_source(repo_source: str | None) -> str:
    if repo_source == "recent project":
        return "最近项目"
    if repo_source == "explicit":
        return "本次指定"
    return "已确认"


def _extract_devflow_budget(review_summary: str) -> tuple[str | None, str | None]:
    agents = None
    calls = None
    for line in (review_summary or "").splitlines():
        line = line.strip().lstrip("-").strip()
        if line.startswith("Estimated agents:"):
            agents = line.split(":", 1)[1].strip()
        elif line.startswith("Estimated MiniMax-M3 calls:"):
            calls = line.split(":", 1)[1].strip()
    return agents, calls


def format_devflow_budget_zh(estimate: DevFlowBudgetEstimate) -> str:
    return "\n".join(
        [
            f"智能体  {estimate.agent_count_min}-{estimate.agent_count_max} 个",
            f"M3 调用  {estimate.minimax_calls_min}-{estimate.minimax_calls_max} 次",
        ]
    )


def render_devflow_budget_prompt(
    *,
    job_id: str,
    task: str,
    repo: str | Path,
    risk: str,
    repo_source: str,
    board: str | None = None,
    estimate: DevFlowBudgetEstimate,
    text_prefix: str = "/",
) -> str:
    """Render the user-facing DevFlow budget confirmation card."""
    return "\n".join(
        [
            "✦ **DevFlow Command Center**",
            "━━━━━━━━━━━━━━━━",
            "状态  待确认",
            f"风险  {_zh_risk_label(risk)}",
            f"预算  M3 {estimate.minimax_calls_min}-{estimate.minimax_calls_max} 次",
            "",
            "**任务**",
            f"「{task}」",
            "",
            "**项目**",
            f"`{repo}`",
            f"来源  {_zh_repo_source(repo_source)}",
            *( [f"Kanban  `{board}`"] if board else [] ),
            "",
            "**执行链路**",
            "OpenSquilla 预审",
            "Kanban 智能体军团",
            "Telegram 进度回传",
            "",
            "**预算预估**",
            format_devflow_budget_zh(estimate),
            "",
            "**操作**",
            "启动本次  开始预审并创建 Kanban 任务",
            "暂不启动  不消耗 OpenSquilla / Kanban 预算",
            "",
            "━━━━━━━━━━━━━━━━",
            f"ID  `{job_id}`",
            f"备用指令  `{text_prefix}approve` / `{text_prefix}cancel`",
        ]
    )


def _devflow_budget_lines_from_summary(review_summary: str) -> list[str]:
    agents, calls = _extract_devflow_budget(review_summary)
    lines = []
    if agents:
        lines.append(f"智能体  {agents} 个")
    if calls:
        lines.append(f"M3 调用  {calls} 次")
    return lines


def _short_path(path: str, max_len: int = 44) -> str:
    if len(path) <= max_len:
        return path
    keep = max_len - 1
    return "…" + path[-keep:]


def _task_line(task: str, max_len: int = 54) -> str:
    task = (task or "").strip()
    if len(task) <= max_len:
        return task
    return task[: max_len - 1] + "…"


def render_devflow_resolved_banner(choice: str) -> str:
    if choice == "cancel":
        return "\n".join(
            [
                "✦ DevFlow",
                "状态  已暂停",
                "预算  未消耗",
            ]
        )
    if choice == "always":
        return "\n".join(
            [
                "✦ DevFlow",
                "状态  已启动",
                "权限  始终允许",
            ]
        )
    return "\n".join(
        [
            "✦ DevFlow",
            "状态  已启动",
        ]
    )


def render_devflow_button_labels() -> dict[str, str]:
    return {
        "once": "启动本次",
        "always": "始终允许",
        "cancel": "暂不启动",
    }


def is_devflow_confirm_message(message: str) -> bool:
    return "DevFlow Command Center" in (message or "")


def is_devflow_confirm_title(title: str) -> bool:
    return "DevFlow" in (title or "")


def render_devflow_started_card(
    *,
    job: DevRunJob,
    repo_source: str | None = None,
) -> str:
    budget_lines = _devflow_budget_lines_from_summary(job.review_summary)
    if not budget_lines:
        budget_lines.append("预算  按当前路线执行")

    return "\n".join(
        [
            "✦ **DevFlow Command Center**",
            "━━━━━━━━━━━━━━━━",
            "状态  已启动",
            f"风险  {_zh_risk_label(job.risk)}",
            "路径  预审 → Kanban",
            "",
            "**任务**",
            f"「{_task_line(job.task)}」",
            "",
            "**项目**",
            f"`{_short_path(job.repo)}`",
            *( [f"来源  {_zh_repo_source(repo_source)}"] if repo_source else [] ),
            *( [f"Kanban  `{job.board}`"] if job.board else [] ),
            "",
            "**预算预估**",
            *budget_lines,
            "",
            "━━━━━━━━━━━━━━━━",
            f"进度  `/devstatus {job.job_id}`",
        ]
    )


def render_devflow_cancelled_card(job: DevRunJob) -> str:
    budget_lines = _devflow_budget_lines_from_summary(job.review_summary)

    return "\n".join(
        [
            "✦ **DevFlow Command Center**",
            "━━━━━━━━━━━━━━━━",
            "状态  已暂停",
            "预算  未消耗",
            f"风险  {_zh_risk_label(job.risk)}",
            "",
            "**任务**",
            f"「{_task_line(job.task)}」",
            "",
            "**项目**",
            f"`{_short_path(job.repo)}`",
            *( [f"Kanban  `{job.board}`"] if job.board else [] ),
            *( ["", "**原预算**", *budget_lines] if budget_lines else [] ),
            "",
            "**回执**",
            "OpenSquilla 未启动",
            "Kanban 未创建",
            "MiniMax-M3 未继续消耗",
            "",
            "━━━━━━━━━━━━━━━━",
            f"ID  `{job.job_id}`",
        ]
    )


def render_devflow_status_card(job: DevRunJob) -> str:
    if job.status == "blocked" and job.cancelled:
        return render_devflow_cancelled_card(job)

    budget_lines = _devflow_budget_lines_from_summary(job.review_summary)

    lines = [
        "✦ **DevFlow Command Center**",
        "━━━━━━━━━━━━━━━━",
        f"状态  {_zh_status_label(job.status)}",
        f"风险  {_zh_risk_label(job.risk)}",
        "",
        "**任务**",
        f"「{_task_line(job.task)}」",
        "",
        "**项目**",
        f"`{_short_path(job.repo)}`",
    ]
    if job.board:
        lines.append(f"Kanban  `{job.board}`")
    if job.review_verdict == "DEVFLOW":
        lines.extend(["", "**路径**", "OpenSquilla 预审", "Kanban 智能体军团"])
    if budget_lines:
        lines.extend(["", "**预算**", *budget_lines])
    if job.test_results:
        lines.extend(["", "**进展**", job.test_results[:700]])
    if job.error:
        lines.extend(["", f"提示  {job.error}"])
    lines.extend(["", "━━━━━━━━━━━━━━━━", f"ID  `{job.job_id}`"])
    return "\n".join(lines)


def make_job_id() -> str:
    return f"dev_{time.strftime('%Y%m%d_%H%M%S')}_{os.urandom(3).hex()}"


def save_job(job: DevRunJob) -> DevRunJob:
    job.updated_at = time.time()
    path = _job_path(job.job_id)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(job), ensure_ascii=False, indent=2), encoding="utf-8")
    _chmod_best_effort(tmp, 0o600)
    tmp.replace(path)
    _chmod_best_effort(path, 0o600)
    return job


def load_job(job_id: str) -> DevRunJob:
    path = _job_path(job_id)
    if not path.exists():
        raise KeyError(job_id)
    data = json.loads(path.read_text(encoding="utf-8"))
    return DevRunJob(**data)


def list_jobs(limit: int = 20) -> list[DevRunJob]:
    files = sorted(_jobs_dir().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    jobs: list[DevRunJob] = []
    for path in files[: max(1, limit)]:
        try:
            jobs.append(DevRunJob(**json.loads(path.read_text(encoding="utf-8"))))
        except Exception:
            continue
    return jobs


def cancel_job(job_id: str) -> DevRunJob:
    job = load_job(job_id)
    if job.status not in _DEFAULT_CANCELLED_STATUSES:
        job.cancelled = True
        job.status = "blocked"
        job.error = "Cancelled by user."
        job.telegram_messages.append("Job cancelled by user.")
        save_job(job)
    return job


def update_job(job_id: str, **changes: Any) -> DevRunJob:
    job = load_job(job_id)
    for key, value in changes.items():
        if hasattr(job, key):
            setattr(job, key, value)
    return save_job(job)


def find_sidecar_command() -> list[str] | None:
    configured = os.environ.get("HERMES_DEVRUN_OPENSQUILLA_WORKER", "").strip()
    candidates = [configured] if configured else []
    candidates.extend(
        [
            str(Path.home() / "codex001" / "opensquilla_worker.mjs"),
            "/Users/yuxiansheng/codex001/opensquilla_worker.mjs",
            "/Users/yuxiansheng/opensquilla_worker.mjs",
        ]
    )
    node = _which("node")
    for candidate in candidates:
        if candidate and node and Path(candidate).expanduser().exists():
            return [
                node,
                str(Path(candidate).expanduser()),
                "--worker",
                "review",
            ]
    return None


def find_worker_command(worker: str) -> list[str] | None:
    command = find_sidecar_command()
    if not command:
        return None
    if "--worker" not in command:
        return [*command, "--worker", worker]
    adjusted = list(command)
    try:
        index = adjusted.index("--worker")
        adjusted[index + 1] = worker
    except (ValueError, IndexError):
        adjusted.extend(["--worker", worker])
    return adjusted


def _which(name: str) -> str | None:
    for folder in os.environ.get("PATH", "").split(os.pathsep):
        path = Path(folder) / name
        if path.exists() and os.access(path, os.X_OK):
            return str(path)
    return None


def build_review_packet(job: DevRunJob) -> str:
    mode = task_mode(job)
    proposed_execution = (
        [
            "- Hermes background agent may inspect files only.",
            "- This task is read-only / plan-only: no file writes, edits, renames, deletes, test execution, or generated diffs are allowed.",
            "- The expected deliverable is an in-chat report or recommendation list only.",
        ]
        if mode == "read_only"
        else [
            "- Hermes background agent may inspect files, make low-risk scoped edits, and run tests when the task permits it.",
            "- If the task is explicitly read-only or no-edit, that stricter task boundary wins.",
        ]
    )
    return "\n".join(
        [
            "# DevRun Review Packet",
            "",
            f"Job ID: {job.job_id}",
            f"Repo: {job.repo}",
            f"Risk: {job.risk}",
            f"Mode: {mode}",
            "",
            "## Task",
            job.task,
            "",
            "## Boundaries",
            "- This is a pre-execution risk gate, not a final implementation review.",
            "- OpenSquilla is advisory only.",
            "- Do not modify files or run commands from the review.",
            "- Return PASS, NEEDS_CHANGES, or BLOCKED with risks and tests.",
            "- PASS means it is safe to begin the bounded execution; it does not certify final correctness.",
            "- Do not require source file contents unless their absence creates a concrete safety blocker.",
            "",
            "## Proposed Execution",
            *proposed_execution,
            "- Service restarts, deletes, database migrations, credential/config edits, git push/commit, and broad rewrites require user approval.",
        ]
    )


def collect_devreview_context(repo: str | Path, *, max_files: int = 80, max_file_chars: int = 1800) -> str:
    """Collect a bounded, redacted repo snapshot for review-only sidecar use."""
    root = Path(repo)
    sections: list[str] = []
    tree: list[str] = []
    snippets: list[str] = []

    try:
        resolved = root.resolve()
    except OSError:
        resolved = root

    if not resolved.exists() or not resolved.is_dir():
        return "(Repo context unavailable: path does not exist or is not a directory.)"

    visited = 0
    for path in sorted(resolved.rglob("*")):
        if visited >= max_files:
            tree.append(f"... truncated after {max_files} entries")
            break
        try:
            rel = path.relative_to(resolved)
        except ValueError:
            continue
        parts = set(rel.parts)
        if parts.intersection(_DEVREVIEW_EXCLUDED_DIRS):
            continue
        if path.is_dir():
            continue
        visited += 1
        tree.append(str(rel))
        if path.name in _DEVREVIEW_CONTEXT_FILES and path.stat().st_size <= 80_000:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            text = _redact_secret_lines(text)
            snippets.append(f"### {rel}\n\n```text\n{text[:max_file_chars]}\n```")

    sections.append("## Repository Snapshot")
    sections.append("")
    sections.append("### File Tree (bounded)")
    sections.append("")
    sections.append("```text")
    sections.extend(tree or ["(No non-excluded files found.)"])
    sections.append("```")
    if snippets:
        sections.append("")
        sections.append("### Selected Manifest/README Snippets")
        sections.append("")
        sections.extend(snippets[:8])
    return "\n".join(sections)


def _redact_secret_lines(text: str) -> str:
    lines: list[str] = []
    for line in (text or "").splitlines():
        if _SECRET_LINE_RE.search(line):
            lines.append("[REDACTED possible secret/config line]")
        else:
            lines.append(line)
    return "\n".join(lines)


def run_sidecar_review(job: DevRunJob, timeout: int = 600) -> tuple[str, str]:
    """Run an optional OpenSquilla sidecar review for a DevRun job.

    Returns ``(verdict, summary)``. Verdict is PASS, NEEDS_CHANGES, BLOCKED, or
    UNAVAILABLE. The sidecar receives only a bounded packet written under the
    Hermes state dir; it is not asked to inspect the workspace directly.
    """
    command = find_sidecar_command()
    if not command:
        return "UNAVAILABLE", "OpenSquilla sidecar command is not configured on this host."

    packet_dir = devrun_state_dir() / "review_packets"
    _ensure_private_dir(packet_dir)
    packet_path = packet_dir / f"{job.job_id}.md"
    packet_path.write_text(build_review_packet(job), encoding="utf-8")
    _chmod_best_effort(packet_path, 0o600)

    full_command = [
        *command,
        "--task",
        "Review this DevRun packet and return PASS, NEEDS_CHANGES, or BLOCKED.",
        "--topic",
        f"DevRun {job.job_id}",
        "--input",
        str(packet_path),
        "--timeout-ms",
        str(timeout * 1000),
    ]

    try:
        result = subprocess.run(
            full_command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        return "BLOCKED", f"OpenSquilla review failed to start: {exc}"

    output = _sidecar_output_text(result.stdout)
    error = (result.stderr or "").strip()
    combined = "\n".join(part for part in (output, error) if part).strip()
    if result.returncode != 0:
        return "BLOCKED", (combined or f"OpenSquilla exited with {result.returncode}")[:4000]

    verdict = extract_review_verdict(combined)
    return verdict, combined[:4000]


def build_devreview_packet(job: DevRunJob) -> str:
    mode = task_mode(job)
    repo_context = collect_devreview_context(job.repo)
    return "\n".join(
        [
            "# DevReview Packet",
            "",
            f"Job ID: {job.job_id}",
            f"Repo: {job.repo}",
            f"Risk: {job.risk}",
            f"Mode: {mode}",
            "",
            "## Task",
            job.task,
            "",
            "## Scope",
            "- This is a review-only mobile development council request.",
            "- Do not modify files, run commands, or inspect the workspace directly.",
            "- Base the answer only on this packet.",
            "- Treat missing code evidence as missing evidence, not as permission to invent details.",
            "",
            "## Required Lenses",
            "- Requirements: clarify goal, users, scope, non-goals, acceptance criteria.",
            "- Architecture: identify likely components, boundaries, integration risks, and simpler alternatives.",
            "- Testing: define high-value unit/integration/manual checks and regression risks.",
            "- Security: flag secrets, auth, permission, data, destructive action, and deployment risks.",
            "- Final acceptance: say what must be true before implementation can start.",
            "",
            repo_context,
        ]
    )


def run_devreview_worker(job: DevRunJob, worker: str, label: str, packet_path: Path, timeout: int) -> tuple[str, str]:
    command = find_worker_command(worker)
    if not command:
        reason = "OpenSquilla worker command is not configured on this host."
        fallback = _run_devreview_auxiliary_fallback(job, worker, label, packet_path, reason, timeout)
        if fallback:
            return label, fallback
        return label, _devreview_fallback_unavailable_report(job, label, reason)

    full_command = [
        *command,
        "--task",
        f"{label}: review this bounded DevReview packet. Do not modify files or call tools.",
        "--topic",
        f"DevReview {job.job_id} {label}",
        "--input",
        str(packet_path),
        "--timeout-ms",
        str(timeout * 1000),
    ]
    try:
        result = subprocess.run(
            full_command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        return label, f"WORKER_FAILED: {label} failed to start: {exc}"

    output = _sidecar_output_text(result.stdout)
    error = (result.stderr or "").strip()
    combined = "\n".join(part for part in (output, error) if part).strip()
    if result.returncode != 0:
        combined = _devreview_failure_text(combined, result.returncode)
        fallback = _run_devreview_auxiliary_fallback(job, worker, label, packet_path, combined, timeout)
        if fallback:
            return label, fallback
        if _should_devreview_auxiliary_fallback(combined):
            return label, _devreview_fallback_unavailable_report(job, label, combined)
    return label, combined[:5000]


def _should_devreview_auxiliary_fallback(reason: str) -> bool:
    return bool(_REVIEW_FALLBACK_ERROR_RE.search(reason or ""))


def _devreview_fallback_prompt(job: DevRunJob, worker: str, label: str, packet: str, reason: str) -> str:
    return "\n".join(
        [
            "You are a bounded DevReview fallback reviewer for Hermes DevFlow.",
            "",
            "The primary OpenSquilla sidecar worker could not produce a report because the model channel failed.",
            f"Primary failure summary: {_redact_worker_error(reason)[:800]}",
            "",
            "Review only the packet below. Do not call tools. Do not inspect the filesystem. Do not modify files.",
            "Return a concise Simplified Chinese report with:",
            "1. Verdict: PASS, PASS_WITH_RISK, NEEDS_CHANGES, or BLOCKED",
            "2. Findings by severity",
            "3. Missing evidence and assumptions",
            "4. Recommended fixes or decisions",
            "5. Items Codex should verify",
            "",
            f"Worker: {worker}",
            f"Label: {label}",
            f"Job ID: {job.job_id}",
            "",
            "DevReview packet:",
            packet,
        ]
    )


def _devreview_fallback_unavailable_report(job: DevRunJob, label: str, reason: str) -> str:
    safe_reason = _redact_worker_error(reason)[:900] or "reviewer model channel unavailable"
    return "\n\n".join(
        [
            "WORKER_FAILED: DevReview reviewer model channel unavailable.",
            "Verdict: BLOCKED",
            "Severity: BLOCKER",
            f"Worker: {label}",
            f"Job ID: {job.job_id}",
            f"Reason: {safe_reason}",
            "Impact: OpenSquilla/M3 did not produce an independent review, and auxiliary fallback also could not produce a usable review.",
            "Required next action: keep the DevFlow/Kanban route in human/specifier review until at least one reviewer channel returns a non-empty report.",
            "Safety: no repository files were modified by this review fallback path.",
        ]
    )[:5000]


def _run_devreview_auxiliary_fallback(
    job: DevRunJob,
    worker: str,
    label: str,
    packet_path: Path,
    reason: str,
    timeout: int,
) -> str:
    if not _should_devreview_auxiliary_fallback(reason):
        return ""
    try:
        packet = packet_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.info("DevReview fallback skipped for %s: packet read failed: %s", job.job_id, exc)
        return ""
    try:
        from agent.auxiliary_client import call_llm, extract_content_or_reasoning
    except Exception as exc:
        logger.info("DevReview fallback unavailable for %s: %s", job.job_id, exc)
        return ""
    try:
        response = call_llm(
            task="devreview_fallback",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict, read-only independent reviewer. "
                        "You never use tools and never modify files."
                    ),
                },
                {
                    "role": "user",
                    "content": _devreview_fallback_prompt(job, worker, label, packet[:12000], reason),
                },
            ],
            temperature=0.2,
            max_tokens=2500,
            timeout=max(30, min(timeout, 120)),
        )
        text = (extract_content_or_reasoning(response) or "").strip()
    except Exception as exc:
        logger.info("DevReview fallback LLM failed for %s (%s): %s", job.job_id, label, exc)
        return ""
    if not text:
        return ""
    return "\n\n".join(
        [
            f"Fallback reviewer used after OpenSquilla worker channel failure ({label}).",
            text[:4500],
        ]
    )[:5000]


def run_devreview_council(job: DevRunJob, timeout_per_worker: int = 240) -> str:
    packet_dir = devrun_state_dir() / "review_packets"
    _ensure_private_dir(packet_dir)
    packet_path = packet_dir / f"{job.job_id}_devreview.md"
    packet_path.write_text(build_devreview_packet(job), encoding="utf-8")
    _chmod_best_effort(packet_path, 0o600)

    sections: list[str] = []
    failures = 0
    for worker, label in DEVREVIEW_WORKERS:
        _label, report = run_devreview_worker(job, worker, label, packet_path, timeout_per_worker)
        if report.startswith("WORKER_FAILED:"):
            failures += 1
        sections.append(f"## {_label}\n\n{report or '(No report generated.)'}")

    status_line = "all workers completed"
    if failures == len(DEVREVIEW_WORKERS):
        status_line = "all workers failed"
    elif failures:
        status_line = f"{failures} worker(s) failed"

    return "\n\n".join(
        [
            f"# DevReview Council Report: {job.job_id}",
            f"- Repo: {job.repo}",
            f"- Risk: {job.risk}",
            f"- Mode: {task_mode(job)}",
            f"- Worker status: {status_line}",
            "",
            *sections,
        ]
    )[:16000]


def run_devflow_preflight(job: DevRunJob, timeout: int = 35) -> str:
    """Run a short, non-blocking-enough DevFlow review before Kanban triage."""
    packet_dir = devrun_state_dir() / "review_packets"
    _ensure_private_dir(packet_dir)
    packet_path = packet_dir / f"{job.job_id}_devflow.md"
    packet_path.write_text(build_devreview_packet(job), encoding="utf-8")
    _chmod_best_effort(packet_path, 0o600)

    label, report = run_devreview_worker(job, "review", "DevFlow 快速门禁审查", packet_path, timeout)
    failed = report.startswith("WORKER_FAILED:")
    status_line = "all workers failed" if failed else "all workers completed"
    return "\n\n".join(
        [
            f"# DevFlow Preflight Report: {job.job_id}",
            f"- Repo: {job.repo}",
            f"- Risk: {job.risk}",
            f"- Mode: {task_mode(job)}",
            f"- Worker status: {status_line}",
            "",
            f"## {label}",
            "",
            report or "(No report generated.)",
        ]
    )[:6000]


def devreview_report_failed(report: str) -> bool:
    return "Worker status: all workers failed" in (report or "")


def _devreview_failure_text(combined: str, returncode: int) -> str:
    readable = _extract_worker_error(combined)
    if readable:
        return f"WORKER_FAILED: {readable}"
    return f"WORKER_FAILED: OpenSquilla worker exited with {returncode} before producing a report."


def _extract_worker_error(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped:
        return ""
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return _redact_worker_error(stripped)[:1200]

    candidates: list[str] = []
    for key in ("error", "errorMessage", "stderrTail", "stdoutTail"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())

    receipt_path = payload.get("receiptPath")
    if isinstance(receipt_path, str) and receipt_path:
        try:
            receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8", errors="replace"))
            candidates.extend(_receipt_error_messages(receipt))
        except OSError:
            pass
        except json.JSONDecodeError:
            pass

    for candidate in candidates:
        redacted = _redact_worker_error(candidate)
        if redacted:
            return redacted[:1200]
    return ""


def _receipt_error_messages(receipt: dict[str, Any]) -> list[str]:
    messages: list[str] = []
    events = receipt.get("events")
    if isinstance(events, dict):
        raw_events = events.get("events")
        if isinstance(raw_events, list):
            for event in raw_events:
                payload = event.get("payload") if isinstance(event, dict) else None
                if not isinstance(payload, dict):
                    continue
                for key in ("error_message", "terminal_message", "message"):
                    value = payload.get(key)
                    if isinstance(value, str) and value.strip():
                        messages.append(value.strip())
    return messages


def _redact_worker_error(text: str) -> str:
    text = re.sub(r"(api[_-]?key|token|secret|password)=\S+", r"\1=[REDACTED]", text, flags=re.IGNORECASE)
    text = re.sub(r"sk-[A-Za-z0-9_-]+", "sk-[REDACTED]", text)
    return text.strip()


def _sidecar_output_text(stdout: str | None) -> str:
    """Return the human review report from a worker stdout payload when present."""
    raw = (stdout or "").strip()
    if not raw:
        return ""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw

    output_path = payload.get("outputPath")
    if isinstance(output_path, str) and output_path:
        try:
            report_path = Path(output_path).expanduser().resolve()
            if report_path.is_file():
                return report_path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            pass
    return raw


def extract_review_verdict(text: str) -> str:
    upper = (text or "").upper()
    lines = upper.splitlines()
    for line in lines:
        value = _extract_explicit_verdict_line(line)
        if value:
            return value
    for index, line in enumerate(lines):
        heading = re.sub(r"^[\s#>*`_\-\d.]+", "", line).strip(":* \t")
        if heading == "VERDICT" or heading.endswith(" VERDICT") or heading in {"结论", "裁决"} or heading.endswith(" 结论"):
            window = "\n".join(lines[index + 1 : index + 8])
            value = _extract_verdict_token(window)
            if value:
                return value
    explicit_match = re.search(r"(VERDICT|结论|裁决)[\s\S]{0,160}?(PASS_WITH_RISK|NEEDS_CHANGES|NEEDS CHANGES|BLOCKED|PASS)\b", upper)
    if explicit_match:
        return _normalize_verdict_token(explicit_match.group(2))
    value = _extract_verdict_token(upper[:400])
    if value:
        return value
    if "BLOCKED" in upper:
        return "BLOCKED"
    if "NEEDS_CHANGES" in upper or "NEEDS CHANGES" in upper:
        return "NEEDS_CHANGES"
    if "PASS" in upper:
        return "PASS"
    return "NEEDS_CHANGES"


def _extract_verdict_token(text: str) -> str:
    match = re.search(r"\b(PASS_WITH_RISK|NEEDS_CHANGES|NEEDS CHANGES|BLOCKED|PASS)\b", text or "")
    if not match:
        return ""
    return _normalize_verdict_token(match.group(1))


def _extract_explicit_verdict_line(text: str) -> str:
    match = re.search(r"\bVERDICT\s*[:：]\s*[*_` ]*(PASS_WITH_RISK|NEEDS_CHANGES|NEEDS CHANGES|BLOCKED|PASS)\b", text or "")
    if not match:
        return ""
    return _normalize_verdict_token(match.group(1))


def _normalize_verdict_token(value: str) -> str:
    if value == "NEEDS CHANGES":
        return "NEEDS_CHANGES"
    if value == "PASS_WITH_RISK":
        return "PASS"
    return value


def collect_repo_changes(repo: str | Path) -> tuple[list[str], str]:
    """Return git-changed files for a repo without reading file contents."""
    command = ["git", "-C", str(repo), "status", "--porcelain=v1"]
    try:
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
            check=False,
        )
    except Exception as exc:
        return [], f"git status unavailable: {exc}"

    if result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        return [], (message or f"git status exited with {result.returncode}")[:600]

    changed: list[str] = []
    for line in (result.stdout or "").splitlines():
        if len(line) < 4:
            continue
        path = line[3:].strip()
        if " -> " in path:
            path = path.rsplit(" -> ", 1)[-1].strip()
        if path and path not in changed:
            changed.append(path)
    return changed, "git status --porcelain=v1"


def record_repo_baseline(job: DevRunJob) -> DevRunJob:
    files, summary = collect_repo_changes(job.repo)
    encoded = json.dumps(files, ensure_ascii=False, separators=(",", ":"))
    job.commands = [item for item in job.commands if not item.startswith("baseline git status: ")]
    job.commands.append(f"baseline git status: {encoded}")
    if summary and not job.test_results:
        job.test_results = summary
    return job


def baseline_changed_files(job: DevRunJob) -> set[str]:
    for item in reversed(job.commands):
        if not item.startswith("baseline git status: "):
            continue
        raw = item.split(": ", 1)[1]
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return set()
        if isinstance(parsed, list):
            return {str(value) for value in parsed}
    return set()


def collect_new_repo_changes(job: DevRunJob) -> tuple[list[str], str]:
    files, summary = collect_repo_changes(job.repo)
    baseline = baseline_changed_files(job)
    new_files = [path for path in files if path not in baseline]
    if baseline and not new_files:
        summary = f"{summary}; no new DevRun changes since baseline"
    return new_files, summary


def build_execution_prompt(job: DevRunJob) -> str:
    mode = task_mode(job)
    mode_rules = (
        "- This task is READ-ONLY / PLAN-ONLY. Do not write, edit, rename, delete, format, or generate files.\n"
        "- Do not run tests, package managers, formatters, build tools, or commands that can modify the workspace.\n"
        "- You may inspect files and run read-only status/listing commands only.\n"
        "- Deliver the result in chat as recommendations, findings, or a plan. Do not produce a diff."
        if mode == "read_only"
        else
        "- Low-risk code edits may be applied when they are small, reversible, and directly tied to the task.\n"
        "- Use existing tests or the smallest meaningful verification."
    )
    return f"""You are Hermes DevRun, a cautious mobile-triggered development executor.

Job ID: {job.job_id}
Workspace: {job.repo}
Risk: {job.risk}
Mode: {mode}

Task:
{job.task}

Operating rules:
- Work only inside this workspace unless the user explicitly approves otherwise: {job.repo}
- Start by inspecting the relevant files and confirming scope.
{mode_rules}
- Before any high-risk action, ask for approval instead of doing it. High-risk actions include service restarts, deleting files, database migrations, editing secrets or .env/config credentials, git commit/push/reset/rebase, broad multi-file rewrites, Hermes/OpenClaw/Kanban core config changes, and cross-workspace writes.
- Do not reveal secrets. Redact tokens, keys, cookies, and .env content.
- At the end, report: changed files, commands/tests run, result, residual risk, and whether user approval is still needed.
"""


def job_display_label(job: DevRunJob) -> str:
    if job.review_verdict == "DEVREVIEW":
        return "DevReview"
    if job.review_verdict == "DEVFLOW" or (job.review_summary or "").startswith("DevFlow route:"):
        return "DevFlow"
    return "DevRun"


def summarize_job(job: DevRunJob) -> str:
    label = job_display_label(job)
    if label == "DevFlow":
        return render_devflow_status_card(job)
    review_limit = 2200 if label in {"DevReview", "DevFlow"} else 600
    lines = [
        f"{label} {job.job_id}",
        f"Status: {job.status}",
        f"Risk: {job.risk}",
        f"Repo: {job.repo}",
        f"Task: {job.task}",
    ]
    if job.review_verdict:
        if job.review_verdict == "DEVFLOW":
            lines.append("Flow: DEVFLOW")
        else:
            lines.append(f"Review: {job.review_verdict}")
    if job.review_summary:
        lines.append("Review summary: " + job.review_summary[:review_limit])
    if job.error:
        lines.append(f"Error: {job.error}")
    if job.changed_files:
        lines.append("Changed files: " + ", ".join(job.changed_files[:10]))
    if job.test_results:
        lines.append("Tests: " + job.test_results[:600])
    if job.status == "done":
        if job.review_verdict == "DEVREVIEW":
            lines.append("Note: DevReview is review-only; no files were changed by this command.")
        elif job.review_verdict == "DEVFLOW":
            lines.append("Note: DevFlow finished its routing step; check Tests/commands for DevRun or Kanban details.")
        elif label == "DevFlow":
            lines.append("Note: DevFlow routed this task to the DevRun executor; done means the scoped execution finished.")
        else:
            lines.append("Note: done means the background DevRun flow finished; check the assistant report for test/pass details.")
    return "\n".join(lines)


def render_jobs(jobs: Iterable[DevRunJob]) -> str:
    rows = list(jobs)
    if not rows:
        return "No DevRun/DevReview/DevFlow jobs found."
    return "\n".join(
        f"{job_display_label(job)} {job.job_id} | {job.status} | {job.risk} | {job.task[:80]}"
        for job in rows
    )
