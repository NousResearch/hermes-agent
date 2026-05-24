"""File-backed Work Case CLI for durable Hermes work coordination."""

from __future__ import annotations

import argparse
import json
import re
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from hermes_constants import get_hermes_home

msvcrt = None
try:
    import fcntl
except ImportError:  # pragma: no cover - platform-specific fallback
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        pass


VALID_KINDS = {"debug", "feature", "ci-repair", "review", "research", "ops"}
VALID_SURFACES = {"backend", "frontend-ui", "macos", "infra-ci", "e2e-mixed", "docs", "other"}
VALID_PHASES = {
    "intake",
    "repro",
    "diagnose",
    "plan",
    "implement",
    "verify",
    "review",
    "closed",
    "escalated",
}
CASE_ID_RE = re.compile(r"^CASE-(\d{4})-(\d{5})$")
LADDER_ID_RE = re.compile(r"^[a-z][a-z0-9-]*$")
BUILTIN_LADDER_IDS = frozenset({"generic-debug", "generic-ci-repair"})
DEFAULT_LADDER_BY_KIND = {
    "debug": "generic-debug",
    "ci-repair": "generic-ci-repair",
}
VERIFICATION_ARTIFACT = Path("artifacts") / "verification.json"
MAX_OPEN_RETRIES = 5

PHASE_TRANSITIONS = {
    "intake": {"repro", "diagnose", "plan", "review", "verify", "escalated", "closed"},
    "repro": {"diagnose", "implement", "verify", "escalated", "closed"},
    "diagnose": {"implement", "verify", "escalated", "closed"},
    "plan": {"implement", "review", "verify", "escalated", "closed"},
    "implement": {"verify", "review", "diagnose", "escalated", "closed"},
    "verify": {"implement", "review", "diagnose", "escalated", "closed"},
    "review": {"implement", "verify", "escalated", "closed"},
    "escalated": {"repro", "diagnose", "plan", "implement", "verify", "review", "closed"},
    "closed": set(),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def cases_root() -> Path:
    return get_hermes_home() / "cases"


def case_dir(case_id: str) -> Path:
    _validate_case_id(case_id)
    return cases_root() / case_id


def _validate_case_id(case_id: str) -> None:
    if not CASE_ID_RE.match(case_id):
        raise ValueError(f"invalid case id {case_id!r}; expected CASE-YYYY-NNNNN")


def _validate_choice(name: str, value: str, allowed: set[str]) -> None:
    if value not in allowed:
        expected = ", ".join(sorted(allowed))
        raise ValueError(f"invalid {name} {value!r}; expected one of: {expected}")


def _validate_ladder_id(ladder_id: str) -> None:
    if not LADDER_ID_RE.match(ladder_id):
        raise ValueError(
            f"invalid ladder id {ladder_id!r}; expected lowercase letters, digits, and hyphens"
        )
    if ladder_id not in BUILTIN_LADDER_IDS:
        raise ValueError(
            f"unknown ladder id {ladder_id!r}; known built-in ladders: "
            f"{', '.join(sorted(BUILTIN_LADDER_IDS))}"
        )


def _default_ladder_for_kind(kind: str) -> str | None:
    return DEFAULT_LADDER_BY_KIND.get(kind)


def _resolve_ladder_id(kind: str, ladder_id: str | None) -> str | None:
    if ladder_id is not None:
        _validate_ladder_id(ladder_id)
        return ladder_id
    return _default_ladder_for_kind(kind)


@contextmanager
def _case_allocate_lock():
    """Serialize case ID allocation across concurrent callers."""
    root = cases_root()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".allocate.lock"

    if fcntl is None and msvcrt is None:
        yield
        return

    if msvcrt and (not lock_path.exists() or lock_path.stat().st_size == 0):
        lock_path.write_text(" ", encoding="utf-8")

    fd = open(lock_path, "r+" if msvcrt else "a+", encoding="utf-8")
    try:
        if fcntl:
            fcntl.flock(fd, fcntl.LOCK_EX)
        else:
            fd.seek(0)
            msvcrt.locking(fd.fileno(), msvcrt.LK_LOCK, 1)
        yield
    finally:
        if fcntl:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        elif msvcrt:
            try:
                fd.seek(0)
                msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        fd.close()


def _scan_max_seq_for_year(year: int) -> int:
    max_seen = 0
    root = cases_root()
    if not root.exists():
        return 0
    for path in root.iterdir():
        if not path.is_dir():
            continue
        match = CASE_ID_RE.match(path.name)
        if not match or int(match.group(1)) != year:
            continue
        max_seen = max(max_seen, int(match.group(2)))
    return max_seen


def _next_case_id(now: datetime | None = None) -> str:
    year = (now or datetime.now(timezone.utc)).year
    root = cases_root()
    root.mkdir(parents=True, exist_ok=True)
    counter_path = root / ".counters.yaml"

    with _case_allocate_lock():
        counters = _read_yaml(counter_path)
        stored = int(counters.get(str(year), 0) or 0)
        next_seq = max(stored, _scan_max_seq_for_year(year)) + 1
        counters[str(year)] = next_seq
        _write_yaml(counter_path, counters)

    return f"CASE-{year}-{next_seq:05d}"


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _append_event(case_path: Path, event_type: str, message: str, **extra: Any) -> None:
    event = {"ts": utc_now(), "type": event_type, "message": message}
    event.update({k: v for k, v in extra.items() if v is not None})
    events_path = case_path / "events.jsonl"
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def _load_case(case_id: str) -> tuple[Path, dict[str, Any]]:
    path = case_dir(case_id)
    case_file = path / "case.yaml"
    if not case_file.exists():
        raise FileNotFoundError(f"case not found: {case_id}")
    return path, _read_yaml(case_file)


def _verification_artifact_exists(case_path: Path) -> bool:
    return (case_path / VERIFICATION_ARTIFACT).is_file()


def _assert_close_allowed(path: Path, data: dict[str, Any], *, force_close: bool) -> None:
    verification = data.get("verification")
    if not isinstance(verification, dict):
        verification = {}
    if not verification.get("required_before_close", True) or force_close:
        return
    if _verification_artifact_exists(path):
        return
    raise ValueError(
        "cannot close case without verification evidence; attach "
        f"{VERIFICATION_ARTIFACT.as_posix()} or pass --force"
    )


def open_case(
    *,
    title: str,
    kind: str,
    surface: str,
    risk: str = "medium",
    ladder_id: str | None = None,
    case_id: str | None = None,
) -> dict[str, Any]:
    _validate_choice("kind", kind, VALID_KINDS)
    _validate_choice("surface", surface, VALID_SURFACES)
    ladder = _resolve_ladder_id(kind, ladder_id)

    last_error: Exception | None = None
    for _attempt in range(MAX_OPEN_RETRIES):
        candidate_id = case_id or _next_case_id()
        try:
            return _create_case(
                case_id=candidate_id,
                title=title,
                kind=kind,
                surface=surface,
                risk=risk,
                ladder_id=ladder,
            )
        except FileExistsError as exc:
            if case_id is not None:
                raise
            last_error = exc
            continue

    if last_error is not None:
        raise last_error
    raise RuntimeError("failed to allocate a unique Work Case ID")


def _create_case(
    *,
    case_id: str,
    title: str,
    kind: str,
    surface: str,
    risk: str,
    ladder_id: str | None,
) -> dict[str, Any]:
    _validate_case_id(case_id)
    path = case_dir(case_id)
    if path.exists():
        raise FileExistsError(f"case already exists: {case_id}")

    now = utc_now()
    case_data: dict[str, Any] = {
        "case_id": case_id,
        "title": title,
        "kind": kind,
        "surface": surface,
        "status": "open",
        "phase": "intake",
        "created_at": now,
        "updated_at": now,
        "owner_profile": "dev",
        "source": {
            "user": "Felipe",
            "kanban_task_id": None,
            "linear_issue": None,
            "source_links": [],
        },
        "classification": {
            "risk": risk,
            "independence": "single worker",
            "billing_pool": None,
        },
        "verification": {
            "ladder_id": ladder_id,
            "required_before_close": True,
        },
        "artifacts": [],
    }
    carry_forward = {
        "case_id": case_id,
        "rejected_hypotheses": [],
        "confirmed_hypotheses": [],
        "commands_already_run": [],
        "blockers": [],
        "trace_ids": [],
        "reroute_notes": [],
    }

    (path / "artifacts" / "traces").mkdir(parents=True, exist_ok=True)
    _write_yaml(path / "case.yaml", case_data)
    _write_yaml(path / "carry_forward.yaml", carry_forward)
    _append_event(path, "case_opened", f"Opened Work Case: {title}", phase="intake")
    return case_data


def show_case(case_id: str) -> dict[str, Any]:
    path, data = _load_case(case_id)
    data = dict(data)
    data["_path"] = str(path)
    return data


def advance_case(
    case_id: str,
    phase: str,
    message: str | None = None,
    *,
    force_close: bool = False,
) -> dict[str, Any]:
    _validate_choice("phase", phase, VALID_PHASES)
    path, data = _load_case(case_id)
    current = str(data.get("phase", ""))
    if phase not in PHASE_TRANSITIONS.get(current, set()):
        raise ValueError(f"cannot advance {case_id} from {current!r} to {phase!r}")

    if phase == "closed":
        _assert_close_allowed(path, data, force_close=force_close)

    data["phase"] = phase
    data["updated_at"] = utc_now()
    if phase == "closed":
        data["status"] = "closed"
    elif phase == "escalated":
        data["status"] = "blocked"
    elif data.get("status") == "blocked":
        data["status"] = "open"

    _write_yaml(path / "case.yaml", data)
    _append_event(
        path,
        "phase_advanced",
        message or f"Advanced case from {current} to {phase}",
        previous_phase=current,
        phase=phase,
    )
    return data


def attach_artifact(
    case_id: str,
    artifact: str,
    *,
    kind: str = "reference",
    description: str = "",
) -> dict[str, Any]:
    path, data = _load_case(case_id)
    entry = {
        "kind": kind,
        "path": artifact,
        "description": description,
        "created_at": utc_now(),
    }
    artifacts = data.setdefault("artifacts", [])
    if not isinstance(artifacts, list):
        raise ValueError("case.yaml artifacts must be a list")
    artifacts.append(entry)
    data["updated_at"] = utc_now()
    _write_yaml(path / "case.yaml", data)
    _append_event(path, "artifact_attached", f"Attached artifact: {artifact}", artifact=entry)
    return entry


def add_event(case_id: str, message: str, event_type: str = "note") -> None:
    path, _data = _load_case(case_id)
    _append_event(path, event_type, message)


def list_cases() -> list[dict[str, Any]]:
    root = cases_root()
    if not root.exists():
        return []
    cases: list[dict[str, Any]] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or not CASE_ID_RE.match(path.name):
            continue
        case_file = path / "case.yaml"
        if not case_file.exists():
            continue
        data = _read_yaml(case_file)
        data["_path"] = str(path)
        cases.append(data)
    return cases


def _print_yaml(data: Any) -> None:
    print(yaml.safe_dump(data, sort_keys=False).rstrip())


def build_parser(parent_subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = parent_subparsers.add_parser(
        "case",
        help="Manage durable Work Cases",
        description="Create and update file-backed Work Cases under HERMES_HOME.",
    )
    sub = parser.add_subparsers(dest="case_action")

    p_open = sub.add_parser("open", help="Open a new Work Case")
    p_open.add_argument("--title", required=True, help="Short case title")
    p_open.add_argument("--kind", required=True, choices=sorted(VALID_KINDS))
    p_open.add_argument("--surface", required=True, choices=sorted(VALID_SURFACES))
    p_open.add_argument("--risk", default="medium", choices=("trivial", "low", "medium", "high"))
    p_open.add_argument("--ladder", dest="ladder_id", default=None, help="Verification ladder ID")
    p_open.add_argument("--case-id", default=None, help="Explicit CASE-YYYY-NNNNN ID")
    p_open.add_argument("--json", action="store_true")

    p_show = sub.add_parser("show", help="Show a Work Case")
    p_show.add_argument("case_id")
    p_show.add_argument("--json", action="store_true")

    p_list = sub.add_parser("list", aliases=["ls"], help="List Work Cases")
    p_list.add_argument("--json", action="store_true")

    p_advance = sub.add_parser("advance", help="Advance a case to another phase")
    p_advance.add_argument("case_id")
    p_advance.add_argument("--phase", required=True, choices=sorted(VALID_PHASES))
    p_advance.add_argument("--message", default=None)
    p_advance.add_argument(
        "--force",
        action="store_true",
        help="Allow closing without verification evidence",
    )
    p_advance.add_argument("--json", action="store_true")

    p_attach = sub.add_parser("attach", help="Attach an artifact reference to a case")
    p_attach.add_argument("case_id")
    p_attach.add_argument("--artifact", required=True, help="Artifact path or external reference")
    p_attach.add_argument("--kind", default="reference")
    p_attach.add_argument("--description", default="")
    p_attach.add_argument("--json", action="store_true")

    p_event = sub.add_parser("event", help="Append an event to a case")
    p_event.add_argument("case_id")
    p_event.add_argument("--message", required=True)
    p_event.add_argument("--type", dest="event_type", default="note")

    parser.set_defaults(_case_parser=parser)
    return parser


def case_command(args: argparse.Namespace) -> int:
    action = getattr(args, "case_action", None)
    if not action:
        parser = getattr(args, "_case_parser", None)
        if parser is not None:
            parser.print_help()
        else:
            print("usage: hermes case <action> [options]", file=sys.stderr)
        return 0

    try:
        if action == "open":
            data = open_case(
                title=args.title,
                kind=args.kind,
                surface=args.surface,
                risk=args.risk,
                ladder_id=args.ladder_id,
                case_id=args.case_id,
            )
            if args.json:
                print(json.dumps(data, indent=2, sort_keys=True))
            else:
                print(f"Opened {data['case_id']} ({data['kind']}, {data['surface']})")
                print(f"Path: {case_dir(data['case_id'])}")
            return 0

        if action == "show":
            data = show_case(args.case_id)
            print(json.dumps(data, indent=2, sort_keys=True) if args.json else yaml.safe_dump(data, sort_keys=False).rstrip())
            return 0

        if action in {"list", "ls"}:
            data = list_cases()
            if args.json:
                print(json.dumps(data, indent=2, sort_keys=True))
            else:
                if not data:
                    print("No Work Cases found.")
                for item in data:
                    print(
                        f"{item.get('case_id')}  {item.get('status')}  "
                        f"{item.get('phase')}  {item.get('kind')}  {item.get('title')}"
                    )
            return 0

        if action == "advance":
            data = advance_case(
                args.case_id,
                args.phase,
                args.message,
                force_close=getattr(args, "force", False),
            )
            if args.json:
                print(json.dumps(data, indent=2, sort_keys=True))
            else:
                print(f"Advanced {data['case_id']} to {data['phase']} ({data['status']})")
            return 0

        if action == "attach":
            entry = attach_artifact(
                args.case_id,
                args.artifact,
                kind=args.kind,
                description=args.description,
            )
            if args.json:
                print(json.dumps(entry, indent=2, sort_keys=True))
            else:
                print(f"Attached artifact to {args.case_id}: {entry['path']}")
            return 0

        if action == "event":
            add_event(args.case_id, args.message, args.event_type)
            print(f"Added event to {args.case_id}")
            return 0

    except (FileExistsError, FileNotFoundError, ValueError, OSError) as exc:
        print(f"case: {exc}", file=sys.stderr)
        return 1

    print(f"case: unknown action {action!r}", file=sys.stderr)
    return 2


def run_slash(rest: str) -> str:
    """Execute a `/case ...` string and return captured output."""
    import contextlib
    import io
    import shlex

    tokens = shlex.split(rest) if rest and rest.strip() else []
    wrap = argparse.ArgumentParser(prog="/case-wrap", add_help=False)
    top_sub = wrap.add_subparsers(dest="_top")
    parser = build_parser(top_sub)
    parser.prog = "/case"

    buf_out = io.StringIO()
    buf_err = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            args = parser.parse_args(tokens)
            case_command(args)
    except SystemExit:
        pass
    out = buf_out.getvalue().rstrip()
    err = buf_err.getvalue().rstrip()
    if err and out:
        return f"{out}\n{err}"
    return err if err else (out or "(no output)")
