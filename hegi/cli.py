"""HEGI command line interface."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .config import default_config_path, load_config, validate_config
from .memory import DraftGate, MCPMemoryBackend, MemoryEvaluator
from .pipeline import HegiPipeline, episode_from_dict, minutes_from_dict
from .state import StateStore

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore[assignment]


def _json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _pipeline(args: argparse.Namespace, *, state: StateStore | None = None) -> HegiPipeline:
    return HegiPipeline(load_config(args.config), state=state)


def cmd_doctor(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
    except Exception as exc:
        print(f"FAIL config: {exc}")
        return 2
    errors = validate_config(config, require_runtime=True)
    checks = {
        "config": str(config.path),
        "enabled": config.enabled,
        "chat_id": config.chat_id,
        "chat_id_configured": bool(config.chat_id),
        "telegram_enabled": bool(config.section("telegram").get("enabled")),
        "curator_env": str(config.curator_env),
        "professor_user_ids": [
            str(item)
            for item in config.section("memory").get("professor_user_ids", [])
        ],
        "agents": [
            {"name": agent.name, "db_path": str(agent.db_path)}
            for agent in config.agents
        ],
        "state_parent_writable": os.access(config.state_db.parent, os.W_OK)
        if config.state_db.parent.exists()
        else os.access(config.state_db.parent.parent, os.W_OK),
        "agent_databases": {
            agent.name: agent.db_path.is_file() for agent in config.agents
        },
        "telegram_env_present": config.curator_env.is_file(),
        "auto_commit": False,
        "approval_required": True,
        "errors": errors,
    }
    _json(checks)
    return 1 if errors else 0


def cmd_collect(args: argparse.Namespace) -> int:
    messages = _pipeline(args).collect(since=args.since)
    _json(
        {
            "count": len(messages),
            "messages": [
                {
                    "agent": item.source_agent,
                    "message_id": item.message_id,
                    "role": item.role,
                    "timestamp": item.timestamp,
                }
                for item in messages
            ],
        }
    )
    return 0


def cmd_run_once(args: argparse.Namespace) -> int:
    if args.send:
        results = _pipeline(args).run_once(dry_run=False)
    else:
        with tempfile.TemporaryDirectory(prefix="hegi-dry-run-") as temporary:
            state = StateStore(Path(temporary) / "state.db")
            results = _pipeline(args, state=state).run_once(dry_run=True)
    _json(results)
    return 0 if all(item.get("status") != "failed" for item in results) else 3


def cmd_analyze(args: argparse.Namespace) -> int:
    pipeline = _pipeline(args)
    row = pipeline.state.episode_by_id(args.meeting_id)
    if row is None:
        print(f"meeting_id를 찾을 수 없습니다: {args.meeting_id}")
        return 2
    episode = episode_from_dict(json.loads(row["episode_json"]))
    minutes = pipeline.analyze_episode(episode)
    _json(minutes.to_dict())
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    state = StateStore(config.state_db)
    with state.connect() as connection:
        counts = {
            row["status"]: row["count"]
            for row in connection.execute(
                "SELECT status, COUNT(*) AS count FROM processed_episodes GROUP BY status"
            )
        }
        pending = connection.execute(
            "SELECT COUNT(*) FROM message_buffer WHERE consumed=0"
        ).fetchone()[0]
        dead_letters = connection.execute(
            "SELECT COUNT(*) FROM dead_letter"
        ).fetchone()[0]
    _json(
        {
            "enabled": config.enabled,
            "state_db": str(config.state_db),
            "episode_counts": counts,
            "pending_messages": pending,
            "dead_letters": dead_letters,
            "approval_jobs": state.approval_job_counts(),
            "daemon": _daemon_status(config.state_db.parent),
        }
    )
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    pipeline = _pipeline(args)
    row = pipeline.state.episode_by_id(args.meeting_id)
    if row is None or not row.get("minutes_json"):
        print("저장된 회의록이 없습니다.")
        return 2
    from .notify import TelegramReporter, load_env_value

    minutes = minutes_from_dict(json.loads(row["minutes_json"]))
    token = load_env_value(pipeline.config.curator_env, "TELEGRAM_BOT_TOKEN")
    ids = TelegramReporter(
        pipeline.state, token=token, chat_id=pipeline.config.chat_id
    ).send(minutes, dry_run=not args.send)
    _json({"meeting_id": args.meeting_id, "message_ids": ids, "dry_run": not args.send})
    return 0


def cmd_approve(args: argparse.Namespace) -> int:
    pipeline = _pipeline(args)
    memory_cfg = pipeline.config.section("memory")
    backend = MCPMemoryBackend(
        read_server=str(memory_cfg.get("read_server", "memory-forest-read")),
        search_tool=str(memory_cfg.get("search_tool", "")),
        draft_server=str(memory_cfg.get("draft_server", "")),
        draft_tool=str(memory_cfg.get("draft_tool", "")),
    )
    gate = DraftGate(
        pipeline.state,
        backend,
        professor_user_ids=[str(item) for item in memory_cfg.get("professor_user_ids", [])],
    )
    command = gate.approve(
        meeting_id=args.meeting_id,
        text=args.text,
        user_id=args.user_id,
        platform_message_id=args.message_id,
    )
    result: dict[str, Any] = {"approved_command": command}
    if command in {"remember", "draft", "merge"}:
        row = pipeline.state.episode_by_id(args.meeting_id)
        if row is None or not row.get("minutes_json"):
            raise ValueError("승인 대상 회의록이 없습니다.")
        minutes = minutes_from_dict(json.loads(row["minutes_json"]))
        evaluation = MemoryEvaluator(backend).evaluate(minutes)
        project = args.project or str(memory_cfg.get("default_project", "")).strip()
        if not project:
            raise ValueError("memory.default_project 또는 --project가 필요합니다.")
        result["draft"] = gate.create_draft_after_recheck(
            minutes, evaluation, project=project
        )
        result["commit"] = "not_performed"
    _json(result)
    return 0


def _atomic_runtime_file(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def _daemon_status(state_dir: Path) -> dict[str, Any]:
    pidfile = state_dir / "daemon.pid"
    readyfile = state_dir / "daemon.ready"
    pid: int | None = None
    alive = False
    try:
        pid = int(pidfile.read_text(encoding="ascii").strip())
        os.kill(pid, 0)
        alive = True
    except (OSError, ValueError):
        pass
    ready: dict[str, Any] = {}
    if readyfile.is_file():
        try:
            ready = json.loads(readyfile.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            ready = {}
    return {"pid": pid, "alive": alive, "ready": ready}


def _acquire_daemon_lock(state_dir: Path):
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / "daemon.lock"
    if fcntl is None:
        try:
            descriptor = os.open(
                lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600
            )
        except FileExistsError:
            try:
                pid = int(lock_path.read_text(encoding="ascii").strip())
                os.kill(pid, 0)
            except (OSError, ValueError):
                lock_path.unlink(missing_ok=True)
                return _acquire_daemon_lock(state_dir)
            raise RuntimeError(f"HEGI daemon이 이미 실행 중입니다: pid={pid}")
        stream = os.fdopen(descriptor, "w+", encoding="ascii")
        stream.write(str(os.getpid()))
        stream.flush()
        (state_dir / "daemon.pid").write_text(str(os.getpid()), encoding="ascii")
        return stream
    stream = lock_path.open("a+", encoding="ascii")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.seek(0)
        pid = stream.read().strip() or "unknown"
        stream.close()
        raise RuntimeError(f"HEGI daemon이 이미 실행 중입니다: pid={pid}")
    stream.seek(0)
    stream.truncate()
    stream.write(str(os.getpid()))
    stream.flush()
    os.fsync(stream.fileno())
    (state_dir / "daemon.pid").write_text(str(os.getpid()), encoding="ascii")
    return stream


def cmd_daemon(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    errors = validate_config(config, require_runtime=True)
    if errors:
        raise ValueError("; ".join(errors))
    state_dir = config.state_db.parent
    pidfile = state_dir / "daemon.pid"
    readyfile = state_dir / "daemon.ready"
    lock_stream = _acquire_daemon_lock(state_dir)
    running = True

    def stop(_signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        pipeline = HegiPipeline(config)
        _atomic_runtime_file(
            readyfile,
            {
                "pid": os.getpid(),
                "started_at": time.time(),
                "send": bool(args.send),
                "config": str(config.path),
            },
        )
        while running:
            try:
                pipeline.run_once(dry_run=not args.send)
            except Exception as exc:
                pipeline.state.add_dead_letter("daemon", {}, str(exc))
            if args.send:
                try:
                    from .approval import process_pending_approvals

                    process_pending_approvals(config)
                except Exception as exc:
                    pipeline.state.add_dead_letter("approval_daemon", {}, str(exc))
            deadline = time.monotonic() + int(
                config.section("daemon").get("poll_seconds", 60)
            )
            while running and time.monotonic() < deadline:
                time.sleep(min(1, deadline - time.monotonic()))
    finally:
        readyfile.unlink(missing_ok=True)
        pidfile.unlink(missing_ok=True)
        if fcntl is not None:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
        lock_stream.close()
        if fcntl is None:
            (state_dir / "daemon.lock").unlink(missing_ok=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HEGI v2 AI Research Secretary")
    parser.add_argument("--config", default=str(default_config_path()))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor").set_defaults(func=cmd_doctor)
    collect = subparsers.add_parser("collect")
    collect.add_argument("--chat-id", help="설정 chat_id와 일치해야 함")
    collect.add_argument("--since", type=float)
    collect.set_defaults(func=cmd_collect)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--meeting-id", required=True)
    analyze.set_defaults(func=cmd_analyze)
    run_once = subparsers.add_parser("run-once")
    run_once.add_argument("--send", action="store_true", help="실제 Telegram 전송 허용")
    run_once.set_defaults(func=cmd_run_once)
    daemon = subparsers.add_parser("daemon")
    daemon.add_argument("--send", action="store_true", help="실제 Telegram 전송 허용")
    daemon.set_defaults(func=cmd_daemon)
    subparsers.add_parser("status").set_defaults(func=cmd_status)
    replay = subparsers.add_parser("replay")
    replay.add_argument("--meeting-id", required=True)
    replay.add_argument("--send", action="store_true")
    replay.set_defaults(func=cmd_replay)
    approve = subparsers.add_parser("approve")
    approve.add_argument("--meeting-id", required=True)
    approve.add_argument("--text", required=True)
    approve.add_argument("--user-id", required=True)
    approve.add_argument("--message-id")
    approve.add_argument("--project")
    approve.set_defaults(func=cmd_approve)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    if config_path.parent.name == "hegi":
        os.environ["HERMES_HOME"] = str(config_path.parent.parent)
    if getattr(args, "chat_id", None):
        config = load_config(args.config)
        if str(args.chat_id) != config.chat_id:
            parser.error("--chat-id는 설정된 telegram.chat_id와 일치해야 합니다.")
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"HEGI 오류: {exc}", file=sys.stderr)
        return 1
