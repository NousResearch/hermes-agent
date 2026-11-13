"""Advisory-only Jev observations for Kanban routing decisions.

The observer runs after the decomposer has successfully applied its assignees.
It records Jev's independent recommendation beside the actual route and never
changes, approves, blocks, or dispatches work.
"""
from __future__ import annotations

import importlib.util
import json
import os
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


_DEFAULT_LOG = "~/.hermes/logs/kanban_jev_observer.jsonl"


def _load_config() -> dict:
    from hermes_cli.config import load_config_readonly

    cfg = load_config_readonly()
    return cfg if isinstance(cfg, dict) else {}


def _observer_config() -> dict:
    cfg = _load_config()
    kanban = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    observer = kanban.get("jev_observer", {}) if isinstance(kanban, dict) else {}
    return observer if isinstance(observer, dict) else {}


def _build_request(
    candidates: list[dict], roster: list[dict], *, max_candidates: int = 8
) -> tuple[dict, dict]:
    limit = max(1, min(int(max_candidates), 32))
    selected = candidates[:limit]
    profiles = []
    criteria: dict[str, str | None] = {}
    for entry in roster:
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        description = str(entry.get("description") or "")[:1000]
        profiles.append({"name": name, "description": description})
        criteria[name] = description or None
    criteria["ask_user"] = "Routing is ambiguous or required context is missing"

    state = {
        "purpose": "Independent advisory comparison of Kanban routing; do not authorize execution.",
        "available_profiles": profiles,
        "candidate_tasks": [
            {
                "index": index,
                "title": str(item.get("title") or "")[:200],
                "body": str(item.get("body") or "")[:6000],
            }
            for index, item in enumerate(selected)
        ],
    }
    questions: dict[str, dict] = {}
    for index, _item in enumerate(selected):
        questions[f"route_{index}"] = {
            "type": "choice",
            "instructions": f"Which available profile should handle candidate task {index}?",
            "criteria": criteria,
        }
        questions[f"risk_{index}"] = {
            "type": "choice",
            "instructions": f"Classify the operational risk of candidate task {index}.",
            "criteria": {
                "low": "Read-only, local, reversible, or routine work",
                "medium": "Writes or changes with bounded reversible impact",
                "high": "External, destructive, financial, privileged, or hard-to-reverse impact",
            },
        }
        questions[f"approval_{index}"] = {
            "type": "noul",
            "instructions": (
                f"Does candidate task {index} require human approval before execution because it is "
                "external, destructive, financial, privileged, or hard to reverse?"
            ),
        }
        questions[f"complexity_{index}"] = {
            "type": "choice",
            "instructions": f"Classify the reasoning complexity of candidate task {index}.",
            "criteria": {
                "simple": "Clear, mechanical, narrowly scoped work",
                "moderate": "Several steps or some judgment required",
                "complex": "Ambiguous, cross-system, architecture-heavy, or deep reasoning work",
            },
        }
    return state, questions


def _choice(answers: dict, key: str) -> tuple[Any, Any]:
    answer = answers.get(key, {}) if isinstance(answers, dict) else {}
    if not isinstance(answer, dict):
        return None, None
    return answer.get("choice"), answer.get("confidence")


def _summarize_result(task_id: str, candidates: list[dict], result: dict) -> dict:
    answers = result.get("answers", {}) if isinstance(result, dict) else {}
    decisions = []
    for index, item in enumerate(candidates):
        jev_assignee, route_confidence = _choice(answers, f"route_{index}")
        risk, risk_confidence = _choice(answers, f"risk_{index}")
        complexity, complexity_confidence = _choice(answers, f"complexity_{index}")
        approval = answers.get(f"approval_{index}", {}) if isinstance(answers, dict) else {}
        approval_probability = approval.get("noul") if isinstance(approval, dict) else None
        actual_assignee = str(item.get("assignee") or "")
        decisions.append(
            {
                "candidate_index": index,
                "title": str(item.get("title") or "")[:200],
                "actual_assignee": actual_assignee,
                "jev_assignee": jev_assignee,
                "route_agrees": jev_assignee == actual_assignee,
                "route_confidence": route_confidence,
                "risk": risk,
                "risk_confidence": risk_confidence,
                "requires_human_approval_probability": approval_probability,
                "complexity": complexity,
                "complexity_confidence": complexity_confidence,
            }
        )
    usage = result.get("usage", {}) if isinstance(result, dict) else {}
    safe_usage = {
        key: value
        for key, value in usage.items()
        if key in {"input_tokens", "output_tokens"} and type(value) is int and value >= 0
    } if isinstance(usage, dict) else {}
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "status": "ok",
        "mode": "observe",
        "task_id": task_id,
        "model": str(result.get("model") or "")[:128],
        "backend": str(result.get("backend") or "")[:32],
        "latency_ms": result.get("latency_ms") if type(result.get("latency_ms")) in (int, float) else None,
        "usage": safe_usage,
        "decisions": decisions,
        "advisory_only": True,
        "execution_authorized": False,
    }


def _profile_home(profile: str) -> Path:
    from hermes_constants import PROFILE_ID_RE, get_default_hermes_root, named_profile_is_live

    if not PROFILE_ID_RE.fullmatch(profile):
        raise ValueError("invalid Hermes profile name")
    if profile != "milly":
        raise ValueError("configured Jev profile must be milly")
    root = Path(get_default_hermes_root()).resolve(strict=False)
    profiles_root = (root / "profiles").resolve(strict=False)
    candidate = root / "profiles" / profile
    resolved = candidate.resolve(strict=False)
    if resolved.parent != profiles_root or not named_profile_is_live(candidate):
        raise FileNotFoundError("configured Hermes profile is unavailable")
    return resolved


def _profile_resource(home: Path, *parts: str) -> Path:
    """Resolve a profile-owned resource without permitting symlink escape."""
    resolved_home = home.resolve(strict=True)
    resolved = resolved_home.joinpath(*parts).resolve(strict=False)
    try:
        resolved.relative_to(resolved_home)
    except ValueError as exc:
        raise ValueError("configured Jev resource escapes configured profile") from exc
    return resolved


def _load_plugin_client(home: Path):
    client_path = _profile_resource(home, "plugins", "jev", "client.py")
    if not client_path.is_file():
        raise FileNotFoundError("Jev client is not installed in the configured profile")
    module_name = "_hermes_jev_observer_client"
    spec = importlib.util.spec_from_file_location(module_name, client_path)
    if spec is None or spec.loader is None:
        raise ImportError("Jev client cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _evaluate(*, profile: str, state: dict, questions: dict, timeout: float) -> dict:
    from dotenv import dotenv_values
    from utils import fast_safe_load

    home = _profile_home(profile)
    profile_cfg_path = _profile_resource(home, "config.yaml")
    profile_cfg = (
        fast_safe_load(profile_cfg_path.read_text(encoding="utf-8"))
        if profile_cfg_path.is_file()
        else {}
    )
    profile_cfg = profile_cfg if isinstance(profile_cfg, dict) else {}
    plugins = profile_cfg.get("plugins", {})
    entries = plugins.get("entries", {}) if isinstance(plugins, dict) else {}
    jev = entries.get("jev", {}) if isinstance(entries, dict) else {}
    settings = jev.get("settings", {}) if isinstance(jev, dict) else {}
    connection = settings.get("connection", {}) if isinstance(settings, dict) else {}
    connection = connection if isinstance(connection, dict) else {}
    backend = str(connection.get("backend") or "typesafe")
    secret_name = "TYPESAFE_API_KEY" if backend == "typesafe" else "CLOUDFLARE_JEV_API_TOKEN"
    secrets = dotenv_values(_profile_resource(home, ".env"), interpolate=False)
    token = secrets.get(secret_name)
    if not isinstance(token, str) or not token:
        raise RuntimeError("configured Jev credential is unavailable")

    client = _load_plugin_client(home)
    kwargs = {
        "backend": backend,
        "token": token,
        "state": state,
        "questions": questions,
        "account_id": str(connection.get("account_id") or ""),
        "model": connection.get("model") or None,
        "timeout": timeout,
    }
    gateway_id = connection.get("gateway_id")
    if gateway_id is not None:
        kwargs["gateway_id"] = gateway_id
    return client.evaluate(**kwargs)


def _append_event(path: Path, event: dict) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    line = (json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError("Jev audit target is not a regular file")
        os.fchmod(fd, 0o600)
        os.write(fd, line)
    finally:
        os.close(fd)


def observe_routing(*, task_id: str, candidates: list[dict], roster: list[dict]) -> dict | None:
    """Record one independent Jev comparison and always fail open.

    No returned value is consumed by the decomposer. Errors are reduced to a
    type-only local audit event so credentials, task bodies, and provider text
    cannot leak through exception strings.
    """
    config = _observer_config()
    if not config.get("enabled") or str(config.get("mode") or "observe") != "observe":
        return None
    audit_path = Path(str(config.get("audit_log") or _DEFAULT_LOG))
    try:
        max_candidates = int(config.get("max_candidates", 8))
        selected = candidates[: max(1, min(max_candidates, 32))]
        if not selected:
            return None
        state, questions = _build_request(selected, roster, max_candidates=max_candidates)
        result = _evaluate(
            profile=str(config.get("profile") or "milly"),
            state=state,
            questions=questions,
            timeout=float(config.get("timeout_seconds", 15)),
        )
        event = _summarize_result(task_id, selected, result)
        _append_event(audit_path, event)
        return event
    except Exception as exc:
        try:
            _append_event(
                audit_path,
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "status": "error",
                    "mode": "observe",
                    "task_id": task_id,
                    "error_type": type(exc).__name__,
                    "advisory_only": True,
                    "execution_authorized": False,
                },
            )
        except Exception:
            # Audit storage is advisory too: a full/read-only disk must never
            # break Kanban decomposition or expose the original exception.
            pass
        return None
