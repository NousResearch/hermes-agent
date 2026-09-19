"""TypeSafe Jev skill suggestion — cookbook call 1 only."""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

PLUGIN_NAME = "typesafe-jev-skill-routing"
DEFAULT_MODEL = "jev-latest"
DEFAULT_BASE_URL = "https://api.typesafe.ai/v1"
HOOK_BUDGET_SEC = 2.5
MAX_CHOICES = 240
GATE_THRESHOLD = 0.30
SUGGEST_CHAR_LIMIT = 4000

GATE_QUESTIONS = {
    "acts_on_user_system": (
        "Is the assistant being asked to act on the user's files, accounts, devices, "
        "or online services, rather than only to explain or advise?"
    ),
    "would_follow_documented_procedure": (
        "Would a careful expert answering this consult a specific documented procedure "
        "or set of commands, rather than answering from general understanding?"
    ),
    "prose_suffices": (
        "Could a knowledgeable generalist fully satisfy this request in prose, with no "
        "tools, no documentation, and no access to the user's files or accounts?"
    ),
}
_INVERTED_GATE = frozenset({"prose_suffices"})


@dataclass(frozen=True)
class Skill:
    name: str
    description: str


@dataclass(frozen=True)
class SuggestResult:
    skill: str
    gate: float
    probability: float
    elapsed_sec: float
    model: str

    def block(self) -> str:
        return "\n".join(
            (
                "<skill_relevance>",
                f"Relevant to the current request: {self.skill}. Ignore this if it does "
                "not fit what the user actually asked for.",
                "</skill_relevance>",
            )
        )


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_mode(value: Any) -> str:
    text = str(value or "auto").strip().lower()
    return text if text in {"auto", "on", "off"} else "auto"


def resolve_settings(raw: Mapping[str, Any] | None) -> Dict[str, Any]:
    settings = dict(raw or {})
    return {
        "mode": normalize_mode(settings.get("mode", "auto")),
        "gate": _as_float(settings.get("gate"), GATE_THRESHOLD),
        "timeout_sec": max(0.5, _as_float(settings.get("timeout_sec"), HOOK_BUDGET_SEC)),
        "model": str(settings.get("model") or DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        "base_url": str(settings.get("base_url") or "").strip(),
        "suggest_chars": max(0, int(_as_float(settings.get("suggest_chars"), SUGGEST_CHAR_LIMIT))),
    }


def gate_mean(answers: Mapping[str, Any]) -> float:
    values: List[float] = []
    for key in GATE_QUESTIONS:
        row = answers.get(f"gate::{key}") or {}
        noul = float(row.get("noul") or 0.0)
        values.append(1.0 - noul if key in _INVERTED_GATE else noul)
    return sum(values) / len(values) if values else 0.0


def should_skip_request(text: str, *, suggest_chars: int = SUGGEST_CHAR_LIMIT) -> bool:
    stripped = text.strip()
    if not stripped or stripped.startswith("/"):
        return True
    if suggest_chars and len(stripped) > suggest_chars:
        return True
    if "<skill_relevance>" in stripped:
        return True
    return False


def load_roster_from_dirs(
    dirs: Sequence[Path],
    *,
    disabled: frozenset[str] | None = None,
    platform_ok: Callable[[Dict[str, Any]], bool] | None = None,
    parse_frontmatter: Callable[[str], tuple[Dict[str, Any], str]] | None = None,
    iter_skill_files: Callable[[Path, str], Any] | None = None,
) -> List[Skill]:
    if parse_frontmatter is None or iter_skill_files is None:
        from agent.skill_utils import iter_skill_index_files, parse_frontmatter as _parse

        parse_frontmatter = _parse
        iter_skill_files = iter_skill_index_files

    disabled = disabled or frozenset()
    platform_ok = platform_ok or (lambda _fm: True)
    seen: set[str] = set()
    skills: List[Skill] = []

    for root in dirs:
        if not root.is_dir():
            continue
        try:
            skill_files = list(iter_skill_files(root, "SKILL.md"))
        except Exception:
            continue
        for skill_file in skill_files:
            try:
                frontmatter, _ = parse_frontmatter(skill_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            skill_name = str(frontmatter.get("name") or skill_file.parent.name).strip()
            if not skill_name or skill_name in disabled or skill_name in seen:
                continue
            if not platform_ok(frontmatter):
                continue
            if len(skill_name) > 64:
                continue
            desc = str(frontmatter.get("description") or skill_name).strip()
            desc = " ".join(desc.split())[:120] or skill_name
            seen.add(skill_name)
            skills.append(Skill(name=skill_name, description=desc))
            if len(skills) >= MAX_CHOICES:
                return skills
    return skills


def default_roster_dirs() -> List[Path]:
    from agent.skill_utils import get_all_skills_dirs, get_project_skills_dirs

    dirs = list(get_all_skills_dirs())
    try:
        for proj in get_project_skills_dirs():
            if proj.is_dir() and proj not in dirs:
                dirs.append(proj)
    except Exception:
        pass
    return dirs


def load_live_roster(*, platform: str = "") -> List[Skill]:
    from agent.skill_utils import (
        extract_skill_conditions,
        get_disabled_skill_names,
        skill_matches_platform,
    )

    disabled = get_disabled_skill_names(platform or None)

    def platform_ok(frontmatter: Dict[str, Any]) -> bool:
        if not skill_matches_platform(frontmatter):
            return False
        conditions = extract_skill_conditions(frontmatter)
        wanted = [
            str(p).strip().lower()
            for p in (conditions.get("session_platforms") or [])
            if str(p).strip()
        ]
        if wanted and platform and platform.strip().lower() not in wanted:
            return False
        return True

    return load_roster_from_dirs(
        default_roster_dirs(),
        disabled=frozenset(disabled),
        platform_ok=platform_ok,
    )


def resolve_api_key() -> tuple[str, str]:
    key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    base_url = os.environ.get("TYPESAFE_BASE_URL", "").strip()
    if key:
        return key, (base_url or DEFAULT_BASE_URL).rstrip("/")

    try:
        from hermes_cli.env_loader import load_hermes_dotenv

        load_hermes_dotenv()
        key = os.environ.get("TYPESAFE_API_KEY", "").strip()
        base_url = os.environ.get("TYPESAFE_BASE_URL", "").strip()
    except Exception:
        key = key or ""

    if key:
        return key, (base_url or DEFAULT_BASE_URL).rstrip("/")

    try:
        from hermes_cli.auth import resolve_api_key_provider_credentials

        creds = resolve_api_key_provider_credentials("jev")
        return str(creds.get("api_key") or "").strip(), str(creds.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
    except Exception:
        return "", DEFAULT_BASE_URL


def systemone_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        return f"{root}/systemone"
    return f"{root}/v1/systemone"


def post_systemone(
    *,
    api_key: str,
    base_url: str,
    model: str,
    state: Mapping[str, Any],
    questions: Mapping[str, Any],
    timeout_sec: float,
    opener: Callable[..., Any] | None = None,
) -> Dict[str, Any]:
    payload = json.dumps({"state": dict(state), "model": model, "questions": dict(questions)}).encode("utf-8")
    request = urllib.request.Request(
        systemone_url(base_url),
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    open_fn = opener or urllib.request.urlopen
    with open_fn(request, timeout=timeout_sec) as response:
        body = response.read().decode("utf-8")
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise ValueError("TypeSafe response was not a JSON object")
    return parsed


def suggest_skill(
    prompt: str,
    skills: Sequence[Skill],
    *,
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    gate_threshold: float = GATE_THRESHOLD,
    timeout_sec: float = HOOK_BUDGET_SEC,
    post_fn: Callable[..., Dict[str, Any]] | None = None,
) -> Optional[SuggestResult]:
    if should_skip_request(prompt):
        return None
    if len(skills) < 2:
        return None
    if not api_key:
        return None

    criteria = {skill.name: skill.description for skill in skills}
    questions: Dict[str, Any] = {
        "which": {
            "type": "choice",
            "instructions": (
                "Which of these skills, if any, is the right one to load to help with "
                "the user's latest request?"
            ),
            "criteria": criteria,
        },
    }
    for key, text in GATE_QUESTIONS.items():
        questions[f"gate::{key}"] = {"type": "noul", "instructions": text}

    started = time.perf_counter()
    post = post_fn or post_systemone
    body = post(
        api_key=api_key,
        base_url=base_url,
        model=model,
        state={"request": prompt.strip()[:SUGGEST_CHAR_LIMIT], "recent_context": ""},
        questions=questions,
        timeout_sec=timeout_sec,
    )
    elapsed = time.perf_counter() - started

    answers = body.get("answers") if isinstance(body.get("answers"), dict) else {}
    which = answers.get("which") if isinstance(answers.get("which"), dict) else {}
    choice = str(which.get("choice") or "").strip()
    probs = which.get("probabilities") if isinstance(which.get("probabilities"), dict) else {}
    probability = float(probs.get(choice) or 0.0)
    gate = gate_mean(answers)

    if not choice or gate < gate_threshold:
        return None

    return SuggestResult(
        skill=choice,
        gate=gate,
        probability=probability,
        elapsed_sec=elapsed,
        model=str(body.get("model") or model),
    )


def routing_enabled(settings: Mapping[str, Any], *, api_key_present: bool) -> bool:
    mode = normalize_mode(settings.get("mode", "auto"))
    if mode == "off":
        return False
    if mode == "on":
        return True
    return api_key_present


def append_log(log_path: Path, kind: str, detail: str) -> None:
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{stamp}\t{kind}\t{detail}\n")
    except OSError:
        pass
