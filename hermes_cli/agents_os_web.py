"""Local-only Agents OS Mission Control web surface.

This module intentionally serves only loopback HTTP and exposes read-only/operator
planning payloads by default. Draft/action endpoints create local runtime records
or approval drafts; they do not execute outbound/public/security/financial work.
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sqlite3
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from hermes_cli.agents_os import (
    AgentsOSPaths,
    AgentsOSService,
    connect,
    log_event,
    resolve_paths,
    row_to_dict,
    slugify,
    utc_now,
)
from hermes_cli.agents_os_idea_factory import draft_idea, idea_factory_schema
from hermes_cli.agents_os_seo import seo_mission_control_payload

LOCAL_HOSTS = {"127.0.0.1", "localhost"}
SOURCE_DEFAULTS = {
    "video:q13OqknCh-c": "https://youtu.be/q13OqknCh-c",
    "transcript:q13OqknCh-c": "/mnt/d/HermesAgent/home/transcripts/q13OqknCh-c_transcript.txt",
    "youtube-note:q13OqknCh-c": "/mnt/d/Obsidian_Vault_v2/Hermes-Agent-Doni/01-INBOX/YouTube/2026-06-08-q13OqknCh-c-claude-agent-operating-system.md",
    "plan:parity-q13OqknCh-c": "/mnt/d/Obsidian_Vault_v2/Hermes-Agent-Doni/08-OPERATIONS/ACTIVE-WORK/2026-06-08-agent-os-parity-build-plan-q13OqknCh-c.md",
    "plan:full-product": "/mnt/d/Obsidian_Vault_v2/Hermes-Agent-Doni/08-OPERATIONS/ACTIVE-WORK/2026-06-08-agent-os-full-product-plan.md",
    "contract:idea-factory-v0": "/mnt/d/Obsidian_Vault_v2/Hermes-Agent-Doni/08-OPERATIONS/ACTIVE-WORK/2026-06-08-idea-factory-v0-contract.md",
}
SOURCE_ENV = {
    "transcript:q13OqknCh-c": "AGENTS_OS_SOURCE_TRANSCRIPT",
    "plan:full-product": "AGENTS_OS_SOURCE_FULL_PLAN",
}
MEDIA_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp3", ".wav", ".ogg", ".mp4", ".webm", ".mov"}
ARTIFACT_SUFFIXES = {".md", ".txt", ".json", ".log", ".html", ".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mp3", ".wav", ".ogg"}


def _json_safe(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("content-length") or 0)
    if length <= 0:
        return {}
    raw = handler.rfile.read(length).decode("utf-8")
    return json.loads(raw or "{}")


def _send_json(handler: BaseHTTPRequestHandler, payload: dict[str, Any] | list[Any], status: int = 200) -> None:
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _send_html(handler: BaseHTTPRequestHandler, html: str, status: int = 200) -> None:
    data = html.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _parse_caps(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
        return value if isinstance(value, list) else []
    except json.JSONDecodeError:
        return []


def _path_info(path: str) -> dict[str, Any]:
    p = Path(path)
    return {
        "path": str(p),
        "exists": p.exists(),
        "suffix": p.suffix.lower(),
        "kind": "directory" if p.exists() and p.is_dir() else "file",
        "size_bytes": p.stat().st_size if p.exists() and p.is_file() else None,
    }


def _default_agent_cards(paths: AgentsOSPaths) -> list[dict[str, Any]]:
    return [
        {
            "id": "doni-local",
            "name": "Doni Local",
            "status": "available",
            "capabilities": ["local planning", "TDD", "reports", "Mission Control"],
            "runtime_home": str(paths.home),
            "reference_home": str(paths.root),
            "memory_boundary": "Doni Hermes home only",
            "auth_boundary": "Doni auth only; credentials are never displayed",
            "allowed_actions": ["safe local files", "tests", "local API smoke", "local reports"],
            "approval_gates": ["deploy", "public send", "credential use", "gateway restart", "destructive changes"],
        },
        {
            "id": "kodi-codex",
            "name": "Kodi/Codex",
            "status": "reference",
            "capabilities": ["coding delegate", "review", "patch suggestions"],
            "runtime_home": "external/approval-gated",
            "reference_home": "repo-local only when explicitly invoked",
            "memory_boundary": "no Doni memory merge",
            "auth_boundary": "no credential sharing from Doni",
            "allowed_actions": ["local branch work after explicit routing"],
            "approval_gates": ["push", "PR", "public GitHub action", "credential use"],
        },
        {
            "id": "marija-profile",
            "name": "Marija profile",
            "status": "separate_profile",
            "capabilities": ["separate Hermes profile"],
            "runtime_home": "/home/goran/.hermes-marija-clean",
            "reference_home": "read-only boundary reference",
            "memory_boundary": "separate personal memory; no merge",
            "auth_boundary": "separate profile auth; no cross-copy",
            "allowed_actions": ["status reference only"],
            "approval_gates": ["any profile write", "auth change", "gateway lifecycle"],
        },
        {
            "id": "ero-openclaw",
            "name": "ERO/OpenClaw reference layer",
            "status": "separate_runtime",
            "capabilities": ["reference bridge", "source artefacts"],
            "runtime_home": "/home/goran/.openclaw/workspace",
            "reference_home": "/mnt/d/AI_Memory/communication",
            "memory_boundary": "separate runtime memory; reference only",
            "auth_boundary": "no auth/session sharing",
            "allowed_actions": ["read-only reference when useful"],
            "approval_gates": ["runtime write", "bridge mutation", "memory import/export"],
        },
        {
            "id": "candidate-agent",
            "name": "Candidate agents",
            "status": "candidate",
            "capabilities": ["future plugin/worker slots"],
            "runtime_home": "not assigned",
            "reference_home": "Mission Control registry",
            "memory_boundary": "must declare before activation",
            "auth_boundary": "must declare before activation",
            "allowed_actions": ["registration draft only"],
            "approval_gates": ["activation", "tool access", "credential access"],
        },
    ]


def agents_registry_payload(paths: AgentsOSPaths) -> dict[str, Any]:
    cards = {card["id"]: card for card in _default_agent_cards(paths)}
    with connect(paths) as conn:
        for row in conn.execute("SELECT * FROM agents ORDER BY created_at ASC").fetchall():
            item = row_to_dict(row)
            base = cards.get(item["id"], {})
            cards[item["id"]] = {
                **base,
                "id": item["id"],
                "name": item.get("name") or item["id"],
                "status": item.get("status") or base.get("status", "available"),
                "capabilities": _parse_caps(item.get("capabilities")) or base.get("capabilities", []),
                "runtime_home": base.get("runtime_home", str(paths.root)),
                "reference_home": base.get("reference_home", str(paths.root)),
                "memory_boundary": base.get("memory_boundary", "declared local boundary required"),
                "auth_boundary": base.get("auth_boundary", "credentials are never displayed"),
                "allowed_actions": base.get("allowed_actions", ["safe local tasks"]),
                "approval_gates": base.get("approval_gates", ["public", "credential", "deploy", "destructive"]),
            }
    return {"local_only": True, "agents": list(cards.values())}


def knowledge_index_payload(paths: AgentsOSPaths) -> dict[str, Any]:
    nodes = []
    for node_id, default_path in SOURCE_DEFAULTS.items():
        value = os.environ.get(SOURCE_ENV.get(node_id, ""), default_path)
        info = _path_info(value) if not value.startswith("http") else {"path": value, "exists": True, "kind": "url", "size_bytes": None, "suffix": ""}
        kind = node_id.split(":", 1)[0]
        nodes.append({"id": node_id, "kind": kind, "label": node_id, "weight": 10 if info["exists"] else 3, **info})
    with connect(paths) as conn:
        for row in conn.execute("SELECT id,title,path,kind,task_id,workflow,created_at FROM artifacts ORDER BY created_at DESC LIMIT 40").fetchall():
            item = row_to_dict(row)
            info = _path_info(item["path"])
            nodes.append({"id": f"artifact:{item['id']}", "kind": "artifact", "label": item["title"], "weight": 7 if info["exists"] else 2, "task_id": item.get("task_id"), "workflow": item.get("workflow"), **info})
    edges = [
        {"from": "video:q13OqknCh-c", "to": "transcript:q13OqknCh-c", "relation": "has_transcript"},
        {"from": "video:q13OqknCh-c", "to": "youtube-note:q13OqknCh-c", "relation": "intake_note"},
        {"from": "youtube-note:q13OqknCh-c", "to": "plan:parity-q13OqknCh-c", "relation": "informed_plan"},
        {"from": "plan:parity-q13OqknCh-c", "to": "plan:full-product", "relation": "expands_to"},
        {"from": "contract:idea-factory-v0", "to": "plan:full-product", "relation": "implements_slice"},
    ]
    return {"local_only": True, "runtime_memory_merge": False, "note": "vault/reference graph, not runtime memory merge", "nodes": nodes, "edges": edges}


def artifacts_payload(paths: AgentsOSPaths) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    with connect(paths) as conn:
        for row in conn.execute("SELECT * FROM artifacts ORDER BY created_at DESC LIMIT 100").fetchall():
            item = row_to_dict(row)
            info = _path_info(item["path"])
            item.update(info)
            item["preview_type"] = "markdown" if info["suffix"] == ".md" else ("json" if info["suffix"] == ".json" else ("media" if info["suffix"] in MEDIA_SUFFIXES else "text"))
            items.append(item)
    seen = {item["path"] for item in items}
    for root in [paths.artifacts, paths.vault_root]:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if len(items) >= 160:
                break
            if p.is_file() and p.suffix.lower() in ARTIFACT_SUFFIXES and str(p) not in seen:
                info = _path_info(str(p))
                items.append({"id": f"file:{slugify(str(p))}", "kind": "file", "title": p.name, "task_id": None, "workflow": None, **info})
    return {"local_only": True, "items": items}


def media_assets_payload(paths: AgentsOSPaths) -> dict[str, Any]:
    assets = []
    for root in [paths.artifacts, paths.vault_root]:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if len(assets) >= 80:
                break
            if p.is_file() and p.suffix.lower() in MEDIA_SUFFIXES:
                mime, _ = mimetypes.guess_type(str(p))
                assets.append({"path": str(p), "title": p.name, "mime": mime or "application/octet-stream", "size_bytes": p.stat().st_size, "read_only": True})
    return {"local_only": True, "generation_enabled": False, "posting_enabled": False, "assets": assets}


def _bounded_file_count(root: Path, *, name: str | None = None, max_scan: int = 600) -> tuple[int, bool]:
    """Fast bounded local status count; avoids slow recursive scans in UI requests."""
    if not root.exists():
        return 0, False
    count = 0
    scanned = 0
    stack = [root]
    while stack and scanned < max_scan:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            scanned += 1
            if scanned >= max_scan:
                break
            if entry.is_dir():
                stack.append(entry)
            elif name is None or entry.name == name:
                count += 1
    return count, bool(stack or scanned >= max_scan)


def redacted_manage_status_payload(paths: AgentsOSPaths) -> dict[str, Any]:
    skills_dir = paths.home / "skills"
    plugins_dir = paths.home / "plugins"
    cron_dir = paths.home / "cron"
    skill_count, skill_truncated = _bounded_file_count(skills_dir, name="SKILL.md")
    cron_count, cron_truncated = _bounded_file_count(cron_dir)
    try:
        plugin_count = len(list(plugins_dir.iterdir())) if plugins_dir.exists() else 0
    except OSError:
        plugin_count = 0
    return {
        "local_only": True,
        "credentials_visible": False,
        "hermes": {"home": str(paths.home), "agents_os_home": str(paths.root), "state_db": str(paths.db), "gateway_restart": False},
        "skills": {"path": str(skills_dir), "count": skill_count, "truncated": skill_truncated},
        "plugins": {"path": str(plugins_dir), "count": plugin_count},
        "cron": {"path": str(cron_dir), "status_only": True, "count": cron_count, "truncated": cron_truncated},
        "mcp": {"status_only": True, "note": "Use hermes mcp test/list outside this read-only panel when needed."},
        "model_provider": "redacted",
        "candidate_integrations": ["desktop shell", "voice dry-run", "read-only project library"],
    }


def voice_status_payload(paths: AgentsOSPaths) -> dict[str, Any]:
    cache_audio = paths.home / "cache" / "audio"
    audio_files = list(cache_audio.glob("*"))[-5:] if cache_audio.exists() else []
    return {
        "local_only": True,
        "stt_status": "detectable" if audio_files else "not_detected_or_no_recent_audio",
        "tts_status": "configured_by_runtime_or_tool_provider",
        "recent_audio_count": len(audio_files),
        "jarvis_dry_run_design": [
            "transcribe local voice input",
            "classify intent through Idea Factory risk gate",
            "show command draft and required approval badge",
            "execute only safe local/read-only actions after explicit local UI confirmation",
        ],
        "computer_control": "approval_gated_unexecuted",
    }


def jarvis_briefing_payload(paths: AgentsOSPaths) -> dict[str, Any]:
    """Safe-local Jarvis/Oracle briefing contract for Mission Control.

    This is a read-only/dry-run payload. It summarizes local state and declares
    command modes without enabling microphone, wake-word, browser, computer, or
    public side effects.
    """
    with connect(paths) as conn:
        task_rows = conn.execute("SELECT status, COUNT(*) AS count FROM tasks GROUP BY status").fetchall()
        approval_rows = conn.execute("SELECT status, COUNT(*) AS count FROM approvals GROUP BY status").fetchall()
        artifact_count = conn.execute("SELECT COUNT(*) AS count FROM artifacts").fetchone()["count"]
        recent_artifacts = [
            row_to_dict(row)
            for row in conn.execute(
                "SELECT id,kind,title,path,task_id,workflow,created_at FROM artifacts ORDER BY created_at DESC LIMIT 5"
            ).fetchall()
        ]
    task_counts = {row["status"]: row["count"] for row in task_rows}
    approval_counts = {row["status"]: row["count"] for row in approval_rows}
    open_task_count = sum(task_counts.get(status, 0) for status in ("new", "pending", "routed", "ready", "in_progress", "needs_approval", "blocked", "review"))
    return {
        "local_only": True,
        "execution_created": False,
        "always_on_microphone": False,
        "wake_word_enabled": False,
        "computer_control": "approval_gated_unexecuted",
        "briefing": {
            "timestamp": utc_now(),
            "agents_os_home": str(paths.root),
            "state_db": str(paths.db),
            "open_task_count": open_task_count,
            "completed_task_count": task_counts.get("completed", 0),
            "pending_approval_count": approval_counts.get("pending", 0),
            "artifact_count": artifact_count,
            "recent_artifacts": recent_artifacts,
        },
        "commands": [
            {"name": "wake", "mode": "read_only_briefing", "approval_required": False, "does": "Boot/status briefing only."},
            {"name": "show", "mode": "read_only_retrieval", "approval_required": False, "does": "Show local tasks, artifacts, notes, and reference graph."},
            {"name": "build", "mode": "safe_local_draft", "approval_required": False, "does": "Create local draft artifacts/tasks only."},
            {"name": "act", "mode": "approval_draft_only", "approval_required": True, "does": "Prepare risky action for explicit approval; do not execute."},
        ],
        "approval_gates": [
            "microphone_wake_word",
            "computer_control",
            "external_open",
            "deploy_publish",
            "credentials",
            "cross_agent_memory_merge",
        ],
        "wall_mode_contract": {
            "enabled_for_display": True,
            "execution_from_wall_mode": False,
            "description": "Large-screen Mission Control display; action execution remains gated.",
        },
    }


def _jarvis_slug_from_time() -> str:
    return utc_now().replace(":", "").replace("-", "").replace(".", "")[:15]


def _decode_optional_audio(data: dict[str, Any]) -> bytes:
    raw = data.get("audio_base64") or ""
    if not raw:
        return b""
    if "," in raw and raw.split(",", 1)[0].startswith("data:"):
        raw = raw.split(",", 1)[1]
    return base64.b64decode(raw)


def _jarvis_audio_suffix(mime: str | None) -> str:
    mime = (mime or "").lower()
    if "wav" in mime:
        return ".wav"
    if "ogg" in mime:
        return ".ogg"
    if "mpeg" in mime or "mp3" in mime:
        return ".mp3"
    return ".webm"


def _jarvis_preview_from_text(transcript_text: str) -> dict[str, Any]:
    draft = draft_idea(transcript_text)
    normalized = transcript_text.lower()
    if any(token in normalized for token in ["prikaži", "prikazi", "show", "status", "stanje", "zadnje", "otvori lokalni", "local status"]):
        draft["classification"] = "research_intake"
        draft["risk_class"] = "safe_local"
        draft["recommended_lane"] = "read-only-status"
        draft["approval_required"] = False
        draft["plan_steps"] = [
            "Dohvatiti lokalni status ili postojeći artefakt.",
            "Prikazati rezultat u command preview kartici.",
            "Ne izvršiti nikakvu vanjsku ili rizičnu akciju.",
        ]
    if any(token in normalized for token in ["sigurnosni", "security", "pentest", "penetration", "ranjiv", "vulnerability", "exploit", "scan klijent", "skeniraj"]):
        draft["classification"] = "security_gated"
        draft["risk_class"] = "security_gated"
        draft["recommended_lane"] = "security-scope-gate"
        draft["approval_required"] = True
        draft["plan_steps"] = [
            "Prikazati security scope i authorization zahtjev u command preview kartici.",
            "Ne pokretati aktivni scan ili test bez eksplicitnog scope/legal approvala.",
            "Dopustiti samo jasno označene read-only provjere nakon approval gatea.",
        ]
    if any(token in normalized for token in ["deploy", "deployaj", "push", "pr ", "pull request", "objavi", "pošalji", "posalji", "email"]):
        draft["classification"] = "public_outbound_gated"
        draft["risk_class"] = "public_gated"
        draft["recommended_lane"] = "public-action-approval"
        draft["approval_required"] = True
        draft["plan_steps"] = [
            "Prikazati namjeru i rizičnu radnju u command preview kartici.",
            "Ne izvršiti javnu, deploy, push ili outbound akciju iz glasa.",
            "Čekati eksplicitno Goranovo odobrenje prije side-effecta.",
        ]
    return draft


def jarvis_preview_payload(paths: AgentsOSPaths, data: dict[str, Any]) -> dict[str, Any]:
    transcript_text = (data.get("transcript_text") or data.get("text") or "").strip()
    if not transcript_text:
        raise ValueError("transcript_text is required")
    draft = _jarvis_preview_from_text(transcript_text)
    command_card = {
        "heard": transcript_text,
        "interpreted_intent": draft["classification"],
        "risk_class": draft["risk_class"],
        "proposed_action": draft["recommended_lane"],
        "approval_required": draft["approval_required"],
        "expected_output": draft["expected_artifacts"],
        "execution_created": False,
        "allowed_now": draft["risk_class"] == "safe_local",
    }
    return {
        "local_only": True,
        "execution_created": False,
        "transcript_text": transcript_text,
        "command_card": command_card,
        "draft": draft,
        "audit": {"agents_os_home": str(paths.root), "created_at": utc_now(), "policy": "preview_only"},
    }


def _jarvis_stt_payload(data: dict[str, Any], audio_path: Path | None = None) -> dict[str, Any]:
    provided = (data.get("transcript_text") or data.get("text") or "").strip()
    if provided:
        return {"provider": "provided_transcript", "text": provided, "confidence": None, "status": "provided"}
    stt_result = data.get("stt_result") if isinstance(data.get("stt_result"), dict) else {}
    stt_text = (stt_result.get("text") or "").strip()
    if stt_text:
        return {
            "provider": stt_result.get("provider") or "external_stt_adapter",
            "text": stt_text,
            "confidence": stt_result.get("confidence"),
            "status": "transcribed",
        }
    if data.get("use_local_stt") and audio_path is not None:
        try:
            return _transcribe_with_local_faster_whisper(
                str(audio_path),
                model=str(data.get("stt_model") or "base"),
                language=str(data.get("stt_language") or "hr"),
            )
        except Exception as exc:
            return {
                "provider": "local-faster-whisper",
                "text": "[stt_pending] Local STT failed; audio artifact was saved for retry.",
                "confidence": None,
                "status": "error",
                "error": exc.__class__.__name__,
                "message": str(exc),
            }
    return {
        "provider": "stub_pending",
        "text": "[stt_pending] Audio captured; STT backend not connected in this local slice.",
        "confidence": None,
        "status": "pending",
    }


def _transcribe_with_local_faster_whisper(audio_path: str, *, model: str = "base", language: str = "hr") -> dict[str, Any]:
    from faster_whisper import WhisperModel  # type: ignore[import-not-found]

    whisper = WhisperModel(model, device="cpu", compute_type="int8")
    segments, info = whisper.transcribe(audio_path, beam_size=5, language=language or None, vad_filter=True)
    text = " ".join(segment.text.strip() for segment in segments).strip()
    return {
        "provider": "local-faster-whisper",
        "text": text or "[stt_empty] Local STT produced no transcript.",
        "confidence": getattr(info, "language_probability", None),
        "status": "transcribed" if text else "empty",
        "language": getattr(info, "language", None),
        "model": model,
    }


def jarvis_model_advisor_payload(paths: AgentsOSPaths, data: dict[str, Any]) -> dict[str, Any]:
    transcript_text = (data.get("transcript_text") or data.get("text") or "").strip()
    if not transcript_text:
        raise ValueError("transcript_text is required")
    deterministic = data.get("deterministic_preview") if isinstance(data.get("deterministic_preview"), dict) else jarvis_preview_payload(paths, {"transcript_text": transcript_text})
    command_card = dict(deterministic.get("command_card") or {})
    model_result = data.get("model_result") if isinstance(data.get("model_result"), dict) else {}
    model_risk = model_result.get("risk_class")
    authoritative_risk = command_card.get("risk_class")
    semantic_intent = model_result.get("semantic_intent") or command_card.get("interpreted_intent")
    voice_reply = model_result.get("voice_reply_short") or _jarvis_voice_reply(command_card)
    command_card["semantic_intent"] = semantic_intent
    command_card["voice_reply_short"] = voice_reply
    command_card["risk_class"] = authoritative_risk
    command_card["approval_required"] = bool(command_card.get("approval_required"))
    command_card["execution_created"] = False
    return {
        "local_only": True,
        "execution_created": False,
        "provider": data.get("provider") or "deterministic",
        "model": data.get("model") or "none",
        "transcript_text": transcript_text,
        "authoritative_risk_class": authoritative_risk,
        "model_risk_class": model_risk,
        "risk_disagreement": bool(model_risk and model_risk != authoritative_risk),
        "command_card": command_card,
        "model_result": model_result,
        "audit": {"agents_os_home": str(paths.root), "created_at": utc_now(), "policy": "deterministic_gate_authoritative"},
    }


def _jarvis_voice_reply(command_card: dict[str, Any]) -> str:
    if command_card.get("approval_required"):
        return "Ovo treba odobrenje. Pripremio sam preview, ništa ne izvršavam."
    return "Ovo je sigurno lokalno. Pripremio sam preview, bez izvršavanja."


def jarvis_transcribe_payload(paths: AgentsOSPaths, data: dict[str, Any]) -> dict[str, Any]:
    """Persist a local push-to-talk artefact and return transcript + intent preview.

    This v0.1 endpoint accepts browser audio plus an optional transcript stub. It
    deliberately does not execute commands; real STT can replace the transcript
    stub behind the same payload contract.
    """
    stamp = _jarvis_slug_from_time()
    audio_bytes = _decode_optional_audio(data)
    suffix = _jarvis_audio_suffix(data.get("audio_mime"))
    audio_dir = paths.artifacts / "jarvis_audio"
    transcript_dir = paths.artifacts / "jarvis_transcripts"
    audio_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir / f"{stamp}-jarvis-command{suffix}"
    transcript_path = transcript_dir / f"{stamp}-jarvis-transcript.md"
    audio_path.write_bytes(audio_bytes)
    stt = _jarvis_stt_payload(data, audio_path)
    transcript_text = stt["text"]
    preview = jarvis_preview_payload(paths, {"transcript_text": transcript_text})
    advisor = jarvis_model_advisor_payload(paths, {"transcript_text": transcript_text, "deterministic_preview": preview})
    transcript_body = {
        "local_only": True,
        "execution_created": False,
        "stt": stt,
        "advisor": {"provider": advisor["provider"], "model": advisor["model"], "risk_disagreement": advisor["risk_disagreement"]},
        "transcript": {"text": transcript_text, "source": stt["provider"], "created_at": utc_now()},
        "intent_preview": advisor["command_card"],
        "audio_artifact_path": str(audio_path),
    }
    transcript_path.write_text(f"# Jarvis transcript\n\n```json\n{json.dumps(transcript_body, ensure_ascii=False, indent=2)}\n```\n", encoding="utf-8")
    artifact_id = f"artifact-jarvis-transcript-{stamp}"
    with connect(paths) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO artifacts(id,kind,title,path,task_id,workflow,created_at) VALUES(?,?,?,?,?,?,?)",
            (artifact_id, "jarvis_transcript", "Jarvis transcript", str(transcript_path), None, "jarvis-push-to-talk", utc_now()),
        )
        log_event(conn, "jarvis_transcribed", payload={"artifact_id": artifact_id, "audio_path": str(audio_path), "execution_created": False})
        conn.commit()
    return {
        "status": "transcribed",
        "local_only": True,
        "execution_created": False,
        "audio_artifact_path": str(audio_path),
        "transcript_artifact_path": str(transcript_path),
        "transcript": transcript_body["transcript"],
        "stt": stt,
        "advisor": transcript_body["advisor"],
        "intent_preview": advisor["command_card"],
        "command_card": advisor["command_card"],
        "artifact_id": artifact_id,
    }


def operator_loop_payload(service: AgentsOSService) -> dict[str, Any]:
    dashboard = service.dashboard_payload()
    tasks = dashboard.get("tasks", [])
    reviews = dashboard.get("reviews", [])
    events = dashboard.get("events", [])
    judge_events = [event for event in events if "judge" in event.get("event_type", "") or "review" in event.get("event_type", "")]
    return {
        "local_only": True,
        "acceptance_criteria": ["evidence exists", "tests/smoke recorded", "approval gates respected"],
        "task_detail_available": True,
        "tasks": tasks,
        "reviews": reviews,
        "evidence_links": dashboard.get("recent_completions", []),
        "judge_status": "ready" if reviews or judge_events else "pending",
        "judge_results_faked": False,
        "blocked_reason": None,
    }


def _write_artifact(paths: AgentsOSPaths, title: str, body: str, *, kind: str, task_id: str | None = None, workflow: str | None = None) -> tuple[str, str]:
    artifact_id = f"artifact-{slugify(title)[:24]}-{utc_now().replace(':','').replace('-','')[:15]}"
    target = paths.artifacts / kind / f"{utc_now().split('T', 1)[0]}-{slugify(title)}.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    text = f"# {title}\n\n{body}\n"
    target.write_text(text, encoding="utf-8")
    with connect(paths) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO artifacts(id,kind,title,path,task_id,workflow,created_at) VALUES(?,?,?,?,?,?,?)",
            (artifact_id, kind, title, str(target), task_id, workflow, utc_now()),
        )
        log_event(conn, "artifact_created", task_id=task_id, payload={"artifact_id": artifact_id, "path": str(target), "source": "mission_control_web"})
        conn.commit()
    return artifact_id, str(target)


def create_idea_action(service: AgentsOSService, data: dict[str, Any]) -> dict[str, Any]:
    draft = service.idea_factory_draft_payload(data)
    paths = service.paths
    title = data.get("title") or f"Idea Factory: {data.get('idea_text', '')[:70]}"
    task_id = f"task-{draft['idea_id'].replace('idea-', '')}"
    now = utc_now()
    approval_id = None
    mode = "safe_local_task"
    status = "pending"
    approval_required = 0
    if draft["approval_required"]:
        mode = "approval_draft"
        status = "needs_approval"
        approval_required = 1
        approval_id = f"approval-{draft['idea_id'].replace('idea-', '')}"
    body = json.dumps({"idea": data, "draft": draft, "mode": mode, "execution_created": False}, ensure_ascii=False, indent=2)
    artifact_id, artifact_path = _write_artifact(paths, title, f"```json\n{body}\n```", kind="idea_factory", task_id=task_id, workflow=draft["recommended_lane"])
    with connect(paths) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO tasks(id,title,status,workflow,priority,created_at,updated_at,notes,route,approval_required) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (task_id, title, status, draft["recommended_lane"], 2, now, now, data.get("idea_text", ""), "approval_gate" if approval_required else "local:direct", approval_required),
        )
        log_event(conn, "idea_action_created", task_id=task_id, payload={"mode": mode, "draft": draft, "artifact_id": artifact_id})
        if approval_id:
            conn.execute(
                "INSERT OR REPLACE INTO approvals(id,title,status,risk,task_id,payload,created_at) VALUES(?,?,?,?,?,?,?)",
                (approval_id, f"Approval draft: {title}", "pending", draft["risk_class"], task_id, body, now),
            )
            log_event(conn, "approval_requested", task_id=task_id, payload={"approval_id": approval_id, "risk": draft["risk_class"], "execution_created": False})
        conn.commit()
    return {"mode": mode, "task_id": task_id, "approval_id": approval_id, "artifact_id": artifact_id, "artifact_path": artifact_path, "draft": draft, "execution_created": False}


def task_detail_payload(paths: AgentsOSPaths, task_id: str) -> dict[str, Any]:
    with connect(paths) as conn:
        task = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if task is None:
            return {"status": "not_found", "task_id": task_id}
        approvals = [row_to_dict(r) for r in conn.execute("SELECT * FROM approvals WHERE task_id=? ORDER BY created_at DESC", (task_id,)).fetchall()]
        runs = [row_to_dict(r) for r in conn.execute("SELECT * FROM runs WHERE task_id=? ORDER BY created_at DESC", (task_id,)).fetchall()]
        events = [row_to_dict(r) for r in conn.execute("SELECT * FROM events WHERE task_id=? ORDER BY created_at DESC", (task_id,)).fetchall()]
        artifacts = [row_to_dict(r) for r in conn.execute("SELECT * FROM artifacts WHERE task_id=? ORDER BY created_at DESC", (task_id,)).fetchall()]
        reviews = [row_to_dict(r) for r in conn.execute("SELECT * FROM reviews WHERE task_id=? ORDER BY created_at DESC", (task_id,)).fetchall()]
    return {"status": "ok", "task": row_to_dict(task), "acceptance_criteria": ["planned", "implemented", "verified", "evidence linked"], "approvals": approvals, "runs": runs, "events": events, "artifacts": artifacts, "reviews": reviews, "judge_status": "pending" if not reviews else "review_available"}


def mission_control_html(service: AgentsOSService) -> str:
    status = service.status_payload()
    dashboard = service.dashboard_payload()
    bootstrap = {"status": status, "dashboard": dashboard, "knowledge_note": "vault/reference graph, not runtime memory merge"}
    return f"""<!doctype html>
<html lang=\"hr\">
<head>
<meta charset=\"utf-8\" />
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
<title>Agents OS Mission Control</title>
<style>
:root {{ color-scheme: dark; --bg:#080b14; --panel:#101827; --panel2:#142035; --text:#e8f0ff; --muted:#93a4bd; --accent:#66d9ff; --warn:#ffc857; --ok:#7dffb2; --bad:#ff6b7a; }}
* {{ box-sizing:border-box; }} body {{ margin:0; font-family:Inter, ui-sans-serif, system-ui, Segoe UI, Arial; background:radial-gradient(circle at 20% 0%, #13284a 0, var(--bg) 38%); color:var(--text); }}
header {{ padding:28px 34px; border-bottom:1px solid #24344f; }} h1 {{ margin:0; letter-spacing:.02em; }} .sub {{ color:var(--muted); margin-top:8px; }}
.tabs {{ display:flex; gap:8px; flex-wrap:wrap; padding:18px 34px; border-bottom:1px solid #1e2c44; position:sticky; top:0; background:#080b14dd; backdrop-filter:blur(8px); z-index:5; }}
button {{ background:#16243a; color:var(--text); border:1px solid #2a4166; border-radius:12px; padding:10px 13px; cursor:pointer; }} button.active, button:hover {{ border-color:var(--accent); box-shadow:0 0 0 1px #66d9ff55 inset; }}
main {{ padding:24px 34px 60px; }} section {{ display:none; }} section.active {{ display:block; }} .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:16px; }}
.card {{ background:linear-gradient(180deg,var(--panel),var(--panel2)); border:1px solid #263a5b; border-radius:18px; padding:16px; box-shadow:0 20px 50px #0007; }}
.kv {{ color:var(--muted); font-size:13px; }} .ok {{ color:var(--ok); }} .warn {{ color:var(--warn); }} .bad {{ color:var(--bad); }} textarea {{ width:100%; min-height:96px; border-radius:14px; background:#09111f; color:var(--text); border:1px solid #2a4166; padding:12px; }} pre {{ overflow:auto; background:#070b12; border:1px solid #1d2c45; border-radius:14px; padding:12px; }} .pill {{ display:inline-block; border:1px solid #35537e; border-radius:999px; padding:3px 8px; color:var(--muted); margin:2px; }}
</style>
</head>
<body>
<header><h1>Agents OS Mission Control</h1><div class=\"sub\">Local-only Doni operator cockpit · gateway restart: false · vault/reference graph, not runtime memory merge</div></header>
<nav class=\"tabs\">
<button data-tab=\"overview\" class=\"active\">Overview</button><button data-tab=\"idea\">Idea Factory</button><button data-tab=\"agents\">Agent Registry</button><button data-tab=\"knowledge\">Knowledge Galaxy</button><button data-tab=\"artifacts\">Artifact Library</button><button data-tab=\"seo\">SEO Mission Control</button><button data-tab=\"operator\">Operator Loop</button><button data-tab=\"media\">Media Studio</button><button data-tab=\"manage\">Manage / Status</button><button data-tab=\"voice\">Voice / Jarvis</button>
</nav>
<main>
<section id=\"overview\" class=\"active\"><div class=\"grid\"><div class=\"card\"><h2>HEALTH <span class=\"ok\">OK</span></h2><div class=\"kv\">State DB: {status.get('state_db')}</div><div class=\"kv\">Schema: {status.get('schema_version')}</div></div><div class=\"card\"><h2>Queue</h2><pre id=\"queueSummary\"></pre></div></div></section>
<section id=\"idea\"><div class=\"card\"><h2>Idea Factory</h2><textarea id=\"ideaText\">Obradi YouTube video</textarea><p><button id=\"draftIdea\">Draft only</button> <button id=\"createIdea\">Create safe task / approval draft</button></p><pre id=\"ideaResult\"></pre><div class=\"kv\">Fields: classification · risk class · recommended lane · plan steps · approval badge · expected artifacts · acceptance criteria</div></div></section>
<section id=\"agents\"><h2>Paperclip Agent Registry</h2><div id=\"agentsList\" class=\"grid\"></div></section>
<section id=\"knowledge\"><h2>Knowledge / Memory Galaxy v0</h2><div class=\"kv\">Read-only vault/reference graph, not runtime memory merge.</div><div id=\"knowledgeList\" class=\"grid\"></div></section>
<section id=\"artifacts\"><h2>Artifact / Project Library</h2><div id=\"artifactList\" class=\"grid\"></div></section>
<section id=\"seo\"><h2>SEO Mission Control</h2><div class=\"card\"><h3>Draft-only SEO/AISO lane</h3><div class=\"kv\">publish disabled · outreach disabled · live metrics require approval-gated credentials</div><pre id=\"seoPayload\"></pre></div><div id=\"seoList\" class=\"grid\"></div></section>
<section id=\"operator\"><h2>Operator Loop / Judge / Evidence</h2><pre id=\"operatorPayload\"></pre></section>
<section id=\"media\"><h2>Media Studio Browser v0</h2><div class=\"kv\">Read-only. No generation. No posting.</div><div id=\"mediaList\" class=\"grid\"></div></section>
<section id=\"manage\"><h2>Manage / Update / Status</h2><pre id=\"managePayload\"></pre></section>
<section id=\"voice\"><h2>Voice / Jarvis gated panel</h2><div class=\"card\"><h3>Jarvis / Oracle Briefing</h3><div class=\"kv\">wake/show/build/act · dry-run only · no always-on microphone · no computer-control</div><pre id=\"jarvisPayload\"></pre></div><div class=\"card\"><h3>Push-to-talk v0.1</h3><div class=\"kv\">Record command captures local browser audio, stores local artefacts, returns transcript + intent preview. No execution.</div><p><button id=\"recordJarvis\">Record command</button> <button id=\"stopJarvis\" disabled>Stop</button> <button id=\"previewJarvis\">Preview typed command</button></p><textarea id=\"jarvisTranscript\">Prikaži zadnje BP24 stanje</textarea><h3>Command Preview</h3><pre id=\"jarvisCommandCard\"></pre></div><pre id=\"voicePayload\"></pre></section>
</main>
<script id=\"bootstrap\" type=\"application/json\">{_json_safe(bootstrap)}</script>
<script>
const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));
const asCard = (title, body) => '<div class="card"><h3>' + escapeHtml(title) + '</h3>' + body + '</div>';
function pill(v) {{ return '<span class="pill">' + escapeHtml(v) + '</span>'; }}
function escapeHtml(v) {{ return String(v ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c])); }}
async function j(url, opts={{}}) {{ const r = await fetch(url, {{headers:{{'content-type':'application/json'}}, ...opts}}); return await r.json(); }}
function showPre(sel, obj) {{ $(sel).textContent = JSON.stringify(obj, null, 2); }}
$$('button[data-tab]').forEach(b => b.addEventListener('click', () => {{ $$('button[data-tab]').forEach(x=>x.classList.remove('active')); $$('section').forEach(x=>x.classList.remove('active')); b.classList.add('active'); $('#' + b.dataset.tab).classList.add('active'); }}));
async function loadAll() {{
 const boot = JSON.parse($('#bootstrap').textContent); showPre('#queueSummary', boot.dashboard.queue_summary || {{}});
 const agents = await j('/api/agents'); $('#agentsList').innerHTML = agents.agents.map(a => asCard(a.name, '<div class="kv">' + escapeHtml(a.id) + ' · ' + escapeHtml(a.status) + '</div><p>' + (a.capabilities||[]).map(pill).join('') + '</p><div class="kv">Memory: ' + escapeHtml(a.memory_boundary) + '</div><div class="kv">Auth: ' + escapeHtml(a.auth_boundary) + '</div><div class="kv">Gates: ' + (a.approval_gates||[]).map(escapeHtml).join(', ') + '</div>')).join('');
 const knowledge = await j('/api/knowledge/index'); $('#knowledgeList').innerHTML = knowledge.nodes.map(n => asCard(n.label, '<div class="kv">' + escapeHtml(n.kind) + ' · exists=' + escapeHtml(n.exists) + '</div><div class="kv">' + escapeHtml(n.path) + '</div>')).join('');
 const artifacts = await j('/api/artifacts'); $('#artifactList').innerHTML = artifacts.items.slice(0,40).map(a => asCard(a.title, '<div class="kv">' + escapeHtml(a.kind) + ' · ' + escapeHtml(a.preview_type||a.suffix) + '</div><div class="kv">' + escapeHtml(a.path) + '</div>')).join('');
 const seo = await j('/api/seo'); showPre('#seoPayload', seo); $('#seoList').innerHTML = ['goals','keyword_queue','draft_queue','review_gates'].map(k => asCard(k, '<div class="kv">' + escapeHtml((seo[k]||[]).length) + ' item(s)</div>')).join('');
 const operator = await j('/api/operator-loop'); showPre('#operatorPayload', operator);
 const media = await j('/api/media'); $('#mediaList').innerHTML = media.assets.map(m => asCard(m.title, '<div class="kv">' + escapeHtml(m.mime) + ' · ' + escapeHtml(m.size_bytes) + ' bytes</div><div class="kv">' + escapeHtml(m.path) + '</div>')).join('') || '<div class="card kv">No local media assets found.</div>';
 showPre('#managePayload', await j('/api/manage/status')); showPre('#voicePayload', await j('/api/voice/status')); showPre('#jarvisPayload', await j('/api/jarvis/briefing'));
}}
$('#draftIdea').addEventListener('click', async () => showPre('#ideaResult', await j('/api/idea-factory/draft', {{method:'POST', body:JSON.stringify({{idea_text:$('#ideaText').value}})}})));
$('#createIdea').addEventListener('click', async () => showPre('#ideaResult', await j('/api/idea-factory/action', {{method:'POST', body:JSON.stringify({{idea_text:$('#ideaText').value}})}})));
let jarvisRecorder = null; let jarvisChunks = [];
async function previewJarvisCommand() {{ showPre('#jarvisCommandCard', await j('/api/jarvis/preview', {{method:'POST', body:JSON.stringify({{transcript_text:$('#jarvisTranscript').value}})}})); }}
$('#previewJarvis').addEventListener('click', previewJarvisCommand);
$('#recordJarvis').addEventListener('click', async () => {{
 if (!navigator.mediaDevices || !window.MediaRecorder) {{ showPre('#jarvisCommandCard', {{status:'browser_audio_unavailable', execution_created:false}}); return; }}
 const stream = await navigator.mediaDevices.getUserMedia({{audio:true}}); jarvisChunks = []; jarvisRecorder = new MediaRecorder(stream);
 jarvisRecorder.ondataavailable = e => {{ if (e.data && e.data.size) jarvisChunks.push(e.data); }};
 jarvisRecorder.onstop = async () => {{
   stream.getTracks().forEach(t => t.stop());
   const blob = new Blob(jarvisChunks, {{type: jarvisRecorder.mimeType || 'audio/webm'}});
   const reader = new FileReader();
   reader.onloadend = async () => showPre('#jarvisCommandCard', await j('/api/jarvis/transcribe', {{method:'POST', body:JSON.stringify({{audio_base64:String(reader.result), audio_mime:blob.type, transcript_text:$('#jarvisTranscript').value}})}}));
   reader.readAsDataURL(blob); $('#recordJarvis').disabled = false; $('#stopJarvis').disabled = true;
 }};
 jarvisRecorder.start(); $('#recordJarvis').disabled = true; $('#stopJarvis').disabled = false; showPre('#jarvisCommandCard', {{status:'recording', execution_created:false}});
}});
$('#stopJarvis').addEventListener('click', () => {{ if (jarvisRecorder && jarvisRecorder.state !== 'inactive') jarvisRecorder.stop(); }});
loadAll().catch(e => showPre('#queueSummary', {{error:String(e)}}));
</script>
</body></html>"""


class MissionControlHandler(BaseHTTPRequestHandler):
    service: AgentsOSService

    def log_message(self, fmt: str, *args: Any) -> None:  # keep smoke output clean
        return

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path == "/":
                _send_html(self, mission_control_html(self.service))
            elif path == "/api/status":
                payload = self.service.status_payload()
                payload["operator_ui"] = {"product": "Agents OS Mission Control", "local_only": True, "gateway_restart": False}
                _send_json(self, payload)
            elif path == "/api/dashboard":
                _send_json(self, self.service.dashboard_payload())
            elif path == "/api/idea-factory/schema":
                _send_json(self, self.service.idea_factory_schema_payload())
            elif path == "/api/agents":
                _send_json(self, agents_registry_payload(self.service.paths))
            elif path == "/api/knowledge/index":
                _send_json(self, knowledge_index_payload(self.service.paths))
            elif path == "/api/artifacts":
                _send_json(self, artifacts_payload(self.service.paths))
            elif path == "/api/seo":
                _send_json(self, seo_mission_control_payload(self.service.paths))
            elif path == "/api/operator-loop":
                _send_json(self, operator_loop_payload(self.service))
            elif path.startswith("/api/tasks/"):
                _send_json(self, task_detail_payload(self.service.paths, path.rsplit("/", 1)[-1]))
            elif path == "/api/media":
                _send_json(self, media_assets_payload(self.service.paths))
            elif path == "/api/manage/status":
                _send_json(self, redacted_manage_status_payload(self.service.paths))
            elif path == "/api/voice/status":
                _send_json(self, voice_status_payload(self.service.paths))
            elif path == "/api/jarvis/briefing":
                _send_json(self, jarvis_briefing_payload(self.service.paths))
            else:
                _send_json(self, {"status": "not_found", "path": path}, 404)
        except Exception as exc:  # deterministic local error payload
            _send_json(self, {"status": "error", "error": exc.__class__.__name__, "message": str(exc)}, 500)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            data = _read_json_body(self)
            if path == "/api/idea-factory/draft":
                _send_json(self, self.service.idea_factory_draft_payload(data))
            elif path == "/api/idea-factory/action":
                _send_json(self, create_idea_action(self.service, data))
            elif path == "/api/jarvis/preview":
                _send_json(self, jarvis_preview_payload(self.service.paths, data))
            elif path == "/api/jarvis/advisor":
                _send_json(self, jarvis_model_advisor_payload(self.service.paths, data))
            elif path == "/api/jarvis/transcribe":
                _send_json(self, jarvis_transcribe_payload(self.service.paths, data))
            else:
                _send_json(self, {"status": "not_found", "path": path}, 404)
        except ValueError as exc:
            _send_json(self, {"status": "error", "error": "bad_request", "message": str(exc)}, 400)
        except Exception as exc:
            _send_json(self, {"status": "error", "error": exc.__class__.__name__, "message": str(exc)}, 500)


def run_server(host: str, port: int, service: AgentsOSService) -> None:
    if host not in LOCAL_HOSTS:
        raise ValueError("Agents OS web may bind only to 127.0.0.1/localhost")
    handler = type("BoundMissionControlHandler", (MissionControlHandler,), {"service": service})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Agents OS Mission Control: http://{host}:{port}", flush=True)
    server.serve_forever()


def web_cmd(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    service = AgentsOSService(paths)
    url = f"http://{args.host}:{args.port}"
    payload = {"status": "ready", "url": url, "local_only": True, "gateway_restart": False, "state_db": str(paths.db)}
    if getattr(args, "json", False):
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if args.host not in LOCAL_HOSTS:
        print("Agents OS web may bind only to 127.0.0.1/localhost", file=sys.stderr)
        return 2
    if getattr(args, "open", False):
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    run_server(args.host, args.port, service)
    return 0


# Attach thin service helpers here to avoid widening the core runtime surface too much.
def _service_idea_factory_schema_payload(self: AgentsOSService) -> dict[str, Any]:
    return idea_factory_schema()


def _service_idea_factory_draft_payload(self: AgentsOSService, data: dict[str, Any]) -> dict[str, Any]:
    payload = draft_idea(
        data.get("idea_text") or data.get("idea") or "",
        context=data.get("context"),
        desired_output=data.get("desired_output"),
        urgency=data.get("urgency", "normal"),
        source_links=data.get("source_links") or data.get("source_link") or [],
    )
    payload["execution_created"] = False
    return payload


def _service_operator_loop_payload(self: AgentsOSService) -> dict[str, Any]:
    return operator_loop_payload(self)


AgentsOSService.idea_factory_schema_payload = _service_idea_factory_schema_payload  # type: ignore[attr-defined]
AgentsOSService.idea_factory_draft_payload = _service_idea_factory_draft_payload  # type: ignore[attr-defined]
AgentsOSService.operator_loop_payload = _service_operator_loop_payload  # type: ignore[attr-defined]
