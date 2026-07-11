from __future__ import annotations

import concurrent.futures
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from hermes_constants import get_hermes_home
except Exception:
    def get_hermes_home() -> Path:
        return Path.home() / ".hermes"


PLUGIN_ID = "openmanus"
TOOLSET = "openmanus"
MAX_PROMPT_CHARS = 12_000
MAX_ITEMS = 20
MAX_PARALLEL = 4
MAX_TIMEOUT_SECONDS = 3_600
MAX_STEPS = 40
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")


CAPABILITIES_SCHEMA = {
    "name": "openmanus_capabilities",
    "description": "Report the pinned OpenManus revision, available agent modes, and the Hermes safety boundary.",
    "parameters": {"type": "object", "properties": {}},
}

RUN_SCHEMA = {
    "name": "openmanus_run",
    "description": (
        "Delegate one bounded task to the pinned OpenManus agent. The default is a "
        "dry-run plan; live execution requires an authorised workspace, explicit "
        "side-effect acknowledgement, and an isolated OpenManus runtime. Suitable "
        "for Hermes agents and MoA workers."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "Task for OpenManus."},
            "workspace": {"type": "string", "description": "Directory inside the configured workspace root."},
            "dry_run": {"type": "boolean", "description": "Plan only. Defaults to true."},
            "allow_side_effects": {"type": "boolean", "description": "Must be true for live execution."},
            "acknowledge_side_effects": {"type": "boolean", "description": "Explicit confirmation for live execution."},
            "allow_network": {"type": "boolean", "description": "Request browser/network tools; config must also allow them."},
            "agent_mode": {"type": "string", "enum": ["manus", "data_analysis"]},
            "max_steps": {"type": "integer", "minimum": 1, "maximum": MAX_STEPS},
            "timeout_seconds": {"type": "integer", "minimum": 10, "maximum": MAX_TIMEOUT_SECONDS},
        },
        "required": ["task"],
    },
}

WIDE_RESEARCH_SCHEMA = {
    "name": "openmanus_wide_research",
    "description": (
        "Run independent OpenManus workers in a bounded parallel fan-out, then "
        "optionally ask the active Hermes host LLM to synthesise their redacted receipts."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_ITEMS,
                "items": {"type": "string"},
                "description": "Independent task prompts.",
            },
            "workspace": {"type": "string", "description": "Authorised workspace root for isolated worker directories."},
            "dry_run": {"type": "boolean", "description": "Plan only. Defaults to true."},
            "allow_side_effects": {"type": "boolean", "description": "Must be true for live execution."},
            "acknowledge_side_effects": {"type": "boolean", "description": "Explicit confirmation for live execution."},
            "allow_network": {"type": "boolean", "description": "Request browser/network tools; config must also allow them."},
            "max_parallel": {"type": "integer", "minimum": 1, "maximum": MAX_PARALLEL},
            "synthesize": {"type": "boolean", "description": "Use Hermes host LLM to combine worker outputs."},
        },
        "required": ["items"],
    },
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def plugin_dir() -> Path:
    return Path(__file__).resolve().parent


def source_root() -> Path:
    return repo_root() / "vendor" / "openmanus"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _load_entry() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly() or {}
    except Exception:
        return {}
    plugins = cfg.get("plugins") if isinstance(cfg, dict) else {}
    entries = plugins.get("entries") if isinstance(plugins, dict) else {}
    entry = entries.get(PLUGIN_ID) if isinstance(entries, dict) else {}
    return entry if isinstance(entry, dict) else {}


def _llm_entry(entry: dict[str, Any]) -> dict[str, Any]:
    value = entry.get("llm")
    return value if isinstance(value, dict) else {}


def _api_key_env(entry: dict[str, Any]) -> str:
    name = str(_llm_entry(entry).get("api_key_env") or "OPENMANUS_API_KEY").strip()
    return name if _ENV_NAME.fullmatch(name) else "OPENMANUS_API_KEY"


def _workspace_root(entry: dict[str, Any]) -> Path:
    raw = str(entry.get("workspace_root") or "").strip()
    if not raw:
        raise ValueError(
            "OpenManus is not configured: set plugins.entries.openmanus.workspace_root"
        )
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"configured OpenManus workspace does not exist: {root}")
    return root


def resolve_workspace(requested: str | None = None) -> Path:
    root = _workspace_root(_load_entry())
    candidate = Path(requested).expanduser() if requested else root
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("workspace escapes the configured OpenManus workspace root") from exc
    if not resolved.is_dir():
        raise ValueError(f"workspace directory does not exist: {resolved}")
    return resolved


def _source_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(source_root()), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def check_available() -> bool:
    return source_root().is_dir() and (source_root() / "main.py").is_file()


def _runtime_command(entry: dict[str, Any]) -> list[str]:
    configured = entry.get("runtime_command")
    if isinstance(configured, list) and configured and all(isinstance(x, str) for x in configured):
        return [str(x) for x in configured]
    if isinstance(configured, str) and configured.strip():
        return shlex.split(configured, posix=False)
    if shutil.which("uv"):
        return ["uv", "run", "--with-requirements", str(source_root() / "requirements.txt")]
    return [sys.executable]


def _build_command(entry: dict[str, Any], workspace: Path, run_root: Path, args: dict[str, Any]) -> list[str]:
    llm = _llm_entry(entry)
    return _runtime_command(entry) + [
        str(plugin_dir() / "runner.py"),
        "--source-root", str(source_root()),
        "--workspace-root", str(workspace),
        "--run-root", str(run_root),
        "--model", str(llm.get("model") or ""),
        "--base-url", str(llm.get("base_url") or ""),
        "--api-type", str(llm.get("api_type") or "openai"),
        "--api-key-env", _api_key_env(entry),
        "--agent-mode", str(args.get("agent_mode") or "manus"),
        "--max-steps", str(max(1, min(int(args.get("max_steps") or 20), MAX_STEPS))),
        "--allow-network" if bool(args.get("allow_network")) else "--no-network",
    ]


def _safe_environment(entry: dict[str, Any]) -> dict[str, str]:
    keep = {
        "PATH",
        "SystemRoot",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "HOME",
        "LANG",
        "APPDATA",
        "LOCALAPPDATA",
        "ProgramFiles",
        "ProgramFiles(x86)",
    }
    env = {key: value for key, value in os.environ.items() if key in keep}
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    secret_name = _api_key_env(entry)
    secret = os.environ.get(secret_name)
    if secret:
        env[secret_name] = secret
    return env


def _redact(text: str, entry: dict[str, Any]) -> str:
    secret = os.environ.get(_api_key_env(entry))
    return text.replace(secret, "[REDACTED]") if secret else text


def _write_receipt(run_root: Path, payload: dict[str, Any]) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)
    receipt = run_root / "receipt.json"
    receipt.write_text(_json(payload) + "\n", encoding="utf-8")
    return receipt


def _new_run_root() -> tuple[str, Path]:
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:10]}"
    return run_id, get_hermes_home() / "openmanus" / "runs" / run_id


def _validate_request(args: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    entry = _load_entry()
    if not check_available():
        raise ValueError("OpenManus submodule is missing or has no main.py")
    task = str(args.get("task") or args.get("prompt") or "").strip()
    if not task:
        raise ValueError("task is required")
    if len(task) > MAX_PROMPT_CHARS:
        raise ValueError(f"task exceeds the {MAX_PROMPT_CHARS}-character limit")
    dry_run = bool(args.get("dry_run", True))
    if not dry_run and not (args.get("allow_side_effects") and args.get("acknowledge_side_effects")):
        raise PermissionError("live OpenManus execution requires allow_side_effects and acknowledge_side_effects")
    if bool(args.get("allow_network")) and not bool(entry.get("allow_network", False)):
        raise PermissionError("network/browser execution is disabled; enable it explicitly in plugins.entries.openmanus")
    workspace = resolve_workspace(args.get("workspace"))
    llm = _llm_entry(entry)
    if not dry_run:
        if not llm.get("model") or not llm.get("base_url"):
            raise ValueError("configure plugins.entries.openmanus.llm.model and llm.base_url")
        if not os.environ.get(_api_key_env(entry)):
            raise ValueError(f"secret environment variable {_api_key_env(entry)} is not set")
    normalized = dict(args)
    normalized["task"] = task
    normalized["dry_run"] = dry_run
    normalized["timeout_seconds"] = max(10, min(int(args.get("timeout_seconds") or 600), MAX_TIMEOUT_SECONDS))
    return normalized, workspace


def run_task(args: dict[str, Any] | None = None) -> dict[str, Any]:
    args = args or {}
    try:
        normalized, workspace = _validate_request(args)
        run_id, run_root = _new_run_root()
        entry = _load_entry()
        command = _build_command(entry, workspace, run_root, normalized)
        payload: dict[str, Any] = {
            "ok": True,
            "run_id": run_id,
            "status": "planned" if normalized["dry_run"] else "running",
            "workspace": str(workspace),
            "source_revision": _source_revision(),
            "agent_mode": normalized.get("agent_mode") or "manus",
            "max_steps": max(1, min(int(normalized.get("max_steps") or 20), MAX_STEPS)),
            "command": command,
            "receipt_path": str(run_root / "receipt.json"),
        }
        if normalized["dry_run"]:
            _write_receipt(run_root, payload)
            return payload

        try:
            completed = subprocess.run(
                command,
                input=normalized["task"],
                capture_output=True,
                text=True,
                cwd=str(source_root()),
                env=_safe_environment(entry),
                timeout=normalized["timeout_seconds"],
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            payload.update(
                {
                    "status": "completed" if completed.returncode == 0 else "failed",
                    "exit_code": completed.returncode,
                    "stdout": _redact(completed.stdout[-20_000:], entry),
                    "stderr": _redact(completed.stderr[-10_000:], entry),
                }
            )
        except subprocess.TimeoutExpired as exc:
            payload.update(
                {
                    "status": "timed_out",
                    "exit_code": None,
                    "stdout": _redact(str(exc.stdout or ""), entry),
                    "stderr": _redact(str(exc.stderr or ""), entry),
                }
            )
        except Exception as exc:
            payload.update({"status": "failed", "exit_code": None, "error": _redact(str(exc), entry)})
        payload["ok"] = payload["status"] == "completed"
        _write_receipt(run_root, payload)
        return payload
    except Exception as exc:
        return {"ok": False, "status": "blocked", "error": str(exc)}


def handle_run(args: dict[str, Any] | None = None, **_: Any) -> str:
    return _json(run_task(args))


def _item_prompt(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        return str(item.get("task") or item.get("prompt") or "").strip()
    return ""


def wide_research(args: dict[str, Any] | None = None, *, host_llm: Any = None) -> dict[str, Any]:
    args = args or {}
    raw_items = args.get("items") or []
    if not isinstance(raw_items, list) or not 1 <= len(raw_items) <= MAX_ITEMS:
        return {"ok": False, "status": "blocked", "error": f"items must contain 1-{MAX_ITEMS} prompts"}
    prompts = [_item_prompt(item) for item in raw_items]
    if any(not prompt for prompt in prompts):
        return {"ok": False, "status": "blocked", "error": "every wide-research item needs a non-empty prompt"}
    try:
        base_workspace = resolve_workspace(args.get("workspace"))
    except Exception as exc:
        return {"ok": False, "status": "blocked", "error": str(exc)}
    dry_run = bool(args.get("dry_run", True))
    if not dry_run and not (args.get("allow_side_effects") and args.get("acknowledge_side_effects")):
        return {"ok": False, "status": "blocked", "error": "live wide research requires explicit side-effect acknowledgement"}
    configured_parallel = max(1, min(int(_load_entry().get("max_parallel") or MAX_PARALLEL), MAX_PARALLEL))
    parallel = max(1, min(int(args.get("max_parallel") or configured_parallel), configured_parallel, len(prompts)))
    parent_id, parent_root = _new_run_root()

    def one(index_prompt: tuple[int, str]) -> dict[str, Any]:
        index, prompt = index_prompt
        item_args = dict(args)
        item_args["task"] = prompt
        item_args["dry_run"] = dry_run
        if not dry_run:
            item_workspace = base_workspace / ".hermes-openmanus" / parent_id / f"item-{index + 1:02d}"
            item_workspace.mkdir(parents=True, exist_ok=True)
            item_args["workspace"] = str(item_workspace)
        return run_task(item_args)

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel, thread_name_prefix="openmanus") as executor:
        results = list(executor.map(one, enumerate(prompts)))
    payload: dict[str, Any] = {
        "ok": all(result.get("ok") for result in results),
        "status": "completed" if all(result.get("status") == "completed" for result in results) else "partial",
        "run_id": parent_id,
        "parallelism": parallel,
        "items": results,
        "receipt_path": str(parent_root / "receipt.json"),
        "feature": "bounded_parallel_research",
    }
    if bool(args.get("synthesize")) and host_llm and results:
        evidence = "\n\n".join(
            f"Item {idx + 1}:\n{str(result.get('stdout') or result.get('error') or '')[-6000:]}"
            for idx, result in enumerate(results)
        )[:30_000]
        try:
            synthesis = host_llm.complete(
                messages=[
                    {"role": "system", "content": "Synthesize independent OpenManus worker receipts. Preserve uncertainty and do not invent missing evidence."},
                    {"role": "user", "content": evidence},
                ],
                max_tokens=2500,
                purpose="openmanus-wide-research-synthesis",
            )
            payload["synthesis"] = {"text": synthesis.text, "provider": synthesis.provider, "model": synthesis.model}
        except Exception as exc:
            payload["synthesis_error"] = str(exc)
    _write_receipt(parent_root, payload)
    return payload


def handle_wide_research(args: dict[str, Any] | None = None, *, host_llm: Any = None, **_: Any) -> str:
    return _json(wide_research(args, host_llm=host_llm))


def capabilities() -> dict[str, Any]:
    source = source_root()
    return {
        "ok": True,
        "plugin": PLUGIN_ID,
        "submodule": "FoundationAgents/OpenManus",
        "source_path": str(source),
        "source_revision": _source_revision(),
        "available": check_available(),
        "agent_modes": ["manus", "data_analysis"],
        "hermes_integration": ["openmanus_run", "openmanus_wide_research", "host_llm_synthesis"],
        "safety": {
            "dry_run_default": True,
            "workspace_confinement": True,
            "symlink_escape_rejected": True,
            "live_side_effect_ack_required": True,
            "network_disabled_by_default": True,
            "mcp_not_auto_attached": True,
            "redacted_receipts": True,
        },
    }


def handle_capabilities(args: dict[str, Any] | None = None, **_: Any) -> str:
    return _json(capabilities())


def handle_slash(command: str = "", **_: Any) -> str:
    action = (command or "").strip().split(maxsplit=1)[0].lower() if command else "status"
    if action in {"status", "capabilities"}:
        return _json(capabilities())
    return "Use /openmanus status or /openmanus capabilities. Live runs require the openmanus tool with explicit acknowledgement."
