"""Top-level CLI for local PII redaction.

The runtime redaction implementation lives in ``agent.pii_redaction``.  This
module intentionally imports it lazily so the CLI can land before the core
worker's module is present, and so normal ``hermes`` startup does not pay for
the redaction backend.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


CONFIG_SECTION = "pii_redaction"
SECURITY_SECTION = "security"
# These are optional runtime npm packages installed by `hermes redact setup`
# into the active Hermes home, not Python install-time dependencies.
RAMPART_PACKAGE_SPEC = "@nationaldesignstudio/rampart@0.1.1"
TRANSFORMERS_PACKAGE_SPEC = "@huggingface/transformers@3.7.5"


def register_redact_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "redact",
        help="Configure and run local PII redaction",
        description="Configure and run local PII redaction before model calls.",
    )
    redact_subparsers = parser.add_subparsers(dest="redact_command")

    config = redact_subparsers.add_parser(
        "config",
        help="Show or update local PII redaction configuration",
    )
    enabled = config.add_mutually_exclusive_group()
    enabled.add_argument("--enable", action="store_true", help="Enable local PII redaction")
    enabled.add_argument("--disable", action="store_true", help="Disable local PII redaction")
    endpoint_scope = config.add_mutually_exclusive_group()
    endpoint_scope.add_argument(
        "--hosted-only",
        action="store_true",
        help="Only redact requests sent to hosted model endpoints",
    )
    endpoint_scope.add_argument(
        "--all-endpoints",
        action="store_true",
        help="Redact requests for hosted and local endpoints",
    )
    failure_mode = config.add_mutually_exclusive_group()
    failure_mode.add_argument(
        "--fail-closed",
        action="store_true",
        help="Block model calls when redaction fails",
    )
    failure_mode.add_argument(
        "--fail-open",
        action="store_true",
        help="Allow model calls when redaction fails",
    )
    heuristics = config.add_mutually_exclusive_group()
    heuristics.add_argument(
        "--heuristics-only",
        action="store_true",
        help="Use built-in heuristic redaction instead of Rampart",
    )
    heuristics.add_argument(
        "--no-heuristics-only",
        action="store_true",
        help="Use the configured backend instead of heuristics-only mode",
    )
    config.add_argument("--model", metavar="<id-or-path>", help="Rampart model id or local path")

    setup = redact_subparsers.add_parser(
        "setup",
        help="Prepare the local Rampart backend under the active Hermes home",
    )
    setup.add_argument(
        "--enable",
        action="store_true",
        help="Enable local PII redaction after Rampart setup succeeds",
    )

    run = redact_subparsers.add_parser(
        "run",
        help="Redact one text input using the configured backend",
    )
    run.add_argument("--text", help="Text to redact. Reads stdin when omitted.")
    run.add_argument("--json", action="store_true", help="Emit JSON with text and stats")

    parser.set_defaults(func=cmd_redact)
    return parser


def cmd_redact(args) -> int:
    command = getattr(args, "redact_command", None)
    if command == "config" or command is None:
        return _cmd_config(args)
    if command == "setup":
        return _cmd_setup(args)
    if command == "run":
        return _cmd_run(args)
    print(f"hermes redact: unknown subcommand: {command}", file=sys.stderr)
    return 2


def _cmd_config(args) -> int:
    from hermes_cli.config import load_config, save_config

    config = load_config()
    section = _ensure_section(config)
    changed = False

    if getattr(args, "enable", False):
        section["enabled"] = True
        changed = True
    if getattr(args, "disable", False):
        section["enabled"] = False
        changed = True
    if getattr(args, "hosted_only", False):
        section["hosted_only"] = True
        changed = True
    if getattr(args, "all_endpoints", False):
        section["hosted_only"] = False
        changed = True
    if getattr(args, "fail_closed", False):
        section["fail_closed"] = True
        changed = True
    if getattr(args, "fail_open", False):
        section["fail_closed"] = False
        changed = True
    if getattr(args, "heuristics_only", False):
        _ensure_rampart(section)["heuristics_only"] = True
        section["provider"] = "rampart"
        changed = True
    if getattr(args, "no_heuristics_only", False):
        _ensure_rampart(section)["heuristics_only"] = False
        section["provider"] = "rampart"
        changed = True
    if getattr(args, "model", None):
        _ensure_rampart(section)["model"] = args.model
        changed = True

    if changed:
        save_config(config)
        config = load_config()

    _print_status(config, saved=changed)
    return 0


def _cmd_setup(args) -> int:
    from hermes_cli.config import get_hermes_home, load_config, save_config

    node = shutil.which("node")
    npm = shutil.which("npm")
    if not node:
        print("hermes redact setup: node is not on PATH.", file=sys.stderr)
        return 1
    if not npm:
        print("hermes redact setup: npm is not on PATH.", file=sys.stderr)
        return 1

    hermes_home = get_hermes_home()
    support_dir = _support_dir(hermes_home)
    support_dir.mkdir(parents=True, exist_ok=True)
    worker_path = _prepare_profile_worker(support_dir)

    print(f"Node: {node}")
    print(f"npm:  {npm}")
    print(f"Rampart support directory: {support_dir}")
    print(f"Installing {RAMPART_PACKAGE_SPEC} and {TRANSFORMERS_PACKAGE_SPEC} ...")

    try:
        subprocess.run(
            [npm, "install", RAMPART_PACKAGE_SPEC, TRANSFORMERS_PACKAGE_SPEC],
            cwd=str(support_dir),
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"hermes redact setup: npm install failed with exit code {exc.returncode}.", file=sys.stderr)
        return exc.returncode or 1
    except OSError as exc:
        print(f"hermes redact setup: failed to run npm: {exc}", file=sys.stderr)
        return 1

    config = load_config()
    section = _ensure_section(config)
    section["provider"] = "rampart"
    rampart = _ensure_rampart(section)
    rampart["command"] = shlex.join([node, str(worker_path)])
    rampart["package"] = RAMPART_PACKAGE_SPEC
    rampart["transformers_package"] = TRANSFORMERS_PACKAGE_SPEC
    rampart["install_dir"] = str(support_dir)
    if getattr(args, "enable", False):
        section["enabled"] = True
    save_config(config)

    print("Rampart setup complete.")
    if getattr(args, "enable", False):
        print("Local PII redaction enabled.")
    else:
        print("Enable with: hermes redact config --enable")
    return 0


def _cmd_run(args) -> int:
    text = getattr(args, "text", None)
    if text is None:
        if sys.stdin.isatty():
            print("hermes redact run: provide --text or pipe input on stdin.", file=sys.stderr)
            return 2
        text = sys.stdin.read()

    try:
        redacted, stats = _redact_text(text)
    except RuntimeError as exc:
        print(f"hermes redact run: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"hermes redact run: redaction failed: {exc}", file=sys.stderr)
        return 1

    if stats.get("skipped"):
        reason = stats.get("skipped_reason") or "unknown"
        print(f"hermes redact run: redaction skipped ({reason}).", file=sys.stderr)
    if getattr(args, "json", False):
        print(json.dumps({"text": redacted, "stats": stats}, indent=2, sort_keys=True))
    else:
        print(redacted, end="" if redacted.endswith("\n") else "\n")
    return 0


def _ensure_section(config: dict[str, Any]) -> dict[str, Any]:
    security = config.get(SECURITY_SECTION)
    if not isinstance(security, dict):
        security = {}
        config[SECURITY_SECTION] = security
    section = security.get(CONFIG_SECTION)
    if not isinstance(section, dict):
        section = {}
        security[CONFIG_SECTION] = section
    return section


def _read_section(config: dict[str, Any]) -> dict[str, Any]:
    security = config.get(SECURITY_SECTION)
    if isinstance(security, dict):
        section = security.get(CONFIG_SECTION)
        if isinstance(section, dict):
            return dict(section)
    return {}


def _ensure_rampart(section: dict[str, Any]) -> dict[str, Any]:
    rampart = section.get("rampart")
    if not isinstance(rampart, dict):
        rampart = {}
        section["rampart"] = rampart
    return rampart


def _support_dir(hermes_home: Path) -> Path:
    return hermes_home / "support" / "pii-redaction" / "rampart"


def _prepare_profile_worker(support_dir: Path) -> Path:
    source = Path(__file__).resolve().parents[1] / "agent" / "rampart_pii_worker.mjs"
    target = support_dir / "rampart_pii_worker.mjs"
    shutil.copyfile(source, target)
    return target


def _load_runtime_module():
    try:
        from agent import pii_redaction  # type: ignore[attr-defined]
    except Exception as exc:
        raise RuntimeError(
            "agent.pii_redaction is not available in this checkout yet. "
            "Install or update the Hermes runtime redaction module, then retry."
        ) from exc
    return pii_redaction


def _runtime_config(config: dict[str, Any] | None = None) -> Any:
    module = _load_runtime_module()
    loader = getattr(module, "load_pii_redaction_config", None)
    if not callable(loader):
        return (config or {}).get(CONFIG_SECTION, {})
    return loader(config) if config is not None else loader()


def _redact_text(text: str) -> tuple[str, dict[str, Any]]:
    module = _load_runtime_module()
    redactor = getattr(module, "redact_text_for_llm", None)
    if not callable(redactor):
        raise RuntimeError("agent.pii_redaction does not expose redact_text_for_llm.")

    result = redactor(text, pii_config=_runtime_config())
    return _normalize_redaction_result(result, fallback_text=text)


def _normalize_redaction_result(result: Any, *, fallback_text: str) -> tuple[Any, dict[str, Any]]:
    if isinstance(result, tuple) and result:
        text = result[0]
        stats = _stats_from_obj(result[1]) if len(result) > 1 else {}
        return text, stats
    if isinstance(result, dict):
        text = result.get("text", result.get("redacted_text", fallback_text))
        stats = _stats_from_obj(result.get("stats", result))
        return text, stats
    text = getattr(result, "text", getattr(result, "redacted_text", result))
    stats = _stats_from_obj(getattr(result, "stats", result))
    return text, stats


def _stats_from_obj(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {
            str(k): v
            for k, v in value.items()
            if k not in {"text", "redacted_text", "messages", "redacted_messages"}
        }
    data: dict[str, Any] = {}
    for key in ("replacements", "entities", "backend", "duration_ms", "failed", "error"):
        if hasattr(value, key):
            data[key] = getattr(value, key)
    return data


def _print_status(config: dict[str, Any], *, saved: bool) -> None:
    raw = _read_section(config)
    try:
        effective = _config_to_dict(_runtime_config(config))
    except RuntimeError:
        effective = raw
    except Exception as exc:
        effective = raw
        print(f"Runtime config unavailable: {exc}", file=sys.stderr)

    status = {**raw, **effective}
    ready = _setup_ready(status)

    if saved:
        print("Saved local PII redaction config.")
    print(f"Enabled:        {_format_bool(status.get('enabled', False))}")
    print(f"Backend:        {_default_backend(status)}")
    print(f"Model:          {_rampart_value(status, 'model') or '(default)'}")
    print(f"Hosted only:    {_format_bool(status.get('hosted_only', True))}")
    print(f"Fail closed:    {_format_bool(status.get('fail_closed', True))}")
    print(f"Heuristics only: {_format_bool(_rampart_value(status, 'heuristics_only', False))}")
    print(f"Setup ready:    {_format_bool(ready)}")
    if not ready:
        print("Setup command:  hermes redact setup")


def _setup_ready(status: dict[str, Any]) -> bool:
    if "setup_ready" in status:
        return bool(status["setup_ready"])
    if status.get("provider") == "heuristics":
        return True

    def installed(base: Path) -> bool:
        rampart_ok = (base / "node_modules" / "@nationaldesignstudio" / "rampart").exists()
        if _rampart_value(status, "heuristics_only"):
            return rampart_ok
        transformers_ok = (base / "node_modules" / "@huggingface" / "transformers").exists()
        return rampart_ok and transformers_ok

    command = _rampart_value(status, "command")
    if command:
        try:
            parts = shlex.split(str(command))
        except ValueError:
            parts = []
        if len(parts) >= 2 and Path(parts[1]).exists():
            worker = Path(parts[1])
            return installed(worker.parent)
    rampart_dir = _rampart_value(status, "install_dir")
    if rampart_dir:
        base = Path(str(rampart_dir))
    else:
        try:
            from hermes_cli.config import get_hermes_home

            base = _support_dir(get_hermes_home())
        except Exception:
            return False
    return installed(base)


def _config_to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    data: dict[str, Any] = {}
    for key in (
        "enabled",
        "provider",
        "model",
        "hosted_only",
        "fail_closed",
        "heuristics_only",
        "rampart",
        "setup_ready",
    ):
        if hasattr(value, key):
            data[key] = getattr(value, key)
    return data


def _default_backend(status: dict[str, Any]) -> str:
    provider = status.get("provider") or status.get("backend")
    return str(provider or "rampart")


def _rampart_value(status: dict[str, Any], key: str, default: Any = None) -> Any:
    rampart = status.get("rampart")
    if isinstance(rampart, dict) and key in rampart:
        return rampart.get(key)
    return status.get(key, default)


def _format_bool(value: Any) -> str:
    return "yes" if bool(value) else "no"
