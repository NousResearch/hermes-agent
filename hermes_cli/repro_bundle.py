"""``hermes debug reproduce`` — local reproduction capsule.

Bundles what a maintainer needs to reproduce a bug locally into a single
``.zip``: build info, the redacted effective config, a plugin/skill
inventory, correlated log excerpts, and an OPT-IN sanitized session
export. Local-only — this module never uploads anything, never executes
bundle contents, and session data is included only when explicitly
requested.

Extends the existing ``hermes_cli/debug.py`` collection machinery
(``collect_share_bundle``) rather than building a parallel redactor/log
collector — the report/log files in this bundle are byte-identical to
what ``hermes debug share`` would upload.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["build_repro_bundle", "run_debug_reproduce"]

_BUNDLE_FORMAT = "hermes-repro/1"


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _build_manifest(*, redacted: bool, session_included: bool) -> Dict[str, Any]:
    return {
        "format": _BUNDLE_FORMAT,
        "created": datetime.now(timezone.utc).isoformat(),
        "build_sha": _git_sha(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "redacted": redacted,
        "session_included": session_included,
    }


def _plugin_inventory() -> List[Dict[str, str]]:
    """Name/version/kind/source for bundled + user plugin manifests.

    Read-only manifest parsing only — never drives plugin discovery/load
    (which imports and executes plugin modules).
    """
    from utils import fast_safe_load

    from hermes_cli.plugins import get_bundled_plugins_dir
    from hermes_constants import get_hermes_home

    entries: List[Dict[str, str]] = []

    def _scan(root: Path, source: str, *, skip: frozenset = frozenset()) -> None:
        if not root.is_dir():
            return
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name in skip:
                continue
            manifest_file = child / "plugin.yaml"
            if not manifest_file.exists():
                manifest_file = child / "plugin.yml"
            if manifest_file.exists():
                try:
                    data = fast_safe_load(manifest_file.read_text(encoding="utf-8")) or {}
                except Exception:
                    continue
                entries.append({
                    "name": str(data.get("name", child.name)),
                    "version": str(data.get("version", "")),
                    "kind": str(data.get("kind", "standalone")),
                    "source": source,
                })
                continue
            for grandchild in sorted(child.iterdir()):
                if not grandchild.is_dir():
                    continue
                gc_manifest = grandchild / "plugin.yaml"
                if not gc_manifest.exists():
                    gc_manifest = grandchild / "plugin.yml"
                if not gc_manifest.exists():
                    continue
                try:
                    data = fast_safe_load(gc_manifest.read_text(encoding="utf-8")) or {}
                except Exception:
                    continue
                entries.append({
                    "name": str(data.get("name", grandchild.name)),
                    "version": str(data.get("version", "")),
                    "kind": str(data.get("kind", "standalone")),
                    "source": source,
                })

    repo_plugins = get_bundled_plugins_dir()
    _scan(repo_plugins, "bundled", skip=frozenset({"memory", "context_engine", "platforms", "model-providers"}))
    _scan(repo_plugins / "platforms", "bundled")
    _scan(get_hermes_home() / "plugins", "user")
    return entries


def _redacted_config_yaml() -> str:
    import yaml

    from agent.redact import redact_sensitive_text
    from hermes_cli.config import load_config_readonly

    config = load_config_readonly()
    raw = yaml.dump(config, sort_keys=True, default_flow_style=False)
    return redact_sensitive_text(raw, force=True)


def _sanitized_session_export(session_id: str) -> Optional[List[Dict[str, Any]]]:
    from agent.redact import redact_sensitive_text
    from hermes_state import SessionDB

    db = SessionDB(read_only=True)
    messages = db.get_messages_as_conversation(session_id)
    if not messages:
        return None
    sanitized: List[Dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            content = redact_sensitive_text(content, force=True)
        elif isinstance(content, list):
            content = [
                {**part, "text": redact_sensitive_text(part["text"], force=True)}
                if isinstance(part, dict) and isinstance(part.get("text"), str)
                else part
                for part in content
            ]
        sanitized.append({**msg, "content": content})
    return sanitized


def build_repro_bundle(
    output_path: Path, *, session_id: Optional[str] = None,
    log_lines: int = 200, redact: bool = True,
) -> Dict[str, Any]:
    """Build the local reproduction bundle at ``output_path``. Returns a summary."""
    from hermes_cli.debug import collect_share_bundle

    share = collect_share_bundle(log_lines=log_lines, redact=redact)
    plugins = _plugin_inventory()
    config_yaml = _redacted_config_yaml()

    session_data: Optional[List[Dict[str, Any]]] = None
    if session_id:
        session_data = _sanitized_session_export(session_id)

    manifest = _build_manifest(redacted=redact, session_included=session_data is not None)

    readme_lines = [
        f"# Hermes reproduction bundle ({manifest['created']})",
        "",
        f"- build_sha: {manifest['build_sha']}",
        f"- os: {manifest['os']}",
        f"- python: {manifest['python']}",
        f"- redacted: {manifest['redacted']}",
        f"- session included: {manifest['session_included']}",
        "",
        "## Contents",
        "",
        "- `manifest.json` — build/environment metadata",
        "- `report.txt` — system dump + log tails",
        "- `logs/*.log` — full log files (redacted unless --no-redact was passed)",
        "- `plugins.json` — bundled + user plugin inventory (name/version/kind, no code)",
        "- `config_redacted.yaml` — effective config, redacted",
    ]
    if session_data is not None:
        readme_lines.append("- `session.json` — sanitized conversation export (opt-in)")
    else:
        readme_lines.append("- session data: NOT included (pass --session <id> to include)")
    readme_lines += [
        "",
        "## Reproduction steps",
        "",
        "1. Check out `build_sha` above.",
        "2. Review `config_redacted.yaml` for the relevant provider/model/plugin settings.",
        "3. Review `report.txt` and `logs/*.log` for the failure context.",
        "4. This bundle never auto-executes anything — inspect before running "
        "any command it references.",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        zf.writestr("report.txt", share.get("report", ""))
        for label, text in share.items():
            if label == "report":
                continue
            zf.writestr(f"logs/{label}", text)
        zf.writestr("plugins.json", json.dumps(plugins, indent=2))
        zf.writestr("config_redacted.yaml", config_yaml)
        if session_data is not None:
            zf.writestr("session.json", json.dumps(session_data, indent=2))
        zf.writestr("README.md", "\n".join(readme_lines))

    return {"manifest": manifest, "plugins": len(plugins), "output": str(output_path)}


def inspect_bundle(bundle_path: Path) -> Dict[str, Any]:
    """List a bundle's contents and manifest WITHOUT executing anything in it."""
    with zipfile.ZipFile(bundle_path, "r") as zf:
        names = zf.namelist()
        manifest: Dict[str, Any] = {}
        if "manifest.json" in names:
            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
    return {"files": names, "manifest": manifest}


def run_debug_reproduce(args: argparse.Namespace) -> None:
    action = getattr(args, "reproduce_action", None) or "build"

    if action == "inspect":
        bundle_path = Path(args.bundle)
        if not bundle_path.exists():
            print(f"error: bundle not found: {bundle_path}", file=sys.stderr)
            sys.exit(1)
        info = inspect_bundle(bundle_path)
        print(f"format: {info['manifest'].get('format', 'unknown')}")
        print(f"created: {info['manifest'].get('created', 'unknown')}")
        print(f"redacted: {info['manifest'].get('redacted', 'unknown')}")
        print(f"session included: {info['manifest'].get('session_included', 'unknown')}")
        print("files:")
        for name in info["files"]:
            print(f"  - {name}")
        return

    output = getattr(args, "output", None)
    output_path = Path(output) if output else Path(
        f"hermes-repro-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.zip"
    )
    session_id = getattr(args, "session", None)
    log_lines = getattr(args, "lines", 200)
    redact = not getattr(args, "no_redact", False)

    if session_id:
        print(f"including session {session_id!r} — sanitized, opt-in")
    else:
        print("no --session given — session data NOT included")

    summary = build_repro_bundle(
        output_path, session_id=session_id, log_lines=log_lines, redact=redact
    )
    print(f"wrote {summary['output']} ({summary['plugins']} plugins inventoried)")
