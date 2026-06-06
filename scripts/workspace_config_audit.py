#!/usr/bin/env python3
"""Read-only audit for `.env` and `.gitignore` drift across local projects.

The audit records environment variable names only. It deliberately never stores
or prints values from `.env` files.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".next",
    "dist",
    "build",
    "coverage",
    ".cache",
    ".turbo",
}

SOURCE_REF_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".next",
    "dist",
    "build",
    "coverage",
    ".cache",
    ".turbo",
    "docs",
    "test",
    "tests",
}

SOURCE_EXTENSIONS = {
    ".bash",
    ".cjs",
    ".js",
    ".jsx",
    ".mjs",
    ".py",
    ".sh",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
}

EXAMPLE_ENV_SUFFIXES = (
    ".example",
    ".sample",
    ".template",
    ".tmpl",
    ".dist",
    ".defaults",
)

ENV_ASSIGN_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")
ENV_KEY_PATTERN = r"[A-Z_][A-Z0-9_]*"
PY_ENV_RE = re.compile(
    rf"(?:os\.)?(?:getenv|environ\.get)\(\s*['\"]({ENV_KEY_PATTERN})['\"]"
    rf"|os\.environ\[\s*['\"]({ENV_KEY_PATTERN})['\"]\s*\]"
)
JS_ENV_RE = re.compile(
    rf"(?:process|import\.meta)\.env\.({ENV_KEY_PATTERN})"
    rf"|process\.env\[\s*['\"]({ENV_KEY_PATTERN})['\"]\s*\]"
)
SHELL_ENV_RE = re.compile(rf"\$\{{({ENV_KEY_PATTERN})(?::-[^}}]*)?\}}")
COMMON_ENV_KEYS = {
    "CI",
    "COLORFGBG",
    "COLORTERM",
    "DBUS_SESSION_BUS_ADDRESS",
    "EDITOR",
    "HOME",
    "LANG",
    "LC_ALL",
    "OLDPWD",
    "PAGER",
    "PATH",
    "PWD",
    "SHELL",
    "TERM",
    "TMP",
    "TMPDIR",
    "TZ",
    "USER",
    "USERNAME",
    "VIRTUAL_ENV",
    "VISUAL",
    "XDG_RUNTIME_DIR",
    "XDG_STATE_HOME",
}
CONFIG_KEY_HINTS = (
    "API",
    "AUTH",
    "BASE",
    "CONFIG",
    "CREDENTIAL",
    "CWD",
    "DATABASE",
    "DB",
    "DEBUG",
    "ENABLED",
    "ENV",
    "HOST",
    "ID",
    "KEY",
    "MODEL",
    "PASSWORD",
    "PATH",
    "PORT",
    "PROVIDER",
    "PROXY",
    "REGION",
    "SECRET",
    "TOKEN",
    "TIMEOUT",
    "URL",
    "WORKSPACE",
)


def _read_text(path: Path, max_bytes: int = 400_000) -> str:
    try:
        with path.open("rb") as fh:
            return fh.read(max_bytes).decode("utf-8", errors="replace")
    except (FileNotFoundError, IsADirectoryError, OSError):
        return ""


def _run_git(root: Path, args: list[str]) -> list[str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except Exception:
        return []
    if proc.returncode != 0:
        return []
    return [line for line in proc.stdout.splitlines() if line.strip()]


def _is_example_env(path: Path) -> bool:
    name = path.name.lower()
    return name == ".env.example" or name.endswith(EXAMPLE_ENV_SUFFIXES)


def _is_env_file(path: Path) -> bool:
    name = path.name
    return name == ".env" or name.startswith(".env.")


def _iter_project_files(root: Path):
    try:
        children = list(root.iterdir()) if root.exists() else []
    except OSError:
        return
    for child in children:
        if child.is_symlink():
            continue
        if child.name in SKIP_DIRS:
            continue
        try:
            is_dir = child.is_dir()
        except OSError:
            continue
        if is_dir:
            yield from _iter_project_files(child)
        else:
            yield child


def _iter_source_files(root: Path):
    try:
        children = list(root.iterdir()) if root.exists() else []
    except OSError:
        return
    for child in children:
        if child.is_symlink():
            continue
        if child.name in SOURCE_REF_SKIP_DIRS:
            continue
        try:
            is_dir = child.is_dir()
        except OSError:
            continue
        if is_dir:
            yield from _iter_source_files(child)
        else:
            yield child


def _is_probable_config_key(key: str) -> bool:
    if key in COMMON_ENV_KEYS:
        return False
    return "_" in key or any(hint in key for hint in CONFIG_KEY_HINTS)


def _parse_env_keys(paths: list[Path]) -> dict[str, list[str]]:
    keys: dict[str, list[str]] = {}
    for path in paths:
        for line in _read_text(path).splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            match = ENV_ASSIGN_RE.match(line)
            if not match:
                continue
            key = match.group(1)
            keys.setdefault(key, []).append(str(path))
    return keys


def _extract_env_refs(root: Path) -> dict[str, list[str]]:
    refs: dict[str, list[str]] = {}
    for path in _iter_source_files(root):
        if _is_env_file(path):
            continue
        if path.suffix.lower() not in SOURCE_EXTENSIONS:
            continue
        text = _read_text(path)
        if not text:
            continue
        found: set[str] = set()
        for regex in (PY_ENV_RE, JS_ENV_RE, SHELL_ENV_RE):
            for match in regex.finditer(text):
                key = next((group for group in match.groups() if group), "")
                if key and _is_probable_config_key(key):
                    found.add(key)
        for key in sorted(found):
            refs.setdefault(key, []).append(str(path))
    return refs


def _gitignore_covers_env(root: Path) -> bool:
    gitignore = root / ".gitignore"
    lines = []
    for raw in _read_text(gitignore).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    if ".env" in lines or "/.env" in lines:
        return True
    return any(line in {".env*", ".env.*", "/.env*", "/.env.*"} for line in lines)


def _finding(severity: str, code: str, message: str, evidence: str = "") -> dict[str, str]:
    return {
        "severity": severity,
        "code": code,
        "message": message,
        "evidence": evidence,
    }


def _project_markers(root: Path) -> bool:
    if not root.is_dir():
        return False
    if (root / ".git").exists() or (root / ".gitignore").exists():
        return True
    try:
        return any(_is_env_file(child) for child in root.iterdir() if child.is_file())
    except OSError:
        return False


def discover_projects(root: str | Path, max_depth: int = 4) -> list[Path]:
    base = Path(root).expanduser().resolve()
    if not base.exists():
        return []
    if _project_markers(base):
        return [base]

    projects: list[Path] = []

    def walk(path: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            children = list(path.iterdir())
        except OSError:
            return
        for child in children:
            if child.is_symlink() or child.name in SKIP_DIRS:
                continue
            try:
                is_dir = child.is_dir()
            except OSError:
                continue
            if not is_dir:
                continue
            if _project_markers(child):
                projects.append(child.resolve())
                continue
            walk(child, depth + 1)

    walk(base, 1)
    return sorted(dict.fromkeys(projects))


def inspect_project(project: str | Path) -> dict[str, Any]:
    root = Path(project).expanduser().resolve()
    all_files = list(_iter_project_files(root))
    env_files = sorted(path for path in all_files if _is_env_file(path) and not _is_example_env(path))
    example_files = sorted(path for path in all_files if _is_env_file(path) and _is_example_env(path))
    env_keys = _parse_env_keys(env_files)
    example_keys = _parse_env_keys(example_files)
    code_refs = _extract_env_refs(root)
    findings: list[dict[str, str]] = []

    if env_files and not _gitignore_covers_env(root):
        findings.append(
            _finding(
                "warning",
                "gitignore-env-missing",
                ".gitignore does not clearly ignore secret-bearing .env files.",
                str(root / ".gitignore"),
            )
        )

    tracked_env = [
        line for line in _run_git(root, ["ls-files"]) if _is_env_file(Path(line)) and not _is_example_env(Path(line))
    ]
    for path in tracked_env:
        findings.append(
            _finding(
                "critical",
                "tracked-env-file",
                "A secret-bearing env file appears to be tracked by git.",
                path,
            )
        )

    for key, locations in sorted(env_keys.items()):
        if key not in code_refs and key not in example_keys:
            findings.append(
                _finding(
                    "info",
                    "env-key-unused",
                    "Key appears in secret env files but was not found in code references or examples.",
                    f"{key} in {len(locations)} env file(s)",
                )
            )

    for key, locations in sorted(code_refs.items()):
        if key not in example_keys:
            findings.append(
                _finding(
                    "warning",
                    "code-key-missing-from-env-example",
                    "Code references an env key that is missing from env example/template files.",
                    f"{key} referenced in {len(locations)} file(s)",
                )
            )

    return {
        "project": str(root),
        "ok": not findings,
        "summary": {
            "findings": len(findings),
            "env_files": len(env_files),
            "example_files": len(example_files),
            "env_keys": len(env_keys),
            "example_keys": len(example_keys),
            "code_refs": len(code_refs),
        },
        "env_files": [str(path) for path in env_files],
        "example_files": [str(path) for path in example_files],
        "env_keys": sorted(env_keys),
        "example_keys": sorted(example_keys),
        "code_refs": sorted(code_refs),
        "findings": findings,
    }


def inspect_workspace(root: str | Path, max_depth: int = 4) -> dict[str, Any]:
    projects = [inspect_project(project) for project in discover_projects(root, max_depth=max_depth)]
    findings = sum(project["summary"]["findings"] for project in projects)
    severity_counts: dict[str, int] = {}
    for project in projects:
        for finding in project["findings"]:
            severity = finding["severity"]
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
    return {
        "root": str(Path(root).expanduser().resolve()),
        "ok": findings == 0,
        "summary": {
            "projects": len(projects),
            "findings": findings,
            "projects_with_findings": sum(1 for project in projects if project["findings"]),
            "severity_counts": dict(sorted(severity_counts.items())),
        },
        "projects": projects,
    }


def _render_project_text(report: dict[str, Any], max_findings: int = 200) -> list[str]:
    lines = [
        "Workspace config audit",
        f"Project: {report['project']}",
        (
            "Summary: "
            f"{report['summary']['findings']} finding(s), "
            f"{report['summary']['env_keys']} env key(s), "
            f"{report['summary']['code_refs']} code reference(s)"
        ),
    ]
    if not report["findings"]:
        lines.append("[OK] no findings")
        return lines
    visible_findings = report["findings"][:max_findings]
    for finding in visible_findings:
        evidence = f" ({finding['evidence']})" if finding.get("evidence") else ""
        lines.append(
            f"[{finding['severity'].upper()}] {finding['code']}: {finding['message']}{evidence}"
        )
    hidden = len(report["findings"]) - len(visible_findings)
    if hidden > 0:
        lines.append(f"... {hidden} more finding(s) omitted from text output; use --format json for full detail.")
    return lines


def render_report(report: dict[str, Any], fmt: str, max_findings: int = 200) -> str:
    if fmt == "json":
        return json.dumps(report, ensure_ascii=False, indent=2) + "\n"

    if fmt == "summary":
        if "projects" not in report:
            return (
                "Workspace config audit summary\n"
                f"Project: {report['project']}\n"
                f"Findings: {report['summary']['findings']}\n"
                f"Env keys: {report['summary']['env_keys']}\n"
                f"Code references: {report['summary']['code_refs']}\n"
            )
        return (
            "Workspace config audit summary\n"
            f"Root: {report['root']}\n"
            f"Projects: {report['summary']['projects']}\n"
            f"Projects with findings: {report['summary']['projects_with_findings']}\n"
            f"Total findings: {report['summary']['findings']}\n"
            f"Severity counts: {json.dumps(report['summary']['severity_counts'], sort_keys=True)}\n"
        )

    if fmt == "markdown":
        if "projects" not in report:
            lines = [
                "# Workspace Config Audit",
                "",
                f"Project: `{report['project']}`",
                "",
                "| Findings | Env Keys | Code References |",
                "|---:|---:|---:|",
                f"| {report['summary']['findings']} | {report['summary']['env_keys']} | {report['summary']['code_refs']} |",
                "",
            ]
            for finding in report["findings"][:max_findings]:
                evidence = f" Evidence: `{finding['evidence']}`" if finding.get("evidence") else ""
                lines.append(f"- **{finding['severity']}** `{finding['code']}` - {finding['message']}{evidence}")
            hidden = len(report["findings"]) - min(len(report["findings"]), max_findings)
            if hidden > 0:
                lines.append(f"- {hidden} more finding(s) omitted; use JSON for full detail.")
            return "\n".join(lines) + "\n"

        lines = [
            "# Workspace Config Audit",
            "",
            f"Root: `{report['root']}`",
            "",
            "| Projects | Projects With Findings | Total Findings |",
            "|---:|---:|---:|",
            (
                f"| {report['summary']['projects']} | "
                f"{report['summary']['projects_with_findings']} | "
                f"{report['summary']['findings']} |"
            ),
            "",
            "| Severity | Count |",
            "|---|---:|",
        ]
        for severity, count in report["summary"]["severity_counts"].items():
            lines.append(f"| {severity} | {count} |")
        lines.append("")
        for project in report["projects"]:
            project_name = Path(project["project"]).name
            lines.extend(
                [
                    f"## {project_name}",
                    "",
                    f"Path: `{project['project']}`",
                    "",
                    "| Findings | Env Keys | Code References |",
                    "|---:|---:|---:|",
                    f"| {project['summary']['findings']} | {project['summary']['env_keys']} | {project['summary']['code_refs']} |",
                    "",
                ]
            )
            for finding in project["findings"][:max_findings]:
                evidence = f" Evidence: `{finding['evidence']}`" if finding.get("evidence") else ""
                lines.append(f"- **{finding['severity']}** `{finding['code']}` - {finding['message']}{evidence}")
            hidden = len(project["findings"]) - min(len(project["findings"]), max_findings)
            if hidden > 0:
                lines.append(f"- {hidden} more finding(s) omitted; use JSON for full detail.")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    if "projects" not in report:
        return "\n".join(_render_project_text(report, max_findings=max_findings)) + "\n"

    lines = [
        "Workspace config audit",
        f"Root: {report['root']}",
        (
            "Summary: "
            f"{report['summary']['projects']} project(s), "
            f"{report['summary']['projects_with_findings']} with finding(s), "
            f"{report['summary']['findings']} total finding(s)"
        ),
        "",
    ]
    for project in report["projects"]:
        lines.extend(_render_project_text(project, max_findings=max_findings))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Workspace or project root to inspect.")
    parser.add_argument("--max-depth", type=int, default=4, help="Project discovery depth for workspace roots.")
    parser.add_argument("--project", action="store_true", help="Treat --root as one project instead of discovering projects.")
    parser.add_argument("--format", choices=("text", "json", "markdown", "summary"), default="text", help="Output format.")
    parser.add_argument("--summary-only", action="store_true", help="Shortcut for --format summary.")
    parser.add_argument("--max-findings", type=int, default=200, help="Maximum findings per project in text output.")
    args = parser.parse_args(argv)

    if args.project:
        report = inspect_project(args.root)
    else:
        report = inspect_workspace(args.root, max_depth=args.max_depth)

    fmt = "summary" if args.summary_only else args.format
    print(render_report(report, fmt, max_findings=args.max_findings), end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
