#!/usr/bin/env python3
"""Build a private DuckDB workflow-lab database from local history/session traces.

This utility is intentionally local-first: it ingests shell history, Atuin
metadata, Codex/Claude/Hermes JSONL session summaries, and git repository
metadata into a DuckDB database, then writes aggregate Markdown findings. It
redacts obvious secrets and avoids emitting raw history payloads in the report.

Example:
    python scripts/workflow_lab_analyze.py --output ~/.hermes/workflow-lab/$(date +%F)
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import pathlib
import re
import sqlite3
import subprocess
import sys
from typing import Any, Iterable

try:
    import duckdb
except ImportError:  # pragma: no cover - exercised by users without dependency
    print(
        "duckdb Python package is required. Install with `uv pip install duckdb` "
        "or use a virtualenv for local workflow-lab analysis.",
        file=sys.stderr,
    )
    raise SystemExit(2)

HOME = pathlib.Path.home()
SECRET_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in [
        r"(api[_-]?key|token|secret|password|authorization|bearer)\s*[:=]\s*[^\s]+",
        r"gho_[A-Za-z0-9_]+",
        r"github_pat_[A-Za-z0-9_]+",
        r"sk-[A-Za-z0-9_-]+",
        r"xox[baprs]-[A-Za-z0-9-]+",
    ]
]


def redact(value: Any) -> str:
    """Best-effort redaction for command/session summaries."""
    if value is None:
        return ""
    text = str(value).replace("\x00", "")
    for pattern in SECRET_PATTERNS:
        text = pattern.sub(
            lambda match: f"{match.group(1)}=[REDACTED]" if match.lastindex else "[REDACTED]",
            text,
        )
    text = re.sub(
        r"(?i)(--?(?:api-key|token|password|secret)\s+)(\S+)",
        r"\1[REDACTED]",
        text,
    )
    return text[:2000]


def display_path(path: pathlib.Path | str | None) -> str:
    if not path:
        return ""
    return str(path).replace(str(HOME), "~")


def command_path(name: str) -> str | None:
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = pathlib.Path(directory) / name
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def run_quiet(command: list[str], cwd: pathlib.Path | None = None) -> str:
    try:
        return subprocess.check_output(
            command,
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=20,
        ).strip()
    except Exception:
        return ""


def add_event(
    rows: list[dict[str, Any]],
    source: str,
    timestamp: str | None,
    text: Any,
    cwd: Any = "",
    status: Any = None,
    duration_ms: Any = None,
    model: str = "",
    agent: str = "",
    kind: str = "command",
    path: pathlib.Path | str | None = None,
) -> None:
    summary = redact(text).strip()
    if not summary:
        return
    rows.append(
        {
            "source": source,
            "ts": timestamp or "",
            "text": summary,
            "cwd": redact(cwd),
            "repo": "",
            "status": "" if status is None else str(status),
            "duration_ms": "" if duration_ms is None else str(duration_ms),
            "model": model or "",
            "agent": agent or "",
            "kind": kind,
            "path": display_path(path),
            "summary": summary[:240],
        }
    )


def ingest_shell_history(rows: list[dict[str, Any]]) -> None:
    for path in [HOME / ".zsh_history", HOME / ".bash_history"]:
        if not path.exists():
            continue
        for line in path.read_text(errors="ignore").splitlines()[-20000:]:
            timestamp = ""
            command = line
            match = re.match(r"^: (\d+):\d+;(.*)$", line)
            if match:
                timestamp = dt.datetime.fromtimestamp(
                    int(match.group(1)), dt.timezone.utc
                ).isoformat()
                command = match.group(2)
            add_event(rows, path.name, timestamp, command, path=path)

    fish = HOME / ".local/share/fish/fish_history"
    if not fish.exists():
        return
    current: dict[str, str] = {}
    for line in fish.read_text(errors="ignore").splitlines()[-40000:]:
        if line.startswith("- cmd:"):
            if current:
                add_event(rows, "fish_history", current.get("ts"), current.get("cmd", ""), path=fish)
            current = {"cmd": line.split(":", 1)[1].strip()}
        elif line.strip().startswith("when:"):
            current["ts"] = dt.datetime.fromtimestamp(
                int(line.split(":", 1)[1]), dt.timezone.utc
            ).isoformat()
    if current:
        add_event(rows, "fish_history", current.get("ts"), current.get("cmd", ""), path=fish)


def ingest_atuin(rows: list[dict[str, Any]]) -> None:
    for path in [HOME / ".local/share/atuin/history.db", HOME / ".atuin/history.db"]:
        if not path.exists():
            continue
        try:
            con = sqlite3.connect(path)
            con.row_factory = sqlite3.Row
            columns = [record[1] for record in con.execute("pragma table_info(history)")]
            query = (
                "select * from history order by timestamp desc limit 20000"
                if "timestamp" in columns
                else "select * from history limit 20000"
            )
            for record in con.execute(query):
                item = dict(record)
                add_event(
                    rows,
                    "atuin",
                    item.get("timestamp"),
                    item.get("command") or item.get("cmd") or "",
                    item.get("cwd", ""),
                    item.get("exit") or item.get("exit_code"),
                    item.get("duration"),
                    path=path,
                )
        except Exception as exc:
            add_event(rows, "analysis_note", "", f"atuin parse failed: {type(exc).__name__}", path=path)


def iter_jsonl(root: pathlib.Path, limit: int = 3000) -> Iterable[pathlib.Path]:
    if not root.exists():
        return []
    try:
        return sorted(root.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    except Exception:
        return []


def summarize_content(content: Any) -> str:
    if isinstance(content, list):
        bits = []
        for item in content[:6]:
            if isinstance(item, dict):
                bits.append(str(item.get("text") or item.get("name") or item.get("type") or ""))
            else:
                bits.append(str(item))
        return " ".join(bits)
    if isinstance(content, dict):
        return str(content.get("text") or content.get("type") or content.get("name") or "")
    return str(content or "")


def ingest_sessions(rows: list[dict[str, Any]], roots: list[tuple[pathlib.Path, str]]) -> None:
    for root, agent in roots:
        for file in iter_jsonl(root):
            first_ts = ""
            model = ""
            cwd = ""
            textbits: list[str] = []
            try:
                for count, line in enumerate(file.open(errors="ignore"), start=1):
                    if count > 600:
                        break
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    first_ts = first_ts or obj.get("timestamp") or obj.get("time") or obj.get("created_at") or ""
                    meta = obj.get("session_meta") if isinstance(obj.get("session_meta"), dict) else obj
                    if obj.get("type") == "session_meta" or obj.get("session_meta"):
                        cwd = cwd or meta.get("cwd", "") or meta.get("workdir", "")
                        model = model or meta.get("model", "")
                    message = obj.get("message") or obj.get("msg") or obj.get("event") or obj
                    if isinstance(message, dict):
                        model = model or message.get("model", "")
                        content = (
                            message.get("content")
                            or message.get("text")
                            or message.get("role")
                            or message.get("type")
                        )
                    else:
                        content = message
                    summary = summarize_content(content)
                    if summary:
                        textbits.append(summary[:300])
                add_event(
                    rows,
                    agent,
                    first_ts,
                    " | ".join(textbits[:12]) or file.name,
                    cwd,
                    model=model,
                    agent=agent,
                    kind="session",
                    path=file,
                )
            except Exception as exc:
                add_event(
                    rows,
                    "analysis_note",
                    "",
                    f"{agent} parse failed {display_path(file)}: {type(exc).__name__}",
                    path=file,
                )


def ingest_git_repos(rows: list[dict[str, Any]], root: pathlib.Path) -> None:
    skip = {
        "node_modules",
        ".cache",
        "vendor",
        "build",
        "dist",
        ".venv",
        "venv",
        "target",
        "Library",
        "Applications",
        ".Trash",
        ".cargo",
        ".rustup",
    }
    repos: list[pathlib.Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        if ".git" in dirnames or ".git" in filenames:
            repo = pathlib.Path(dirpath)
            repos.append(repo)
            if ".git" in dirnames:
                dirnames.remove(".git")
        dirnames[:] = [name for name in dirnames if name not in skip and not name.startswith(".npm")]
        if len(repos) >= 250:
            break
    for repo in repos:
        branch = run_quiet(["git", "branch", "--show-current"], cwd=repo)
        dirty = run_quiet(["git", "status", "--short"], cwd=repo).splitlines()
        remote = run_quiet(["git", "remote", "get-url", "origin"], cwd=repo)
        add_event(
            rows,
            "git_repo",
            "",
            f"repo={repo} branch={branch} dirty_files={len(dirty)} remote={remote}",
            str(repo),
            kind="repo",
            path=repo,
        )


def write_rows_jsonl(rows: list[dict[str, Any]], output: pathlib.Path) -> pathlib.Path:
    path = output / "events_redacted.jsonl"
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def create_duckdb(rows_path: pathlib.Path, db_path: pathlib.Path) -> duckdb.DuckDBPyConnection:
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    escaped = str(rows_path).replace("'", "''")
    con.execute(f"create table events as select * from read_json_auto('{escaped}')")
    con.execute(
        "update events set repo = regexp_extract(cwd, '(/Users/[^/]+/[^/]+/[^/]+)', 1) "
        "where cwd like '/Users/%'"
    )
    return con


def markdown_table(con: duckdb.DuckDBPyConnection, sql: str) -> str:
    rows = con.execute(sql).fetchall()
    columns = [description[0] for description in con.description]
    if not rows:
        return "_No rows._"
    widths = [len(name) for name in columns]
    for row in rows:
        widths = [max(width, len(str(value))) for width, value in zip(widths, row)]
    header = "| " + " | ".join(name.ljust(width) for name, width in zip(columns, widths)) + " |"
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = ["| " + " | ".join(str(value).ljust(width) for value, width in zip(row, widths)) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def build_report(con: duckdb.DuckDBPyConnection, rows: list[dict[str, Any]], discovery: str) -> str:
    command_text = [row["text"] for row in rows if row["kind"] == "command"]
    patterns = {
        "manual_git_status": r"^git status",
        "manual_git_diff": r"^git diff",
        "pytest_or_run_tests": r"pytest|scripts/run_tests\.sh",
        "npm_tests_lints": r"npm (run )?(test|lint|typecheck)",
        "gh_pr_checks_runs": r"gh pr checks|gh run",
        "searches": r"\b(rg|grep|fd|find)\b",
        "python_scripts": r"^python(3)?\b",
        "uv": r"\buv\b",
    }
    counts = {name: sum(1 for command in command_text if re.search(pattern, command)) for name, pattern in patterns.items()}
    candidates: list[tuple[str, str, str, str]] = []
    if counts["manual_git_status"] + counts["manual_git_diff"] > 50:
        candidates.append(("Repo context/status digest", "High", "Medium", "Low"))
    if counts["pytest_or_run_tests"] + counts["npm_tests_lints"] > 20:
        candidates.append(("Test/lint command recommender", "High", "Low", "Low"))
    if counts["searches"] > 100:
        candidates.append(("Search pattern snippets/runbook", "Medium", "Low", "Low"))
    if counts["gh_pr_checks_runs"] > 10:
        candidates.append(("PR CI watcher/check-rollup helper", "Medium", "Low", "Low"))
    candidates.append(("Reusable DuckDB workflow-lab analyzer", "High", "Low", "Low"))

    candidate_rows = "\n".join("|" + "|".join(candidate) + "|" for candidate in candidates)
    pattern_rows = "\n".join(f"|{name}|{count}|" for name, count in counts.items())
    return "\n\n".join(
        [
            f"# Workflow Lab Local Analysis\n\nRows ingested: {len(rows)}",
            "## Discovery\n" + discovery,
            "## Sources\n" + markdown_table(con, "select source, count(*) as n from events group by 1 order by n desc"),
            "## Frequent command prefixes (redacted)\n"
            + markdown_table(
                con,
                "select regexp_extract(text, '^[^ ]+', 0) as cmd, count(*) as n "
                "from events where kind='command' group by 1 order by n desc limit 25",
            ),
            "## Repeated failed commands (only where exit status was captured)\n"
            + markdown_table(
                con,
                "select left(text, 120) as command, count(*) as n from events "
                "where status <> '' and try_cast(status as bigint) <> 0 "
                "group by 1 having count(*) > 1 order by n desc limit 20",
            ),
            "## Agent/session sources\n"
            + markdown_table(
                con,
                "select agent, count(*) as sessions, count(distinct path) as files "
                "from events where kind='session' group by 1 order by sessions desc",
            ),
            "## Repos discovered\n"
            + markdown_table(
                con,
                "select left(cwd, 100) as repo, count(*) as n from events "
                "where kind='repo' group by 1 order by n desc limit 50",
            ),
            "## Friction pattern counts\n|pattern|count|\n|-|-|\n" + pattern_rows,
            "## Candidate improvements\n|candidate|impact|effort|risk|\n|-|-|-|-|\n" + candidate_rows,
            "## Privacy note\nRaw commands/session payloads stay local. The Markdown report uses aggregate counts and redacted summaries only.",
        ]
    )


def build_discovery_note() -> str:
    commands = [
        "duckdb",
        "python3",
        "uv",
        "sqlite3",
        "jq",
        "rg",
        "fd",
        "git",
        "gh",
        "atuin",
        "codex",
        "claude",
        "ladybugs",
        "ladybug",
        "lbug",
        "suv",
        "pxh",
        "shq",
    ]
    command_lines = [f"- {name}: `{path}`" if (path := command_path(name)) else f"- {name}: not found" for name in commands]
    package_lines = []
    for package in ["ladybugs", "ladybug", "ladybug_core", "lbug"]:
        spec = importlib.util.find_spec(package)
        package_lines.append(f"- {package}: {spec.origin if spec else 'not importable'}")
    return "\n".join(["### Local command availability", *command_lines, "", "### Ladybugs package check", *package_lines])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, default=HOME / ".hermes/workflow-lab" / dt.date.today().isoformat())
    parser.add_argument("--repo-root", type=pathlib.Path, default=HOME, help="Root to scan for git repositories")
    parser.add_argument("--no-git-scan", action="store_true", help="Skip local git repository metadata scan")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    ingest_shell_history(rows)
    ingest_atuin(rows)
    ingest_sessions(
        rows,
        [
            (HOME / ".codex", "codex"),
            (HOME / ".claude", "claude"),
            (HOME / ".hermes/sessions", "hermes"),
            (pathlib.Path.cwd() / ".codex", "codex-local"),
            (pathlib.Path.cwd() / ".claude", "claude-local"),
            (pathlib.Path.cwd() / ".hermes", "hermes-local"),
        ],
    )
    if not args.no_git_scan:
        ingest_git_repos(rows, args.repo_root.expanduser())
    rows_path = write_rows_jsonl(rows, output)
    con = create_duckdb(rows_path, output / "workflow.duckdb")
    discovery = build_discovery_note()
    (output / "analysis.md").write_text(build_report(con, rows, discovery))
    print(output / "workflow.duckdb")
    print(output / "analysis.md")


if __name__ == "__main__":
    main()
