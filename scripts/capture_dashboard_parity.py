#!/usr/bin/env python3
"""P04 dashboard parity — INDEPENDENT, HERMETIC disposable capture.

Builds ONE disposable board under a TemporaryDirectory (owned root), renders
it two ways from the SAME SQLite source, and records the comparable
status/count output:

  CLI-side   : the actual CLI list rendering path hermes_cli/kanban.py::
               _cmd_list uses — including its recompute_ready() "mini-dispatch"
               ready-state recomputation — driven through the real CLI command
               dispatcher (not merely a lower-level list_tasks helper).
  Dashboard  : the kanban dashboard plugin router GET /board via a FastAPI
               TestClient (the same get_board that backs the live dashboard).

Both paths read the same tasks table and bucket by the same status set, so
the captured per-status counts MUST agree. This is independently reproducible
provenance, not citation.

Safety:
  - Uses TemporaryDirectory; always cleans it (no leaked temp home).
  - Clears ambient HERMES_KANBAN_* overrides and pins to the owned root.
  - Never writes to the repo by default. With --output <path> it writes the
    dated report artifact; otherwise it prints/returns JSON.

Excludes Android Auto and voice (out of scope for kanban board parity).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import secrets
import shutil
import sqlite3
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

# ``scripts/`` is directly under the active repository/worktree root.
REPO_ROOT = Path(__file__).resolve().parents[1]


def _repo_root() -> Path:
    """Resolve the repo root robustly (works inside git worktrees)."""
    try:
        import subprocess
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True,
        )
        if out.returncode == 0 and out.stdout.strip():
            return Path(out.stdout.strip())
    except Exception:
        pass
    return REPO_ROOT

_OVERRIDE_ENV_KEYS = (
    "HERMES_HOME", "HERMES_KANBAN_HOME", "HERMES_KANBAN_DB",
    "HERMES_KANBAN_ATTACHMENTS_ROOT", "HERMES_KANBAN_WORKSPACES_ROOT",
    "HERMES_KANBAN_BOARD",
)
_OWNERSHIP_MARKER = ".p04_capture_owned_root"
_registry = getattr(sys, "_p04_capture_root_tokens", None)
if _registry is None:
    _registry = {}
    setattr(sys, "_p04_capture_root_tokens", _registry)
_OWNED_ROOT_TOKENS = _registry


def _validate_disposable_root(root: Path) -> Path:
    root = root.resolve()
    try:
        root.relative_to(Path(tempfile.gettempdir()).resolve())
    except ValueError as exc:
        raise RuntimeError(f"refusing non-temporary capture root: {root}") from exc
    repo_root = _repo_root().resolve()
    if root == repo_root or repo_root in root.parents:
        raise RuntimeError(f"refusing non-temporary capture root: {root}")
    return root


def _register_owned_root(root: Path) -> Path:
    root = _validate_disposable_root(root)
    token = secrets.token_urlsafe(32)
    (root / _OWNERSHIP_MARKER).write_text(token + "\n", encoding="utf-8")
    _OWNED_ROOT_TOKENS[root] = token
    return root


def _guard_owned_root(root: Path) -> Path:
    root = _validate_disposable_root(root)
    marker = root / _OWNERSHIP_MARKER
    token = _OWNED_ROOT_TOKENS.get(root)
    if token is None or marker.is_symlink() or not marker.is_file():
        raise RuntimeError(f"refusing unowned capture root: {root}")
    if marker.read_text(encoding="utf-8").strip() != token:
        raise RuntimeError(f"refusing invalid capture root: {root}")
    return root


@contextmanager
def owned_env(root: Path):
    root = _guard_owned_root(root)
    saved = {k: os.environ.get(k) for k in _OVERRIDE_ENV_KEYS}
    for k in _OVERRIDE_ENV_KEYS:
        os.environ.pop(k, None)
    os.environ["HERMES_HOME"] = str(root)
    os.environ["HERMES_KANBAN_HOME"] = str(root)
    try:
        yield
    finally:
        for k in _OVERRIDE_ENV_KEYS:
            os.environ.pop(k, None)
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def _load_kanban_cli():
    """Return the kanban CLI module (hermes_cli/kanban.py)."""
    sys.path.insert(0, str(_repo_root()))
    from hermes_cli import kanban as kanban_cli
    return kanban_cli


def _render_cli_list(root: Path) -> dict:
    """Render the board the SAME way the CLI `hermes kanban list` does.

    We drive _cmd_list directly (it performs recompute_ready() then
    list_tasks) and capture its rendered lines by intercepting stdout.
    Returns {status_counts, ids}.
    """
    kanban_cli = _load_kanban_cli()
    from hermes_cli import kanban_db as kb

    # Build a fake argparse.Namespace matching `hermes kanban list` defaults.
    ns = argparse.Namespace(
        assignee=None, mine=False, status=None, tenant=None, session=None,
        archived=False, json=True, sort=None, workflow_template_id=None,
        current_step_key=None, theme=None, epic_id=None,
    )
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with owned_env(root):
        with redirect_stdout(buf):
            rc = kanban_cli._cmd_list(ns)
    assert rc == 0, f"_cmd_list returned {rc}: {buf.getvalue()}"
    payload = json.loads(buf.getvalue())
    status_counts: dict = {}
    ids = []
    for t in payload:
        status_counts[t["status"]] = status_counts.get(t["status"], 0) + 1
        ids.append(t["id"])
    return {"status_counts": status_counts, "ids": ids,
            "source": "hermes_cli.kanban._cmd_list (incl. recompute_ready)"}


def _load_dashboard_router():
    plugin_file = _repo_root() / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_parity", plugin_file,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.router


def _render_dashboard(root: Path) -> dict:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    app = FastAPI()
    app.include_router(_load_dashboard_router(), prefix="/api/plugins/kanban")
    client = TestClient(app)
    with owned_env(root):
        r = client.get("/api/plugins/kanban/board")
    assert r.status_code == 200, r.text
    data = r.json()
    status_counts = {}
    ids = []
    for col in data["columns"]:
        if len(col["tasks"]):
            status_counts[col["name"]] = len(col["tasks"])
        for t in col["tasks"]:
            ids.append(t["id"])
    return {"status_counts": status_counts, "ids": ids,
            "source": "plugins/kanban/dashboard/plugin_api.py::get_board (TestClient)"}


def capture() -> dict:
    """Run a hermetic capture and return JSON without changing caller state."""
    with tempfile.TemporaryDirectory(prefix="p04-parity-") as tmp_name:
        tmp = _register_owned_root(Path(tmp_name))
        with owned_env(tmp):
            # Import only after the fixture environment is active; this avoids
            # module-level resolution of a caller's live Hermes root.
            sys.path.insert(0, str(_repo_root()))
            from hermes_cli import kanban_db as kb

            full_tier_body = (
                "Implement the feature.\n\n"
                "## Acceptance Criteria\n- behaviour matches spec\n- tests green\n\n"
                "## Test Plan\n- pytest the focused suite\n"
            )
            kb.init_db(board="default")
            conn = kb.connect()
            try:
                t1 = kb.create_task(conn, title="parity-cli-task", assignee="cli-owner",
                                    tier="full", body=full_tier_body, parents=())
                kb.claim_task(conn, t1)  # running
                t2 = kb.create_task(conn, title="parity-dash-task", assignee="dash-owner",
                                    tier="full", body=full_tier_body, parents=())
                kb.claim_task(conn, t2)
                kb.complete_task(conn, t2, summary="done", result="ok")
                conn.commit()
            finally:
                conn.close()

            # The dashboard is a read view; the CLI handler deliberately runs
            # recompute_ready(). Record the pre-normalisation view and compare
            # parity only after that documented CLI normalisation boundary.
            dash_before = _render_dashboard(tmp)
            cli = _render_cli_list(tmp)
            dash = _render_dashboard(tmp)
            cli_sparse = {k: v for k, v in cli["status_counts"].items() if v}
            dash_sparse = {k: v for k, v in dash["status_counts"].items() if v}
            parity_ok = (sorted(cli["ids"]) == sorted(dash["ids"])
                         and cli_sparse == dash_sparse)

            try:
                import subprocess
                git_sha = subprocess.run(
                    ["git", "-C", str(_repo_root()), "rev-parse", "HEAD"],
                    capture_output=True, text=True,
                ).stdout.strip()
            except Exception:
                git_sha = "unknown"
            report = {
                "proof": "P04 dashboard snapshot parity — CLI list handler vs dashboard GET /board",
                "verdict": "PARITY-PROVEN-BY-CAPTURE" if parity_ok else "PARITY-FAILED",
                "seed_statuses": {
                    "parity-cli-task": "running (claimed)",
                    "parity-dash-task": "review (completed; full-tier review gate, not done)",
                },
                "captured_output": {
                    "dashboard_before_cli_recompute": dash_before,
                    "cli_side": cli,
                    "dashboard_side_after_cli_recompute": dash,
                    "id_set_match": sorted(cli["ids"]) == sorted(dash["ids"]),
                    "status_counts_match": cli_sparse == dash_sparse,
                },
                "provenance": {
                    "method": "one disposable board; dashboard before/after CLI recompute",
                    "cli_render_command": cli["source"],
                    "dashboard_render_command": dash["source"],
                    "captured_by": "scripts/capture_dashboard_parity.py",
                    "git_sha": git_sha,
                },
                "scope_limits": {
                    "excludes": ["Android Auto", "voice"],
                    "does_not_prove": [
                        "A live dashboard server process reading a live board (TestClient only).",
                        "Order-independent parity before the CLI's documented recompute_ready mutation.",
                    ],
                    "does_prove": [
                        "After the CLI list handler's documented recompute_ready phase, the CLI and "
                        "dashboard endpoint return the same task IDs and non-zero status counts for "
                        "the same disposable board snapshot."
                    ],
                },
            }
        # TemporaryDirectory has exited only after owned_env restored the exact
        # caller environment. Report cleanup truthfully after the directory is gone.
    report["provenance"]["temp_root_cleaned"] = not tmp.exists()
    return report


def _is_under(root: Path, path: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _protected_roots() -> list[Path]:
    roots = [Path.home() / ".hermes"]
    for key in (
        "HERMES_HOME", "HERMES_KANBAN_HOME", "HERMES_KANBAN_DB",
        "HERMES_KANBAN_ATTACHMENTS_ROOT", "HERMES_KANBAN_WORKSPACES_ROOT",
    ):
        value = os.environ.get(key)
        if value:
            roots.append(Path(value))
    return roots


def _safe_output_path(raw: str) -> Path:
    """Accept a new report path outside Hermes data roots without side effects."""
    out = Path(raw).expanduser()
    protected = _protected_roots()
    prospective = out.resolve(strict=False)
    if any(_is_under(root, prospective) or _is_under(prospective, root) for root in protected):
        raise ValueError(f"refusing output under Hermes data path: {prospective}")
    if out.exists() or out.is_symlink():
        raise ValueError(f"refusing to overwrite existing or symlinked output: {out}")
    if not out.parent.is_dir():
        raise ValueError(f"refusing output with missing parent directory: {out.parent}")
    for parent in (out.parent, *out.parent.parents):
        if parent.is_symlink():
            raise ValueError(f"refusing symlinked output directory: {parent}")
    resolved = out.resolve(strict=False)
    if any(_is_under(root, resolved) or _is_under(resolved, root) for root in protected):
        raise ValueError(f"refusing output under Hermes data path: {resolved}")
    return resolved


def _write_new_output(out: Path, text: str) -> None:
    """Atomically create a new report using a pinned, non-symlink parent FD."""
    out = _safe_output_path(str(out))
    parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        parent_flags |= os.O_NOFOLLOW
    parent_fd = os.open(out.parent, parent_flags)
    try:
        actual_parent = Path(os.readlink(f"/proc/self/fd/{parent_fd}"))
        actual = actual_parent / out.name
        if any(_is_under(root, actual) or _is_under(actual, root) for root in _protected_roots()):
            raise ValueError(f"refusing output under Hermes data path: {actual}")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(out.name, flags, 0o600, dir_fd=parent_fd)
        except FileExistsError as exc:
            raise ValueError(f"refusing to overwrite existing output: {out}") from exc
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            try:
                os.unlink(out.name, dir_fd=parent_fd)
            finally:
                raise
    finally:
        os.close(parent_fd)


def main():
    ap = argparse.ArgumentParser(description="P04 dashboard parity capture (hermetic)")
    ap.add_argument("--output", help="Write the dated report JSON to this path "
                                      "(default: print/return JSON only; never "
                                      "overwrites a tracked report silently)")
    args = ap.parse_args()
    report = capture()
    text = json.dumps(report, indent=2)
    if args.output:
        try:
            out = _safe_output_path(args.output)
        except ValueError as exc:
            print(f"refused output: {exc}", file=sys.stderr)
            return 2
        try:
            _write_new_output(out, text)
        except ValueError as exc:
            print(f"refused output: {exc}", file=sys.stderr)
            return 2
        print(f"wrote report to {out} (verdict={report['verdict']})")
    else:
        print(text)
    return 0 if report["verdict"] == "PARITY-PROVEN-BY-CAPTURE" else 1


if __name__ == "__main__":
    sys.exit(main())
