"""Hermetic tests for scripts/mw/backend_check.py (MW-P3-I2e).

Local in-process mock HTTP server + temp sqlite DB via sqlite3 CLI.
No real network, no secrets committed, stdlib only.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import pytest

# ---------------------------------------------------------------------------
# load module under test
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_CHECK_PATH = REPO_ROOT / "scripts" / "mw" / "backend_check.py"

_spec = importlib.util.spec_from_file_location("mw_backend_check", BACKEND_CHECK_PATH)
assert _spec and _spec.loader
bc = importlib.util.module_from_spec(_spec)
sys.modules["mw_backend_check"] = bc
_spec.loader.exec_module(bc)

# ---------------------------------------------------------------------------
# sqlite3 CLI availability
# ---------------------------------------------------------------------------

SQLITE3 = shutil.which("sqlite3")
pytestmark = pytest.mark.skipif(SQLITE3 is None, reason="sqlite3 CLI required")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _seed_db(db_path: Path) -> None:
    """Create tables + baseline rows for all check types."""
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.executescript(
            """
            CREATE TABLE sys_site (
              id INTEGER PRIMARY KEY,
              site_id TEXT NOT NULL
            );
            CREATE TABLE contact (
              id INTEGER PRIMARY KEY,
              name TEXT,
              msg TEXT,
              site_id TEXT
            );
            CREATE TABLE contact_src (
              id INTEGER PRIMARY KEY,
              name TEXT,
              msg TEXT,
              site_id TEXT
            );
            CREATE TABLE broken_report (
              id INTEGER PRIMARY KEY,
              menu TEXT,
              url TEXT,
              site_id TEXT
            );
            CREATE TABLE content (
              id INTEGER PRIMARY KEY,
              site_id TEXT,
              asset_path TEXT
            );
            CREATE TABLE users (
              id INTEGER PRIMARY KEY,
              site_id TEXT,
              name TEXT
            );
            """
        )
        cur.execute("INSERT INTO sys_site (id, site_id) VALUES (1, 'rsf')")
        # parity baseline: matching contact / contact_src
        for i, name in enumerate(["Alice", "Bob", "Carol", "Dave", "Eve"], start=1):
            cur.execute(
                "INSERT INTO contact (id, name, msg, site_id) VALUES (?,?,?,?)",
                (i, name, f"m{i}", "rsf"),
            )
            cur.execute(
                "INSERT INTO contact_src (id, name, msg, site_id) VALUES (?,?,?,?)",
                (i, name, f"m{i}", "rsf"),
            )
        # enough rows for sampling (need >= 20 for min sample floor when testing)
        for i in range(6, 31):
            cur.execute(
                "INSERT INTO contact (id, name, msg, site_id) VALUES (?,?,?,?)",
                (i, f"User{i}", f"m{i}", "rsf"),
            )
            cur.execute(
                "INSERT INTO contact_src (id, name, msg, site_id) VALUES (?,?,?,?)",
                (i, f"User{i}", f"m{i}", "rsf"),
            )
        cur.execute(
            "INSERT INTO broken_report (id, menu, url, site_id) VALUES (1, 'docs', '/old.pdf', 'rsf')"
        )
        cur.execute(
            "INSERT INTO content (id, site_id, asset_path) VALUES (1, 'rsf', '/assets/rsf/a.png')"
        )
        cur.execute(
            "INSERT INTO content (id, site_id, asset_path) VALUES (2, 'rsf', '/assets/rsf/b.pdf')"
        )
        for i in range(1, 4):
            cur.execute(
                "INSERT INTO users (id, site_id, name) VALUES (?,?,?)",
                (i, "rsf", f"u{i}"),
            )
        conn.commit()
    finally:
        conn.close()


class MockState:
    """Shared mutable state for the mock API (backed by sqlite + in-memory)."""

    def __init__(self, db_path: Path, site_id: str = "rsf"):
        self.db_path = db_path
        self.site_id = site_id
        self.break_admin = False  # form_cycle admin → 404
        self.break_verify = False  # delete row after insert so verify fails
        self.dashboard_overreport: Optional[Dict[str, Any]] = None
        self.search_not_found_500 = False
        self.last_auth: Optional[str] = None
        # Override created id returned by POST /api/forms/contact (regression FIX 2)
        self.form_id_override: Any = None
        self.search_docs = [
            {"id": 1, "title": "ทางเดิน", "body": "เอกสารทาง"},
            {"id": 2, "title": "news", "body": "hello"},
        ]

    def conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(str(self.db_path))
        c.row_factory = sqlite3.Row
        return c


def _make_handler(state: MockState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # quiet
            return

        def _read_json(self) -> Any:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))

        def _send(self, code: int, payload: Any = None, raw: Optional[str] = None) -> None:
            body = raw if raw is not None else json.dumps(payload if payload is not None else {})
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            state.last_auth = self.headers.get("Authorization")
            parsed = urlparse(self.path)
            path = parsed.path
            qs = parse_qs(parsed.query)

            if path.startswith("/api/admin/forms/"):
                if state.break_admin:
                    self._send(404, {"error": "not found"})
                    return
                fid = path.rsplit("/", 1)[-1]
                with state.conn() as c:
                    row = c.execute(
                        "SELECT id, name, msg, site_id FROM contact WHERE id=?", (fid,)
                    ).fetchone()
                if not row:
                    self._send(404, {"error": "missing"})
                    return
                self._send(
                    200,
                    {
                        "id": row["id"],
                        "name": row["name"],
                        "status": "new",
                        "site_id": row["site_id"],
                    },
                )
                return

            if path == "/api/admin/broken":
                with state.conn() as c:
                    rows = c.execute(
                        "SELECT id, menu, url FROM broken_report ORDER BY id"
                    ).fetchall()
                items = [{"id": r["id"], "menu": r["menu"], "url": r["url"]} for r in rows]
                self._send(200, {"items": items, "urls": [r["url"] for r in rows]})
                return

            if path == "/api/dashboard/summary":
                with state.conn() as c:
                    users = c.execute(
                        "SELECT COUNT(*) AS n FROM users WHERE site_id=?",
                        (state.site_id,),
                    ).fetchone()["n"]
                    reports = c.execute(
                        "SELECT COUNT(*) AS n FROM broken_report"
                    ).fetchone()["n"]
                    forms = c.execute("SELECT COUNT(*) AS n FROM contact").fetchone()["n"]
                totals = {
                    "users": int(users),
                    "reports": int(reports),
                    "forms": int(forms),
                }
                if state.dashboard_overreport:
                    for k, v in state.dashboard_overreport.items():
                        totals[k] = v
                self._send(200, {"totals": totals})
                return

            if path == "/api/search":
                q = (qs.get("q") or [""])[0]
                if state.search_not_found_500 and q == "zzqxnotfound":
                    self._send(500, {"error": "search failed"})
                    return
                hits = [d for d in state.search_docs if q and q in (d["title"] + d["body"])]
                self._send(200, {"hits": hits, "q": q})
                return

            self._send(404, {"error": f"no route {path}"})

        def do_POST(self) -> None:  # noqa: N802
            state.last_auth = self.headers.get("Authorization")
            parsed = urlparse(self.path)
            path = parsed.path
            body = self._read_json()

            if path == "/api/forms/contact":
                name = body.get("name", "")
                msg = body.get("msg", "")
                with state.conn() as c:
                    cur = c.execute(
                        "INSERT INTO contact (name, msg, site_id) VALUES (?,?,?)",
                        (name, msg, state.site_id),
                    )
                    new_id = cur.lastrowid
                    # Keep contact_src in lockstep so data_parity still holds
                    # after form_cycle mutations in the full suite.
                    c.execute(
                        "INSERT INTO contact_src (id, name, msg, site_id) VALUES (?,?,?,?)",
                        (new_id, name, msg, state.site_id),
                    )
                    c.commit()
                    if state.break_verify:
                        c.execute("DELETE FROM contact WHERE id=?", (new_id,))
                        c.execute("DELETE FROM contact_src WHERE id=?", (new_id,))
                        c.commit()
                # form_id_override lets tests inject invalid / malicious ids (FIX 2)
                resp_id = (
                    state.form_id_override
                    if state.form_id_override is not None
                    else new_id
                )
                self._send(200, {"data": {"id": resp_id}, "ok": True})
                return

            if path == "/api/report-broken":
                menu = body.get("menu", "")
                url = body.get("url", "")
                with state.conn() as c:
                    cur = c.execute(
                        "INSERT INTO broken_report (menu, url, site_id) VALUES (?,?,?)",
                        (menu, url, state.site_id),
                    )
                    new_id = cur.lastrowid
                    c.commit()
                self._send(200, {"data": {"id": new_id}, "ok": True})
                return

            self._send(404, {"error": f"no route {path}"})

    return Handler


@pytest.fixture
def env(tmp_path: Path, monkeypatch):
    """Temp DB + mock HTTP server + base config factory."""
    # make_config configures auth via MW_BACKEND_TOKEN → run authenticated by default
    # (in-process _run_mod reads os.environ). auth-missing tests clear it explicitly.
    monkeypatch.setenv("MW_BACKEND_TOKEN", "test-token")
    db = tmp_path / "test.db"
    _seed_db(db)
    state = MockState(db)
    handler = _make_handler(state)
    server = HTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"

    def make_config(**overrides: Any) -> Path:
        cfg: Dict[str, Any] = {
            "api_base": base,
            "site_id": "rsf",
            "auth": {"header": "Authorization", "value_env": "MW_BACKEND_TOKEN"},
            "query_cmd": [SQLITE3, "-json", "{db}"],
            "db_path": str(db),
            "checks": [
                {
                    "id": "site_exists",
                    "type": "query_rows",
                    "sql": "SELECT id FROM sys_site WHERE site_id='{site_id}'",
                    "expect_min_rows": 1,
                },
                {
                    "id": "form_cycle",
                    "type": "form_cycle",
                    "submit": {
                        "method": "POST",
                        "path": "/api/forms/contact",
                        "body": {"name": "MW probe", "msg": "hello"},
                    },
                    "id_json_path": "data.id",
                    "verify_sql": (
                        "SELECT name FROM contact WHERE id={id} AND site_id='{site_id}'"
                    ),
                    "verify_expect": {"name": "MW probe"},
                    "admin": {"method": "GET", "path": "/api/admin/forms/{id}"},
                    "admin_expect_json": {"status": "new"},
                },
                {
                    "id": "broken_file_report",
                    "type": "form_cycle",
                    "submit": {
                        "method": "POST",
                        "path": "/api/report-broken",
                        "body": {"menu": "docs", "url": "/x.pdf"},
                    },
                    "id_json_path": "data.id",
                    "verify_sql": (
                        "SELECT url FROM broken_report WHERE id={id} "
                        "AND site_id='{site_id}'"
                    ),
                    "verify_expect": {"url": "/x.pdf"},
                    "admin": {"method": "GET", "path": "/api/admin/broken"},
                    "admin_contains": "/x.pdf",
                },
                {
                    "id": "siteid_isolation",
                    "type": "siteid_isolation",
                    "sql": (
                        "SELECT site_id, asset_path FROM content WHERE site_id='{site_id}'"
                    ),
                    "forbid_other_site_rows": True,
                    "asset_prefix_ok": "/assets/{site_id}/",
                },
                {
                    "id": "dashboard_parity",
                    "type": "dashboard_parity",
                    "dashboard": {"method": "GET", "path": "/api/dashboard/summary"},
                    "points": [
                        {
                            "json_path": "totals.users",
                            "sql": "SELECT COUNT(*) AS n FROM users WHERE site_id='{site_id}'",
                            "sql_field": "n",
                        },
                        {
                            "json_path": "totals.reports",
                            "sql": "SELECT COUNT(*) AS n FROM broken_report",
                            "sql_field": "n",
                        },
                        {
                            "json_path": "totals.forms",
                            "sql": "SELECT COUNT(*) AS n FROM contact",
                            "sql_field": "n",
                        },
                    ],
                },
                {
                    "id": "es_search",
                    "type": "es_search",
                    "search": {"method": "GET", "path": "/api/search?q={q}&page=1"},
                    "query_word": "ทาง",
                    "expect_min_results": 1,
                    "results_json_path": "hits",
                    "not_found": {"q": "zzqxnotfound", "expect_empty": True},
                },
                {
                    "id": "data_parity",
                    "type": "data_parity",
                    "tables": [
                        {
                            "name": "contact",
                            "source_count_sql": "SELECT COUNT(*) AS n FROM contact_src",
                            "target_count_sql": "SELECT COUNT(*) AS n FROM contact",
                            "key": "id",
                            "sample_source_sql": (
                                "SELECT id,name FROM contact_src ORDER BY id"
                            ),
                            "sample_target_sql": (
                                "SELECT id,name FROM contact ORDER BY id"
                            ),
                        }
                    ],
                    "sample_pct": 5,
                },
            ],
        }
        # shallow + deep merges for overrides
        for k, v in overrides.items():
            if k == "checks":
                cfg["checks"] = v
            else:
                cfg[k] = v
        path = tmp_path / "backend-check.yaml"
        # write as JSON-compatible YAML (flow style via json dump works for mini-yaml?
        # Prefer PyYAML-friendly block via json for nested — use yaml if available,
        # else write a simple JSON-as-YAML isn't ideal. Use text construction with
        # PyYAML if present, else dump via json and wrap — actually load_yaml uses
        # PyYAML when available. Write with yaml or a careful dump.
        _dump_yaml(path, cfg)
        return path

    yield {
        "tmp": tmp_path,
        "db": db,
        "state": state,
        "base": base,
        "make_config": make_config,
        "port": port,
    }
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def _dump_yaml(path: Path, cfg: Dict[str, Any]) -> None:
    """Write config; prefer PyYAML, else a compact JSON (PyYAML may load it? no).

    Use a minimal recursive YAML emitter good enough for our schema.
    """
    try:
        import yaml  # type: ignore

        path.write_text(
            yaml.safe_dump(cfg, default_flow_style=False, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return
    except ImportError:
        pass

    def emit(obj: Any, indent: int = 0) -> List[str]:
        sp = "  " * indent
        lines: List[str] = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"{sp}{k}:")
                    lines.extend(emit(v, indent + 1))
                elif isinstance(v, bool):
                    lines.append(f"{sp}{k}: {'true' if v else 'false'}")
                elif v is None:
                    lines.append(f"{sp}{k}: null")
                elif isinstance(v, (int, float)):
                    lines.append(f"{sp}{k}: {v}")
                else:
                    s = str(v).replace('"', '\\"')
                    lines.append(f'{sp}{k}: "{s}"')
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    first = True
                    for k, v in item.items():
                        prefix = "- " if first else "  "
                        first = False
                        if isinstance(v, (dict, list)):
                            lines.append(f"{sp}{prefix}{k}:")
                            lines.extend(emit(v, indent + 2 if prefix == "- " else indent + 1))
                            # fix indent for continuation after dash
                            if prefix == "- ":
                                # re-emit nested with indent+1 from list base... simplify:
                                pass
                        elif isinstance(v, bool):
                            lines.append(f"{sp}{prefix}{k}: {'true' if v else 'false'}")
                        elif isinstance(v, (int, float)):
                            lines.append(f"{sp}{prefix}{k}: {v}")
                        else:
                            s = str(v).replace('"', '\\"')
                            lines.append(f'{sp}{prefix}{k}: "{s}"')
                else:
                    lines.append(f"{sp}- {json.dumps(item, ensure_ascii=False)}")
        return lines

    # The manual emitter above is fragile for nested list-of-dict with nested
    # lists. Prefer json.dumps wrapped — our loaders use PyYAML in CI usually.
    # Fallback: write Python-eval-safe via a pure JSON file and force load path.
    # Simpler reliable fallback without PyYAML:
    path.write_text(_emit_yaml_safe(cfg), encoding="utf-8")


def _emit_yaml_safe(obj: Any, indent: int = 0) -> str:
    """Deterministic YAML subset writer for tests (no PyYAML required)."""

    def scalar(v: Any) -> str:
        if v is True:
            return "true"
        if v is False:
            return "false"
        if v is None:
            return "null"
        if isinstance(v, int) and not isinstance(v, bool):
            return str(v)
        if isinstance(v, float):
            return str(v)
        s = str(v)
        if any(c in s for c in ":#{}[]&*!|>%@`'\"\n") or s != s.strip():
            return json.dumps(s, ensure_ascii=False)
        return s

    lines: List[str] = []

    def walk(o: Any, level: int, list_item: bool = False) -> None:
        sp = "  " * level
        if isinstance(o, dict):
            first = True
            for k, v in o.items():
                prefix = f"{sp}- " if list_item and first else (sp if not (list_item and first) else sp)
                if list_item and first:
                    key_prefix = f"{sp}- "
                elif list_item and not first:
                    key_prefix = f"{sp}  "
                else:
                    key_prefix = sp
                first = False
                if isinstance(v, dict):
                    if not v:
                        lines.append(f"{key_prefix}{k}: {{}}")
                    else:
                        lines.append(f"{key_prefix}{k}:")
                        walk(v, level + (2 if list_item else 1), False)
                elif isinstance(v, list):
                    if not v:
                        lines.append(f"{key_prefix}{k}: []")
                    else:
                        lines.append(f"{key_prefix}{k}:")
                        for item in v:
                            if isinstance(item, (dict, list)):
                                walk(item, level + (2 if list_item else 1), True)
                            else:
                                lines.append(
                                    f"{'  ' * (level + (2 if list_item else 1))}- {scalar(item)}"
                                )
                else:
                    lines.append(f"{key_prefix}{k}: {scalar(v)}")
        elif isinstance(o, list):
            for item in o:
                walk(item, level, True)
        else:
            lines.append(f"{sp}- {scalar(o)}")

    walk(obj, 0, False)
    return "\n".join(lines) + "\n"


def _run(
    config: Path,
    *,
    only: Optional[str] = None,
    as_json: bool = False,
    timeout: float = 10,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[int, str, str]:
    argv = [str(BACKEND_CHECK_PATH), "--config", str(config), "--timeout", str(timeout)]
    if only:
        argv.extend(["--only", only])
    if as_json:
        argv.append("--json")
    run_env = os.environ.copy()
    # default: run authenticated (make_config configures auth via MW_BACKEND_TOKEN).
    # a test can override with env={"MW_BACKEND_TOKEN": ""} to exercise auth-missing.
    run_env.setdefault("MW_BACKEND_TOKEN", "test-token")
    if env:
        run_env.update(env)
    proc = subprocess.run(
        [sys.executable, *argv],
        capture_output=True,
        text=True,
        env=run_env,
        cwd=str(config.parent),
    )
    return proc.returncode, proc.stdout, proc.stderr


def _run_mod(
    config: Path,
    *,
    only: Optional[List[str]] = None,
    timeout: float = 10,
) -> bc.RunReport:
    cfg = bc.load_config(config)
    bc.validate_config(cfg)
    return bc.run_checks(cfg, only=only, timeout=timeout)


# ---------------------------------------------------------------------------
# unit: sample_keys deterministic
# ---------------------------------------------------------------------------


def test_sample_keys_deterministic():
    keys = list(range(1, 50))
    a = bc.sample_keys(keys, 5, min_rows=20)
    b = bc.sample_keys(keys, 5, min_rows=20)
    assert a == b
    assert len(a) == 20  # max(20, ceil(5% of 49)) = 20
    # ordered by sha256
    assert a == sorted(a, key=lambda k: __import__("hashlib").sha256(str(k).encode()).hexdigest())[:20]


def test_sample_keys_pct_large():
    keys = list(range(100))
    picked = bc.sample_keys(keys, 50, min_rows=20)
    assert len(picked) == 50


# ---------------------------------------------------------------------------
# site_exists
# ---------------------------------------------------------------------------


def test_site_exists_pass(env):
    cfg = env["make_config"]()
    code, out, err = _run(cfg, only="site_exists")
    assert code == 0, (out, err)
    assert "site_exists: PASS" in out
    # --only is a partial run: must not claim full HEALTHY=YES (FIX 7)
    assert "PARTIAL" in out
    assert "HEALTHY=YES" not in out


def test_site_exists_missing_fail(env):
    # wipe site rows
    with sqlite3.connect(str(env["db"])) as c:
        c.execute("DELETE FROM sys_site")
        c.commit()
    cfg = env["make_config"]()
    code, out, err = _run(cfg, only="site_exists")
    assert code == 1
    assert "site_exists: FAIL" in out
    # --only subset: PARTIAL marker (not bare HEALTHY=YES); fail still exit 1
    assert "PARTIAL" in out
    assert "HEALTHY=YES" not in out


# ---------------------------------------------------------------------------
# form_cycle
# ---------------------------------------------------------------------------


def test_form_cycle_pass(env):
    cfg = env["make_config"]()
    code, out, err = _run(cfg, only="form_cycle")
    assert code == 0, (out, err)
    assert "form_cycle: PASS" in out


def test_form_cycle_admin_404_fail(env):
    env["state"].break_admin = True
    cfg = env["make_config"]()
    code, out, err = _run(cfg, only="form_cycle")
    assert code == 1
    assert "form_cycle: FAIL" in out
    assert "admin" in out.lower()


def test_form_cycle_verify_missing_row_fail(env):
    env["state"].break_verify = True
    cfg = env["make_config"]()
    code, out, err = _run(cfg, only="form_cycle")
    assert code == 1
    assert "form_cycle: FAIL" in out
    assert "verify" in out.lower()


# ---------------------------------------------------------------------------
# broken_file_report
# ---------------------------------------------------------------------------


def test_broken_file_report_pass(env):
    cfg = env["make_config"]()
    code, out, err = _run(cfg, only="broken_file_report")
    assert code == 0, (out, err)
    assert "broken_file_report: PASS" in out


# ---------------------------------------------------------------------------
# siteid_isolation
# ---------------------------------------------------------------------------


def test_siteid_isolation_pass(env):
    cfg = env["make_config"]()
    code, out, err = _run(cfg, only="siteid_isolation")
    assert code == 0, (out, err)
    assert "siteid_isolation: PASS" in out


def test_siteid_isolation_leak_fail(env):
    with sqlite3.connect(str(env["db"])) as c:
        # change SQL to select ALL content so leak is visible — inject foreign row
        # and widen check sql in config
        c.execute(
            "INSERT INTO content (id, site_id, asset_path) VALUES (99, 'other', '/assets/other/x.png')"
        )
        c.commit()
    # query only current site would hide leak; use unfiltered sql
    cfg = env["make_config"](
        checks=[
            {
                "id": "siteid_isolation",
                "type": "siteid_isolation",
                "sql": "SELECT site_id, asset_path FROM content",
                "forbid_other_site_rows": True,
                "asset_prefix_ok": "/assets/{site_id}/",
            }
        ]
    )
    code, out, err = _run(cfg, only="siteid_isolation")
    assert code == 1
    assert "siteid_isolation: FAIL" in out
    assert "leak" in out.lower() or "other" in out


def test_siteid_isolation_bad_asset_fail(env):
    with sqlite3.connect(str(env["db"])) as c:
        c.execute(
            "INSERT INTO content (id, site_id, asset_path) VALUES (50, 'rsf', '/assets/other/x.png')"
        )
        c.commit()
    cfg = env["make_config"]()
    code, out, err = _run(cfg, only="siteid_isolation")
    assert code == 1
    assert "siteid_isolation: FAIL" in out
    assert "asset" in out.lower() or "prefix" in out.lower() or "/assets/other" in out


# ---------------------------------------------------------------------------
# dashboard_parity
# ---------------------------------------------------------------------------


def test_dashboard_parity_pass(env):
    cfg = env["make_config"]()
    code, out, err = _run(cfg, only="dashboard_parity")
    assert code == 0, (out, err)
    assert "dashboard_parity: PASS" in out


def test_dashboard_parity_mismatch_fail(env):
    env["state"].dashboard_overreport = {"users": 999}
    cfg = env["make_config"]()
    code, out, err = _run(cfg, only="dashboard_parity")
    assert code == 1
    assert "dashboard_parity: FAIL" in out
    assert "users" in out or "999" in out


def test_dashboard_parity_lt3_points_exit2(env):
    cfg = env["make_config"](
        checks=[
            {
                "id": "dashboard_parity",
                "type": "dashboard_parity",
                "dashboard": {"method": "GET", "path": "/api/dashboard/summary"},
                "points": [
                    {
                        "json_path": "totals.users",
                        "sql": "SELECT COUNT(*) AS n FROM users",
                        "sql_field": "n",
                    },
                    {
                        "json_path": "totals.reports",
                        "sql": "SELECT COUNT(*) AS n FROM broken_report",
                        "sql_field": "n",
                    },
                ],
            }
        ]
    )
    code, out, err = _run(cfg, only="dashboard_parity")
    assert code == 2
    assert "config error" in err.lower() or "points" in err.lower()


# ---------------------------------------------------------------------------
# es_search
# ---------------------------------------------------------------------------


def test_es_search_pass(env):
    """Thai query_word (ทาง) must round-trip: URL-encoded request → mock hits."""
    cfg = env["make_config"]()
    # In-process path: assert genuine hits against mock with Thai keyword.
    report = _run_mod(cfg, only=["es_search"])
    # --only is partial: all_ran_passed, not full healthy
    assert report.all_ran_passed, report.to_dict()
    assert report.partial is True
    assert report.healthy is False
    assert len(report.checks) == 1
    ch = report.checks[0]
    assert ch.ok and ch.id == "es_search"
    assert ch.detail.get("hits", 0) >= 1, ch.detail  # mock docs match "ทาง"
    # CLI path (same Thai query end-to-end via subprocess)
    code, out, err = _run(cfg, only="es_search")
    assert code == 0, (out, err)
    assert "es_search: PASS" in out


def test_es_search_not_found_500_fail(env):
    env["state"].search_not_found_500 = True
    cfg = env["make_config"]()
    code, out, err = _run(cfg, only="es_search")
    assert code == 1
    assert "es_search: FAIL" in out
    assert "not_found" in out.lower() or "500" in out or "HTTP" in out


# ---------------------------------------------------------------------------
# data_parity
# ---------------------------------------------------------------------------


def test_data_parity_pass(env):
    cfg = env["make_config"]()
    code, out, err = _run(cfg, only="data_parity")
    assert code == 0, (out, err)
    assert "data_parity: PASS" in out


def test_data_parity_count_mismatch_fail(env):
    with sqlite3.connect(str(env["db"])) as c:
        c.execute("DELETE FROM contact WHERE id=1")
        c.commit()
    cfg = env["make_config"]()
    code, out, err = _run(cfg, only="data_parity")
    assert code == 1
    assert "data_parity: FAIL" in out
    assert "count" in out.lower() or "mismatch" in out.lower()


def test_data_parity_field_mismatch_fail(env):
    with sqlite3.connect(str(env["db"])) as c:
        # change a name that will be in the sample set — change ALL target names
        # so any sampled key mismatches
        c.execute("UPDATE contact SET name='TAMPERED' WHERE id=1")
        c.commit()
    # force sample to include key 1 by using high sample_pct
    cfg_path = env["make_config"]()
    cfg = bc.load_config(cfg_path)
    for ch in cfg["checks"]:
        if ch.get("id") == "data_parity":
            ch["sample_pct"] = 100
    _dump_yaml(cfg_path, cfg)
    code, out, err = _run(cfg_path, only="data_parity")
    assert code == 1
    assert "data_parity: FAIL" in out


def test_data_parity_sampling_deterministic(env):
    cfg = env["make_config"]()
    r1 = _run_mod(cfg, only=["data_parity"])
    r2 = _run_mod(cfg, only=["data_parity"])
    assert r1.checks[0].ok and r2.checks[0].ok
    keys1 = r1.checks[0].detail["tables"][0].get("sampled_keys")
    keys2 = r2.checks[0].detail["tables"][0].get("sampled_keys")
    assert keys1 == keys2
    assert keys1  # non-empty


# ---------------------------------------------------------------------------
# auth redaction
# ---------------------------------------------------------------------------


def test_auth_token_attached_never_logged(env):
    secret = "super-secret-token-xyz-999"
    cfg = env["make_config"]()
    code, out, err = _run(
        cfg,
        only="form_cycle",
        as_json=True,
        env={"MW_BACKEND_TOKEN": secret},
    )
    assert code == 0, (out, err)
    assert secret not in out
    assert secret not in err
    # mock received Authorization
    assert env["state"].last_auth == secret
    # json: all ran passed, but --only ⇒ partial (not full healthy)
    data = json.loads(out)
    assert data["total"]["fail"] == 0
    assert data["partial"] is True
    assert data["healthy"] is False
    assert secret not in json.dumps(data)


# ---------------------------------------------------------------------------
# config / usage errors + unreachable
# ---------------------------------------------------------------------------


def test_config_missing_exit2(tmp_path):
    missing = tmp_path / "nope.yaml"
    code, out, err = _run(missing)
    assert code == 2
    assert "config" in err.lower() or "not found" in err.lower()


def test_unknown_check_type_exit2(env):
    cfg = env["make_config"](
        checks=[{"id": "x", "type": "not_a_real_type", "sql": "SELECT 1"}]
    )
    code, out, err = _run(cfg)
    assert code == 2
    assert "unknown" in err.lower() or "type" in err.lower()


def test_unreachable_api_fail_not_crash(env):
    # point at a closed port
    cfg = env["make_config"](api_base="http://127.0.0.1:1")
    code, out, err = _run(cfg, only="form_cycle", timeout=2)
    assert code == 1
    assert "form_cycle: FAIL" in out
    # should not traceback crash (exit 2 with stack) — exit 1 fail closed
    assert "Traceback" not in err


def test_json_output_shape(env):
    cfg = env["make_config"]()
    code, out, err = _run(cfg, only="site_exists", as_json=True)
    assert code == 0, (out, err)
    data = json.loads(out)
    assert data["site_id"] == "rsf"
    # --only subset: partial, not full-backend healthy
    assert data["partial"] is True
    assert data["healthy"] is False
    assert data["ran"] == 1
    assert data["configured_total"] == 7
    assert data["total"]["pass"] == 1
    assert data["total"]["fail"] == 0
    assert data["total"]["count"] == 1
    assert data["failed"] == []
    assert data["checks"][0]["id"] == "site_exists"
    assert data["checks"][0]["status"] == "PASS"


def test_full_suite_healthy(env):
    cfg = env["make_config"]()
    code, out, err = _run(cfg)
    assert code == 0, (out, err)
    assert "HEALTHY=YES" in out
    assert "7/7" in out or "backend: 7/7" in out


def test_default_config_path(env, monkeypatch):
    """When --config omitted, load <cwd>/.work/backend-check.yaml."""
    work = env["tmp"] / ".work"
    work.mkdir(parents=True, exist_ok=True)
    cfg = env["make_config"]()
    default = work / "backend-check.yaml"
    default.write_text(cfg.read_text(encoding="utf-8"), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(BACKEND_CHECK_PATH), "--only", "site_exists"],
        capture_output=True,
        text=True,
        cwd=str(env["tmp"]),
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert "site_exists: PASS" in proc.stdout


def test_redact_secrets_helper():
    assert bc.redact_secrets("token=abc123 end", ["abc123"]) == f"token={bc._TOKEN_REDACT} end"


def test_ascii_safe_url_encodes_thai_query():
    """Thai query values must be percent-encoded (ASCII-only URL for urllib)."""
    raw = "http://127.0.0.1:9/api/search?q=ทาง&page=1"
    safe = bc.ascii_safe_url(raw)
    assert "ทาง" not in safe
    assert "%E0%B8%97%E0%B8%B2%E0%B8%87" in safe  # ทาง
    assert safe.encode("ascii")  # must not raise
    # path segment encoding
    path_raw = "http://127.0.0.1:9/api/admin/forms/ทาง"
    path_safe = bc.ascii_safe_url(path_raw)
    assert "ทาง" not in path_safe
    assert path_safe.encode("ascii")


def test_json_path_get():
    data = {"data": {"id": 7}, "totals": {"users": 3}}
    assert bc.json_path_get(data, "data.id") == 7
    assert bc.json_path_get(data, "totals.users") == 3
    assert bc.json_path_get(data, "missing.path") is None


# ---------------------------------------------------------------------------
# FIX 1–7 regressions (false-healthy paths from cross-vendor review)
# ---------------------------------------------------------------------------


def test_siteid_isolation_null_site_id_fail(env):
    """FIX 1: site_id=NULL is a leak (unscoped), not a skip."""
    with sqlite3.connect(str(env["db"])) as c:
        c.execute(
            "INSERT INTO content (id, site_id, asset_path) VALUES (77, NULL, '/assets/rsf/x.png')"
        )
        c.commit()
    cfg = env["make_config"](
        checks=[
            {
                "id": "siteid_isolation",
                "type": "siteid_isolation",
                "sql": "SELECT site_id, asset_path FROM content",
                "forbid_other_site_rows": True,
                "asset_prefix_ok": "/assets/{site_id}/",
            }
        ]
    )
    code, out, err = _run(cfg, only="siteid_isolation")
    assert code == 1, (out, err)
    assert "siteid_isolation: FAIL" in out
    assert "leak" in out.lower() or "NULL" in out or "unscoped" in out.lower()


def test_siteid_isolation_null_asset_path_fail(env):
    """FIX 1: asset_path=NULL under required prefix → FAIL (not skip)."""
    with sqlite3.connect(str(env["db"])) as c:
        c.execute(
            "INSERT INTO content (id, site_id, asset_path) VALUES (78, 'rsf', NULL)"
        )
        c.commit()
    cfg = env["make_config"]()
    code, out, err = _run(cfg, only="siteid_isolation")
    assert code == 1, (out, err)
    assert "siteid_isolation: FAIL" in out
    assert "asset" in out.lower() or "NULL" in out or "missing" in out.lower()


def test_form_cycle_invalid_created_ids_fail(env):
    """FIX 2a/c: id=0 / false / '' / SQL-meta string → fail closed."""
    cfg = env["make_config"]()
    for bad_id in (0, False, "", "1; DROP TABLE", "x'OR'1"):
        env["state"].form_id_override = bad_id
        code, out, err = _run(cfg, only="form_cycle")
        assert code == 1, (bad_id, out, err)
        assert "form_cycle: FAIL" in out
        assert "invalid created id" in out.lower()
    env["state"].form_id_override = None


def test_form_cycle_verify_sql_must_be_site_scoped_exit2(env):
    """FIX 2b: verify_sql without {site_id} → ConfigError exit 2."""
    cfg = env["make_config"](
        checks=[
            {
                "id": "form_cycle",
                "type": "form_cycle",
                "submit": {
                    "method": "POST",
                    "path": "/api/forms/contact",
                    "body": {"name": "MW probe", "msg": "hello"},
                },
                "id_json_path": "data.id",
                "verify_sql": "SELECT name FROM contact WHERE id={id}",
                "verify_expect": {"name": "MW probe"},
                "admin": {"method": "GET", "path": "/api/admin/forms/{id}"},
                "admin_expect_json": {"status": "new"},
            }
        ]
    )
    code, out, err = _run(cfg, only="form_cycle")
    assert code == 2, (out, err)
    assert "config error" in err.lower()
    assert "site-scoped" in err.lower() or "{site_id}" in err


def test_form_cycle_admin_without_assertion_exit2(env):
    """FIX 3: admin block present but no assertion → ConfigError exit 2."""
    cfg = env["make_config"](
        checks=[
            {
                "id": "form_cycle",
                "type": "form_cycle",
                "submit": {
                    "method": "POST",
                    "path": "/api/forms/contact",
                    "body": {"name": "MW probe", "msg": "hello"},
                },
                "id_json_path": "data.id",
                "verify_sql": (
                    "SELECT name FROM contact WHERE id={id} AND site_id='{site_id}'"
                ),
                "verify_expect": {"name": "MW probe"},
                "admin": {"method": "GET", "path": "/api/admin/forms/{id}"},
                # deliberately no admin_expect_json / admin_contains
            }
        ]
    )
    code, out, err = _run(cfg, only="form_cycle")
    assert code == 2, (out, err)
    assert "config error" in err.lower()
    assert "admin_expect_json" in err or "admin_contains" in err


def test_dashboard_parity_bool_not_coerced_to_count(env):
    """FIX 4: dashboard true must not match SQL 1 (bool rejected by _to_int)."""
    # totals.users becomes JSON true; SQL still returns integer 3
    env["state"].dashboard_overreport = {"users": True}
    cfg = env["make_config"]()
    code, out, err = _run(cfg, only="dashboard_parity")
    assert code == 1, (out, err)
    assert "dashboard_parity: FAIL" in out
    assert "users" in out or "True" in out or "true" in out.lower() or "mismatch" in out.lower()


def test_data_parity_sampled_label_and_full_field_check(env):
    """FIX 6: default reason says SAMPLED + counts; full_field_check catches unsampled corruption."""
    # --- default sample path: honest SAMPLED label ---
    cfg = env["make_config"]()
    report = _run_mod(cfg, only=["data_parity"])
    assert report.all_ran_passed, report.to_dict()
    ch = report.checks[0]
    assert "SAMPLED" in ch.reason, ch.reason
    assert "count-parity FULL" in ch.reason
    assert "sampled_count" in ch.detail
    assert "total_rows" in ch.detail
    sc = ch.detail["sampled_count"]
    tr = ch.detail["total_rows"]
    assert sc < tr or sc == tr  # sample may equal total if tiny set; here 20 of 30
    assert f"{sc}/{tr}" in ch.reason
    # table detail too
    t0 = ch.detail["tables"][0]
    assert t0.get("field_check") == "SAMPLED"
    assert t0.get("sampled_count") == sc
    assert t0.get("total_rows") == tr

    # --- full_field_check: corrupt a key that default sample may miss → FAIL ---
    # Find a key that sample_keys would NOT pick at 5%/min20 on current keys
    keys = list(range(1, 31))
    sampled = set(bc.sample_keys(keys, 5, min_rows=20))
    unsampled = [k for k in keys if k not in sampled]
    assert unsampled, "need an unsampled key for this regression"
    victim = unsampled[0]
    with sqlite3.connect(str(env["db"])) as c:
        c.execute("UPDATE contact SET name='TAMPERED_FULL' WHERE id=?", (victim,))
        c.commit()

    cfg_full = env["make_config"](
        checks=[
            {
                "id": "data_parity",
                "type": "data_parity",
                "tables": [
                    {
                        "name": "contact",
                        "source_count_sql": "SELECT COUNT(*) AS n FROM contact_src",
                        "target_count_sql": "SELECT COUNT(*) AS n FROM contact",
                        "key": "id",
                        "sample_source_sql": "SELECT id,name FROM contact_src ORDER BY id",
                        "sample_target_sql": "SELECT id,name FROM contact ORDER BY id",
                        "full_field_check": True,
                    }
                ],
                "sample_pct": 5,
            }
        ]
    )
    code, out, err = _run(cfg_full, only="data_parity")
    assert code == 1, (out, err)
    assert "data_parity: FAIL" in out


def test_only_subset_reports_partial_not_full_healthy(env):
    """FIX 7: --only one of 7 checks → partial JSON + PARTIAL human, no bare HEALTHY=YES."""
    cfg = env["make_config"]()
    code, out, err = _run(cfg, only="site_exists", as_json=True)
    assert code == 0, (out, err)
    data = json.loads(out)
    assert data["partial"] is True
    assert data["ran"] == 1
    assert data["configured_total"] == 7
    assert data["healthy"] is False  # full healthy only when all configured ran
    assert data["total"]["pass"] == 1

    code2, out2, err2 = _run(cfg, only="site_exists", as_json=False)
    assert code2 == 0, (out2, err2)
    assert "PARTIAL" in out2
    assert "PARTIAL RUN" in out2 or "PARTIAL:" in out2
    assert "HEALTHY=YES" not in out2


def test_validate_created_id_unit():
    """Unit: accepted scalars vs rejected garbage (FIX 2)."""
    ok, s, _ = bc.validate_created_id(42)
    assert ok and s == "42"
    ok, s, _ = bc.validate_created_id("abc-1")
    assert ok and s == "abc-1"
    for bad in (None, 0, -1, False, True, "", "  ", [], {}, "a'b", "x;y", "/*x*/"):
        ok, _, reason = bc.validate_created_id(bad)
        assert not ok, bad
        assert "invalid" in reason.lower()


def test_to_int_rejects_bool():
    """Unit: True/False must not become 1/0 (FIX 4)."""
    assert bc._to_int(True) is None
    assert bc._to_int(False) is None
    assert bc._to_int(1) == 1
    assert bc._to_int("3") == 3
    assert bc._values_equal(True, 1) is False
    assert bc._values_equal(1, 1) is True


# --- hardening after GPT-5 round 2 (Opus-authored; Grok hit round cap) ---------

def test_auth_configured_but_token_missing_exit2(env):
    """FIX B: auth configured but ENV token empty → refuse to run unauthenticated."""
    cfg = env["make_config"]()
    code, out, err = _run(cfg, env={"MW_BACKEND_TOKEN": ""})
    assert code == 2, (out, err)
    assert "unauthenticated" in (out + err)


def test_form_cycle_admin_empty_block_exit2(env):
    """FIX C: an empty admin block (no path/assertion) is a config error, not a runtime fail."""
    cfg = env["make_config"](
        checks=[
            {
                "id": "form_cycle",
                "type": "form_cycle",
                "submit": {"method": "POST", "path": "/api/forms/contact", "body": {"name": "x", "msg": "y"}},
                "id_json_path": "data.id",
                "verify_sql": "SELECT name FROM contact WHERE id={id} AND site_id='{site_id}'",
                "verify_expect": {"name": "x"},
                "admin": {},
            }
        ]
    )
    code, out, err = _run(cfg)
    assert code == 2, (out, err)


def test_full_field_check_incomplete_coverage_fail(env):
    """FIX A: full_field_check must NOT claim FULL when the field query returns fewer rows than count."""
    cfg = env["make_config"](
        checks=[
            {
                "id": "data_parity",
                "type": "data_parity",
                "tables": [
                    {
                        "name": "contact",
                        "source_count_sql": "SELECT COUNT(*) AS n FROM contact_src",
                        "target_count_sql": "SELECT COUNT(*) AS n FROM contact",
                        "key": "id",
                        # LIMIT makes the field query return fewer rows than the count → incomplete coverage
                        "sample_source_sql": "SELECT id,name FROM contact_src ORDER BY id LIMIT 5",
                        "sample_target_sql": "SELECT id,name FROM contact ORDER BY id LIMIT 5",
                        "full_field_check": True,
                    }
                ],
                "sample_pct": 5,
            }
        ]
    )
    code, out, err = _run(cfg, only="data_parity")
    assert code == 1, (out, err)
    assert "incomplete field coverage" in out.lower() or "incomplete field coverage" in err.lower()
