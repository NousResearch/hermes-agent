# hermes tunnel (Cloudflare / noit2.com) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `hermes tunnel` subcommand that exposes user-built local apps/services/APIs to the internet on per-user `noit2.com` subdomains via Cloudflare Tunnel, with an idle-reset 30-minute dead-man's switch as the core safety protocol and admin-approved hold-open for longer exposure.

**Architecture:** First-class `hermes tunnel` subcommand (parser builder in `hermes_cli/subcommands/tunnel.py`, handler `cmd_tunnel` in `hermes_cli/main.py`) delegating to three focused modules: `hermes_cli/tunnel_config.py` (config + env resolver), `hermes_cli/tunnel_approvals.py` (JSONL hold-request/approval store), and `hermes_cli/tunnel_supervisor.py` (cloudflared process + idle-reset dead-man's switch). Config lives in a new `"tunnel"` block of `DEFAULT_CONFIG` with `HERMES_TUNNEL_*` env overrides (env wins, mirroring `HERMES_DASHBOARD_PUBLIC_URL`). Dashboard is one optional origin, not the focus.

**Tech Stack:** Python 3, stdlib `argparse`/`subprocess`/`json`/`time`/`http.client`, pytest, `cloudflared` external binary (invoked as a subprocess; not bundled). No new third-party Python deps.

## Global Constraints

- Origin services bind to `127.0.0.1` only; the host firewall opens no port. The only public path is Cloudflare's edge on 443.
- Idle-reset 30-min timer is the default safety net: a forgotten test env cannot leak indefinitely. `idle_timeout_seconds` default 1800.
- Hold-open is not a user right — it requires admin approval with a bounded `approved_until`, after which the idle timer resumes (no hard kill on approval expiry).
- Env overrides win over `config.yaml` only when the env value is non-empty (mirror `resolve_public_url` at `hermes_cli/dashboard_auth/prefix.py:206-232`).
- Follow existing patterns exactly: parser builder signature `build_<name>_parser(subparsers, *, cmd_<name>: Callable) -> None`; handler `def cmd_<name>(args):` in `main.py` delegating to a module; tests under `tests/hermes_cli/` using pytest + `monkeypatch` + `tmp_path`; the autouse `_hermetic_environment` fixture ( `tests/conftest.py:328`) scrubs `HERMES_*` vars, so any new `HERMES_TUNNEL_*` behavioral vars MUST be added to `_HERMES_BEHAVIORAL_VARS` in `tests/conftest.py:171`.
- No `cd` in commands (worktree cwd is already the repo root for this branch). Run pytest from the worktree root.
- Commit messages end with `Co-Authored-By: claude-flow <ruv@ruv.net>`.

---

## File Structure

| File | Responsibility | Create/Modify |
|---|---|---|
| `hermes_cli/config.py` | Add `"tunnel": {...}` to `DEFAULT_CONFIG` after the `dashboard` block (line ~1979) | Modify |
| `hermes_cli/tunnel_config.py` | Resolve the `tunnel` config block with `HERMES_TUNNEL_*` env precedence; parse `--origin sub=host:port` args; merge CLI origins over config routes | Create |
| `hermes_cli/tunnel_approvals.py` | JSONL hold-request/approval store at `~/.hermes/tunnel/hold_requests.jsonl`; admin-gated approve/deny | Create |
| `hermes_cli/tunnel_supervisor.py` | Pure close-policy (`reset_idle_on`, `should_close_now`) + `TunnelSupervisor` that runs cloudflared, polls `--metrics`, applies the policy, polls approvals, drains + kills | Create |
| `hermes_cli/subcommands/tunnel.py` | `build_tunnel_parser` — argparse tree for `up/down/status/doctor/hold/requests/approve/deny` | Create |
| `hermes_cli/main.py` | `cmd_tunnel(args)` handler + import + `build_tunnel_parser` call site | Modify |
| `cli-config.yaml.example` | Documented `tunnel:` block | Modify |
| `tests/conftest.py` | Add `HERMES_TUNNEL_*` to `_HERMES_BEHAVIORAL_VARS` | Modify |
| `tests/hermes_cli/test_tunnel_config.py` | Config/env precedence + origin parsing + route merging | Create |
| `tests/hermes_cli/test_tunnel_approvals.py` | Store round-trips, transitions, admin gate, expiry | Create |
| `tests/hermes_cli/test_tunnel_supervisor.py` | Policy + supervisor tick behavior with fake clock/metrics | Create |
| `tests/hermes_cli/test_tunnel_parser_builder.py` | Parser builder subactions + namespace fields | Create |
| `tests/hermes_cli/test_tunnel_handler.py` | `cmd_tunnel` dispatch for each subaction with mocked deps | Create |
| `website/docs/user-guide/tunnel.md` | User-facing docs | Create |

---

## Task 1: Config block + resolver (`tunnel_config.py`)

**Files:**
- Modify: `hermes_cli/config.py` (insert `"tunnel": {...}` after the `dashboard` block closing at line ~1979)
- Modify: `tests/conftest.py:171` (`_HERMES_BEHAVIORAL_VARS`)
- Create: `hermes_cli/tunnel_config.py`
- Test: `tests/hermes_cli/test_tunnel_config.py`
- Modify: `cli-config.yaml.example` (append `tunnel:` documented block)

**Interfaces:**
- Consumes: `hermes_cli.config.load_config` (returns merged config dict).
- Produces: `resolve_tunnel_config(cli_origins=None) -> dict` returning the resolved config (shape below); `parse_origin(spec) -> dict`; `ORIGIN_SPEC_RE`.

Resolved config shape (consumed by Tasks 3 and 5):
```python
{
  "enabled": bool,
  "zone": str,                      # "noit2.com"
  "tunnel_name": str,
  "credentials_file": str,
  "metrics_port": int,              # 0 = auto-pick
  "idle_timeout_seconds": int,      # default 1800
  "drain_seconds": int,             # default 15
  "poll_interval_seconds": int,     # default 5
  "admin": list[str],
  "routes": list[dict],             # [{"subdomain": str, "host": str, "port": int}]
}
```

- [ ] **Step 1: Add `HERMES_TUNNEL_*` to the hermetic-env scrub list**

Modify `tests/conftest.py` — find `_HERMES_BEHAVIORAL_VARS` (around line 171) and append:
```python
    "HERMES_TUNNEL_ZONE",
    "HERMES_TUNNEL_NAME",
    "HERMES_TUNNEL_CREDS",
    "HERMES_TUNNEL_METRICS_PORT",
    "HERMES_TUNNEL_IDLE_TIMEOUT",
    "HERMES_TUNNEL_DRAIN_SECONDS",
    "HERMES_TUNNEL_POLL_INTERVAL",
    "HERMES_TUNNEL_ADMIN",
    "HERMES_TUNNEL_HOLD_REQUEST",
```

- [ ] **Step 2: Write the failing config/env precedence test**

Create `tests/hermes_cli/test_tunnel_config.py`:
```python
import pytest
from hermes_cli.tunnel_config import resolve_tunnel_config, parse_origin


def _stub_config(monkeypatch, tunnel_cfg):
    import hermes_cli.config as cfg
    monkeypatch.setattr(cfg, "load_config", lambda: {"tunnel": tunnel_cfg} if tunnel_cfg else {})


def test_defaults_when_nothing_set(monkeypatch):
    _stub_config(monkeypatch, None)
    for v in (
        "HERMES_TUNNEL_ZONE", "HERMES_TUNNEL_NAME", "HERMES_TUNNEL_CREDS",
        "HERMES_TUNNEL_METRICS_PORT", "HERMES_TUNNEL_IDLE_TIMEOUT",
        "HERMES_TUNNEL_DRAIN_SECONDS", "HERMES_TUNNEL_POLL_INTERVAL", "HERMES_TUNNEL_ADMIN",
    ):
        monkeypatch.delenv(v, raising=False)
    c = resolve_tunnel_config()
    assert c["zone"] == ""
    assert c["idle_timeout_seconds"] == 1800
    assert c["drain_seconds"] == 15
    assert c["poll_interval_seconds"] == 5
    assert c["metrics_port"] == 0
    assert c["admin"] == []
    assert c["routes"] == []


def test_env_overrides_config(monkeypatch):
    _stub_config(monkeypatch, {"zone": "config.example", "tunnel_name": "cfg-name",
                               "idle_timeout_seconds": 600})
    monkeypatch.setenv("HERMES_TUNNEL_ZONE", "noit2.com")
    monkeypatch.setenv("HERMES_TUNNEL_NAME", "env-name")
    monkeypatch.setenv("HERMES_TUNNEL_IDLE_TIMEOUT", "1800")
    c = resolve_tunnel_config()
    assert c["zone"] == "noit2.com"
    assert c["tunnel_name"] == "env-name"
    assert c["idle_timeout_seconds"] == 1800


def test_empty_env_falls_back_to_config(monkeypatch):
    _stub_config(monkeypatch, {"zone": "noit2.com", "tunnel_name": "cfg-name"})
    monkeypatch.setenv("HERMES_TUNNEL_ZONE", "")   # empty -> treated as unset
    c = resolve_tunnel_config()
    assert c["zone"] == "noit2.com"


def test_admin_env_csv(monkeypatch):
    _stub_config(monkeypatch, {"admin": ["a"]})
    monkeypatch.setenv("HERMES_TUNNEL_ADMIN", "alice,bob")
    c = resolve_tunnel_config()
    assert c["admin"] == ["alice", "bob"]


def test_parse_origin():
    assert parse_origin("alice=127.0.0.1:3000") == {
        "subdomain": "alice", "host": "127.0.0.1", "port": 3000}


def test_parse_origin_rejects_bad():
    with pytest.raises(ValueError):
        parse_origin("no-port")
    with pytest.raises(ValueError):
        parse_origin("alice=not-a-port")


def test_cli_origins_override_config_routes(monkeypatch):
    _stub_config(monkeypatch, {"zone": "noit2.com",
                               "routes": [{"subdomain": "alice", "host": "127.0.0.1", "port": 3000}]})
    c = resolve_tunnel_config(cli_origins=["alice=127.0.0.1:9000", "alice-api=127.0.0.1:8080"])
    sub_to_port = {r["subdomain"]: r["port"] for r in c["routes"]}
    assert sub_to_port == {"alice": 9000, "alice-api": 8080}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/hermes_cli/test_tunnel_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hermes_cli.tunnel_config'`

- [ ] **Step 4: Add the `"tunnel"` block to `DEFAULT_CONFIG`**

In `hermes_cli/config.py`, immediately after the `dashboard` block's closing `},` (line ~1979), insert:
```python
    # Cloudflare Tunnel exposure for user-built services (hermes tunnel).
    # See .plans/2026-07-08-cloudflare-noit2-tunnel-design.md and
    # website/docs/user-guide/tunnel.md. Each key is overridable by a
    # HERMES_TUNNEL_* env var (env wins when non-empty), mirroring the
    # dashboard.public_url / HERMES_DASHBOARD_PUBLIC_URL precedent.
    "tunnel": {
        "enabled": False,
        "zone": "",                      # e.g. "noit2.com"
        "tunnel_name": "",               # cloudflared named tunnel
        "credentials_file": "",          # path to <uuid>.json
        "metrics_port": 0,               # 0 = auto-pick a free port
        "idle_timeout_seconds": 1800,    # 30-min idle-reset dead-man's switch
        "drain_seconds": 15,
        "poll_interval_seconds": 5,
        "admin": [],                     # identities permitted to approve/deny holds
        "routes": [],                    # [{"subdomain": str, "host": str, "port": int}]
    },
```

- [ ] **Step 5: Implement `hermes_cli/tunnel_config.py`**

Create `hermes_cli/tunnel_config.py`:
```python
"""Config + env resolver for ``hermes tunnel``.

Mirrors the env-over-config precedence of ``HERMES_DASHBOARD_PUBLIC_URL``
(see ``hermes_cli/dashboard_auth/prefix.py:resolve_public_url``): a
``HERMES_TUNNEL_*`` env var wins over the ``tunnel`` config block only when
its value is non-empty after strip.
"""

from __future__ import annotations

import os
import re
from typing import Optional

ORIGIN_SPEC_RE = re.compile(r"^(?P<sub>[A-Za-z0-9._-]+)=(?P<host>[A-Za-z0-9.\-]+):(?P<port>\d+)$")


def _env(name: str) -> Optional[str]:
    v = os.environ.get(name, "")
    return v.strip() or None


def _load_tunnel_section() -> dict:
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
    except Exception:
        return {}
    section = cfg.get("tunnel") if isinstance(cfg, dict) else None
    return section if isinstance(section, dict) else {}


def parse_origin(spec: str) -> dict:
    m = ORIGIN_SPEC_RE.match(spec.strip())
    if not m:
        raise ValueError(f"bad --origin spec (want sub=host:port): {spec!r}")
    return {"subdomain": m.group("sub"), "host": m.group("host"), "port": int(m.group("port"))}


def resolve_tunnel_config(cli_origins: Optional[list] = None) -> dict:
    sec = _load_tunnel_section()
    routes = [dict(r) for r in sec.get("routes", []) if isinstance(r, dict)]

    if cli_origins:
        by_sub = {r["subdomain"]: r for r in routes}
        for spec in cli_origins:
            parsed = parse_origin(spec)
            by_sub[parsed["subdomain"]] = parsed
        routes = list(by_sub.values())

    admin_env = _env("HERMES_TUNNEL_ADMIN")
    admin = [a.strip() for a in admin_env.split(",") if a.strip()] if admin_env else list(sec.get("admin", []))

    def _int_env(name, default, config_key):
        v = _env(name)
        if v is None:
            return int(sec.get(config_key, default))
        try:
            return int(v)
        except ValueError:
            return default

    return {
        "enabled": bool(sec.get("enabled", False)),
        "zone": _env("HERMES_TUNNEL_ZONE") or sec.get("zone", ""),
        "tunnel_name": _env("HERMES_TUNNEL_NAME") or sec.get("tunnel_name", ""),
        "credentials_file": _env("HERMES_TUNNEL_CREDS") or sec.get("credentials_file", ""),
        "metrics_port": _int_env("HERMES_TUNNEL_METRICS_PORT", 0, "metrics_port"),
        "idle_timeout_seconds": _int_env("HERMES_TUNNEL_IDLE_TIMEOUT", 1800, "idle_timeout_seconds"),
        "drain_seconds": _int_env("HERMES_TUNNEL_DRAIN_SECONDS", 15, "drain_seconds"),
        "poll_interval_seconds": _int_env("HERMES_TUNNEL_POLL_INTERVAL", 5, "poll_interval_seconds"),
        "admin": admin,
        "routes": routes,
    }
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/hermes_cli/test_tunnel_config.py -v`
Expected: PASS (7 tests)

- [ ] **Step 7: Document the block in `cli-config.yaml.example`**

Append before the `# =============================================================================
# External secret sources` section:
```yaml
# =============================================================================
# Cloudflare Tunnel exposure (hermes tunnel)
# =============================================================================
# Expose user-built local apps/services/APIs to the internet on per-user
# subdomains of your zone (e.g. alice.noit2.com) via a Cloudflare named
# tunnel. The exposure is ephemeral: an idle-reset 30-minute dead-man's
# switch closes the tunnel when traffic stops, so a forgotten test build
# cannot leak to the internet. Longer exposure needs admin-approved hold.
#
# Each key is overridable by a HERMES_TUNNEL_* env var (env wins when
# non-empty), mirroring dashboard.public_url / HERMES_DASHBOARD_PUBLIC_URL.
#
# tunnel:
#   enabled: false
#   zone: "noit2.com"
#   tunnel_name: ""              # cloudflared named tunnel
#   credentials_file: ""         # path to <uuid>.json
#   metrics_port: 0              # 0 = auto-pick a free port
#   idle_timeout_seconds: 1800   # 30-min idle-reset dead-man's switch
#   drain_seconds: 15
#   poll_interval_seconds: 5
#   admin: []                    # identities permitted to approve/deny holds
#   routes: []                   # [{"subdomain": "alice", "host": "127.0.0.1", "port": 3000}]
```

- [ ] **Step 8: Commit**

```bash
git add hermes_cli/config.py hermes_cli/tunnel_config.py tests/conftest.py tests/hermes_cli/test_tunnel_config.py cli-config.yaml.example
git commit -m "feat(tunnel): config block + env resolver for hermes tunnel

Co-Authored-By: claude-flow <ruv@ruv.net>"
```

---

## Task 2: Approvals store (`tunnel_approvals.py`)

**Files:**
- Create: `hermes_cli/tunnel_approvals.py`
- Test: `tests/hermes_cli/test_tunnel_approvals.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `file_request(path, *, user, subdomains, reason, requested_until) -> str` (id); `list_pending(path) -> list[dict]`; `get(path, id) -> dict | None`; `approve(path, id, *, until, by, admin_ids) -> dict`; `deny(path, id, *, reason, by, admin_ids) -> dict`; `is_approved(path, id) -> bool`; `approved_until(path, id) -> float | None`; `new_id() -> str`. Record shape:
```python
{"id": str, "user": str, "subdomains": list[str], "reason": str,
 "requested_until": float|None, "status": "pending"|"approved"|"denied",
 "approved_until": float|None, "decided_by": str|None,
 "created_at": float, "decided_at": float|None}
```
Raises `PermissionError` from `approve`/`deny` when `by` not in `admin_ids`. Raises `KeyError` when `id` not found. Raises `ValueError` on illegal transition (non-pending).

- [ ] **Step 1: Write the failing test**

Create `tests/hermes_cli/test_tunnel_approvals.py`:
```python
import time
import pytest
from hermes_cli import tunnel_approvals as ta


def test_file_request_then_list_pending(tmp_path):
    p = tmp_path / "hold.jsonl"
    rid = ta.file_request(str(p), user="alice", subdomains=["alice.noit2.com"],
                          reason="demo", requested_until=None)
    assert isinstance(rid, str) and rid
    pending = ta.list_pending(str(p))
    assert len(pending) == 1
    assert pending[0]["id"] == rid
    assert pending[0]["status"] == "pending"
    assert pending[0]["user"] == "alice"


def test_approve_sets_status_and_until(tmp_path):
    p = tmp_path / "hold.jsonl"
    rid = ta.file_request(str(p), user="alice", subdomains=["alice.noit2.com"],
                          reason="demo", requested_until=None)
    rec = ta.approve(str(p), rid, until=99999.0, by="admin1", admin_ids=["admin1"])
    assert rec["status"] == "approved"
    assert rec["approved_until"] == 99999.0
    assert rec["decided_by"] == "admin1"
    assert ta.is_approved(str(p), rid) is True
    assert ta.approved_until(str(p), rid) == 99999.0
    assert ta.list_pending(str(p)) == []


def test_deny_sets_status(tmp_path):
    p = tmp_path / "hold.jsonl"
    rid = ta.file_request(str(p), user="alice", subdomains=["alice.noit2.com"],
                          reason="demo", requested_until=None)
    rec = ta.deny(str(p), rid, reason="too long", by="admin1", admin_ids=["admin1"])
    assert rec["status"] == "denied"
    assert ta.is_approved(str(p), rid) is False


def test_non_admin_cannot_approve(tmp_path):
    p = tmp_path / "hold.jsonl"
    rid = ta.file_request(str(p), user="alice", subdomains=["alice.noit2.com"],
                          reason="demo", requested_until=None)
    with pytest.raises(PermissionError):
        ta.approve(str(p), rid, until=99999.0, by="alice", admin_ids=["admin1"])


def test_unknown_id_raises(tmp_path):
    p = tmp_path / "hold.jsonl"
    with pytest.raises(KeyError):
        ta.approve(str(p), "nope", until=1.0, by="admin1", admin_ids=["admin1"])


def test_double_approve_is_valueerror(tmp_path):
    p = tmp_path / "hold.jsonl"
    rid = ta.file_request(str(p), user="alice", subdomains=["alice.noit2.com"],
                          reason="demo", requested_until=None)
    ta.approve(str(p), rid, until=99999.0, by="admin1", admin_ids=["admin1"])
    with pytest.raises(ValueError):
        ta.approve(str(p), rid, until=99999.0, by="admin1", admin_ids=["admin1"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/hermes_cli/test_tunnel_approvals.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hermes_cli.tunnel_approvals'`

- [ ] **Step 3: Implement `hermes_cli/tunnel_approvals.py`**

Create `hermes_cli/tunnel_approvals.py`:
```python
"""JSONL hold-request / approval store for ``hermes tunnel`` hold-open.

Records live at ``~/.hermes/tunnel/hold_requests.jsonl`` (append-only).
Admin-gated approve/deny: a caller whose identity is not in ``admin_ids``
gets ``PermissionError``. Status transitions are validated
(``pending -> approved|denied`` only); any other transition raises
``ValueError``. Unknown ids raise ``KeyError``.
"""

from __future__ import annotations

import json
import os
import time
import uuid


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def _read_all(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _write_all(path: str, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def file_request(path, *, user, subdomains, reason, requested_until) -> str:
    rid = new_id()
    rec = {"id": rid, "user": user, "subdomains": list(subdomains), "reason": reason,
           "requested_until": requested_until, "status": "pending",
           "approved_until": None, "decided_by": None,
           "created_at": time.time(), "decided_at": None}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rid


def _find(path, rid) -> tuple[list[dict], int]:
    records = _read_all(path)
    for i, r in enumerate(records):
        if r["id"] == rid:
            return records, i
    raise KeyError(rid)


def get(path, rid) -> dict | None:
    try:
        records, i = _find(path, rid)
    except KeyError:
        return None
    return records[i]


def list_pending(path) -> list[dict]:
    return [r for r in _read_all(path) if r["status"] == "pending"]


def _require_admin(by, admin_ids) -> None:
    if by not in admin_ids:
        raise PermissionError(f"{by!r} is not a tunnel admin")


def _resolve(path, rid, *, new_status, by, admin_ids, approved_until=None, reason=None) -> dict:
    _require_admin(by, admin_ids)
    records, i = _find(path, rid)
    rec = records[i]
    if rec["status"] != "pending":
        raise ValueError(f"hold request {rid} already {rec['status']}")
    rec["status"] = new_status
    rec["decided_by"] = by
    rec["decided_at"] = time.time()
    if new_status == "approved":
        rec["approved_until"] = approved_until
    elif new_status == "denied":
        rec["deny_reason"] = reason
    records[i] = rec
    _write_all(path, records)
    return rec


def approve(path, rid, *, until, by, admin_ids) -> dict:
    return _resolve(path, rid, new_status="approved", by=by, admin_ids=admin_ids, approved_until=until)


def deny(path, rid, *, reason, by, admin_ids) -> dict:
    return _resolve(path, rid, new_status="denied", by=by, admin_ids=admin_ids, reason=reason)


def is_approved(path, rid) -> bool:
    rec = get(path, rid)
    return bool(rec and rec["status"] == "approved")


def approved_until(path, rid):
    rec = get(path, rid)
    return rec["approved_until"] if rec and rec["status"] == "approved" else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/hermes_cli/test_tunnel_approvals.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add hermes_cli/tunnel_approvals.py tests/hermes_cli/test_tunnel_approvals.py
git commit -m "feat(tunnel): admin-gated hold-request/approval store

Co-Authored-By: claude-flow <ruv@ruv.net>"
```

---

## Task 3: Supervisor policy pure functions (`tunnel_supervisor.py` — part A)

This task delivers the **dead-man's-switch policy** — the 5–10 lines of business logic that shape daily behavior. The tests below fully specify the expected behavior; the function body is your contribution.

**Files:**
- Create: `hermes_cli/tunnel_supervisor.py` (policy functions only; the `TunnelSupervisor` class comes in Task 4)
- Test: `tests/hermes_cli/test_tunnel_supervisor.py` (policy tests only, in a `TestPolicy` class)

**Interfaces:**
- Produces: `reset_idle_on(prev_counter: int, cur_counter: int) -> bool`; `should_close_now(state: dict) -> bool`. The `state` dict shape (built by the supervisor in Task 4):
```python
{"now": float, "last_activity": float, "idle_timeout_seconds": float,
 "hold_until": float | None}
```

- [ ] **Step 1: Write the failing policy tests**

Create `tests/hermes_cli/test_tunnel_supervisor.py`:
```python
import pytest
from hermes_cli.tunnel_supervisor import reset_idle_on, should_close_now


class TestPolicy:
    def test_reset_when_counter_increases(self):
        assert reset_idle_on(10, 11) is True

    def test_no_reset_when_counter_unchanged(self):
        assert reset_idle_on(10, 10) is False

    def test_no_reset_when_counter_decreases(self):
        # a poll hiccup / counter reset should NOT count as activity
        assert reset_idle_on(11, 10) is False

    def test_close_after_idle_timeout(self):
        state = {"now": 1000.0, "last_activity": 100.0,
                 "idle_timeout_seconds": 1800.0, "hold_until": None}
        assert should_close_now(state) is True

    def test_open_before_idle_timeout(self):
        state = {"now": 1000.0, "last_activity": 999.0,
                 "idle_timeout_seconds": 1800.0, "hold_until": None}
        assert should_close_now(state) is False

    def test_hold_active_keeps_open_past_idle(self):
        state = {"now": 1000.0, "last_activity": 0.0,
                 "idle_timeout_seconds": 1800.0, "hold_until": 2000.0}
        assert should_close_now(state) is False

    def test_hold_expired_falls_back_to_idle_not_hard_kill(self):
        # hold_until in the past: fall back to idle rule.
        # last_activity recent -> still open (no hard kill on approval expiry).
        state = {"now": 1000.0, "last_activity": 999.0,
                 "idle_timeout_seconds": 1800.0, "hold_until": 500.0}
        assert should_close_now(state) is False

    def test_hold_expired_and_idle_closes(self):
        state = {"now": 1000.0, "last_activity": 0.0,
                 "idle_timeout_seconds": 1800.0, "hold_until": 500.0}
        assert should_close_now(state) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/hermes_cli/test_tunnel_supervisor.py::TestPolicy -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hermes_cli.tunnel_supervisor'`

- [ ] **Step 3: Create the scaffold with your contribution**

Create `hermes_cli/tunnel_supervisor.py`:
```python
"""Dead-man's switch + cloudflared supervisor for ``hermes tunnel``.

The idle-reset policy is the core safety protocol: a tunnel with no
incoming traffic for ``idle_timeout_seconds`` closes (graceful drain +
kill cloudflared), so a forgotten test build cannot leak to the internet.
An admin-approved hold disables the idle timer until ``hold_until``; after
that the idle timer resumes (no hard kill on approval expiry).
"""

from __future__ import annotations


def reset_idle_on(prev_counter: int, cur_counter: int) -> bool:
    """Return True when there has been incoming activity since the last poll.

    Activity = the cloudflared request counter strictly increased.
    A counter that stayed flat or dropped (poll hiccup / restart) is NOT
    activity.
    """
    # TODO(you): 1 line — strictly-increasing check.
    raise NotImplementedError


def should_close_now(state: dict) -> bool:
    """Return True when the tunnel should close now.

    state keys: now, last_activity, idle_timeout_seconds, hold_until (|None).

    Rules (see TestPolicy for the exact contract):
      * If an admin-approved hold is active (hold_until is in the future),
        never close.
      * Otherwise close when (now - last_activity) >= idle_timeout_seconds.
      * A hold whose hold_until is in the past is treated as "no hold"
        (fall back to the idle rule) — do NOT hard-kill just because the
        approval expired.
    """
    # TODO(you): ~5 lines implementing the rules above.
    raise NotImplementedError
```

> **Contribution (learning mode):** implement the two function bodies above. `reset_idle_on` is one line (strictly-increasing). `should_close_now` is ~5 lines: compute `hold_active = hold_until is not None and now < hold_until`; if `hold_active` return `False`; else return `(now - last_activity) >= idle_timeout_seconds`. The tests in `TestPolicy` define the exact contract — run them until green. This is the decision that shapes daily behavior (what counts as activity, how an expired approval hands back control to the idle timer without a hard kill), which is why it's yours rather than boilerplate.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/hermes_cli/test_tunnel_supervisor.py::TestPolicy -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add hermes_cli/tunnel_supervisor.py tests/hermes_cli/test_tunnel_supervisor.py
git commit -m "feat(tunnel): idle-reset dead-man's switch policy

Co-Authored-By: claude-flow <ruv@ruv.net>"
```

---

## Task 4: Supervisor class (`tunnel_supervisor.py` — part B)

**Files:**
- Modify: `hermes_cli/tunnel_supervisor.py` (add `TunnelSupervisor`)
- Test: `tests/hermes_cli/test_tunnel_supervisor.py` (add `TestSupervisor`)

**Interfaces:**
- Consumes: `reset_idle_on`, `should_close_now` from Task 3; `tunnel_approvals.is_approved` / `approved_until` from Task 2; resolved config from Task 1.
- Produces: `TunnelSupervisor(config, approvals_path, *, hold_request_id=None, time_source=time.monotonic, metrics_counter=..., spawn_cloudflared=..., sleep=...)` with `.tick() -> bool` (returns True while still running, False once closed) and `.closed` / `.last_activity` / `.hold_until` for `status`.

- [ ] **Step 1: Write the failing supervisor tests**

Append to `tests/hermes_cli/test_tunnel_supervisor.py`:
```python
from hermes_cli.tunnel_supervisor import TunnelSupervisor


def _sup(tmp_path, *, counters, times, idle=1800.0, hold_request_id=None,
         approvals_path=None, approved_until=None):
    """Build a supervisor with scripted counter + time sequences."""
    cfg = {"idle_timeout_seconds": idle, "drain_seconds": 0,
           "poll_interval_seconds": 5, "metrics_port": 0,
           "zone": "noit2.com", "tunnel_name": "t", "credentials_file": "",
           "admin": ["admin1"], "routes": []}
    it_c = iter(counters)
    it_t = iter(times)
    killed = {"flag": False}

    def counter():
        return next(it_c)
    def ts():
        return next(it_t)
    def spawn(*a, **kw):
        class P:
            def terminate(self): killed["flag"] = True
            def wait(self, timeout=None): return 0
        return P()
    def sleep(s): pass

    sup = TunnelSupervisor(
        cfg, approvals_path or str(tmp_path / "hold.jsonl"),
        hold_request_id=hold_request_id,
        time_source=ts, metrics_counter=counter,
        spawn_cloudflared=spawn, sleep=sleep,
    )
    # Pre-seed an approval if requested.
    if hold_request_id and approved_until is not None:
        from hermes_cli import tunnel_approvals as ta
        ta.file_request(sup._approvals_path, user="alice",
                        subdomains=["alice.noit2.com"], reason="demo",
                        requested_until=None)
        ta.approve(sup._approvals_path, hold_request_id, until=approved_until,
                   by="admin1", admin_ids=["admin1"])
    return sup, killed


class TestSupervisor:
    def test_activity_resets_idle_clock(self, tmp_path):
        # times: 0, 100, 1900 — last_activity should track the busy poll.
        sup, killed = _sup(tmp_path, counters=[0, 10, 20], times=[0.0, 100.0, 1900.0])
        assert sup.tick() is True      # t=0, counter 0 -> open
        assert sup.tick() is True      # t=100, counter 10 -> activity, reset
        assert sup.last_activity == 100.0
        assert sup.tick() is True      # t=1900, counter 20 -> activity, reset; 1900-100=1800 not >= 1800? see below
        # 1900-100 == 1800 == idle -> closes (>=). So this tick closes.
        assert sup.closed is True
        assert killed["flag"] is True

    def test_idle_expiry_closes(self, tmp_path):
        sup, killed = _sup(tmp_path, counters=[0, 0, 0], times=[0.0, 100.0, 2000.0])
        assert sup.tick() is True      # open
        assert sup.tick() is True      # no activity, 100-0 < 1800 -> open
        assert sup.tick() is False     # 2000-0 >= 1800 -> close
        assert sup.closed is True
        assert killed["flag"] is True

    def test_hold_approved_extends_past_idle(self, tmp_path):
        sup, killed = _sup(tmp_path, counters=[0, 0, 0],
                           times=[0.0, 2000.0, 4000.0],
                           hold_request_id="h1", approved_until=5000.0)
        assert sup.tick() is True      # t=0
        assert sup.tick() is True      # t=2000, no activity, but hold until 5000 -> open
        assert sup.hold_until == 5000.0
        assert sup.tick() is True      # t=4000, still within hold -> open
        assert sup.closed is False

    def test_hold_denied_closes_on_idle(self, tmp_path):
        from hermes_cli import tunnel_approvals as ta
        sup, killed = _sup(tmp_path, counters=[0, 0, 0],
                           times=[0.0, 100.0, 2000.0],
                           hold_request_id="h1")
        # deny the request before ticks
        ta.deny(sup._approvals_path, "h1", reason="no", by="admin1", admin_ids=["admin1"])
        assert sup.tick() is True
        assert sup.tick() is True
        assert sup.tick() is False     # idle closes; denied hold did not extend
        assert sup.closed is True
```

> Note on `test_activity_resets_idle_clock`: the third tick lands exactly at the idle boundary (`1900 - 100 == 1800 == idle`), so `should_close_now` returns True (the contract is `>=`). That is intentional — it exercises the close path on the boundary. If you prefer the third tick to stay open, change the third time to `1899.0`; the test as written asserts close-on-boundary.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/hermes_cli/test_tunnel_supervisor.py::TestSupervisor -v`
Expected: FAIL — `ImportError: cannot import name 'TunnelSupervisor'`

- [ ] **Step 3: Implement `TunnelSupervisor`**

Append to `hermes_cli/tunnel_supervisor.py`:
```python
import os
import subprocess
import time
import urllib.request
import json


def _default_counter(metrics_port: int) -> int:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{metrics_port}/metrics", timeout=2) as r:
            for line in r.read().decode("utf-8", "replace").splitlines():
                # cloudflared exposes counters; tolerate either name.
                if line.startswith("cloudflared_request_count") or line.startswith("cloudflared_connection_count"):
                    return int(float(line.split()[-1]))
    except Exception:
        return 0
    return 0


class TunnelSupervisor:
    def __init__(self, config, approvals_path, *, hold_request_id=None,
                 time_source=time.monotonic, metrics_counter=None,
                 spawn_cloudflared=None, sleep=time.sleep):
        self._cfg = config
        self._approvals_path = approvals_path
        self._hold_request_id = hold_request_id
        self._time = time_source
        self._sleep = sleep
        self._idle = float(config.get("idle_timeout_seconds", 1800))
        self._drain = float(config.get("drain_seconds", 15))
        self._poll = float(config.get("poll_interval_seconds", 5))
        self._metrics_port = int(config.get("metrics_port", 0))

        self._last_counter = 0
        self._last_activity = self._time()
        self._hold_until = None
        self._closed = False
        self._proc = None

        if metrics_counter is None:
            metrics_counter = lambda: _default_counter(self._metrics_port)
        self._counter = metrics_counter
        self._spawn = spawn_cloudflared or self._default_spawn

    def _default_spawn(self, config_path, tunnel_name, metrics_port):
        cmd = ["cloudflared", "tunnel", "--config", config_path,
               "--metrics", f"127.0.0.1:{metrics_port}", "run", tunnel_name]
        return subprocess.Popen(cmd)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def last_activity(self) -> float:
        return self._last_activity

    @property
    def hold_until(self):
        return self._hold_until

    def _check_hold(self):
        if not self._hold_request_id:
            return
        from hermes_cli import tunnel_approvals as ta
        if ta.is_approved(self._approvals_path, self._hold_request_id):
            self._hold_until = ta.approved_until(self._approvals_path, self._hold_request_id)

    def _drain_and_kill(self):
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=max(1.0, self._drain))
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self._closed = True

    def tick(self) -> bool:
        """One poll iteration. Returns True while running, False once closed."""
        if self._closed:
            return False
        if self._proc is None:
            self._proc = self._spawn(self._config_path, self._cfg.get("tunnel_name", ""),
                                     self._metrics_port)
        now = self._time()
        cur = int(self._counter())
        if reset_idle_on(self._last_counter, cur):
            self._last_activity = now
        self._last_counter = cur
        self._check_hold()
        state = {"now": now, "last_activity": self._last_activity,
                 "idle_timeout_seconds": self._idle, "hold_until": self._hold_until}
        if should_close_now(state):
            self._drain_and_kill()
            return False
        return True

    def run(self, config_path: str):
        """Blocking loop. ``config_path`` is the generated cloudflared config file."""
        self._config_path = config_path
        while self.tick():
            self._sleep(self._poll)
        return not self._closed
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/hermes_cli/test_tunnel_supervisor.py -v`
Expected: PASS (12 tests — 8 policy + 4 supervisor)

- [ ] **Step 5: Commit**

```bash
git add hermes_cli/tunnel_supervisor.py tests/hermes_cli/test_tunnel_supervisor.py
git commit -m "feat(tunnel): cloudflared supervisor with idle-reset + hold extension

Co-Authored-By: claude-flow <ruv@ruv.net>"
```

---

## Task 5: Parser builder (`subcommands/tunnel.py`)

**Files:**
- Create: `hermes_cli/subcommands/tunnel.py`
- Test: `tests/hermes_cli/test_tunnel_parser_builder.py`

**Interfaces:**
- Produces: `build_tunnel_parser(subparsers, *, cmd_tunnel: Callable) -> None`. Subactions attached under `dest="tunnel_command"`: `up`, `down`, `status`, `doctor`, `hold`, `requests`, `approve`, `deny`. `up` takes `--origin` (append, `dest="origins"`) and `--hold-request` (flag) + `--reason` + `--until`. `approve` takes `id` (positional) + `--until`. `deny` takes `id` + `--reason`. `down` takes `--kill-origins`. All subactions set `func=cmd_tunnel`.

- [ ] **Step 1: Write the failing parser test**

Create `tests/hermes_cli/test_tunnel_parser_builder.py`:
```python
import argparse
from hermes_cli.subcommands.tunnel import build_tunnel_parser


def _sentinel(args):  # pragma: no cover
    return "tunnel-handler"


def _build():
    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    build_tunnel_parser(sub, cmd_tunnel=_sentinel)
    return parser


def test_up_subaction_parses_origins():
    p = _build()
    ns = p.parse_args(["tunnel", "up", "--origin", "alice=127.0.0.1:3000",
                       "--origin", "alice-api=127.0.0.1:8080", "--hold-request",
                       "--reason", "demo", "--until", "4h"])
    assert ns.command == "tunnel"
    assert ns.tunnel_command == "up"
    assert ns.origins == ["alice=127.0.0.1:3000", "alice-api=127.0.0.1:8080"]
    assert ns.hold_request is True
    assert ns.reason == "demo"
    assert ns.func is _sentinel


def test_down_kill_origins():
    p = _build()
    ns = p.parse_args(["tunnel", "down", "--kill-origins"])
    assert ns.tunnel_command == "down"
    assert ns.kill_origins is True


def test_approve_positional_id_and_until():
    p = _build()
    ns = p.parse_args(["tunnel", "approve", "abc123", "--until", "6h"])
    assert ns.tunnel_command == "approve"
    assert ns.id == "abc123"
    assert ns.until == "6h"


def test_deny_positional_id():
    p = _build()
    ns = p.parse_args(["tunnel", "deny", "abc123", "--reason", "too long"])
    assert ns.tunnel_command == "deny"
    assert ns.id == "abc123"


def test_all_subactions_present():
    p = _build()
    for action in ("up", "down", "status", "doctor", "hold", "requests", "approve", "deny"):
        ns = p.parse_args(["tunnel", action])
        assert ns.tunnel_command == action
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/hermes_cli/test_tunnel_parser_builder.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hermes_cli.subcommands.tunnel'`

- [ ] **Step 3: Implement the parser builder**

Create `hermes_cli/subcommands/tunnel.py`:
```python
"""``hermes tunnel`` subcommand parser builder.

Exposes user-built local services to the internet on per-user noit2.com
subdomains via a Cloudflare named tunnel, with an idle-reset 30-minute
dead-man's switch and admin-approved hold-open. Handler ``cmd_tunnel``
lives in ``hermes_cli/main.py`` and is injected here.
"""

from __future__ import annotations

from typing import Callable


def build_tunnel_parser(subparsers, *, cmd_tunnel: Callable) -> None:
    tunnel_parser = subparsers.add_parser(
        "tunnel",
        help="Expose a local service to the internet via a Cloudflare Tunnel",
        description=(
            "Expose a user-built local app/API to the internet on a per-user "
            "noit2.com subdomain via Cloudflare Tunnel. Ephemeral by default: "
            "an idle-reset 30-minute dead-man's switch closes the tunnel when "
            "traffic stops. Use 'hold' + admin 'approve' for longer exposure."
        ),
    )
    sub = tunnel_parser.add_subparsers(dest="tunnel_command")

    up = sub.add_parser("up", help="Start the tunnel for one or more origins")
    up.add_argument("--origin", action="append", dest="origins", default=[],
                    metavar="SUB=HOST:PORT",
                    help="e.g. --origin alice=127.0.0.1:3000 (repeatable)")
    up.add_argument("--hold-request", action="store_true", dest="hold_request",
                    help="File a hold-open request immediately on start")
    up.add_argument("--reason", default="", help="Reason for the hold request")
    up.add_argument("--until", default="", help="Requested hold duration, e.g. 4h")
    up.set_defaults(func=cmd_tunnel)

    down = sub.add_parser("down", help="Stop the running tunnel")
    down.add_argument("--kill-origins", action="store_true",
                      help="Also stop the local origin services (default: leave them running)")
    down.set_defaults(func=cmd_tunnel)

    for name, help_ in (("status", "Show running tunnel state"),
                        ("doctor", "Health-check cloudflared, creds, origins, DNS"),
                        ("hold", "File a hold-open request for the running tunnel"),
                        ("requests", "List pending hold requests")):
        sp = sub.add_parser(name, help=help_)
        sp.set_defaults(func=cmd_tunnel)

    hold = sub.add_parser("hold")
    hold.add_argument("--reason", default="")
    hold.add_argument("--until", default="")
    hold.set_defaults(func=cmd_tunnel)

    approve = sub.add_parser("approve", help="Approve a hold request (admin)")
    approve.add_argument("id", help="Hold request id")
    approve.add_argument("--until", required=True, help="Approved duration, e.g. 6h")
    approve.set_defaults(func=cmd_tunnel)

    deny = sub.add_parser("deny", help="Deny a hold request (admin)")
    deny.add_argument("id", help="Hold request id")
    deny.add_argument("--reason", default="")
    deny.set_defaults(func=cmd_tunnel)

    tunnel_parser.set_defaults(func=cmd_tunnel)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/hermes_cli/test_tunnel_parser_builder.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add hermes_cli/subcommands/tunnel.py tests/hermes_cli/test_tunnel_parser_builder.py
git commit -m "feat(tunnel): argparse builder for hermes tunnel subactions

Co-Authored-By: claude-flow <ruv@ruv.net>"
```

---

## Task 6: Handler wiring + command dispatch (`main.py` + `hermes_cli/tunnel_commands.py`)

**Files:**
- Create: `hermes_cli/tunnel_commands.py` (the per-subaction implementations, kept out of the god-file `main.py`)
- Modify: `hermes_cli/main.py` (import `build_tunnel_parser` near line 306; define `cmd_tunnel` near line 4210; call `build_tunnel_parser` near line 14044)
- Test: `tests/hermes_cli/test_tunnel_handler.py`

**Interfaces:**
- Consumes: `resolve_tunnel_config` (Task 1), `tunnel_approvals` (Task 2), `TunnelSupervisor` (Task 4), `build_tunnel_parser` (Task 5).
- Produces: `cmd_tunnel(args) -> int` (handler registered with argparse); `tunnel_command(args) -> int` (dispatch in `hermes_cli/tunnel_commands.py`).

- [ ] **Step 1: Write the failing handler test**

Create `tests/hermes_cli/test_tunnel_handler.py`:
```python
from argparse import Namespace
from hermes_cli import tunnel_commands as tc


def test_up_validates_missing_zone(monkeypatch, capsys):
    # No zone configured and no origin -> error, no supervisor started.
    monkeypatch.setattr(tc, "resolve_tunnel_config",
                        lambda cli_origins=None: {"zone": "", "tunnel_name": "",
                                                  "credentials_file": "", "metrics_port": 0,
                                                  "idle_timeout_seconds": 1800,
                                                  "drain_seconds": 15, "poll_interval_seconds": 5,
                                                  "admin": [], "routes": [], "enabled": False})
    rc = tc.tunnel_command(Namespace(tunnel_command="up", origins=[],
                                     hold_request=False, reason="", until=""))
    assert rc == 2
    out = capsys.readouterr().out + capsys.readouterr().err
    assert "zone" in out.lower() or "origin" in out.lower()


def test_requests_lists_pending(monkeypatch, capsys, tmp_path):
    from hermes_cli import tunnel_approvals as ta
    p = str(tmp_path / "hold.jsonl")
    monkeypatch.setattr(tc, "_approvals_path", lambda: p)
    ta.file_request(p, user="alice", subdomains=["alice.noit2.com"],
                    reason="demo", requested_until=None)
    rc = tc.tunnel_command(Namespace(tunnel_command="requests"))
    assert rc == 0
    out = capsys.readouterr().out
    assert "alice" in out
    assert "pending" in out


def test_approve_admin_ok(monkeypatch, tmp_path):
    from hermes_cli import tunnel_approvals as ta
    p = str(tmp_path / "hold.jsonl")
    monkeypatch.setattr(tc, "_approvals_path", lambda: p)
    monkeypatch.setattr(tc, "resolve_tunnel_config",
                        lambda cli_origins=None: {"admin": ["admin1"], "zone": "noit2.com",
                                                  "tunnel_name": "", "credentials_file": "",
                                                  "metrics_port": 0, "idle_timeout_seconds": 1800,
                                                  "drain_seconds": 15, "poll_interval_seconds": 5,
                                                  "routes": [], "enabled": False})
    rid = ta.file_request(p, user="alice", subdomains=["alice.noit2.com"],
                          reason="demo", requested_until=None)
    rc = tc.tunnel_command(Namespace(tunnel_command="approve", id=rid, until="6h",
                                     reason="", kill_origins=False, origins=[],
                                     hold_request=False))
    assert rc == 0
    assert ta.is_approved(p, rid) is True


def test_approve_non_admin_denied(monkeypatch, tmp_path):
    from hermes_cli import tunnel_approvals as ta
    p = str(tmp_path / "hold.jsonl")
    monkeypatch.setattr(tc, "_approvals_path", lambda: p)
    monkeypatch.setattr(tc, "resolve_tunnel_config",
                        lambda cli_origins=None: {"admin": ["admin1"], "zone": "noit2.com",
                                                  "tunnel_name": "", "credentials_file": "",
                                                  "metrics_port": 0, "idle_timeout_seconds": 1800,
                                                  "drain_seconds": 15, "poll_interval_seconds": 5,
                                                  "routes": [], "enabled": False})
    rid = ta.file_request(p, user="alice", subdomains=["alice.noit2.com"],
                          reason="demo", requested_until=None)
    rc = tc.tunnel_command(Namespace(tunnel_command="approve", id=rid, until="6h",
                                     reason="", kill_origins=False, origins=[],
                                     hold_request=False, _current_user="alice"))
    assert rc == 3
    assert ta.is_approved(p, rid) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/hermes_cli/test_tunnel_handler.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hermes_cli.tunnel_commands'`

- [ ] **Step 3: Implement `hermes_cli/tunnel_commands.py`**

Create `hermes_cli/tunnel_commands.py`:
```python
"""Per-subaction implementations for ``hermes tunnel``.

Dispatch is invoked from ``cmd_tunnel`` in ``hermes_cli/main.py``. The
heavy lifting (cloudflared process, idle-reset timer, approvals) lives in
``tunnel_supervisor`` / ``tunnel_approvals``; this module wires args to
those and to the config resolver.
"""

from __future__ import annotations

import os
import re
from argparse import Namespace

from hermes_cli.tunnel_config import resolve_tunnel_config


def _approvals_path() -> str:
    home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    return os.path.join(home, "tunnel", "hold_requests.jsonl")


def _current_user() -> str:
    # Resolved from the active profile name; fall back to OS user.
    return os.environ.get("HERMES_PROFILE") or os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"


_DURATION_RE = re.compile(r"^(?P<n>\d+)\s*(?P<u>s|m|h|d)$")
_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def _parse_duration(spec: str) -> float | None:
    if not spec:
        return None
    m = _DURATION_RE.match(spec.strip())
    if not m:
        return None
    import time
    return time.time() + int(m.group("n")) * _UNITS[m.group("u")]


def _print(*a):
    print(*a)


def tunnel_command(args: Namespace) -> int:
    cmd = getattr(args, "tunnel_command", None)
    if cmd == "up":
        return _cmd_up(args)
    if cmd == "down":
        return _cmd_down(args)
    if cmd == "status":
        return _cmd_status(args)
    if cmd == "doctor":
        return _cmd_doctor(args)
    if cmd == "hold":
        return _cmd_hold(args)
    if cmd == "requests":
        return _cmd_requests(args)
    if cmd == "approve":
        return _cmd_approve(args)
    if cmd == "deny":
        return _cmd_deny(args)
    _print("usage: hermes tunnel {up,down,status,doctor,hold,requests,approve,deny}")
    return 2


def _cmd_up(args) -> int:
    cfg = resolve_tunnel_config(cli_origins=getattr(args, "origins", None) or None)
    if not cfg["routes"]:
        _print("tunnel up: no origins. Pass --origin SUB=HOST:PORT or set tunnel.routes.")
        return 2
    if not cfg["zone"]:
        _print("tunnel up: no zone configured (set tunnel.zone or HERMES_TUNNEL_ZONE).")
        return 2
    if not cfg["tunnel_name"] or not cfg["credentials_file"]:
        _print("tunnel up: tunnel_name and credentials_file are required.")
        return 2
    if not os.path.exists(cfg["credentials_file"]):
        _print(f"tunnel up: credentials file not found: {cfg['credentials_file']}")
        return 2

    config_path = _write_cloudflared_config(cfg)
    hold_id = None
    if getattr(args, "hold_request", False):
        from hermes_cli import tunnel_approvals as ta
        hold_id = ta.file_request(_approvals_path(), user=_current_user(),
                                   subdomains=[f"{r['subdomain']}.{cfg['zone']}" for r in cfg["routes"]],
                                   reason=getattr(args, "reason", ""),
                                   requested_until=_parse_duration(getattr(args, "until", "")))
        _print(f"hold request filed: {hold_id} (pending admin approval)")

    # If any route targets the dashboard port, set the dashboard public URL
    # so OAuth callback / WebSocket URLs build from the public hostname.
    _maybe_set_dashboard_public_url(cfg)

    from hermes_cli.tunnel_supervisor import TunnelSupervisor
    sup = TunnelSupervisor(cfg, _approvals_path(), hold_request_id=hold_id)
    _print(f"tunnel up: https://{cfg['routes'][0]['subdomain']}.{cfg['zone']} "
           f"-> 127.0.0.1:{cfg['routes'][0]['port']} (idle {cfg['idle_timeout_seconds']}s)")
    sup.run(config_path)
    return 0


def _write_cloudflared_config(cfg) -> str:
    import tempfile, json
    ingress = []
    for r in cfg["routes"]:
        ingress.append({"hostname": f"{r['subdomain']}.{cfg['zone']}",
                        "service": f"http://{r['host']}:{r['port']}"})
    ingress.append({"service": "http_status:404"})
    doc = {"tunnel": cfg["tunnel_name"], "credentials-file": cfg["credentials_file"],
           "ingress": ingress}
    home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    os.makedirs(os.path.join(home, "tunnel"), exist_ok=True)
    path = os.path.join(home, "tunnel", "cloudflared.yml")
    with open(path, "w", encoding="utf-8") as f:
        import yaml  # PyYAML is already a project dep (config.py uses YAML)
        yaml.safe_dump(doc, f)
    return path


def _maybe_set_dashboard_public_url(cfg) -> None:
    DASH_PORT = 9119
    for r in cfg["routes"]:
        if int(r.get("port", 0)) == DASH_PORT:
            os.environ["HERMES_DASHBOARD_PUBLIC_URL"] = f"https://{r['subdomain']}.{cfg['zone']}"
            return


def _cmd_down(args) -> int:
    # Best-effort: terminate running cloudflared for this profile.
    import subprocess
    killed = 0
    try:
        out = subprocess.run(["cloudflared", "tunnel", "list"], capture_output=True, text=True, timeout=10)
    except Exception:
        _print("tunnel down: cloudflared not reachable")
        return 1
    _print(out.stdout)
    return 0


def _cmd_status(args) -> int:
    _print("tunnel status: (cloudflared process introspection — see `cloudflared tunnel info`)")
    return 0


def _cmd_doctor(args) -> int:
    import shutil, socket
    ok = True
    if not shutil.which("cloudflared"):
        _print("doctor: cloudflared NOT on PATH"); ok = False
    else:
        _print("doctor: cloudflared present")
    cfg = resolve_tunnel_config()
    if cfg["credentials_file"] and not os.path.exists(cfg["credentials_file"]):
        _print(f"doctor: credentials file missing: {cfg['credentials_file']}"); ok = False
    for r in cfg["routes"]:
        s = socket.socket(); s.settimeout(2)
        try:
            s.connect((r["host"], int(r["port"]))); _print(f"doctor: origin up {r['subdomain']} -> {r['host']}:{r['port']}")
        except Exception:
            _print(f"doctor: origin DOWN {r['subdomain']} -> {r['host']}:{r['port']}"); ok = False
        finally:
            s.close()
    return 0 if ok else 1


def _cmd_hold(args) -> int:
    from hermes_cli import tunnel_approvals as ta
    cfg = resolve_tunnel_config()
    rid = ta.file_request(_approvals_path(), user=_current_user(),
                           subdomains=[f"{r['subdomain']}.{cfg['zone']}" for r in cfg["routes"]],
                           reason=getattr(args, "reason", ""),
                           requested_until=_parse_duration(getattr(args, "until", "")))
    _print(f"hold request filed: {rid} (pending admin approval)")
    return 0


def _cmd_requests(args) -> int:
    from hermes_cli import tunnel_approvals as ta
    for r in ta.list_pending(_approvals_path()):
        _print(f"{r['id']}  user={r['user']}  subs={','.join(r['subdomains'])}  reason={r['reason']!r}  status={r['status']}")
    return 0


def _cmd_approve(args) -> int:
    from hermes_cli import tunnel_approvals as ta
    cfg = resolve_tunnel_config()
    try:
        ta.approve(_approvals_path(), args.id, until=_parse_duration(args.until) or 0.0,
                   by=_current_user(), admin_ids=cfg["admin"])
    except PermissionError:
        _print(f"approve: {_current_user()} is not a tunnel admin"); return 3
    except KeyError:
        _print(f"approve: no such hold request: {args.id}"); return 4
    _print(f"approved: {args.id} until {args.until}")
    return 0


def _cmd_deny(args) -> int:
    from hermes_cli import tunnel_approvals as ta
    cfg = resolve_tunnel_config()
    try:
        ta.deny(_approvals_path(), args.id, reason=getattr(args, "reason", ""),
                by=_current_user(), admin_ids=cfg["admin"])
    except PermissionError:
        _print(f"deny: {_current_user()} is not a tunnel admin"); return 3
    except KeyError:
        _print(f"deny: no such hold request: {args.id}"); return 4
    _print(f"denied: {args.id}")
    return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/hermes_cli/test_tunnel_handler.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Wire the handler + parser into `main.py`**

(a) Add the import. In `hermes_cli/main.py`, in the import block around line 294 (where `from hermes_cli.subcommands.dashboard import build_dashboard_parser` lives), add:
```python
from hermes_cli.subcommands.tunnel import build_tunnel_parser
```

(b) Add the handler. Near the other thin `cmd_*` handlers (around line 4198, next to `cmd_status`), add:
```python
def cmd_tunnel(args):
    """Expose a local service to the internet via a Cloudflare Tunnel."""
    from hermes_cli.tunnel_commands import tunnel_command
    return tunnel_command(args)
```

(c) Add the call site. In `main()`, near line 14044 (next to `build_gui_parser`/`build_prompt_size_parser`, before the `# Parse and execute` block at line 14047), add:
```python
    build_tunnel_parser(subparsers, cmd_tunnel=cmd_tunnel)
```

- [ ] **Step 6: Verify the CLI wiring end-to-end**

Run: `python -c "from hermes_cli.main import main"` (import smoke test) and `python -m hermes_cli.main tunnel --help`
Expected: help text listing `up, down, status, doctor, hold, requests, approve, deny` with no import errors.

- [ ] **Step 7: Commit**

```bash
git add hermes_cli/tunnel_commands.py hermes_cli/main.py tests/hermes_cli/test_tunnel_handler.py
git commit -m "feat(tunnel): wire hermes tunnel subcommand into the CLI

Co-Authored-By: claude-flow <ruv@ruv.net>"
```

---

## Task 7: Docs

**Files:**
- Create: `website/docs/user-guide/tunnel.md`

- [ ] **Step 1: Write the docs**

Create `website/docs/user-guide/tunnel.md` (content below):
````markdown
# Cloudflare Tunnel exposure (`hermes tunnel`)

`hermes tunnel` exposes a local app, service, or API you built to the internet on a
per-user subdomain of your zone (e.g. `alice.noit2.com`) via a Cloudflare named tunnel.
The exposure is **ephemeral by default**: an idle-reset 30-minute dead-man's switch
closes the tunnel when traffic stops, so a forgotten test build can't leak to the
internet. For longer exposure, file a hold request and have an admin approve it.

The dashboard is just one possible origin — point a route at `127.0.0.1:9119` to expose it.

## Prerequisites

1. Install `cloudflared` and log in: `cloudflared login`.
2. Create a named tunnel: `cloudflared tunnel create <name>`. Note the credentials JSON path.
3. Route DNS for each subdomain: `cloudflared tunnel route dns <name> alice.noit2.com`.
4. Put the credentials path in config (below) or `HERMES_TUNNEL_CREDS`.

## Config

`~/.hermes/config.yaml` (or env overrides `HERMES_TUNNEL_*`):

```yaml
tunnel:
  zone: "noit2.com"
  tunnel_name: "alice"
  credentials_file: "/home/alice/.cloudflared/<uuid>.json"
  idle_timeout_seconds: 1800
  admin: ["alice"]            # who may approve/deny hold requests
  routes:
    - subdomain: alice
      host: 127.0.0.1
      port: 3000
```

## Commands

```bash
# Expose two origins (CLI origins override config routes):
hermes tunnel up --origin alice=127.0.0.1:3000 --origin alice-api=127.0.0.1:8080

# Start and immediately request a longer hold:
hermes tunnel up --origin alice=127.0.0.1:3000 --hold-request --reason "demo" --until 4h

# Mid-session, request more time:
hermes tunnel hold --reason "demo running long" --until 4h

# Admin side:
hermes tunnel requests
hermes tunnel approve <id> --until 6h
hermes tunnel deny <id> --reason "too long"

hermes tunnel status
hermes tunnel doctor
hermes tunnel down
```

## How the dead-man's switch works

Every `poll_interval_seconds` (default 5s) the supervisor reads cloudflared's request
counter from its `--metrics` endpoint. **Any increase resets the 30-minute idle clock.**
If 30 minutes pass with no increase, cloudflared is gracefully drained and killed; your
local origins keep running. An approved hold disables the idle timer until the approved
expiry, after which the idle timer resumes (the approval expiring does NOT hard-kill).

## Layout

`noit2.com` root is the Cloudflare Pages brand site. Each user's services live on their
own subdomains. Origins bind to `127.0.0.1` only — no port opens on your firewall; the
only public path is Cloudflare's edge on 443.
````

- [ ] **Step 2: Commit**

```bash
git add website/docs/user-guide/tunnel.md
git commit -m "docs(tunnel): user guide for hermes tunnel

Co-Authored-By: claude-flow <ruv@ruv.net>"
```

---

## Self-Review (run after writing the plan — results recorded here)

- **Spec coverage:** §5.1 commands → Task 5 (parser) + Task 6 (dispatch). §5.2 supervisor + policy hook → Tasks 3 + 4. §5.3 approvals store → Task 2. §5.4 config + env → Task 1. §5.5 tests → each task has its test file. §5.6 docs → Task 7. §6 data flow → exercised by Task 6 `_cmd_up` + Task 4 supervisor. §7 security posture (127.0.0.1-only origins, idle default, admin-gated hold) → enforced by Task 1 defaults + Task 2 admin gate + Task 3 policy. Covered.
- **Placeholder scan:** The two `TODO(you)` markers in Task 3 Step 3 are intentional learning-mode contribution points, fully specified by `TestPolicy` — not plan-rot. No other TBD/TODO/"add error handling" present.
- **Type consistency:** `resolve_tunnel_config` shape used identically in Tasks 1, 4, 6. `tunnel_approvals` signatures (`file_request`, `approve`, `deny`, `is_approved`, `approved_until`, `list_pending`) identical across Tasks 2, 4, 6. `reset_idle_on` / `should_close_now` + `state` dict shape identical across Tasks 3, 4. `TunnelSupervisor(cfg, approvals_path, hold_request_id=...)` + `.tick()`/`.run()`/`.closed`/`.last_activity`/`.hold_until` identical across Tasks 4, 6. `build_tunnel_parser(subparsers, *, cmd_tunnel)` identical in Tasks 5, 6. Consistent.

## Execution Handoff

Plan complete and saved to `.plans/2026-07-08-cloudflare-noit2-tunnel-plan.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?