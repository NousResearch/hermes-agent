#!/usr/bin/env python3
"""Portable backend/DB cycle verifier (mw-backend-check).

Prove that a migrated site's BACKEND cycles work end-to-end against a real
API + DB — never trust the UI alone.  Config-driven and portable.

DB access is via a configurable QUERY COMMAND (argv template), so the tool
stays stdlib-only and DB-driver-free.  Every check FAILS CLOSED.

  exit 0  — all checks pass
  exit 1  — one or more checks fail
  exit 2  — usage / config error

stdlib-only core · optional PyYAML · self-contained mini-YAML fallback ·
Python 3.9+

Task: MW-P3-I2e
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_ERR = 2

DEFAULT_CONFIG_REL = Path(".work") / "backend-check.yaml"
DEFAULT_TIMEOUT = 30

KNOWN_TYPES = frozenset(
    {
        "query_rows",
        "form_cycle",
        "siteid_isolation",
        "dashboard_parity",
        "es_search",
        "data_parity",
    }
)

_TOKEN_REDACT = "***REDACTED***"
_FORCE_MINI_YAML = False

try:
    import yaml as _yaml  # type: ignore
except ImportError:  # pragma: no cover
    _yaml = None


# ---------------------------------------------------------------------------
# minimal YAML subset loader (self-contained; mirror page_check / menu_gate)
# ---------------------------------------------------------------------------


class MiniYamlError(ValueError):
    """Raised when the mini YAML loader hits an ambiguous construct."""


def _split_flow_items(inner: str) -> List[str]:
    """Split comma-separated items inside ``[]`` or ``{}``, respecting quotes."""
    parts: List[str] = []
    buf = ""
    in_q: Optional[str] = None
    depth = 0
    for ch in inner:
        if in_q:
            buf += ch
            if ch == in_q:
                in_q = None
            continue
        if ch in "\"'":
            in_q = ch
            buf += ch
            continue
        if ch in "[{":
            depth += 1
            buf += ch
            continue
        if ch in "]}":
            depth = max(0, depth - 1)
            buf += ch
            continue
        if ch == "," and depth == 0:
            parts.append(buf.strip())
            buf = ""
            continue
        buf += ch
    if buf.strip():
        parts.append(buf.strip())
    return parts


def _scalar(v: str) -> Any:
    """Parse a YAML scalar: quoted string, bool, null, int/float, flow list/map, or raw str."""
    v = v.strip()
    if not v:
        return None
    if len(v) >= 2 and v[0] in "\"'" and v[-1] == v[0]:
        return v[1:-1]
    if " #" in v:
        v = v.split(" #", 1)[0].strip()
    if v.startswith("{") and v.endswith("}"):
        inner = v[1:-1].strip()
        if not inner:
            return {}
        out: Dict[str, Any] = {}
        for part in _split_flow_items(inner):
            if ":" not in part:
                raise MiniYamlError(f"unrepresentable flow mapping entry: {part!r}")
            k, _, raw = part.partition(":")
            out[k.strip()] = _scalar(raw)
        return out
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if not inner:
            return []
        return [_scalar(p) for p in _split_flow_items(inner)]
    low = v.lower()
    if low in ("null", "~"):
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def _mini_yaml(text: str) -> Any:
    """Minimal YAML-subset parser for backend-check config.

    Handles nested mappings by 2-space indent, lists of mappings, quoted/
    unquoted scalars, booleans, null, inline ``[a, b]`` / ``{k: v}``, ``#``
    comments.  Fail closed on unrepresentable constructs (MiniYamlError).
    """
    lines = [
        l
        for l in (raw.rstrip() for raw in text.splitlines())
        if l.strip() and not l.strip().startswith("#")
    ]
    pos = 0

    def indent_of(l: str) -> int:
        return len(l) - len(l.lstrip(" "))

    def _is_list_item(s: str) -> bool:
        return s == "-" or s.startswith("- ")

    def _list_item_body(s: str) -> str:
        if s == "-":
            return ""
        return s[2:].strip()

    def _parse_mapping_continuation(d: Dict[str, Any], base_indent: int) -> None:
        nonlocal pos
        while pos < len(lines):
            nline = lines[pos]
            ni = indent_of(nline)
            if ni <= base_indent:
                break
            ns = nline.strip()
            if _is_list_item(ns):
                break
            if ":" not in ns:
                raise MiniYamlError(
                    f"unrepresentable YAML at line content {ns!r}: "
                    "expected key: value under list item"
                )
            nk, _, nv = ns.partition(":")
            pos += 1
            if nv.strip():
                d[nk.strip()] = _scalar(nv)
            elif pos < len(lines) and indent_of(lines[pos]) > ni:
                d[nk.strip()] = parse_block(indent_of(lines[pos]))
            else:
                d[nk.strip()] = None

    def parse_block(indent: int) -> Any:
        nonlocal pos
        result: Any = None
        while pos < len(lines):
            line = lines[pos]
            cur = indent_of(line)
            if cur < indent:
                break
            if cur > indent:
                raise MiniYamlError(
                    f"orphan deeper indent without a parent key near: {line.strip()!r}"
                )
            s = line.strip()
            if _is_list_item(s):
                if result is None:
                    result = []
                if not isinstance(result, list):
                    raise MiniYamlError(f"mixed mapping/list structure near: {s!r}")
                item = _list_item_body(s)
                pos += 1
                if item == "":
                    if pos < len(lines) and indent_of(lines[pos]) > indent:
                        nested_indent = indent_of(lines[pos])
                        nested_s = lines[pos].strip()
                        if not _is_list_item(nested_s) and ":" in nested_s:
                            d: Dict[str, Any] = {}
                            _parse_mapping_continuation(d, indent)
                            result.append(d)
                        else:
                            result.append(parse_block(nested_indent))
                    else:
                        result.append(None)
                elif ":" in item and item[0] not in "\"'":
                    k, _, v = item.partition(":")
                    d = {k.strip(): _scalar(v) if v.strip() else None}
                    _parse_mapping_continuation(d, indent)
                    if d and list(d.values()) == [None]:
                        only_k = next(iter(d))
                        if pos < len(lines) and indent_of(lines[pos]) > indent:
                            d[only_k] = parse_block(indent_of(lines[pos]))
                    result.append(d)
                else:
                    result.append(_scalar(item))
            else:
                if result is None:
                    result = {}
                if not isinstance(result, dict):
                    raise MiniYamlError(f"mixed list/mapping structure near: {s!r}")
                if ":" not in s:
                    raise MiniYamlError(
                        f"unrepresentable YAML near: {s!r} (expected key: value)"
                    )
                k, _, v = s.partition(":")
                pos += 1
                if v.strip():
                    result[k.strip()] = _scalar(v)
                elif pos < len(lines) and indent_of(lines[pos]) > cur:
                    result[k.strip()] = parse_block(indent_of(lines[pos]))
                else:
                    result[k.strip()] = None
        return result

    return parse_block(0) if lines else {}


def load_yaml_text(text: str, force_mini: bool = False) -> Any:
    """Load YAML text: prefer PyYAML unless forced to mini loader."""
    use_mini = force_mini or _FORCE_MINI_YAML or _yaml is None
    if not use_mini:
        data = _yaml.safe_load(text)
        return data if data is not None else {}
    return _mini_yaml(text) or {}


def load_yaml_file(path: Path, force_mini: bool = False) -> Any:
    text = path.read_text(encoding="utf-8")
    return load_yaml_text(text, force_mini=force_mini)


# ---------------------------------------------------------------------------
# redact / path helpers
# ---------------------------------------------------------------------------


def redact_secrets(text: str, secrets: Sequence[str]) -> str:
    """Replace any known secret substrings in text (fail-safe for output)."""
    out = text
    for s in secrets:
        if s and s in out:
            out = out.replace(s, _TOKEN_REDACT)
    return out


def _as_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v)


def apply_placeholders(value: str, mapping: Dict[str, Any]) -> str:
    """Replace ``{key}`` placeholders. Unknown keys left as-is."""
    if value is None:
        return ""
    s = str(value)

    def _sub(m: re.Match) -> str:
        key = m.group(1)
        if key in mapping and mapping[key] is not None:
            return _as_str(mapping[key])
        return m.group(0)

    return re.sub(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", _sub, s)


def ascii_safe_url(url: str) -> str:
    """Percent-encode path/query (and IDNA host) so urllib gets an ASCII-only URL.

    Placeholder values (Thai keywords, ids, site_id) may be substituted raw into
    path templates or query strings.  urllib.request requires the URL to be
    ASCII-safe; non-ASCII must be percent-encoded (query) or IDNA (host).
    """
    parts = urlsplit(url)
    # Path segments: keep structural reserved chars, encode everything else
    # (including non-ASCII such as Thai).
    path = quote(parts.path, safe="/%!$&'()*+,;=:@")
    if parts.query:
        # parse_qsl accepts unencoded Unicode values; urlencode re-encodes
        # with quote_plus so spaces become '+' and Thai becomes %XX.
        pairs = parse_qsl(parts.query, keep_blank_values=True)
        query = urlencode(pairs, doseq=True)
    else:
        query = ""
    fragment = quote(parts.fragment, safe="!$&'()*+,;=:@/?") if parts.fragment else ""

    netloc = parts.netloc
    if netloc:
        userinfo = ""
        hostport = netloc
        if "@" in netloc:
            userinfo, hostport = netloc.rsplit("@", 1)
            userinfo = userinfo + "@"
        if not hostport.startswith("["):
            host, sep, port = hostport.partition(":")
            if sep and port.isdigit():
                try:
                    host = host.encode("idna").decode("ascii")
                except (UnicodeError, UnicodeDecodeError):
                    pass
                hostport = f"{host}:{port}"
            else:
                try:
                    hostport = hostport.encode("idna").decode("ascii")
                except (UnicodeError, UnicodeDecodeError):
                    pass
        netloc = userinfo + hostport

    return urlunsplit((parts.scheme, netloc, path, query, fragment))


def deep_template(obj: Any, mapping: Dict[str, Any]) -> Any:
    """Recursively apply placeholders to strings in nested structures."""
    if isinstance(obj, str):
        return apply_placeholders(obj, mapping)
    if isinstance(obj, list):
        return [deep_template(x, mapping) for x in obj]
    if isinstance(obj, dict):
        return {k: deep_template(v, mapping) for k, v in obj.items()}
    return obj


def json_path_get(data: Any, path: str) -> Any:
    """Resolve a dotted path like ``data.id`` or ``totals.users``."""
    if not path:
        return data
    cur = data
    for part in str(path).split("."):
        if cur is None:
            return None
        if isinstance(cur, dict):
            if part not in cur:
                return None
            cur = cur[part]
        elif isinstance(cur, list):
            try:
                idx = int(part)
            except ValueError:
                return None
            if idx < 0 or idx >= len(cur):
                return None
            cur = cur[idx]
        else:
            return None
    return cur


def _to_int(v: Any) -> Optional[int]:
    """Coerce to int for count parity. Booleans are NOT numbers (fail closed)."""
    if v is None:
        return None
    # bool is a subclass of int — must reject before isinstance(..., int)
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v == int(v):
        return int(v)
    try:
        s = str(v).strip()
        # reject bool-like strings? keep numeric strings only
        return int(s)
    except (TypeError, ValueError):
        return None


def _values_equal(a: Any, b: Any) -> bool:
    """Loose equality for field compare (string/int/float). Booleans never equal ints."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    # Never treat True/False as 1/0
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b if (isinstance(a, bool) and isinstance(b, bool)) else False
    ia, ib = _to_int(a), _to_int(b)
    if ia is not None and ib is not None and str(a).strip() == str(ia) and str(b).strip() == str(ib):
        return ia == ib
    return str(a) == str(b)


# SQL metacharacters that must not appear in a string created-id substituted into SQL
_ID_SQL_META = ("'", ";", "--", "/*")


def validate_created_id(created_id: Any) -> Tuple[bool, str, str]:
    """Validate form_cycle created id is a safe scalar.

    Returns (ok, id_string, reason). Accepted: int > 0, or non-empty
    non-whitespace string (not bool) without SQL metacharacters.
    """
    if isinstance(created_id, bool):
        return False, "", "invalid created id"
    if isinstance(created_id, int):
        if created_id <= 0:
            return False, "", "invalid created id"
        return True, str(created_id), ""
    if isinstance(created_id, str):
        if not created_id.strip():
            return False, "", "invalid created id"
        for meta in _ID_SQL_META:
            if meta in created_id:
                return False, "", "invalid created id (SQL metacharacters)"
        return True, created_id, ""
    # None, list, dict, float, etc.
    return False, "", "invalid created id"


# ---------------------------------------------------------------------------
# result types
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    id: str
    type: str
    status: str  # "PASS" | "FAIL"
    reason: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "PASS"


@dataclass
class RunReport:
    site_id: str
    checks: List[CheckResult] = field(default_factory=list)
    configured_total: int = 0  # checks present in config (before --only filter)

    @property
    def pass_count(self) -> int:
        return sum(1 for c in self.checks if c.ok)

    @property
    def fail_count(self) -> int:
        return sum(1 for c in self.checks if not c.ok)

    @property
    def count(self) -> int:
        return len(self.checks)

    @property
    def ran(self) -> int:
        return self.count

    @property
    def partial(self) -> bool:
        """True when fewer checks ran than configured (e.g. --only subset)."""
        total = self.configured_total if self.configured_total > 0 else self.count
        return self.count < total

    @property
    def failed_ids(self) -> List[str]:
        return [c.id for c in self.checks if not c.ok]

    @property
    def all_ran_passed(self) -> bool:
        """Every check that actually ran passed (partial subset OK for exit 0)."""
        return self.count > 0 and self.fail_count == 0

    @property
    def healthy(self) -> bool:
        """Full backend healthy: every configured check ran and passed (not partial)."""
        return self.all_ran_passed and not self.partial

    def to_dict(self) -> Dict[str, Any]:
        configured = self.configured_total if self.configured_total > 0 else self.count
        out: Dict[str, Any] = {
            "site_id": self.site_id,
            "checks": [
                {
                    "id": c.id,
                    "type": c.type,
                    "status": c.status,
                    "reason": c.reason,
                    "detail": c.detail,
                }
                for c in self.checks
            ],
            "total": {
                "pass": self.pass_count,
                "fail": self.fail_count,
                "count": self.count,
            },
            "failed": self.failed_ids,
            "healthy": self.healthy,
            "partial": self.partial,
            "ran": self.ran,
            "configured_total": configured,
        }
        return out


# ---------------------------------------------------------------------------
# HTTP + DB runners
# ---------------------------------------------------------------------------


class BackendError(Exception):
    """Non-config runtime failure for a single check link (fail closed)."""

    def __init__(self, link: str, message: str, detail: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.link = link
        self.message = message
        self.detail = detail or {}


class ConfigError(Exception):
    """Usage / config / schema error → exit 2."""


class HttpClient:
    """urllib HTTP client with auth header from ENV (token never stored in logs)."""

    def __init__(
        self,
        api_base: str,
        auth: Optional[Dict[str, Any]] = None,
        timeout: float = DEFAULT_TIMEOUT,
        secrets: Optional[List[str]] = None,
    ):
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.secrets: List[str] = list(secrets or [])
        self._auth_header: Optional[str] = None
        self._auth_value: Optional[str] = None
        if auth:
            header = auth.get("header") or "Authorization"
            env_key = auth.get("value_env")
            if env_key:
                token = os.environ.get(str(env_key), "")
                if not token:
                    # auth ถูกตั้งค่าไว้ แต่ไม่มี token ใน ENV → ห้ามยิงแบบไม่ยืนยันตัวตน
                    # (endpoint สาธารณะอาจตอบผ่านหลอกว่า healthy) — fail closed
                    raise ConfigError(
                        f"auth token env {env_key!r} not set — refusing to run unauthenticated"
                    )
                self._auth_header = str(header)
                self._auth_value = token
                if token not in self.secrets:
                    self.secrets.append(token)

    def request(
        self,
        method: str,
        path: str,
        body: Any = None,
        placeholders: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, str, Optional[Any]]:
        """Perform HTTP request. Returns (status, body_text, parsed_json_or_None).

        Non-2xx raises BackendError. Connection/timeout raises BackendError.
        """
        ph = placeholders or {}
        path_t = apply_placeholders(path, ph)
        if not path_t.startswith("/"):
            path_t = "/" + path_t
        # Encode non-ASCII path/query values (e.g. Thai {q}) before urllib.
        url = ascii_safe_url(self.api_base + path_t)
        method_u = (method or "GET").upper()
        data_bytes: Optional[bytes] = None
        headers: Dict[str, str] = {"Accept": "application/json"}
        if self._auth_header and self._auth_value:
            headers[self._auth_header] = self._auth_value
        if body is not None and method_u in ("POST", "PUT", "PATCH"):
            body_t = deep_template(body, ph)
            # Preserve Thai/Unicode in JSON body; send as UTF-8 bytes.
            data_bytes = json.dumps(body_t, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json; charset=utf-8"
        req = urllib.request.Request(url, data=data_bytes, headers=headers, method=method_u)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                status = getattr(resp, "status", None) or resp.getcode()
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise BackendError(
                "http",
                f"HTTP {e.code} for {method_u} {path_t}",
                {"status": e.code, "path": path_t, "body_snippet": err_body[:300]},
            )
        except urllib.error.URLError as e:
            reason = redact_secrets(str(getattr(e, "reason", e)), self.secrets)
            raise BackendError(
                "http",
                f"connection error for {method_u} {path_t}: {reason}",
                {"path": path_t},
            )
        except TimeoutError:
            raise BackendError(
                "http",
                f"timeout for {method_u} {path_t}",
                {"path": path_t, "timeout": self.timeout},
            )
        except OSError as e:
            raise BackendError(
                "http",
                f"OS error for {method_u} {path_t}: {e}",
                {"path": path_t},
            )

        if not (200 <= int(status) < 300):
            raise BackendError(
                "http",
                f"HTTP {status} for {method_u} {path_t}",
                {"status": status, "path": path_t, "body_snippet": raw[:300]},
            )
        parsed: Optional[Any] = None
        if raw.strip():
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None
        return int(status), raw, parsed


class QueryRunner:
    """Run configurable query_cmd argv (shell=False); parse stdout as JSON rows."""

    def __init__(
        self,
        query_cmd: Sequence[str],
        db_path: str,
        timeout: float = DEFAULT_TIMEOUT,
        secrets: Optional[List[str]] = None,
    ):
        if not query_cmd:
            raise ConfigError("query_cmd is required (argv list)")
        self.query_cmd = list(query_cmd)
        self.db_path = db_path
        self.timeout = timeout
        self.secrets = list(secrets or [])

    def run(self, sql: str, placeholders: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        ph = dict(placeholders or {})
        ph.setdefault("db", self.db_path)
        sql_t = apply_placeholders(sql, ph)
        argv = [apply_placeholders(str(a), ph) for a in self.query_cmd]
        # Append SQL as final arg (sqlite3 -json db "SQL" / psql style)
        argv.append(sql_t)
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                shell=False,
                check=False,
            )
        except FileNotFoundError as e:
            raise BackendError(
                "query",
                f"query_cmd binary not found: {argv[0]!r}",
                {"argv0": argv[0], "error": str(e)},
            )
        except subprocess.TimeoutExpired:
            raise BackendError(
                "query",
                f"query_cmd timed out after {self.timeout}s",
                {"sql_snippet": sql_t[:200]},
            )
        if proc.returncode != 0:
            err = redact_secrets((proc.stderr or proc.stdout or "").strip(), self.secrets)
            raise BackendError(
                "query",
                f"query_cmd exit {proc.returncode}: {err[:300]}",
                {"exit": proc.returncode, "sql_snippet": sql_t[:200]},
            )
        out = (proc.stdout or "").strip()
        if not out:
            return []
        try:
            data = json.loads(out)
        except json.JSONDecodeError as e:
            raise BackendError(
                "query",
                f"query stdout not JSON: {e}",
                {"stdout_snippet": out[:200]},
            )
        if isinstance(data, list):
            rows: List[Dict[str, Any]] = []
            for item in data:
                if isinstance(item, dict):
                    rows.append(item)
                else:
                    rows.append({"value": item})
            return rows
        if isinstance(data, dict):
            # some drivers wrap rows
            if "rows" in data and isinstance(data["rows"], list):
                return [r for r in data["rows"] if isinstance(r, dict)]
            return [data]
        raise BackendError(
            "query",
            f"unexpected JSON type from query: {type(data).__name__}",
            {},
        )


# ---------------------------------------------------------------------------
# seeded sampling (deterministic, hashlib — no randomness)
# ---------------------------------------------------------------------------


def sample_keys(keys: Sequence[Any], sample_pct: float, min_rows: int = 20) -> List[Any]:
    """Pick ceil(max(min_rows, pct% of n)) keys ordered by sha256(str(key)).

    Same keys every run for the same key set and parameters.
    """
    uniq = list(dict.fromkeys(keys))  # preserve first-seen, unique
    n = len(uniq)
    if n == 0:
        return []
    pct = max(0.0, float(sample_pct))
    want = int(math.ceil(max(min_rows, n * pct / 100.0)))
    want = min(want, n)
    ranked = sorted(uniq, key=lambda k: hashlib.sha256(str(k).encode("utf-8")).hexdigest())
    return ranked[:want]


# ---------------------------------------------------------------------------
# check runners
# ---------------------------------------------------------------------------


def _pass(cid: str, ctype: str, reason: str = "", detail: Optional[Dict] = None) -> CheckResult:
    return CheckResult(id=cid, type=ctype, status="PASS", reason=reason, detail=detail or {})


def _fail(cid: str, ctype: str, reason: str, detail: Optional[Dict] = None) -> CheckResult:
    return CheckResult(id=cid, type=ctype, status="FAIL", reason=reason, detail=detail or {})


def check_query_rows(
    check: Dict[str, Any],
    *,
    site_id: str,
    query: QueryRunner,
) -> CheckResult:
    cid = str(check.get("id") or "query_rows")
    ctype = "query_rows"
    sql = check.get("sql")
    if not sql:
        return _fail(cid, ctype, "missing sql")
    expect_min = int(check.get("expect_min_rows", 1))
    try:
        rows = query.run(str(sql), {"site_id": site_id})
    except BackendError as e:
        return _fail(cid, ctype, f"query failed: {e.message}", {"link": e.link, **e.detail})
    n = len(rows)
    if n >= expect_min:
        return _pass(cid, ctype, f"rows={n} >= {expect_min}", {"rows": n})
    return _fail(
        cid,
        ctype,
        f"rows={n} < expect_min_rows={expect_min}",
        {"rows": n, "expect_min_rows": expect_min},
    )


def _admin_has_assertion(check: Dict[str, Any]) -> bool:
    """True if form_cycle has at least one non-empty admin assertion."""
    admin_expect = check.get("admin_expect_json")
    admin_contains = check.get("admin_contains")
    if isinstance(admin_expect, dict) and len(admin_expect) > 0:
        return True
    if admin_contains is not None and str(admin_contains).strip() != "":
        return True
    return False


def check_form_cycle(
    check: Dict[str, Any],
    *,
    site_id: str,
    http: HttpClient,
    query: QueryRunner,
) -> CheckResult:
    cid = str(check.get("id") or "form_cycle")
    ctype = "form_cycle"
    submit = check.get("submit") or {}
    id_path = check.get("id_json_path") or "data.id"
    verify_sql = check.get("verify_sql")
    verify_expect = check.get("verify_expect") or {}
    # admin key present vs omitted: omitted → skip admin link; present → must assert
    admin_key_present = "admin" in check
    admin = check.get("admin") if admin_key_present else None
    admin_expect = check.get("admin_expect_json")
    admin_contains = check.get("admin_contains")

    if not submit.get("path"):
        return _fail(cid, ctype, "missing submit.path")
    if not verify_sql:
        return _fail(cid, ctype, "missing verify_sql")

    # FIX 2b: verify_sql must be site-scoped (include {site_id}) — config error
    verify_sql_s = str(verify_sql)
    if "{site_id}" not in verify_sql_s:
        raise ConfigError(
            "form_cycle.verify_sql must be site-scoped (include {site_id})"
        )

    # FIX 3 + C: admin block ⇒ config-time validation (path + assertion) → exit 2
    if admin_key_present:
        if not isinstance(admin, dict):
            raise ConfigError(
                "form_cycle.admin must be a mapping with path + assertion"
            )
        if not admin.get("path"):
            raise ConfigError("form_cycle.admin needs a path")
        if not _admin_has_assertion(check):
            raise ConfigError(
                "form_cycle.admin needs admin_expect_json or admin_contains"
            )

    # --- link 1: submit ---
    try:
        _st, _raw, parsed = http.request(
            submit.get("method") or "POST",
            str(submit["path"]),
            body=submit.get("body"),
            placeholders={"site_id": site_id},
        )
    except BackendError as e:
        return _fail(
            cid,
            ctype,
            f"submit failed: {e.message}",
            {"link": "submit", **e.detail},
        )
    if parsed is None:
        return _fail(cid, ctype, "submit response not JSON", {"link": "submit"})
    created_id = json_path_get(parsed, str(id_path))
    # FIX 2a/c: scalar id validation + SQL meta guard
    id_ok, id_s, id_reason = validate_created_id(created_id)
    if not id_ok:
        return _fail(
            cid,
            ctype,
            id_reason or "invalid created id",
            {
                "link": "submit",
                "id_json_path": id_path,
                "created_id_repr": repr(created_id)[:80],
            },
        )
    ph = {"site_id": site_id, "id": id_s}

    # --- link 2: verify DB (site-scoped template already enforced) ---
    try:
        rows = query.run(verify_sql_s, ph)
    except BackendError as e:
        return _fail(
            cid,
            ctype,
            f"verify query failed: {e.message}",
            {"link": "verify", "id": id_s, **e.detail},
        )
    if not rows:
        return _fail(
            cid,
            ctype,
            f"verify: no DB row for id {id_s}",
            {"link": "verify", "id": id_s},
        )
    row = rows[0]
    if isinstance(verify_expect, dict):
        for k, expected in verify_expect.items():
            actual = row.get(k)
            if not _values_equal(actual, expected):
                return _fail(
                    cid,
                    ctype,
                    f"verify: field {k!r} mismatch (got {actual!r}, want {expected!r})",
                    {"link": "verify", "id": id_s, "field": k, "got": actual, "want": expected},
                )

    # --- link 3: admin (optional only when admin key omitted) ---
    if admin_key_present and isinstance(admin, dict):
        try:
            _st, body_text, admin_parsed = http.request(
                admin.get("method") or "GET",
                str(admin["path"]),
                placeholders=ph,
            )
        except BackendError as e:
            return _fail(
                cid,
                ctype,
                f"admin failed: {e.message}",
                {"link": "admin", "id": id_s, **e.detail},
            )

        if admin_expect is not None:
            if not isinstance(admin_expect, dict):
                return _fail(
                    cid, ctype, "admin_expect_json must be a mapping", {"link": "admin"}
                )
            if admin_parsed is None or not isinstance(admin_parsed, dict):
                return _fail(
                    cid,
                    ctype,
                    f"admin view missing expected JSON for id {id_s}",
                    {"link": "admin", "id": id_s},
                )
            for k, expected in admin_expect.items():
                actual = admin_parsed.get(k)
                if actual is None and "." in str(k):
                    actual = json_path_get(admin_parsed, str(k))
                if not _values_equal(actual, expected):
                    return _fail(
                        cid,
                        ctype,
                        f"admin: field {k!r} mismatch for id {id_s} "
                        f"(got {actual!r}, want {expected!r})",
                        {
                            "link": "admin",
                            "id": id_s,
                            "field": k,
                            "got": actual,
                            "want": expected,
                        },
                    )

        if admin_contains is not None and str(admin_contains).strip() != "":
            needle = apply_placeholders(str(admin_contains), ph)
            if needle not in body_text:
                return _fail(
                    cid,
                    ctype,
                    f"admin body missing substring {needle!r} (id {id_s})",
                    {"link": "admin", "id": id_s, "admin_contains": needle},
                )

        return _pass(
            cid,
            ctype,
            f"submit→verify→admin ok (id={id_s})",
            {"id": id_s},
        )

    return _pass(
        cid,
        ctype,
        f"submit→verify ok (id={id_s}, no admin)",
        {"id": id_s},
    )


def check_siteid_isolation(
    check: Dict[str, Any],
    *,
    site_id: str,
    query: QueryRunner,
) -> CheckResult:
    cid = str(check.get("id") or "siteid_isolation")
    ctype = "siteid_isolation"
    sql = check.get("sql")
    if not sql:
        return _fail(cid, ctype, "missing sql")
    forbid = bool(check.get("forbid_other_site_rows", True))
    prefix_tpl = check.get("asset_prefix_ok")
    prefix = apply_placeholders(str(prefix_tpl), {"site_id": site_id}) if prefix_tpl else None

    try:
        rows = query.run(str(sql), {"site_id": site_id})
    except BackendError as e:
        return _fail(cid, ctype, f"query failed: {e.message}", {"link": e.link, **e.detail})

    # FIX 1: NULL/missing site_id = leak; NULL/missing asset_path under required prefix = fail
    leaks: List[Dict[str, Any]] = []
    bad_assets: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        if forbid:
            has_sid = "site_id" in row
            rid = row.get("site_id") if has_sid else None
            if rid is None or (isinstance(rid, str) and not str(rid).strip()):
                leaks.append(
                    {
                        "row_index": idx,
                        "site_id": None,
                        "reason": "unscoped (site_id NULL/missing)",
                        "row": {k: row.get(k) for k in list(row.keys())[:8]},
                    }
                )
            elif str(rid) != str(site_id):
                leaks.append(
                    {
                        "row_index": idx,
                        "site_id": rid,
                        "reason": "other_site",
                        "row": {k: row.get(k) for k in list(row.keys())[:8]},
                    }
                )
        if prefix is not None:
            has_ap = "asset_path" in row
            ap = row.get("asset_path") if has_ap else None
            if ap is None or (isinstance(ap, str) and not str(ap).strip()):
                bad_assets.append(
                    {
                        "row_index": idx,
                        "asset_path": None,
                        "reason": "asset_path NULL/missing",
                        "row": {k: row.get(k) for k in list(row.keys())[:8]},
                    }
                )
            elif not str(ap).startswith(prefix):
                bad_assets.append(
                    {
                        "row_index": idx,
                        "asset_path": ap,
                        "reason": "wrong_prefix",
                        "row": {k: row.get(k) for k in list(row.keys())[:8]},
                    }
                )

    if leaks:
        leak_ids = sorted(
            set(
                "NULL" if x.get("site_id") is None else str(x.get("site_id"))
                for x in leaks
            )
        )
        return _fail(
            cid,
            ctype,
            f"data-leak: unscoped or other site_id present: {leak_ids}",
            {"link": "isolation", "leaks": leaks[:20], "leak_site_ids": leak_ids},
        )
    if bad_assets:
        bad_paths = [
            "NULL" if x.get("asset_path") is None else str(x.get("asset_path"))
            for x in bad_assets[:5]
        ]
        return _fail(
            cid,
            ctype,
            f"wrong-site or missing asset_path (prefix {prefix!r}): {bad_paths!r}",
            {
                "link": "asset_prefix",
                "prefix": prefix,
                "bad": bad_assets[:10],
            },
        )
    return _pass(cid, ctype, f"rows={len(rows)} isolated", {"rows": len(rows)})


def check_dashboard_parity(
    check: Dict[str, Any],
    *,
    site_id: str,
    http: HttpClient,
    query: QueryRunner,
) -> CheckResult:
    cid = str(check.get("id") or "dashboard_parity")
    ctype = "dashboard_parity"
    dash = check.get("dashboard") or {}
    points = check.get("points") or []
    if not isinstance(points, list) or len(points) < 3:
        # config schema error — raise so main can exit 2
        raise ConfigError(
            f"check {cid!r}: dashboard_parity requires >=3 points (got {len(points) if isinstance(points, list) else 0})"
        )
    if not dash.get("path"):
        return _fail(cid, ctype, "missing dashboard.path")

    try:
        _st, _raw, parsed = http.request(
            dash.get("method") or "GET",
            str(dash["path"]),
            placeholders={"site_id": site_id},
        )
    except BackendError as e:
        return _fail(cid, ctype, f"dashboard failed: {e.message}", {"link": "dashboard", **e.detail})
    if parsed is None:
        return _fail(cid, ctype, "dashboard response not JSON", {"link": "dashboard"})

    mismatches: List[Dict[str, Any]] = []
    pairs: List[Dict[str, Any]] = []
    for i, pt in enumerate(points):
        if not isinstance(pt, dict):
            raise ConfigError(f"check {cid!r}: points[{i}] must be a mapping")
        jpath = pt.get("json_path")
        sql = pt.get("sql")
        sql_field = pt.get("sql_field") or "n"
        if not jpath or not sql:
            raise ConfigError(f"check {cid!r}: points[{i}] needs json_path and sql")
        dash_val = json_path_get(parsed, str(jpath))
        dash_n = _to_int(dash_val)
        try:
            rows = query.run(str(sql), {"site_id": site_id})
        except BackendError as e:
            return _fail(
                cid,
                ctype,
                f"sql for {jpath} failed: {e.message}",
                {"link": "sql", "json_path": jpath, **e.detail},
            )
        if not rows:
            return _fail(
                cid,
                ctype,
                f"sql for {jpath} returned no rows",
                {"link": "sql", "json_path": jpath},
            )
        sql_val = rows[0].get(sql_field)
        sql_n = _to_int(sql_val)
        pair = {
            "json_path": jpath,
            "dashboard": dash_n if dash_n is not None else dash_val,
            "sql": sql_n if sql_n is not None else sql_val,
            "sql_field": sql_field,
        }
        pairs.append(pair)
        if dash_n is None or sql_n is None or dash_n != sql_n:
            mismatches.append(pair)

    if mismatches:
        first = mismatches[0]
        return _fail(
            cid,
            ctype,
            f"parity mismatch {first['json_path']}: dashboard={first['dashboard']} sql={first['sql']}",
            {"link": "parity", "mismatches": mismatches, "pairs": pairs},
        )
    return _pass(cid, ctype, f"{len(points)} points match", {"pairs": pairs})


def check_es_search(
    check: Dict[str, Any],
    *,
    site_id: str,
    http: HttpClient,
) -> CheckResult:
    cid = str(check.get("id") or "es_search")
    ctype = "es_search"
    search = check.get("search") or {}
    query_word = check.get("query_word")
    expect_min = int(check.get("expect_min_results", 1))
    results_path = check.get("results_json_path") or "hits"
    not_found = check.get("not_found") or {}

    if not search.get("path"):
        return _fail(cid, ctype, "missing search.path")
    if query_word is None:
        return _fail(cid, ctype, "missing query_word")

    q = str(query_word)
    try:
        _st, _raw, parsed = http.request(
            search.get("method") or "GET",
            str(search["path"]),
            placeholders={"site_id": site_id, "q": q},
        )
    except BackendError as e:
        return _fail(cid, ctype, f"search failed: {e.message}", {"link": "search", **e.detail})
    if parsed is None:
        return _fail(cid, ctype, "search response not JSON", {"link": "search"})
    hits = json_path_get(parsed, str(results_path))
    if hits is None:
        return _fail(
            cid,
            ctype,
            f"missing results at {results_path!r}",
            {"link": "search", "results_json_path": results_path},
        )
    if not isinstance(hits, list):
        return _fail(
            cid,
            ctype,
            f"results at {results_path!r} is not a list",
            {"link": "search"},
        )
    if len(hits) < expect_min:
        return _fail(
            cid,
            ctype,
            f"hits={len(hits)} < expect_min_results={expect_min}",
            {"link": "search", "hits": len(hits)},
        )

    # not_found path — must return empty list gracefully (not 5xx)
    nf_q = not_found.get("q", "zzqxnotfound")
    expect_empty = bool(not_found.get("expect_empty", True))
    try:
        _st2, _raw2, parsed2 = http.request(
            search.get("method") or "GET",
            str(search["path"]),
            placeholders={"site_id": site_id, "q": str(nf_q)},
        )
    except BackendError as e:
        return _fail(
            cid,
            ctype,
            f"not_found query failed (must handle empty gracefully): {e.message}",
            {"link": "not_found", **e.detail},
        )
    if parsed2 is None:
        return _fail(cid, ctype, "not_found response not JSON", {"link": "not_found"})
    hits2 = json_path_get(parsed2, str(results_path))
    if hits2 is None:
        return _fail(
            cid,
            ctype,
            f"not_found missing results at {results_path!r}",
            {"link": "not_found"},
        )
    if not isinstance(hits2, list):
        return _fail(cid, ctype, "not_found results not a list", {"link": "not_found"})
    if expect_empty and len(hits2) != 0:
        return _fail(
            cid,
            ctype,
            f"not_found expected empty hits, got {len(hits2)}",
            {"link": "not_found", "hits": len(hits2)},
        )
    return _pass(
        cid,
        ctype,
        f"hits={len(hits)}, not_found empty ok",
        {"hits": len(hits), "not_found_hits": len(hits2)},
    )


def check_data_parity(
    check: Dict[str, Any],
    *,
    site_id: str,
    query: QueryRunner,
) -> CheckResult:
    cid = str(check.get("id") or "data_parity")
    ctype = "data_parity"
    tables = check.get("tables") or []
    sample_pct = float(check.get("sample_pct", 5))
    if not tables:
        return _fail(cid, ctype, "missing tables")

    table_reports: List[Dict[str, Any]] = []
    failed_tables: List[str] = []
    # Aggregate label parts for honest PASS reason
    field_modes: List[str] = []

    for t in tables:
        if not isinstance(t, dict):
            raise ConfigError(f"check {cid!r}: each table must be a mapping")
        name = str(t.get("name") or "?")
        src_sql = t.get("source_count_sql")
        tgt_sql = t.get("target_count_sql")
        key = str(t.get("key") or "id")
        sample_src = t.get("sample_source_sql")
        sample_tgt = t.get("sample_target_sql")
        # FIX 6: optional full field-by-field (not just sample)
        full_field_check = bool(t.get("full_field_check", False))
        if not src_sql or not tgt_sql:
            raise ConfigError(f"check {cid!r} table {name}: need source_count_sql and target_count_sql")
        if not sample_src or not sample_tgt:
            raise ConfigError(f"check {cid!r} table {name}: need sample_source_sql and sample_target_sql")

        try:
            src_rows = query.run(str(src_sql), {"site_id": site_id})
            tgt_rows = query.run(str(tgt_sql), {"site_id": site_id})
        except BackendError as e:
            failed_tables.append(name)
            table_reports.append(
                {"name": name, "status": "FAIL", "reason": f"count query: {e.message}", "link": "count"}
            )
            continue

        def _count(rows: List[Dict[str, Any]]) -> Optional[int]:
            if not rows:
                return 0
            r0 = rows[0]
            for fld in ("n", "count", "cnt", "N"):
                if fld in r0:
                    return _to_int(r0[fld])
            # single-value row
            if len(r0) == 1:
                return _to_int(next(iter(r0.values())))
            return None

        sc = _count(src_rows)
        tc = _count(tgt_rows)
        if sc is None or tc is None:
            failed_tables.append(name)
            table_reports.append(
                {
                    "name": name,
                    "status": "FAIL",
                    "reason": "could not parse count",
                    "source_count": sc,
                    "target_count": tc,
                }
            )
            continue
        if sc != tc:
            failed_tables.append(name)
            table_reports.append(
                {
                    "name": name,
                    "status": "FAIL",
                    "reason": f"count mismatch source={sc} target={tc}",
                    "link": "count",
                    "source_count": sc,
                    "target_count": tc,
                }
            )
            continue

        # field-level compare (sampled by default; FULL when full_field_check)
        try:
            srows = query.run(str(sample_src), {"site_id": site_id})
            trows = query.run(str(sample_tgt), {"site_id": site_id})
        except BackendError as e:
            failed_tables.append(name)
            table_reports.append(
                {"name": name, "status": "FAIL", "reason": f"sample query: {e.message}", "link": "sample"}
            )
            continue

        src_by_key: Dict[str, Dict[str, Any]] = {}
        for r in srows:
            if key in r:
                src_by_key[str(r[key])] = r
        tgt_by_key: Dict[str, Dict[str, Any]] = {}
        for r in trows:
            if key in r:
                tgt_by_key[str(r[key])] = r

        # FIX A: full_field_check ต้องครอบ "ทุกแถวจริง" — field query ต้องคืนจำนวน = count parity,
        # key ต้องไม่ซ้ำ, และชุด key ต้นทาง/ปลายทางต้องตรงกัน — ไม่งั้นห้ามเคลม FULL
        if full_field_check:
            cov_errs = []
            if len(srows) != sc or len(src_by_key) != sc:
                cov_errs.append(f"source field rows {len(srows)} (unique {len(src_by_key)}) != count {sc}")
            if len(trows) != tc or len(tgt_by_key) != tc:
                cov_errs.append(f"target field rows {len(trows)} (unique {len(tgt_by_key)}) != count {tc}")
            if set(src_by_key.keys()) != set(tgt_by_key.keys()):
                cov_errs.append("source/target key sets differ")
            if cov_errs:
                failed_tables.append(name)
                table_reports.append(
                    {
                        "name": name,
                        "status": "FAIL",
                        "reason": "incomplete field coverage: " + "; ".join(cov_errs),
                        "link": "field_full",
                    }
                )
                continue

        all_keys = list(src_by_key.keys())
        total_rows = len(all_keys)
        if full_field_check:
            picked = list(all_keys)
            field_mode = "FULL"
        else:
            picked = sample_keys(all_keys, sample_pct, min_rows=20)
            field_mode = "SAMPLED"
        sampled_count = len(picked)
        field_modes.append(field_mode)

        mismatches: List[Dict[str, Any]] = []
        for k in picked:
            srow = src_by_key.get(k)
            trow = tgt_by_key.get(k)
            if srow is None:
                mismatches.append({"key": k, "reason": "missing in source"})
                continue
            if trow is None:
                mismatches.append({"key": k, "reason": "missing in target"})
                continue
            fields = set(srow.keys()) | set(trow.keys())
            for f in sorted(fields):
                if not _values_equal(srow.get(f), trow.get(f)):
                    mismatches.append(
                        {
                            "key": k,
                            "field": f,
                            "source": srow.get(f),
                            "target": trow.get(f),
                        }
                    )
                    break  # one mismatch per key is enough for reporting density
            if len(mismatches) >= 5 and not full_field_check:
                break

        if mismatches:
            failed_tables.append(name)
            link = "field_full" if full_field_check else "sample"
            table_reports.append(
                {
                    "name": name,
                    "status": "FAIL",
                    "reason": (
                        f"field-check {field_mode} mismatch ({len(mismatches)}+)"
                    ),
                    "link": link,
                    "source_count": sc,
                    "target_count": tc,
                    "sampled": sampled_count,
                    "sampled_count": sampled_count,
                    "total_rows": total_rows,
                    "field_check": field_mode,
                    "sampled_keys": [str(x) for x in picked[:10]],
                    "mismatches": mismatches[:5],
                }
            )
        else:
            table_reports.append(
                {
                    "name": name,
                    "status": "PASS",
                    "source_count": sc,
                    "target_count": tc,
                    "sampled": sampled_count,
                    "sampled_count": sampled_count,
                    "total_rows": total_rows,
                    "field_check": field_mode,
                    "sampled_keys": [str(x) for x in picked[:10]],
                }
            )

    if failed_tables:
        first = next(t for t in table_reports if t.get("status") == "FAIL")
        return _fail(
            cid,
            ctype,
            f"table {first['name']}: {first.get('reason', 'fail')}",
            {"tables": table_reports, "failed": failed_tables},
        )

    # FIX 6: honest label — never read sampled as full field verification
    # Prefer first table's numbers for the summary reason when single table;
    # multi-table: note modes.
    if len(table_reports) == 1:
        tr = table_reports[0]
        scount = tr.get("sampled_count", 0)
        trows = tr.get("total_rows", 0)
        fmode = tr.get("field_check", "SAMPLED")
        if fmode == "FULL":
            reason = f"count-parity FULL + field-check FULL ({scount}/{trows})"
        else:
            reason = (
                f"count-parity FULL + field-check SAMPLED ({scount}/{trows})"
            )
    else:
        modes = sorted(set(field_modes)) or ["SAMPLED"]
        if modes == ["FULL"]:
            reason = f"count-parity FULL + field-check FULL ({len(tables)} tables)"
        elif "SAMPLED" in modes and "FULL" in modes:
            reason = (
                f"count-parity FULL + field-check MIXED "
                f"SAMPLED/FULL ({len(tables)} tables)"
            )
        else:
            # all sampled — still label SAMPLED with aggregate counts
            scount = sum(int(t.get("sampled_count") or 0) for t in table_reports)
            trows = sum(int(t.get("total_rows") or 0) for t in table_reports)
            reason = (
                f"count-parity FULL + field-check SAMPLED ({scount}/{trows})"
            )

    return _pass(
        cid,
        ctype,
        reason,
        {
            "tables": table_reports,
            "sampled_count": sum(int(t.get("sampled_count") or 0) for t in table_reports),
            "total_rows": sum(int(t.get("total_rows") or 0) for t in table_reports),
        },
    )


CHECK_HANDLERS = {
    "query_rows": check_query_rows,
    "form_cycle": check_form_cycle,
    "siteid_isolation": check_siteid_isolation,
    "dashboard_parity": check_dashboard_parity,
    "es_search": check_es_search,
    "data_parity": check_data_parity,
}


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def resolve_config_path(explicit: Optional[str], cwd: Optional[Path] = None) -> Path:
    """--config if given, else <cwd>/.work/backend-check.yaml."""
    if explicit:
        return Path(explicit)
    base = cwd or Path.cwd()
    return base / DEFAULT_CONFIG_REL


def load_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config not found: {path}")
    try:
        data = load_yaml_file(path)
    except MiniYamlError as e:
        raise ConfigError(f"YAML parse error in {path}: {e}") from e
    except Exception as e:
        raise ConfigError(f"failed to load config {path}: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"config root must be a mapping: {path}")
    return data


def validate_config(cfg: Dict[str, Any]) -> None:
    if not cfg.get("api_base"):
        raise ConfigError("config missing api_base")
    if not cfg.get("site_id"):
        raise ConfigError("config missing site_id")
    if not cfg.get("query_cmd"):
        raise ConfigError("config missing query_cmd")
    if not isinstance(cfg.get("query_cmd"), list):
        raise ConfigError("query_cmd must be a list (argv template)")
    if not cfg.get("db_path"):
        raise ConfigError("config missing db_path")
    checks = cfg.get("checks")
    if not checks or not isinstance(checks, list):
        raise ConfigError("config missing checks list")
    for i, c in enumerate(checks):
        if not isinstance(c, dict):
            raise ConfigError(f"checks[{i}] must be a mapping")
        cid = c.get("id")
        ctype = c.get("type")
        if not cid:
            raise ConfigError(f"checks[{i}] missing id")
        if not ctype:
            raise ConfigError(f"check {cid!r} missing type")
        if ctype not in KNOWN_TYPES:
            raise ConfigError(f"unknown check type {ctype!r} for id {cid!r}")


def run_checks(
    cfg: Dict[str, Any],
    *,
    only: Optional[Sequence[str]] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> RunReport:
    site_id = str(cfg["site_id"])
    db_path = str(cfg["db_path"])
    # resolve relative db_path against cwd
    if not os.path.isabs(db_path):
        db_path = str(Path(db_path).resolve())

    secrets: List[str] = []
    auth = cfg.get("auth")
    if isinstance(auth, dict):
        env_key = auth.get("value_env")
        if env_key:
            tok = os.environ.get(str(env_key), "")
            if tok:
                secrets.append(tok)

    http = HttpClient(
        str(cfg["api_base"]),
        auth=auth if isinstance(auth, dict) else None,
        timeout=timeout,
        secrets=secrets,
    )
    query = QueryRunner(
        cfg["query_cmd"],
        db_path,
        timeout=timeout,
        secrets=secrets,
    )

    all_checks = list(cfg["checks"])
    configured_total = len(all_checks)
    checks = all_checks
    if only:
        only_set = set(only)
        checks = [c for c in all_checks if c.get("id") in only_set]
        missing = only_set - {c.get("id") for c in checks}
        if missing:
            raise ConfigError(f"--only unknown check id(s): {sorted(missing)}")

    report = RunReport(site_id=site_id, configured_total=configured_total)
    for c in checks:
        ctype = str(c["type"])
        handler = CHECK_HANDLERS.get(ctype)
        if handler is None:
            raise ConfigError(f"unknown check type {ctype!r}")
        # handlers that need http vs query only
        kwargs: Dict[str, Any] = {"site_id": site_id, "query": query}
        if ctype in ("form_cycle", "dashboard_parity", "es_search"):
            kwargs["http"] = http
        if ctype == "es_search":
            kwargs.pop("query", None)
        result = handler(c, **kwargs)
        # redact secrets from reason/detail strings
        result.reason = redact_secrets(result.reason, secrets)
        result.detail = json.loads(
            redact_secrets(json.dumps(result.detail, default=str), secrets)
        )
        report.checks.append(result)
    return report


def format_human(report: RunReport) -> str:
    lines: List[str] = []
    configured = report.configured_total if report.configured_total > 0 else report.count
    # FIX 7: never claim full HEALTHY on a subset (--only) run
    if report.partial:
        lines.append(
            f"PARTIAL RUN (ran {report.ran} of {configured} checks)"
        )
    for c in report.checks:
        if c.ok:
            lines.append(f"{c.id}: PASS")
        else:
            extra = f" ({c.reason})" if c.reason else ""
            lines.append(f"{c.id}: FAIL{extra}")
    if report.partial:
        # Must not print bare HEALTHY=YES on partial runs
        lines.append(
            f"backend: {report.pass_count}/{report.count} · fail {report.fail_count} · "
            f"PARTIAL: {report.pass_count}/{configured} passed"
        )
    else:
        healthy = "YES" if report.healthy else "NO"
        lines.append(
            f"backend: {report.pass_count}/{report.count} · fail {report.fail_count} · "
            f"HEALTHY={healthy}"
        )
    if report.failed_ids:
        lines.append("failed:")
        for c in report.checks:
            if not c.ok:
                link = (c.detail or {}).get("link", "")
                link_s = f" [{link}]" if link else ""
                lines.append(f"  - {c.id}{link_s}: {c.reason}")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="backend_check.py",
        description="Config-driven backend/DB cycle verifier (mw-backend-check).",
    )
    p.add_argument(
        "--config",
        default=None,
        help=f"Path to YAML config (default: <cwd>/{DEFAULT_CONFIG_REL})",
    )
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    p.add_argument(
        "--only",
        default=None,
        help="Comma-separated check ids to run",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Per-request / query timeout seconds (default {DEFAULT_TIMEOUT})",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg_path = resolve_config_path(args.config)
    try:
        cfg = load_config(cfg_path)
        validate_config(cfg)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return EXIT_ERR
    except Exception as e:
        print(f"config error: {e}", file=sys.stderr)
        return EXIT_ERR

    only: Optional[List[str]] = None
    if args.only:
        only = [x.strip() for x in str(args.only).split(",") if x.strip()]
        if not only:
            print("config error: --only is empty", file=sys.stderr)
            return EXIT_ERR

    try:
        report = run_checks(cfg, only=only, timeout=float(args.timeout))
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return EXIT_ERR
    except Exception as e:
        # unexpected — fail closed as config/runtime error
        print(f"error: {e}", file=sys.stderr)
        return EXIT_ERR

    # collect secrets for final redact pass
    secrets: List[str] = []
    auth = cfg.get("auth") if isinstance(cfg, dict) else None
    if isinstance(auth, dict) and auth.get("value_env"):
        tok = os.environ.get(str(auth["value_env"]), "")
        if tok:
            secrets.append(tok)

    if args.json:
        payload = report.to_dict()
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        print(redact_secrets(text, secrets))
    else:
        print(redact_secrets(format_human(report), secrets))

    if report.count == 0:
        print("config error: no checks run", file=sys.stderr)
        return EXIT_ERR
    # Exit 0 when every check that ran passed (partial subset still exit 0 if all pass).
    # Full HEALTHY=true only when not partial — see RunReport.healthy / format_human.
    return EXIT_OK if report.all_ran_passed else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
