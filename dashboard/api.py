"""
Hermes Dashboard Backend API
~/.hermes/dashboard/api.py

api_server.py의 aiohttp app에 대시보드 라우트를 등록하는 플러그인.
"""

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

try:
    from aiohttp import web
except ImportError:
    web = None  # type: ignore

HERMES_HOME = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
DASHBOARD_DIR = HERMES_HOME / "dashboard"

# FC-RAG 대시보드 (LexDiff 법령 질의 로그 뷰어)는 사용자 맞춤 기능.
# 기본 비활성화. 활성화하려면: HERMES_FC_RAG_ENABLED=1
_FC_RAG_ENABLED = os.getenv("HERMES_FC_RAG_ENABLED", "").lower() in ("1", "true", "yes", "on")

# 시크릿 마스킹: 이 패턴이 키 이름에 포함되면 자동 마스킹
_SECRET_PATTERNS = re.compile(r'TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL', re.IGNORECASE)
_SAFE_NAME = re.compile(r'^[A-Za-z][A-Za-z0-9_]*$')
_SAFE_ID = re.compile(r'^[a-f0-9]{6,20}$')

# Hermes가 json.dumps(ensure_ascii=True)로 저장한 \uXXXX 이스케이프를 실제 유니코드로 복원.
# 로그/tool_result 출력 시 한글 깨짐 복구용.
_UNICODE_ESCAPE_RE = re.compile(r'\\u([0-9a-fA-F]{4})')


def unescape_unicode(s: str) -> str:
    """\\u2550 같은 리터럴 시퀀스를 실제 문자(═)로 복원한다.
    BMP 밖 문자(surrogate pair)는 변환하지 않고 그대로 둔다.
    """
    if not isinstance(s, str) or '\\u' not in s:
        return s
    try:
        return _UNICODE_ESCAPE_RE.sub(lambda m: chr(int(m.group(1), 16)), s)
    except Exception:
        return s


# ── Atomic File I/O ──

def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {} if yaml else {}
    except Exception:
        return {}


def _write_yaml(path: Path, data: dict) -> None:
    if not yaml:
        raise RuntimeError("PyYAML not installed")
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _write_json(path: Path, data: Any) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_env(path: Path) -> dict:
    result = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip()
    return result


def _write_env(path: Path, data: dict) -> None:
    lines = [f"{k}={v}" for k, v in data.items()]
    content = "\n".join(lines) + "\n"
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _mask_secret(v: str) -> str:
    if len(v) <= 8:
        return "****"
    return v[:4] + "..." + v[-3:]


def _is_secret_key(k: str) -> bool:
    return bool(_SECRET_PATTERNS.search(k))


def _safe_within(target: Path, parent: Path) -> bool:
    """Verify resolved path is within parent (path traversal prevention)."""
    try:
        target.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


# ── Handlers ──

async def handle_dashboard(request: web.Request) -> web.Response:
    index = DASHBOARD_DIR / "index.html"
    if not index.exists():
        return web.Response(text="Dashboard not found", status=404)
    return web.Response(text=index.read_text(encoding="utf-8"), content_type="text/html", charset="utf-8")


async def handle_overview(request: web.Request) -> web.json_response:
    config = _read_yaml(HERMES_HOME / "config.yaml")
    state = _read_json(HERMES_HOME / "gateway_state.json")
    jobs = _read_json(HERMES_HOME / "cron" / "jobs.json")
    env = _read_env(HERMES_HOME / ".env")
    mcp = config.get("mcp_servers", {})
    skills_dir = HERMES_HOME / "skills"
    skill_count = sum(1 for _ in skills_dir.rglob("SKILL.md")) if skills_dir.exists() else 0
    sessions_dir = HERMES_HOME / "sessions"
    session_count = len(list(sessions_dir.glob("session_*.json"))) if sessions_dir.exists() else 0
    return web.json_response({
        "model": config.get("model", "unknown"),
        "gateway": {
            "pid": state.get("pid"),
            "state": state.get("gateway_state", "unknown"),
            "platforms": state.get("platforms", {}),
            "active_agents": state.get("active_agents", 0),
        },
        "mcp_servers": {name: {"command": s.get("command", ""), "args": s.get("args", [])} for name, s in mcp.items()},
        "cron_jobs": len(jobs.get("jobs", [])),
        "cron_summary": [
            {"id": j["id"], "name": j["name"], "enabled": j.get("enabled", True),
             "last_status": j.get("last_status"), "next_run": j.get("next_run_at"),
             "deliver": j.get("deliver", "local")}
            for j in jobs.get("jobs", [])
        ],
        "skill_count": skill_count,
        "session_count": session_count,
        "telegram": {
            "home_channel": env.get("TELEGRAM_HOME_CHANNEL", ""),
            "allowed_users": env.get("TELEGRAM_ALLOWED_USERS", ""),
        },
        "features": {
            "fc_rag": _FC_RAG_ENABLED,
        },
    })


async def handle_get_config(request: web.Request) -> web.json_response:
    return web.json_response(_read_yaml(HERMES_HOME / "config.yaml"))


async def handle_set_config(request: web.Request) -> web.json_response:
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "invalid JSON"}, status=400)
    if not isinstance(body, dict):
        return web.json_response({"ok": False, "error": "object expected"}, status=400)
    config = _read_yaml(HERMES_HOME / "config.yaml")
    config.update(body)
    _write_yaml(HERMES_HOME / "config.yaml", config)
    return web.json_response({"ok": True, "restart_needed": True})


async def handle_get_env(request: web.Request) -> web.json_response:
    env = _read_env(HERMES_HOME / ".env")
    masked = {}
    masked_keys = set()
    for k, v in env.items():
        if _is_secret_key(k):
            masked[k] = _mask_secret(v)
            masked_keys.add(k)
        else:
            masked[k] = v
    return web.json_response({"env": masked, "raw_keys": list(env.keys()), "masked_keys": list(masked_keys)})


async def handle_set_env(request: web.Request) -> web.json_response:
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "invalid JSON"}, status=400)
    env = _read_env(HERMES_HOME / ".env")
    for k, v in body.items():
        if not _SAFE_NAME.match(k):
            continue
        if "..." in v or "****" in v:
            continue
        env[k] = v
    _write_env(HERMES_HOME / ".env", env)
    return web.json_response({"ok": True})


async def handle_get_mcp(request: web.Request) -> web.json_response:
    return web.json_response(_read_yaml(HERMES_HOME / "config.yaml").get("mcp_servers", {}))


async def handle_set_mcp(request: web.Request) -> web.json_response:
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "invalid JSON"}, status=400)
    if not isinstance(body, dict):
        return web.json_response({"ok": False, "error": "object expected"}, status=400)
    config = _read_yaml(HERMES_HOME / "config.yaml")
    config["mcp_servers"] = body
    _write_yaml(HERMES_HOME / "config.yaml", config)
    return web.json_response({"ok": True, "restart_needed": True})


async def handle_get_soul(request: web.Request) -> web.json_response:
    return web.json_response({"content": _read_text(HERMES_HOME / "SOUL.md")})


async def handle_set_soul(request: web.Request) -> web.json_response:
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "invalid JSON"}, status=400)
    _write_text(HERMES_HOME / "SOUL.md", body.get("content", ""))
    return web.json_response({"ok": True})


async def handle_get_skills(request: web.Request) -> web.json_response:
    skills_dir = HERMES_HOME / "skills"
    result = []
    if skills_dir.exists():
        for skill_file in sorted(skills_dir.rglob("SKILL.md")):
            rel = skill_file.relative_to(skills_dir)
            category = str(rel.parent.parent) if len(rel.parts) > 2 else str(rel.parent)
            name = rel.parent.name
            lines = skill_file.read_text(encoding="utf-8", errors="replace").split("\n")
            desc = ""
            for line in lines:
                if line.startswith("description:"):
                    desc = line.split(":", 1)[1].strip().strip('"').strip("'")
                    break
            result.append({"name": name, "category": category, "description": desc, "path": str(rel.parent)})
    return web.json_response(result)


async def handle_get_logs(request: web.Request) -> web.json_response:
    try:
        lines_param = min(max(int(request.query.get("lines", "100")), 1), 5000)
    except (ValueError, TypeError):
        lines_param = 100
    log_file = HERMES_HOME / "logs" / "gateway.log"
    if not log_file.exists():
        return web.json_response({"lines": []})
    all_lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
    # \uXXXX 이스케이프 복원 (Hermes가 tool_result를 ensure_ascii=True로 저장)
    decoded = [unescape_unicode(l) for l in all_lines[-lines_param:]]
    return web.json_response({"lines": decoded})


async def handle_get_sessions(request: web.Request) -> web.json_response:
    sessions_dir = HERMES_HOME / "sessions"
    result = []
    if sessions_dir.exists():
        files = sorted(sessions_dir.glob("session_*.json"), key=lambda f: f.stat().st_mtime, reverse=True)[:20]
        for f in files:
            st = f.stat()
            result.append({"name": f.stem, "size": st.st_size, "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))})
    return web.json_response(result)


async def handle_restart_gateway(request: web.Request) -> web.json_response:
    try:
        subprocess.run(["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/ai.hermes.gateway"], capture_output=True, timeout=10)
        return web.json_response({"ok": True})
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=500)


async def handle_get_cron_output(request: web.Request) -> web.json_response:
    job_id = request.match_info["job_id"]
    if not _SAFE_ID.match(job_id):
        return web.json_response({"error": "invalid job_id"}, status=400)
    sessions_dir = HERMES_HOME / "sessions"
    outputs = sorted(sessions_dir.glob(f"session_cron_{job_id}_*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not outputs:
        return web.json_response({"output": None})
    return web.json_response({
        "file": outputs[0].name,
        "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(outputs[0].stat().st_mtime)),
        "size": outputs[0].stat().st_size,
    })


async def handle_set_model(request: web.Request) -> web.json_response:
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "invalid JSON"}, status=400)
    new_model = body.get("model", "").strip()
    if not new_model or len(new_model) > 100:
        return web.json_response({"ok": False, "error": "model required"}, status=400)
    config = _read_yaml(HERMES_HOME / "config.yaml")
    old_model = config.get("model", "unknown")
    config["model"] = new_model
    _write_yaml(HERMES_HOME / "config.yaml", config)
    return web.json_response({"ok": True, "old": old_model, "new": new_model, "restart_needed": True})


async def handle_get_auth_status(request: web.Request) -> web.json_response:
    auth = _read_json(HERMES_HOME / "auth.json")
    result = {"active_provider": auth.get("active_provider", "unknown"), "providers": {}}
    for name, prov in auth.get("providers", {}).items():
        # openai-codex는 OAuth 토큰을 'tokens' dict에 저장 (access_token/id_token/refresh_token)
        tokens = prov.get("tokens") or {}
        has_token = bool(
            prov.get("access_token")
            or prov.get("api_key")
            or (isinstance(tokens, dict) and (tokens.get("access_token") or tokens.get("id_token")))
        )
        result["providers"][name] = {
            "has_token": has_token,
            "last_refresh": prov.get("last_refresh", ""),
            "request_count": prov.get("request_count", 0),
        }
    return web.json_response(result)


async def handle_delete_session(request: web.Request) -> web.json_response:
    name = request.match_info["name"]
    sessions_dir = HERMES_HOME / "sessions"
    target = sessions_dir / f"{name}.json"
    if not _safe_within(target, sessions_dir):
        return web.json_response({"ok": False, "error": "invalid path"}, status=400)
    if target.exists():
        target.unlink()
        return web.json_response({"ok": True})
    return web.json_response({"ok": False, "error": "not found"}, status=404)


async def handle_clear_logs(request: web.Request) -> web.json_response:
    log_file = HERMES_HOME / "logs" / "gateway.log"
    if log_file.exists():
        log_file.write_text("", encoding="utf-8")
    return web.json_response({"ok": True})


def _parse_query_ts(ts: str) -> float | None:
    """fc-rag-queries.jsonl의 ts (ISO UTC with Z)를 epoch seconds로."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _parse_session_ts(ts: str) -> float | None:
    """세션 파일의 naive datetime (로컬 KST)을 epoch seconds로."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.astimezone()  # attach system local tz
        return dt.timestamp()
    except Exception:
        return None


# 세션 인덱스 캐시 (파일명 → (mtime, start_epoch, end_epoch))
_session_index_cache: dict[str, tuple[float, float | None, float | None]] = {}


def _build_session_index() -> list[tuple[float, float, str]]:
    """api 세션 파일들의 (start_epoch, end_epoch, path) 인덱스.

    mtime 변경된 파일만 재파싱한다.
    """
    sessions_dir = HERMES_HOME / "sessions"
    if not sessions_dir.exists():
        return []
    index: list[tuple[float, float, str]] = []
    seen: set[str] = set()
    for p in sessions_dir.glob("session_api-*.json"):
        name = p.name
        seen.add(name)
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        cached = _session_index_cache.get(name)
        if cached and cached[0] == mtime:
            _, s_ep, e_ep = cached
        else:
            s_ep = e_ep = None
            try:
                with open(p, "r", encoding="utf-8") as f:
                    d = json.load(f)
                s_ep = _parse_session_ts(d.get("session_start", ""))
                e_ep = _parse_session_ts(d.get("last_updated", ""))
            except Exception:
                pass
            _session_index_cache[name] = (mtime, s_ep, e_ep)
        if s_ep is not None and e_ep is not None:
            index.append((s_ep, e_ep, str(p)))
    # 스테일 엔트리 제거
    for stale in list(_session_index_cache.keys()):
        if stale not in seen:
            del _session_index_cache[stale]
    return index


def _match_session_for_query(q: dict, session_index: list[tuple[float, float, str]]) -> str | None:
    """query의 (ts=완료시각, durationMs)로 세션 파일을 퍼지 매칭. 일치 시 경로 반환.

    lexdiff의 query-logger는 질의 완료 후 ts를 기록하므로 q.ts ≈ session.last_updated,
    q.ts - durationMs ≈ session.session_start 로 매칭한다.
    """
    q_end = _parse_query_ts(q.get("ts", ""))
    if q_end is None:
        return None
    dur_s = (q.get("durationMs") or 0) / 1000.0
    q_start = q_end - dur_s
    best: tuple[float, str] | None = None  # (score, path)
    for s_start, s_end, path in session_index:
        end_diff = abs(s_end - q_end)
        start_diff = abs(s_start - q_start)
        if end_diff > 15.0:
            continue
        score = end_diff + start_diff * 0.5
        if best is None or score < best[0]:
            best = (score, path)
    return best[1] if best else None


async def handle_fc_rag_queries(request: web.Request) -> web.json_response:
    """FC-RAG 법령 질의 로그 반환 (최근 N건)."""
    try:
        limit = min(max(int(request.query.get("limit", "100")), 1), 1000)
    except (ValueError, TypeError):
        limit = 100
    log_file = HERMES_HOME / "logs" / "fc-rag-queries.jsonl"
    if not log_file.exists():
        return web.json_response({"queries": [], "total": 0})
    lines = log_file.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    queries = []
    for line in reversed(lines[-limit:]):
        try:
            queries.append(json.loads(line))
        except (json.JSONDecodeError, ValueError):
            continue
    # 1) 기존 trace 파일 기반 (구버전 호환)
    trace_file = HERMES_HOME / "logs" / "fc-rag-traces.jsonl"
    trace_ids: set = set()
    if trace_file.exists():
        for tl in trace_file.read_text(encoding="utf-8", errors="replace").strip().splitlines():
            try:
                trace_ids.add(json.loads(tl).get("traceId", ""))
            except Exception:
                continue
    # 2) 세션 파일 퍼지 매칭 (신버전)
    session_index = _build_session_index()
    for q in queries:
        has_explicit = q.get("traceId", "") in trace_ids
        has_session = _match_session_for_query(q, session_index) is not None
        q["hasTrace"] = has_explicit or has_session
    return web.json_response({"queries": queries, "total": len(lines)})


async def handle_fc_rag_stats(request: web.Request) -> web.json_response:
    """FC-RAG 통계: 총 질의 수, 소스별 분포, 평균 소요시간, 일별 추이."""
    log_file = HERMES_HOME / "logs" / "fc-rag-queries.jsonl"
    if not log_file.exists():
        return web.json_response({
            "total": 0, "bySource": {}, "byComplexity": {},
            "avgDurationMs": 0, "errorRate": 0, "daily": [],
        })
    lines = log_file.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    entries = []
    for line in lines:
        try:
            entries.append(json.loads(line))
        except (json.JSONDecodeError, ValueError):
            continue
    if not entries:
        return web.json_response({
            "total": 0, "bySource": {}, "byComplexity": {},
            "avgDurationMs": 0, "errorRate": 0, "daily": [],
        })
    by_source: dict[str, int] = {}
    by_complexity: dict[str, int] = {}
    by_day: dict[str, dict] = {}
    total_duration = 0
    error_count = 0
    for e in entries:
        src = e.get("source", "unknown")
        by_source[src] = by_source.get(src, 0) + 1
        cx = e.get("complexity", "unknown")
        by_complexity[cx] = by_complexity.get(cx, 0) + 1
        total_duration += e.get("durationMs", 0)
        if e.get("error"):
            error_count += 1
        day = e.get("ts", "")[:10]
        if day:
            if day not in by_day:
                by_day[day] = {"date": day, "count": 0, "errors": 0, "avgMs": 0, "_totalMs": 0}
            by_day[day]["count"] += 1
            by_day[day]["_totalMs"] += e.get("durationMs", 0)
            if e.get("error"):
                by_day[day]["errors"] += 1
    daily = sorted(by_day.values(), key=lambda d: d["date"])
    for d in daily:
        d["avgMs"] = round(d["_totalMs"] / d["count"]) if d["count"] else 0
        del d["_totalMs"]
    return web.json_response({
        "total": len(entries),
        "bySource": by_source,
        "byComplexity": by_complexity,
        "avgDurationMs": round(total_duration / len(entries)) if entries else 0,
        "errorRate": round(error_count / len(entries) * 100, 1) if entries else 0,
        "daily": daily[-30:],
    })


def _truncate(value: Any, limit: int = 4000) -> Any:
    """긴 값을 대시보드용으로 자른다. 문자열은 \\uXXXX 이스케이프도 복원."""
    if isinstance(value, str):
        value = unescape_unicode(value)
        return value if len(value) <= limit else value[:limit] + f"… (+{len(value) - limit} chars)"
    if isinstance(value, (dict, list)):
        try:
            s = json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)[:limit]
        return value if len(s) <= limit else json.loads(json.dumps(value, ensure_ascii=False))  # keep full if JSON fits
    return value


def _session_to_trace(session: dict, query: dict) -> dict:
    """Hermes 세션 파일을 FC-RAG trace 포맷으로 변환.

    Events types:
      - system_prompt: 시스템 프롬프트
      - user_query: 사용자 쿼리
      - assistant_reasoning: 사고/추론
      - assistant_message: 답변 텍스트
      - tool_call: 도구 호출 (name, args)
      - tool_result: 도구 결과 (tool_call_id, content)
    """
    trace_id = query.get("traceId", "")
    started_at = query.get("ts") or session.get("session_start", "")
    completed_at = session.get("last_updated", "")
    source = query.get("source", "hermes")

    events: list[dict] = []
    start_epoch = _parse_session_ts(session.get("session_start", "")) or 0.0

    def _ts(offset_idx: int) -> str:
        # 실제 메시지별 타임스탬프가 없으므로 순서만 보존
        return datetime.fromtimestamp(start_epoch + offset_idx * 0.001, tz=timezone.utc).isoformat().replace("+00:00", "Z")

    # 시스템 프롬프트
    sp = session.get("system_prompt")
    if isinstance(sp, list):
        sp = "".join(s.get("text", "") if isinstance(s, dict) else str(s) for s in sp)
    if sp:
        events.append({
            "ts": _ts(0),
            "event": "system_prompt",
            "data": {"content": _truncate(str(sp), 2000)},
        })

    # 메시지 순회
    for i, m in enumerate(session.get("messages", []), start=1):
        role = m.get("role")
        if role == "user":
            c = m.get("content")
            if isinstance(c, list):
                c = "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in c)
            events.append({
                "ts": _ts(i),
                "event": "user_query",
                "data": {"content": _truncate(str(c) if c else "", 2000)},
            })
        elif role == "assistant":
            reasoning = m.get("reasoning")
            if reasoning:
                events.append({
                    "ts": _ts(i),
                    "event": "assistant_reasoning",
                    "data": {"content": _truncate(str(reasoning), 4000)},
                })
            content = m.get("content")
            if isinstance(content, list):
                content = "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in content)
            if content:
                events.append({
                    "ts": _ts(i),
                    "event": "assistant_message",
                    "data": {"content": _truncate(str(content), 6000)},
                })
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        pass
                events.append({
                    "ts": _ts(i),
                    "event": "tool_call",
                    "data": {
                        "id": tc.get("id"),
                        "name": fn.get("name"),
                        "args": _truncate(args, 2000),
                    },
                })
        elif role == "tool":
            c = m.get("content")
            if isinstance(c, list):
                c = "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in c)
            events.append({
                "ts": _ts(i),
                "event": "tool_result",
                "data": {
                    "tool_call_id": m.get("tool_call_id"),
                    "result": _truncate(str(c) if c else "", 4000),
                },
            })

    return {
        "traceId": trace_id,
        "query": query.get("query", ""),
        "startedAt": started_at,
        "completedAt": completed_at,
        "source": source,
        "sessionId": session.get("session_id"),
        "model": session.get("model"),
        "messageCount": session.get("message_count"),
        "events": events,
    }


async def handle_fc_rag_trace(request: web.Request) -> web.json_response:
    """특정 traceId의 상세 trace 이벤트 반환.

    우선순위:
    1. fc-rag-traces.jsonl의 명시적 trace 엔트리
    2. 세션 파일 퍼지 매칭 (session_api-*.json)
    """
    trace_id = request.match_info["trace_id"]

    # (1) 기존 trace 파일 직접 조회
    trace_file = HERMES_HOME / "logs" / "fc-rag-traces.jsonl"
    if trace_file.exists():
        for line in reversed(trace_file.read_text(encoding="utf-8", errors="replace").strip().splitlines()):
            try:
                t = json.loads(line)
                if t.get("traceId") == trace_id:
                    return web.json_response({"trace": t})
            except Exception:
                continue

    # (2) query 로그에서 trace_id 찾기
    log_file = HERMES_HOME / "logs" / "fc-rag-queries.jsonl"
    if not log_file.exists():
        return web.json_response({"error": "no query log"}, status=404)
    query_match: dict | None = None
    for line in reversed(log_file.read_text(encoding="utf-8", errors="replace").strip().splitlines()):
        try:
            q = json.loads(line)
            if q.get("traceId") == trace_id:
                query_match = q
                break
        except Exception:
            continue
    if not query_match:
        return web.json_response({"error": "trace not found"}, status=404)

    # (3) 세션 파일 퍼지 매칭
    session_index = _build_session_index()
    session_path = _match_session_for_query(query_match, session_index)
    if not session_path:
        return web.json_response({"error": "no matching session"}, status=404)
    try:
        with open(session_path, "r", encoding="utf-8") as f:
            session = json.load(f)
    except Exception as e:
        return web.json_response({"error": f"session load failed: {e}"}, status=500)

    trace = _session_to_trace(session, query_match)
    return web.json_response({"trace": trace})


async def handle_disk_usage(request: web.Request) -> web.json_response:
    dirs = {
        "sessions": HERMES_HOME / "sessions", "logs": HERMES_HOME / "logs",
        "skills": HERMES_HOME / "skills", "cron": HERMES_HOME / "cron",
        "state.db": HERMES_HOME / "state.db", "response_store.db": HERMES_HOME / "response_store.db",
    }
    result = {}
    for name, path in dirs.items():
        try:
            if path.is_file():
                result[name] = path.stat().st_size
            elif path.is_dir():
                result[name] = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            else:
                result[name] = 0
        except OSError:
            result[name] = 0
    try:
        result["total"] = sum(f.stat().st_size for f in HERMES_HOME.rglob("*") if f.is_file())
    except OSError:
        result["total"] = 0
    return web.json_response(result)


async def handle_get_skill_detail(request: web.Request) -> web.json_response:
    skill_path = request.match_info["path"]
    skills_dir = HERMES_HOME / "skills"
    skill_file = skills_dir / skill_path / "SKILL.md"
    if not _safe_within(skill_file, skills_dir):
        return web.json_response({"error": "invalid path"}, status=400)
    if not skill_file.exists():
        return web.json_response({"error": "not found"}, status=404)
    return web.json_response({"content": skill_file.read_text(encoding="utf-8", errors="replace")})


async def handle_create_cron(request: web.Request) -> web.json_response:
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "invalid JSON"}, status=400)
    jobs_file = HERMES_HOME / "cron" / "jobs.json"
    jobs_data = _read_json(jobs_file)
    if "jobs" not in jobs_data:
        jobs_data["jobs"] = []
    job_id = hashlib.md5(f"{body.get('name','')}{time.time()}".encode()).hexdigest()[:12]
    now = datetime.now(timezone.utc).astimezone().isoformat()
    schedule = body.get("schedule", "0 9 * * *")
    new_job = {
        "id": job_id, "name": body.get("name", "New Job")[:100],
        "prompt": body.get("prompt", "")[:2000],
        "skills": [body["skill"]] if body.get("skill") else [], "skill": body.get("skill", ""),
        "model": None, "provider": None, "base_url": None, "script": None,
        "schedule": {"kind": "cron", "expr": schedule, "display": schedule},
        "schedule_display": schedule,
        "repeat": {"times": None, "completed": 0},
        "enabled": True, "state": "scheduled",
        "paused_at": None, "paused_reason": None,
        "created_at": now, "next_run_at": None,
        "last_run_at": None, "last_status": None,
        "last_error": None, "last_delivery_error": None,
        "deliver": body.get("deliver", "local"), "origin": None,
    }
    jobs_data["jobs"].append(new_job)
    jobs_data["updated_at"] = now
    _write_json(jobs_file, jobs_data)
    return web.json_response({"ok": True, "id": job_id})


async def handle_edit_cron(request: web.Request) -> web.json_response:
    job_id = request.match_info["job_id"]
    if not _SAFE_ID.match(job_id):
        return web.json_response({"ok": False, "error": "invalid job_id"}, status=400)
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "invalid JSON"}, status=400)
    jobs_file = HERMES_HOME / "cron" / "jobs.json"
    jobs_data = _read_json(jobs_file)
    for job in jobs_data.get("jobs", []):
        if job["id"] == job_id:
            for key in ("name", "prompt", "skill", "deliver", "enabled"):
                if key in body:
                    job[key] = body[key]
                    if key == "skill":
                        job["skills"] = [body[key]] if body[key] else []
            if "schedule" in body:
                job["schedule"] = {"kind": "cron", "expr": body["schedule"], "display": body["schedule"]}
                job["schedule_display"] = body["schedule"]
            jobs_data["updated_at"] = datetime.now(timezone.utc).astimezone().isoformat()
            _write_json(jobs_file, jobs_data)
            return web.json_response({"ok": True})
    return web.json_response({"ok": False, "error": "job not found"}, status=404)


async def handle_delete_cron(request: web.Request) -> web.json_response:
    job_id = request.match_info["job_id"]
    if not _SAFE_ID.match(job_id):
        return web.json_response({"ok": False, "error": "invalid job_id"}, status=400)
    jobs_file = HERMES_HOME / "cron" / "jobs.json"
    jobs_data = _read_json(jobs_file)
    jobs_data["jobs"] = [j for j in jobs_data.get("jobs", []) if j["id"] != job_id]
    jobs_data["updated_at"] = datetime.now(timezone.utc).astimezone().isoformat()
    _write_json(jobs_file, jobs_data)
    return web.json_response({"ok": True})


# ═══════════════════════════════════════════════════════════════
# Chat Menu — 내부 채팅 + 플랫폼 메시지 뷰어
# ═══════════════════════════════════════════════════════════════


async def handle_chat_sessions(request: web.Request) -> web.json_response:
    """모든 플랫폼 세션 목록 반환 (telegram/api/dashboard 등).

    Query: ?platform=telegram|api|dashboard|all (default: all)
    """
    platform_filter = request.query.get("platform", "all").lower()
    sessions_dir = HERMES_HOME / "sessions"
    if not sessions_dir.exists():
        return web.json_response({"sessions": []})
    result = []
    files = sorted(sessions_dir.glob("session_*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    for f in files[:200]:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            continue
        plat = d.get("platform", "unknown")
        if platform_filter != "all" and plat != platform_filter:
            continue
        # 최근 메시지 스니펫
        last_user = last_assistant = ""
        for m in reversed(d.get("messages", [])):
            role = m.get("role")
            if role not in ("user", "assistant"):
                continue
            c = m.get("content", "")
            if isinstance(c, list):
                c = "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in c)
            c = unescape_unicode(str(c or ""))[:120]
            if role == "user" and not last_user:
                last_user = c
            elif role == "assistant" and not last_assistant:
                last_assistant = c
            if last_user and last_assistant:
                break
        result.append({
            "sessionId": d.get("session_id", f.stem),
            "platform": plat,
            "model": d.get("model"),
            "start": d.get("session_start"),
            "updated": d.get("last_updated"),
            "messageCount": len(d.get("messages", [])),
            "lastUser": last_user,
            "lastAssistant": last_assistant,
            "file": f.name,
        })
    return web.json_response({"sessions": result, "total": len(result)})


async def handle_chat_session_detail(request: web.Request) -> web.json_response:
    """특정 세션의 전체 메시지 반환. \\uXXXX 복원 포함."""
    name = request.match_info["name"]
    sessions_dir = HERMES_HOME / "sessions"
    target = sessions_dir / f"{name}.json"
    if not _safe_within(target, sessions_dir):
        return web.json_response({"error": "invalid path"}, status=400)
    if not target.exists():
        return web.json_response({"error": "not found"}, status=404)
    try:
        with open(target, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception as e:
        return web.json_response({"error": f"load failed: {e}"}, status=500)

    messages = []
    for m in d.get("messages", []):
        role = m.get("role")
        c = m.get("content", "")
        if isinstance(c, list):
            c = "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in c)
        item: dict[str, Any] = {
            "role": role,
            "content": unescape_unicode(str(c or "")),
        }
        if m.get("reasoning"):
            item["reasoning"] = unescape_unicode(str(m["reasoning"]))
        if m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        pass
                tcs.append({
                    "id": tc.get("id"),
                    "name": fn.get("name"),
                    "args": _truncate(args, 2000),
                })
            item["toolCalls"] = tcs
        if m.get("tool_call_id"):
            item["toolCallId"] = m["tool_call_id"]
        messages.append(item)

    return web.json_response({
        "sessionId": d.get("session_id"),
        "platform": d.get("platform"),
        "model": d.get("model"),
        "start": d.get("session_start"),
        "updated": d.get("last_updated"),
        "messages": messages,
    })


async def handle_chat_send(request: web.Request) -> web.json_response:
    """내부 채팅 — Hermes /v1/chat/completions 호출해서 응답 반환.

    Request: {"message": "...", "history": [{"role","content"}, ...]?}
    Response: {"reply": "...", "model": "...", "durationMs": ...}
    """
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid json"}, status=400)
    msg = (body.get("message") or "").strip()
    if not msg:
        return web.json_response({"error": "empty message"}, status=400)
    history = body.get("history") or []
    if not isinstance(history, list):
        history = []

    # .env에서 API key 읽기
    env_file = HERMES_HOME / ".env"
    api_key = "lexdiff-hermes-local"  # fallback
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("API_SERVER_KEY="):
                api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                break

    messages = []
    for h in history[-20:]:  # 최근 20개만
        if not isinstance(h, dict):
            continue
        r = h.get("role")
        c = h.get("content", "")
        if r in ("user", "assistant") and isinstance(c, str) and c:
            messages.append({"role": r, "content": c})
    messages.append({"role": "user", "content": msg})

    payload = {
        "model": "gpt-5.4",
        "messages": messages,
        "stream": False,
        "max_tokens": 2000,
        "skip_context_files": True,
    }

    import aiohttp as _aio
    started = time.time()
    try:
        async with _aio.ClientSession() as sess:
            async with sess.post(
                "http://127.0.0.1:8642/v1/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                timeout=_aio.ClientTimeout(total=120),
            ) as resp:
                data = await resp.json()
    except Exception as e:
        return web.json_response({"error": f"hermes call failed: {e}"}, status=502)

    duration_ms = int((time.time() - started) * 1000)
    try:
        reply = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return web.json_response({"error": "invalid hermes response", "raw": data}, status=502)

    return web.json_response({
        "reply": unescape_unicode(str(reply)),
        "model": data.get("model", "gpt-5.4"),
        "durationMs": duration_ms,
        "usage": data.get("usage"),
    })


def register_dashboard_routes(app: "web.Application") -> None:
    app.router.add_get("/dashboard", handle_dashboard)
    app.router.add_get("/dashboard/", handle_dashboard)
    app.router.add_get("/api/dashboard/overview", handle_overview)
    app.router.add_get("/api/dashboard/config", handle_get_config)
    app.router.add_post("/api/dashboard/config", handle_set_config)
    app.router.add_get("/api/dashboard/env", handle_get_env)
    app.router.add_post("/api/dashboard/env", handle_set_env)
    app.router.add_get("/api/dashboard/mcp", handle_get_mcp)
    app.router.add_post("/api/dashboard/mcp", handle_set_mcp)
    app.router.add_get("/api/dashboard/soul", handle_get_soul)
    app.router.add_post("/api/dashboard/soul", handle_set_soul)
    app.router.add_get("/api/dashboard/skills", handle_get_skills)
    app.router.add_get("/api/dashboard/logs", handle_get_logs)
    app.router.add_get("/api/dashboard/sessions", handle_get_sessions)
    app.router.add_post("/api/dashboard/restart", handle_restart_gateway)
    app.router.add_get("/api/dashboard/cron/{job_id}/output", handle_get_cron_output)
    app.router.add_post("/api/dashboard/model", handle_set_model)
    app.router.add_get("/api/dashboard/auth", handle_get_auth_status)
    app.router.add_delete("/api/dashboard/sessions/{name}", handle_delete_session)
    app.router.add_post("/api/dashboard/logs/clear", handle_clear_logs)
    app.router.add_get("/api/dashboard/disk", handle_disk_usage)
    app.router.add_get("/api/dashboard/skills/{path:.+}/detail", handle_get_skill_detail)
    app.router.add_post("/api/dashboard/cron/create", handle_create_cron)
    app.router.add_post("/api/dashboard/cron/{job_id}/edit", handle_edit_cron)
    app.router.add_delete("/api/dashboard/cron/{job_id}", handle_delete_cron)
    # FC-RAG 대시보드 (LexDiff 맞춤 기능) — HERMES_FC_RAG_ENABLED=1 로만 활성화
    if _FC_RAG_ENABLED:
        app.router.add_get("/api/dashboard/fc-rag/queries", handle_fc_rag_queries)
        app.router.add_get("/api/dashboard/fc-rag/stats", handle_fc_rag_stats)
        app.router.add_get("/api/dashboard/fc-rag/trace/{trace_id}", handle_fc_rag_trace)
    # Chat menu
    app.router.add_get("/api/dashboard/chat/sessions", handle_chat_sessions)
    app.router.add_get("/api/dashboard/chat/sessions/{name}", handle_chat_session_detail)
    app.router.add_post("/api/dashboard/chat/send", handle_chat_send)
