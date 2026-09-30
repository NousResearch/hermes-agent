"""AI agent probes (ai_agents family, Windows): Claude Code, Codex, Hermes, MCP inventory, NVIDIA, Ollama.

Registration only at import time. Shared helpers and memoised reads live in apps_dev.
"""
from __future__ import annotations

import collections
import datetime as dt
import glob
import json
import os
import re
import time

from ..registry import probe
from .apps_dev import (AI, _LA, _PD, _RA, _U, _appx_match, _cols, _copy_db, _file_version,
    _isdir, _iso, _ls, _memo, _mtime, _open, _q, _read_json, _redact, _rmdb, _running, _rv, _subkeys, _top,
    _uninst_match, tomllib, winreg)

# ================================================================== ai_agents

def _detect(paths):
    return [p for p in paths if p and os.path.exists(p)]


@probe(id="ai.catalog_absent_checks", level="L1", family=AI, tier="T0", collect="core")
def ai_catalog_absent_checks(h, facts):
    """Path stats for AI tools the user may lack (Cursor, Windsurf, Gemini CLI, opencode, aider, LM Studio...)."""
    U, LA, RA = _U(h), _LA(h), _RA(h)
    npm = os.path.join(RA, "npm", "node_modules")
    cat = {
        "cursor": [os.path.join(U, ".cursor"), os.path.join(RA, "Cursor"), os.path.join(LA, "Programs", "cursor")],
        "windsurf": [os.path.join(U, ".codeium"), os.path.join(U, ".windsurf"), os.path.join(RA, "Windsurf"),
                     os.path.join(LA, "Programs", "Windsurf")],
        "gemini_cli": [os.path.join(U, ".gemini"), os.path.join(npm, "@google", "gemini-cli")],
        "opencode": [os.path.join(U, ".local", "share", "opencode"), os.path.join(U, ".config", "opencode"),
                     os.path.join(npm, "opencode-ai")],
        "aider": [os.path.join(U, ".aider.conf.yml"), os.path.join(U, ".aider"), os.path.join(U, ".aider.chat.history.md")],
        "continue": [os.path.join(U, ".continue")],
        "cline": [os.path.join(U, ".cline"), os.path.join(U, "Documents", "Cline")],
        "chatgpt_desktop": [os.path.join(LA, "Programs", "ChatGPT"), os.path.join(RA, "ChatGPT")],
        "lm_studio": [os.path.join(U, ".lmstudio"), os.path.join(U, ".cache", "lm-studio"), os.path.join(RA, "LM Studio"),
                      os.path.join(LA, "Programs", "LM Studio")],
        "jan": [os.path.join(RA, "Jan"), os.path.join(U, "jan")],
        "gpt4all": [os.path.join(LA, "nomic.ai")],
        "msty": [os.path.join(RA, "Msty")],
        "anythingllm": [os.path.join(RA, "anythingllm-desktop")],
        "hf_hub": [os.path.join(U, ".cache", "huggingface", "hub")],
    }
    found = {k: len(_detect(v)) > 0 for k, v in cat.items()}
    if _appx_match(h, r"^OpenAI\.ChatGPT"):
        found["chatgpt_desktop"] = True
    hf = os.path.join(U, ".cache", "huggingface", "hub")
    hf_models = sum(1 for n in (_ls(hf) or []) if n.startswith("models--"))
    return {"present": True, "found": sorted(k for k, v in found.items() if v),
            "absent": sorted(k for k, v in found.items() if not v), "hf_hub_models": hf_models}


@probe(id="browser.automation", level="L1", family=AI, tier="T0", collect="core")
def browser_automation(h, facts):
    """Browser-automation runtimes on disk: Playwright, camoufox, agent-browser, puppeteer."""
    U, LA, RA = _U(h), _LA(h), _RA(h)
    out = {}
    for label, p in [("playwright", os.path.join(LA, "ms-playwright")), ("playwright_cache", os.path.join(U, ".cache", "ms-playwright")),
                     ("puppeteer", os.path.join(U, ".cache", "puppeteer")), ("camoufox", os.path.join(LA, "camoufox")),
                     ("agent_browser_browsers", os.path.join(U, ".agent-browser", "browsers"))]:
        names = _ls(p, 50)
        if names is not None:
            out[label] = {"entries": sorted(names)[:20], "mtime": _mtime(p)}
    installs = []
    for loc in ([os.path.join(LA, "hermes", "node", "node_modules", "agent-browser"),
                 os.path.join(RA, "npm", "node_modules", "agent-browser")]):
        pj = _read_json(os.path.join(loc, "package.json"))
        if isinstance(pj, dict):
            installs.append({"where": "hermes_bundled" if "hermes" in loc.lower() else "npm_global",
                             "version": pj.get("version")})
    if installs:
        out["agent_browser"] = installs
    return {"present": bool(out), **out}


@probe(id="browser_harness.present", level="L1", family=AI, tier="T0", collect="core")
def browser_harness_present(h, facts):
    """browser-harness config dir and runtime session pid-file count."""
    bh = os.path.join(_U(h), ".config", "browser-harness")
    if not _isdir(bh):
        return None
    pids = glob.glob(os.path.join(bh, "runtime", "*.pid"))
    return {"present": True, "runtime_sessions": len(pids[:500]), "mtime": _mtime(bh)}


# ---------------- Claude Code

def _claude_json(h):
    """Whitelisted view of ~/.claude.json. Secret-bearing keys are reduced to booleans immediately."""
    def build():
        cj = _read_json(os.path.join(_U(h), ".claude.json"), maxb=20_000_000)
        if not isinstance(cj, dict):
            return None
        projects = cj.get("projects") or {}
        out = {"numStartups": cj.get("numStartups"), "installMethod": cj.get("installMethod"),
               "firstStartTime": _iso(cj.get("firstStartTime")), "autoUpdates": cj.get("autoUpdates"),
               "hasCompletedOnboarding": cj.get("hasCompletedOnboarding"),
               "project_count": len(projects) if isinstance(projects, dict) else 0,
               "global_mcp_servers": sorted((cj.get("mcpServers") or {}).keys()),
               "project_mcp_servers": sorted({n for v in (projects.values() if isinstance(projects, dict) else [])
                                              if isinstance(v, dict) for n in (v.get("mcpServers") or {}).keys()}),
               "has_oauth_account": "oauthAccount" in cj, "has_primary_api_key": "primaryApiKey" in cj}
        del cj
        return out
    return _memo(h, "claude_json", build)


@probe(id="claude_code.present", level="L1", family=AI, tier="T0", collect="core")
def claude_code_present(h, facts):
    """Claude Code CLI: ~/.claude home and native claude.exe (with file version)."""
    U = _U(h)
    home = os.path.join(U, ".claude")
    exe = os.path.join(U, ".local", "bin", "claude.exe")
    npm = os.path.join(_RA(h), "npm", "node_modules", "@anthropic-ai", "claude-code")
    if not (_isdir(home) or os.path.exists(exe) or _isdir(npm)):
        return None
    vers = _ls(os.path.join(U, ".local", "share", "claude", "versions")) or []
    return {"present": True, "home": _isdir(home), "native_exe": os.path.exists(exe),
            "exe_version": _file_version(exe), "npm_install": _isdir(npm), "installed_versions": len(vers),
            "home_mtime": _mtime(home)}


@probe(id="claude_code.config", level="L2", family=AI, tier="T0", collect="core", gate="claude_code.present")
def claude_code_config(h, facts):
    """Claude Code settings (whitelisted keys), permission rule counts, skills/commands/agents/plugins."""
    home = os.path.join(_U(h), ".claude")
    if not _isdir(home):
        return None
    s = _read_json(os.path.join(home, "settings.json")) or {}
    sl = _read_json(os.path.join(home, "settings.local.json")) or {}
    km = _read_json(os.path.join(home, "plugins", "known_marketplaces.json")) or {}
    ip = _read_json(os.path.join(home, "plugins", "installed_plugins.json")) or {}
    plugins = (ip.get("plugins") if isinstance(ip.get("plugins"), dict) else ip) if isinstance(ip, dict) else {}
    return {"present": True, "settings_keys": sorted(s.keys()) if isinstance(s, dict) else [],
            "settings": {k: s.get(k) for k in ("model", "autoUpdatesChannel", "theme", "outputStyle")
                         if isinstance(s, dict) and isinstance(s.get(k), (str, bool, int))},
            "hooks_events": sorted((s.get("hooks") or {}).keys()) if isinstance(s, dict) else [],
            "allow_rules": len(((s.get("permissions") or {}).get("allow") or [])) + len(((sl.get("permissions") or {}).get("allow") or [])),
            "skills": len(glob.glob(os.path.join(home, "skills", "*", "SKILL.md"))),
            "commands": len(glob.glob(os.path.join(home, "commands", "*.md"))),
            "agents": len(glob.glob(os.path.join(home, "agents", "*.md"))),
            "plugin_marketplaces": sorted(km.keys()) if isinstance(km, dict) else [],
            "installed_plugins": sorted(plugins.keys())[:30] if isinstance(plugins, dict) else [],
            "claude_md": os.path.exists(os.path.join(home, "CLAUDE.md"))}


@probe(id="claude_code.claude_json", level="L2", family=AI, tier="T1", collect="core", gate="claude_code.present")
def claude_code_claude_json(h, facts):
    """~/.claude.json whitelist: numStartups, installMethod, firstStartTime, MCP names; secrets as booleans."""
    cj = _claude_json(h)
    if not cj:
        return None
    return {"present": True, **cj}


def _cc_scan(h):
    def build():
        pdir = os.path.join(_U(h), ".claude", "projects")
        r = {"projects": 0, "sessions": 0, "sessions_30d": 0, "bytes": 0, "user_msgs": 0, "assistant_msgs": 0,
             "skipped_large": 0}
        cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=30)).strftime("%Y-%m-%d")
        models, entry, versions, days = collections.Counter(), collections.Counter(), collections.Counter(), set()
        first = last = None
        titles = []
        files_seen = 0
        for pd in sorted(glob.glob(os.path.join(pdir, "*")))[:500]:
            files = glob.glob(os.path.join(pd, "*.jsonl"))
            if not files:
                continue
            r["projects"] += 1
            for f in files:
                files_seen += 1
                if files_seen > 5000:
                    break
                r["sessions"] += 1
                try:
                    sz = os.path.getsize(f)
                except OSError:
                    continue
                r["bytes"] += sz
                if sz > 60_000_000:
                    r["skipped_large"] += 1
                    continue
                title = None
                flast = None
                with open(f, encoding="utf-8", errors="replace") as fh:
                    for n, line in enumerate(fh):
                        if n > 200_000:
                            break
                        try:
                            j = json.loads(line)
                        except ValueError:
                            continue
                        if not isinstance(j, dict):
                            continue
                        t = j.get("type")
                        ts = j.get("timestamp")
                        if isinstance(ts, str):
                            flast = ts if flast is None or ts > flast else flast
                            first = ts if first is None or ts < first else first
                            last = ts if last is None or ts > last else last
                            days.add(ts[:10])
                        if j.get("entrypoint"):
                            entry[j["entrypoint"]] += 1
                        if j.get("version"):
                            versions[j["version"]] += 1
                        if t == "user" and not j.get("isMeta") and not j.get("toolUseResult"):
                            r["user_msgs"] += 1
                        elif t == "assistant":
                            r["assistant_msgs"] += 1
                            m = (j.get("message") or {}).get("model") if isinstance(j.get("message"), dict) else None
                            if m:
                                models[m] += 1
                        elif t == "custom-title" and j.get("customTitle"):
                            title = j["customTitle"]
                        elif t == "ai-title" and j.get("aiTitle") and not title:
                            title = j["aiTitle"]
                        elif t == "summary" and j.get("summary") and not title:
                            title = j["summary"]
                if flast and flast[:10] >= cutoff:
                    r["sessions_30d"] += 1
                if title:
                    titles.append((os.path.getmtime(f), title))
        hist = os.path.join(_U(h), ".claude", "history.jsonl")
        hinfo = None
        if os.path.exists(hist) and os.path.getsize(hist) < 50_000_000:
            hts = []
            with open(hist, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    try:
                        v = json.loads(line).get("timestamp")
                    except (ValueError, AttributeError):
                        continue
                    if v:
                        hts.append(v)
            hinfo = {"entries": len(hts), "first": _iso(min(hts)) if hts else None, "last": _iso(max(hts)) if hts else None}
        r.update({"first": _iso(first), "last": _iso(last), "active_days": len(days), "models": _top(models),
                  "entrypoints": _top(entry), "cli_versions": _top(versions, 5), "prompt_history": hinfo})
        return {"agg": r, "titles": [t for _, t in sorted(titles, reverse=True)]}
    return _memo(h, "cc_scan", build)


@probe(id="claude_code.sessions", level="L2", family=AI, tier="T1", collect="extended", gate="claude_code.present",
       timeout_ms=5000)
def claude_code_sessions(h, facts):
    """Claude Code JSONL aggregates: session/message counts, date range, active days, models, entrypoints."""
    if not _isdir(os.path.join(_U(h), ".claude", "projects")):
        return None
    agg = _cc_scan(h)["agg"]
    return {"present": agg["sessions"] > 0, **agg}


@probe(id="claude_code.titles", level="L2", family=AI, tier="T2", collect="deep", gate="claude_code.present")
def claude_code_titles(h, facts):
    """Most recent Claude Code session titles (redacted, truncated). T2 content, opt-in."""
    if not _isdir(os.path.join(_U(h), ".claude", "projects")):
        return None
    t = _cc_scan(h)["titles"]
    return {"present": bool(t), "count_with_title": len(t), "recent": [_redact(x) for x in t[:5]]}


@probe(id="claude_desktop.present", level="L1", family=AI, tier="T0", collect="core")
def claude_desktop_present(h, facts):
    """Claude desktop app: installed now, or evidence of a past install (Claude-3p dir)."""
    LA, RA = _LA(h), _RA(h)
    uninst = [i["version"] for i in _uninst_match(h, r"^Claude\b")]
    appx = _appx_match(h, r"Claude|Anthropic")
    dirs = {"roaming_Claude": _isdir(os.path.join(RA, "Claude")),
            "local_AnthropicClaude": _isdir(os.path.join(LA, "AnthropicClaude")),
            "local_Claude-3p": _isdir(os.path.join(LA, "Claude-3p")),
            "programdata_Claude": _isdir(os.path.join(_PD(), "Claude"))}
    installed = bool(uninst or appx or dirs["local_AnthropicClaude"] or dirs["roaming_Claude"])
    cfg = _read_json(os.path.join(RA, "Claude", "claude_desktop_config.json"))
    mcp = sorted((cfg.get("mcpServers") or {}).keys()) if isinstance(cfg, dict) else []
    past = dirs["local_Claude-3p"] or dirs["programdata_Claude"]
    if not (installed or past):
        return None
    return {"present": True, "installed": installed, "past_install_evidence": bool(past and not installed),
            "uninstall_versions": uninst, "appx": [f"{p['name']} {p['version']}" for p in appx],
            "dirs": dirs, "mcp_servers": mcp, "claude_3p_mtime": _mtime(os.path.join(LA, "Claude-3p"))}


# ---------------- Codex

def _codex_home(h):
    return os.environ.get("CODEX_HOME") or os.path.join(_U(h), ".codex")


@probe(id="codex.present", level="L1", family=AI, tier="T0", collect="core")
def codex_present(h, facts):
    """OpenAI Codex: ~/.codex home and Codex Desktop MSIX (appx repository)."""
    home = _codex_home(h)
    appx = _appx_match(h, r"^OpenAI\.Codex")
    bindir = os.path.join(_LA(h), "OpenAI", "Codex", "bin")
    npm = _isdir(os.path.join(_RA(h), "npm", "node_modules", "@openai", "codex"))
    if not (_isdir(home) or appx or npm):
        return None
    return {"present": True, "home": _isdir(home), "desktop_appx": [{"version": p["version"], "arch": p["arch"]} for p in appx],
            "bundled_cli_builds": len(_ls(bindir) or []), "npm_cli": npm, "home_mtime": _mtime(home)}


@probe(id="codex.auth_presence", level="L1", family=AI, tier="T3", collect="core")
def codex_auth_presence(h, facts):
    """Codex auth.json presence/size/mtime (never opened). Presence means signed in."""
    return h.meta(os.path.join(_codex_home(h), "auth.json"))


@probe(id="codex.config", level="L2", family=AI, tier="T0", collect="core", gate="codex.present")
def codex_config(h, facts):
    """Codex config.toml whitelist: model, provider, sandbox, MCP/plugin names. MCP env tables excluded."""
    p = os.path.join(_codex_home(h), "config.toml")
    if not os.path.exists(p) or tomllib is None:
        return None
    try:
        with open(p, "rb") as f:
            c = tomllib.load(f)
    except Exception as e:
        return {"present": True, "error": type(e).__name__}
    return {"present": True, "model": c.get("model"), "model_provider": c.get("model_provider"),
            "model_reasoning_effort": c.get("model_reasoning_effort"), "approval_policy": c.get("approval_policy"),
            "sandbox_mode": c.get("sandbox_mode"), "windows_sandbox": (c.get("windows") or {}).get("sandbox"),
            "profiles": sorted((c.get("profiles") or {}).keys()),
            "model_providers": sorted((c.get("model_providers") or {}).keys()),
            "mcp_servers": sorted((c.get("mcp_servers") or {}).keys()),
            "plugins_enabled": sorted(k for k, v in (c.get("plugins") or {}).items() if isinstance(v, dict) and v.get("enabled")),
            "trusted_projects": len(c.get("projects") or {})}


@probe(id="codex.usage", level="L2", family=AI, tier="T1", collect="extended", gate="codex.present", timeout_ms=5000)
def codex_usage(h, facts):
    """Codex rollouts (count/bytes/date range/turn models) + state_*.sqlite thread counts + turn durations."""
    home = _codex_home(h)
    if not _isdir(home):
        return None
    files = sorted(glob.glob(os.path.join(home, "sessions", "*", "*", "*", "rollout-*.jsonl")) +
                   glob.glob(os.path.join(home, "archived_sessions", "*.jsonl")))[:3000]
    models, efforts, orig, clis = (collections.Counter() for _ in range(4))
    dates, b, turns = [], 0, 0
    for f in files:
        try:
            b += os.path.getsize(f)
        except OSError:
            continue
        m = re.search(r"rollout-(\d{4}-\d\d-\d\dT\d\d-\d\d-\d\d)", f)
        if m:
            dates.append(m.group(1))
        if os.path.getsize(f) > 80_000_000:
            continue
        with open(f, encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                head = line[:120]
                if i == 0 and '"session_meta"' in head:
                    try:
                        p = json.loads(line)["payload"]
                        orig[p.get("originator")] += 1; clis[p.get("cli_version")] += 1
                    except Exception:
                        pass
                elif '"turn_context"' in head:
                    turns += 1
                    try:
                        p = json.loads(line)["payload"]
                        models[p.get("model")] += 1
                        efforts[p.get("effort") or p.get("reasoning_effort")] += 1
                    except Exception:
                        pass
    dates.sort()
    cutoff = (dt.datetime.now() - dt.timedelta(days=30)).strftime("%Y-%m-%d")
    out = {"present": True, "sessions_30d": sum(1 for d in dates if d[:10] >= cutoff),
           "rollouts": {"files": len(files), "bytes": b, "first": dates[0] if dates else None,
                        "last": dates[-1] if dates else None, "turn_contexts": turns, "models": _top(models),
                        "efforts": _top(efforts), "originators": _top(orig), "cli_versions": _top(clis, 5)}}
    sdb = sorted(glob.glob(os.path.join(home, "state_*.sqlite")), key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1)))
    if sdb:
        dst = _copy_db(h, sdb[-1], "codex-state")
        if dst:
            try:
                cs = _cols(dst, "threads")
                by_src = collections.Counter()
                for (s,) in _q(dst, "select source from threads"):
                    if isinstance(s, str) and s.startswith("{"):
                        try:
                            s = "subagent" if "subagent" in s.lower() else next(iter(json.loads(s)))
                        except Exception:
                            s = "object"
                    by_src[s] += 1
                a, z = _q(dst, "select min(created_at), max(updated_at) from threads")[0]
                out["threads"] = {"count": sum(by_src.values()), "first": _iso(a), "last": _iso(z), "by_source": dict(by_src),
                                  "by_provider": dict(_q(dst, "select model_provider, count(*) from threads group by 1")) if "model_provider" in cs else None,
                                  "by_model": dict(_q(dst, "select model, count(*) from threads group by 1")) if "model" in cs else None,
                                  "tokens_used_total": _q(dst, "select sum(tokens_used) from threads")[0][0] if "tokens_used" in cs else None}
                try:
                    out["threads"]["subagent_edges"] = _q(dst, "select count(*) from thread_spawn_edges")[0][0]
                except Exception:
                    pass
            except Exception as e:
                out["threads"] = {"error": type(e).__name__}
            finally:
                _rmdb(dst)
    th = sorted(glob.glob(os.path.join(home, "thread_history_*.sqlite")))
    if th:
        dst = _copy_db(h, th[-1], "codex-th")
        if dst:
            try:
                out["turns"] = {"count": _q(dst, "select count(*) from thread_turns")[0][0],
                                "agent_minutes": round((_q(dst, "select sum(duration_ms) from thread_turns")[0][0] or 0) / 60000, 1),
                                "items": _q(dst, "select count(*) from thread_items")[0][0]}
            except Exception as e:
                out["turns"] = {"error": type(e).__name__}
            finally:
                _rmdb(dst)
    return out


@probe(id="codex.extras", level="L2", family=AI, tier="T1", collect="extended", gate="codex.present")
def codex_extras(h, facts):
    """Codex skill/plugin counts, dictation-session count, logs db row count and range."""
    home = _codex_home(h)
    if not _isdir(home):
        return None
    sk = os.path.join(home, "skills")
    pc = os.path.join(home, "plugins", "cache")
    out = {"present": True,
           "skills_system": len(glob.glob(os.path.join(sk, ".system", "*", "SKILL.md"))),
           "skills_user": len(glob.glob(os.path.join(sk, "*", "SKILL.md"))),
           "plugin_cache": sum(len(_ls(m) or []) for m in glob.glob(os.path.join(pc, "*"))),
           "dictation_sessions": len(_ls(os.path.join(home, "dictation-history"), 5000) or []),
           "agents_md": os.path.exists(os.path.join(home, "AGENTS.md"))}
    lg = sorted(glob.glob(os.path.join(home, "logs_*.sqlite")))
    if lg:
        dst = _copy_db(h, lg[-1], "codex-logs")
        if dst:
            try:
                n, a, z = _q(dst, "select count(*), min(ts), max(ts) from logs")[0]
                out["logs_db"] = {"rows": n, "first": _iso(a), "last": _iso(z)}
            except Exception as e:
                out["logs_db"] = {"error": type(e).__name__}
            finally:
                _rmdb(dst)
    return out


@probe(id="codex.chatgpt_catalog", level="L2", family=AI, tier="T2", collect="deep", gate="codex.present")
def codex_chatgpt_catalog(h, facts):
    """ChatGPT cloud conversations visible to Codex: count/date range by host kind + recent titles (T2)."""
    dev = os.path.join(_codex_home(h), "sqlite", "codex-dev.db")
    if not os.path.exists(dev):
        return None
    dst = _copy_db(h, dev, "codex-dev")
    if not dst:
        return None
    try:
        hosts = dict(_q(dst, "select host_id, host_kind from local_thread_catalog_hosts"))
        rows = _q(dst, "select host_id, source_kind, count(*), min(source_created_at), max(source_created_at) from local_thread_catalog group by 1,2")
        cat = [{"host_kind": hosts.get(hid, "?"), "source_kind": sk, "count": n, "first": _iso(a), "last": _iso(z)}
               for hid, sk, n, a, z in rows]
        titles = [_redact(t) for (t,) in _q(dst, "select display_title from local_thread_catalog where source_kind='chatgpt' order by source_updated_at desc limit 5")]
    except Exception as e:
        return {"present": True, "error": type(e).__name__}
    finally:
        _rmdb(dst)
    return {"present": bool(cat), "catalog": cat, "recent_titles": titles}


@probe(id="consent.system_ai_models", level="L1", family=AI, tier="T1", collect="core")
def consent_system_ai_models(h, facts):
    """CapabilityAccessManager systemAIModels consent: global value and apps that used on-device AI models."""
    base = r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\systemAIModels"
    out = {}
    for label, hive in (("HKCU", winreg.HKEY_CURRENT_USER), ("HKLM", winreg.HKEY_LOCAL_MACHINE)):
        k = _open(hive, base)
        if k is None:
            continue
        apps = used = 0
        for sk in list(_subkeys(k, 500)):
            if sk == "NonPackaged":
                nk = _open(hive, base + "\\NonPackaged")
                subs = ["NonPackaged\\" + s for s in (list(_subkeys(nk, 500)) if nk else [])]
            else:
                subs = [sk]
            for s in subs:
                ak = _open(hive, base + "\\" + s)
                if ak is None:
                    continue
                apps += 1
                if (_rv(ak, "LastUsedTimeStart") or 0) > 0:
                    used += 1
        out[label] = {"value": _rv(k, "Value"), "apps": apps, "apps_used": used}
    if not out:
        return None
    return {"present": True, **out}


@probe(id="copilot_cli.present", level="L1", family=AI, tier="T0", collect="core")
def copilot_cli_present(h, facts):
    """GitHub Copilot CLI: ~/.copilot, firstLaunchAt, log/session counts, MCP server names."""
    home = os.path.join(_U(h), ".copilot")
    if not _isdir(home):
        return None
    c = _read_json(os.path.join(home, "config.json"), jsonc=True) or {}
    mc = _read_json(os.path.join(home, "mcp-config.json"), jsonc=True) or {}
    return {"present": True, "first_launch": _iso(c.get("firstLaunchAt")) if isinstance(c, dict) else None,
            "logs": len(_ls(os.path.join(home, "logs")) or []),
            "sessions": len(_ls(os.path.join(home, "session-state")) or []) + len(_ls(os.path.join(home, "history-session-state")) or []),
            "mcp_servers": sorted((mc.get("mcpServers") or {}).keys()) if isinstance(mc, dict) else []}


@probe(id="copilot_m365.present", level="L1", family=AI, tier="T0", collect="core")
def copilot_m365_present(h, facts):
    """Microsoft 365 Copilot (OfficeHub) and Windows Copilot appx + running process."""
    m365 = _appx_match(h, r"^Microsoft\.MicrosoftOfficeHub$")
    win = _appx_match(h, r"^Microsoft\.Copilot$")
    run = _running(h, r"^m365copilot\.exe$|^copilot\.exe$")
    if not (m365 or win):
        return None
    return {"present": True, "m365_version": m365[0]["version"] if m365 else None,
            "windows_copilot_version": win[0]["version"] if win else None, "running": run}


@probe(id="cua_driver.present", level="L1", family=AI, tier="T0", collect="core")
def cua_driver_present(h, facts):
    """cua-driver (computer-use): concrete release dirs (junctions fail over ssh), version check, running."""
    home = os.path.join(_U(h), ".cua-driver")
    if not _isdir(home) and not _isdir(os.path.join(_LA(h), "Programs", "Cua")):
        return None
    rel = sorted(_ls(os.path.join(home, "packages", "releases")) or [])
    vc = _read_json(os.path.join(home, "version_check.json")) or {}
    return {"present": True, "releases": rel[-5:], "releases_installed": sorted(_ls(os.path.join(home, ".release_installed")) or [])[-5:],
            "first_install": _mtime(os.path.join(home, ".installation_recorded")),
            "latest_available": vc.get("latest_version") if isinstance(vc, dict) else None,
            "running": bool(_running(h, r"^cua-driver"))}


@probe(id="docker.model_runner", level="L1", family=AI, tier="T0", collect="core")
def docker_model_runner(h, facts):
    """Docker Model Runner models.json model count."""
    dm = _read_json(os.path.join(_U(h), ".docker", "models", "models.json"))
    if dm is None:
        return None
    lst = dm.get("models", []) if isinstance(dm, dict) else dm
    return {"present": True, "models": len(lst) if isinstance(lst, (list, dict)) else 0}


# ---------------- Hermes

def _hermes_homes(h):
    def build():
        env = h.l0.get("hermes_home") or os.environ.get("HERMES_HOME") or ""
        LA, U = _LA(h), _U(h)
        cands = []
        for role, p in (("HERMES_HOME", env), ("localappdata", os.path.join(LA, "hermes")),
                        ("dot_hermes", os.path.join(U, ".hermes"))):
            if p and _isdir(p) and all(os.path.normcase(p) != os.path.normcase(c["path"]) for c in cands):
                cands.append({"role": role, "path": p, "has_state": os.path.exists(os.path.join(p, "state.db"))})
        primary = next((c for c in cands if c["role"] == "HERMES_HOME"), None) or \
            next((c for c in cands if c["has_state"]), None)
        side = [p for p in glob.glob(os.path.join(LA, "hermes-*")) if _isdir(p)][:50]
        return {"env_set": bool(env), "cands": cands, "primary": primary, "side": side}
    return _memo(h, "hermes_homes", build)


@probe(id="hermes.present", level="L1", family=AI, tier="T0", collect="core")
def hermes_present(h, facts):
    """Hermes Agent: HERMES_HOME/default homes, MSIX builds, desktop userData. Side homes flagged as operator."""
    hh = _hermes_homes(h)
    appx = _appx_match(h, r"^NousResearch\.|Hermes")
    ud = glob.glob(os.path.join(_RA(h), "Hermes*"))
    homes = []
    for c in hh["cands"]:
        p = c["path"]
        im = None
        try:
            with open(os.path.join(p, ".install_method"), encoding="utf-8", errors="replace") as f:
                im = f.read(40).strip()
        except OSError:
            pass
        ver = None
        pp = os.path.join(p, "hermes-agent", "pyproject.toml")
        if tomllib and os.path.exists(pp):
            try:
                with open(pp, "rb") as f:
                    ver = tomllib.load(f).get("project", {}).get("version")
            except Exception:
                pass
        homes.append({"role": c["role"], "primary": hh["primary"] is c, "state_db_bytes": (h.meta(os.path.join(p, "state.db")).get("bytes")),
                      "mtime": _mtime(p), "install_method": im, "agent_version": ver,
                      "git_checkout": _isdir(os.path.join(p, "hermes-agent", ".git"))})
    if not (homes or appx or ud or hh["side"]):
        return None
    return {"present": True, "HERMES_HOME_set": hh["env_set"], "user_home": hh["primary"] is not None,
            "homes": homes, "side_homes_operator": len(hh["side"]),
            "appx": [{"name": p["name"], "version": p["version"], "arch": p["arch"]} for p in appx],
            "desktop_userdata_dirs": len(ud), "running": sum(_running(h, r"^hermes").values())}


@probe(id="hermes.auth_presence", level="L1", family=AI, tier="T3", collect="core")
def hermes_auth_presence(h, facts):
    """Primary Hermes home auth.json and .env presence/size (never opened)."""
    p = _hermes_homes(h)["primary"]
    if not p:
        return None
    a, e = h.meta(os.path.join(p["path"], "auth.json")), h.meta(os.path.join(p["path"], ".env"))
    return {"present": a["present"] or e["present"], "auth_json": a, "env": e}


def _parse_hermes_yaml(p):
    out = {"model": {}, "toolsets": [], "mcp_servers": [], "platform_toolsets": [], "top_level_keys": 0}
    cur = None
    with open(p, encoding="utf-8", errors="replace") as fh:
        for n, raw in enumerate(fh):
            if n > 5000:
                break
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            ind = len(line) - len(line.lstrip(" "))
            s = line.strip()
            if ind == 0:
                out["top_level_keys"] += 1
                cur = s.split(":", 1)[0]
                rest = s.split(":", 1)[1].strip() if ":" in s else ""
                if cur == "model" and rest and rest != "{}":
                    out["model"]["default"] = rest.strip("'\"")
                continue
            if cur == "model" and ind == 2 and ":" in s:
                k, v = s.split(":", 1)
                v = v.strip().strip("'\"")
                if k == "base_url":
                    m = re.match(r"https?://([^/:]+)", v)
                    host = m.group(1) if m else ""
                    v = "local" if re.match(r"^(127\.|localhost|0\.0\.0\.0|192\.168\.|10\.)", host) else host
                if k in ("default", "provider", "base_url", "api_mode"):
                    out["model"][k] = v
            elif cur == "toolsets" and s.startswith("- "):
                out["toolsets"].append(s[2:].strip("'\""))
            elif cur == "mcp_servers" and ind == 2 and s.endswith(":"):
                out["mcp_servers"].append(s[:-1].strip("'\""))
            elif cur == "platform_toolsets" and ind == 2 and ":" in s:
                out["platform_toolsets"].append(s.split(":", 1)[0])
    return out


@probe(id="hermes.config", level="L2", family=AI, tier="T0", collect="core", gate="hermes.present")
def hermes_config(h, facts):
    """Primary Hermes config.yaml whitelist: model/provider, base_url host kind, toolsets, MCP names. .env never read."""
    p = _hermes_homes(h)["primary"]
    if not p or not os.path.exists(os.path.join(p["path"], "config.yaml")):
        return None
    try:
        return {"present": True, **_parse_hermes_yaml(os.path.join(p["path"], "config.yaml"))}
    except OSError:
        return None


@probe(id="hermes.skills", level="L2", family=AI, tier="T1", collect="extended", gate="hermes.present")
def hermes_skills(h, facts):
    """Primary Hermes skills: count, categories, archived; top-used names only if in the bundled catalog."""
    p = _hermes_homes(h)["primary"]
    if not p:
        return None
    sk = os.path.join(p["path"], "skills")
    if not _isdir(sk):
        return None
    total, cats = 0, set()
    for dp, dns, fns in os.walk(sk):
        rel = os.path.relpath(dp, sk)
        dns[:] = [d for d in dns if not d.startswith(".")] if rel.count(os.sep) < 4 else []
        if "SKILL.md" in fns:
            total += 1
            if rel != ".":
                cats.add(rel.split(os.sep)[0])
        if total > 5000:
            break
    catalog = set()
    for base in ("skills", "optional-skills"):
        for f in glob.glob(os.path.join(p["path"], "hermes-agent", base, "**", "SKILL.md"), recursive=True)[:3000]:
            catalog.add(os.path.basename(os.path.dirname(f)))
    top, top_plugin, tracked, used = [], [], 0, 0
    u = _read_json(os.path.join(sk, ".usage.json"))
    if isinstance(u, dict):
        items = u.get("skills", u) if isinstance(u.get("skills", u), dict) else {}
        tracked = len(items)

        def score(v):
            if isinstance(v, dict):
                return v.get("use_count") or v.get("count") or v.get("uses") or 0
            return v if isinstance(v, (int, float)) else 0
        ranked = [k for k, v in sorted(items.items(), key=lambda kv: -score(kv[1])) if score(v) > 0]
        top = [k for k in ranked if k in catalog][:5]
        top_plugin = [k.split(":", 1)[1] for k in ranked if ":" in k][:5]
        used = len(ranked)
    cats = {d for d in (_ls(sk) or []) if not d.startswith(".") and _isdir(os.path.join(sk, d))}
    return {"present": True, "skills": total, "categories": len(cats), "skills_used": used,
            "top_used_plugin": top_plugin,
            "archived": len(glob.glob(os.path.join(sk, ".archive", "*", "SKILL.md"))),
            "usage_tracked": tracked, "top_used_catalog": top, "catalog_available": bool(catalog),
            "plugins": len(_ls(os.path.join(p["path"], "plugins")) or [])}


def _hermes_db_summary(h, home, tag, titles=False):
    st = os.path.join(home, "state.db")
    if not os.path.exists(st):
        return None
    dst = _copy_db(h, st, tag)
    if not dst:
        return {"error": "copy_failed"}
    try:
        cs = _cols(dst, "sessions")
        if titles:
            if "title" not in cs:
                return {"titles": []}
            rows = _q(dst, "select title from sessions where title is not null and title != '' "
                           + ("and parent_session_id is null " if "parent_session_id" in cs else "")
                           + "order by started_at desc limit 5")
            return {"titles": [_redact(t) for (t,) in rows]}
        a, z = _q(dst, "select min(started_at), max(coalesce(ended_at, started_at)) from sessions")[0]
        r = {"sessions": _q(dst, "select count(*) from sessions")[0][0],
             "messages": _q(dst, "select count(*) from messages")[0][0],
             "first": _iso(a), "last": _iso(z),
             "by_source": dict(_q(dst, "select coalesce(source,'?'), count(*) from sessions group by 1")),
             "by_month": dict(_q(dst, "select strftime('%Y-%m', started_at, 'unixepoch'), count(*) from sessions group by 1")),
             "by_model": dict(_q(dst, "select coalesce(model,'?'), count(*) from sessions group by 1 order by 2 desc limit 10"))}
        r["sessions_30d"] = _q(dst, "select count(*) from sessions where started_at >= ?", (time.time() - 30 * 86400,))[0][0]
        if "parent_session_id" in cs:
            r["subagent_sessions"] = _q(dst, "select count(*) from sessions where parent_session_id is not null")[0][0]
        if "tool_call_count" in cs:
            r["tool_calls"] = _q(dst, "select sum(tool_call_count) from sessions")[0][0]
        if "estimated_cost_usd" in cs:
            r["est_cost_usd"] = _q(dst, "select round(sum(estimated_cost_usd), 2) from sessions")[0][0]
        try:
            r["model_calls"] = [{"model": m, "provider": p, "calls": c} for m, p, c in
                                _q(dst, "select model, billing_provider, sum(api_call_count) from session_model_usage group by 1,2 order by 3 desc limit 10")]
        except Exception:
            pass
        return r
    except Exception as e:
        return {"error": type(e).__name__}
    finally:
        _rmdb(dst)


@probe(id="hermes.usage", level="L2", family=AI, tier="T1", collect="extended", gate="hermes.present", timeout_ms=5000)
def hermes_usage(h, facts):
    """Primary Hermes state.db (copied with -wal): session/message/tool-call counts, date range, sources, models."""
    hh = _hermes_homes(h)
    p = hh["primary"]
    if not p:
        return None
    r = _hermes_db_summary(h, p["path"], "hermes-state")
    if r is None:
        return None
    prof = []
    for db in glob.glob(os.path.join(p["path"], "profiles", "*", "state.db"))[:20]:
        s = _hermes_db_summary(h, os.path.dirname(db), "hermes-prof") or {}
        prof.append({"sessions": s.get("sessions"), "first": s.get("first"), "last": s.get("last")})
    others = []
    for c in hh["cands"]:
        if c is not p and c["has_state"]:
            s = _hermes_db_summary(h, c["path"], "hermes-other") or {}
            others.append({"role": c["role"], "sessions": s.get("sessions"), "last": s.get("last")})
    return {"present": True, "home_role": p["role"], **r, "profiles": prof, "other_homes": others,
            "side_homes_excluded": len(hh["side"])}


@probe(id="hermes.titles", level="L2", family=AI, tier="T2", collect="deep", gate="hermes.present")
def hermes_titles(h, facts):
    """Most recent top-level Hermes session titles (redacted). T2 content, opt-in."""
    p = _hermes_homes(h)["primary"]
    if not p:
        return None
    r = _hermes_db_summary(h, p["path"], "hermes-titles", titles=True)
    if not r or not r.get("titles"):
        return None
    return {"present": True, "recent": r["titles"]}


# ---------------- MCP inventory, NVIDIA, Ollama

BUNDLED_MCP = {"node_repl"}


@probe(id="l3.mcp_inventory", level="L2", family=AI, tier="T0", collect="core", gate="ai.catalog_absent_checks")
def l3_mcp_inventory(h, facts):
    """Union of whitelisted MCP server names across agents; vendor-bundled servers separated."""
    U, RA = _U(h), _RA(h)
    by = {}
    cj = _claude_json(h)
    if cj:
        by["claude_code"] = sorted(set(cj["global_mcp_servers"]) | set(cj["project_mcp_servers"]))
    cp = os.path.join(_codex_home(h), "config.toml")
    if tomllib and os.path.exists(cp):
        try:
            with open(cp, "rb") as f:
                by["codex"] = sorted((tomllib.load(f).get("mcp_servers") or {}).keys())
        except Exception:
            pass
    for label, path, keys in [("vscode", os.path.join(RA, "Code", "User", "mcp.json"), ("servers", "mcpServers")),
                              ("claude_desktop", os.path.join(RA, "Claude", "claude_desktop_config.json"), ("mcpServers",)),
                              ("cursor", os.path.join(U, ".cursor", "mcp.json"), ("mcpServers",)),
                              ("copilot_cli", os.path.join(U, ".copilot", "mcp-config.json"), ("mcpServers",)),
                              ("gemini_cli", os.path.join(U, ".gemini", "settings.json"), ("mcpServers",)),
                              ("windsurf", os.path.join(U, ".codeium", "windsurf", "mcp_config.json"), ("mcpServers",))]:
        j = _read_json(path, jsonc=True)
        if isinstance(j, dict):
            for k in keys:
                if isinstance(j.get(k), dict):
                    by[label] = sorted(j[k].keys()); break
    p = _hermes_homes(h)["primary"]
    if p and os.path.exists(os.path.join(p["path"], "config.yaml")):
        try:
            by["hermes"] = _parse_hermes_yaml(os.path.join(p["path"], "config.yaml"))["mcp_servers"]
        except OSError:
            pass
    user = sorted({n for v in by.values() for n in v if n not in BUNDLED_MCP})
    bundled = sorted({n for v in by.values() for n in v if n in BUNDLED_MCP})
    nv = _isdir(os.path.join(_LA(h), "NVIDIA Corporation", "NVIDIA App", "McpServer"))
    return {"present": True, "by_agent": {k: v for k, v in by.items() if v}, "user_configured": user,
            "user_configured_total": len(user), "vendor_bundled": bundled + (["nvidia_app_mcp_server"] if nv else [])}


@probe(id="nvidia.app_mcp_server", level="L1", family=AI, tier="T0", collect="core")
def nvidia_app_mcp_server(h, facts):
    """NVIDIA App McpServer dir + server.json key names and non-secret fields (token never emitted)."""
    mcp = os.path.join(_LA(h), "NVIDIA Corporation", "NVIDIA App", "McpServer")
    if not _isdir(mcp):
        return None
    sj = _read_json(os.path.join(mcp, "server.json"))
    keys, info = [], {}
    if isinstance(sj, dict):
        keys = sorted(sj.keys())
        info = {k: sj[k] for k in ("name", "title", "version") if isinstance(sj.get(k), str)}
        del sj
    return {"present": True, "mtime": _mtime(mcp), "files": len(_ls(mcp) or []), "server_json_keys": keys,
            "has_token_key": "token" in keys, **info}


def _ollama_models_root(h):
    return os.environ.get("OLLAMA_MODELS") or os.path.join(_U(h), ".ollama", "models")


@probe(id="ollama.present", level="L1", family=AI, tier="T0", collect="core")
def ollama_present(h, facts):
    """Ollama install (per-user exe, uninstall version), models dir, running process."""
    exe = os.path.join(_LA(h), "Programs", "Ollama", "ollama.exe")
    mroot = _ollama_models_root(h)
    if not (os.path.exists(exe) or _isdir(mroot) or _isdir(os.path.join(_U(h), ".ollama"))):
        return None
    un = _uninst_match(h, r"^Ollama")
    return {"present": True, "exe": os.path.exists(exe), "version": un[0]["version"] if un else _file_version(exe),
            "install_date": un[0]["install_date"] if un else None, "models_dir": _isdir(mroot),
            "OLLAMA_MODELS_set": bool(os.environ.get("OLLAMA_MODELS")), "running": _running(h, r"^ollama")}


@probe(id="ollama.models", level="L2", family=AI, tier="T0", collect="core", gate="ollama.present")
def ollama_models(h, facts):
    """Ollama model manifests (name, size, pulled date) and blob count."""
    mroot = _ollama_models_root(h)
    man = os.path.join(mroot, "manifests")
    models = []
    for f in glob.glob(os.path.join(man, "*", "*", "*", "*"))[:500]:
        if not os.path.isfile(f):
            continue
        parts = f[len(man) + 1:].split(os.sep)
        j = _read_json(f, maxb=1_000_000) or {}
        size = sum(l.get("size", 0) for l in j.get("layers", []) if isinstance(l, dict)) + ((j.get("config") or {}).get("size", 0) or 0)
        name = ("" if parts[1] == "library" else parts[1] + "/") + parts[2] + ":" + parts[3]
        if parts[0] != "registry.ollama.ai":
            name = parts[0] + "/" + name
        models.append({"name": name, "gb": round(size / 1e9, 2), "pulled": _mtime(f)})
    blobs = _ls(os.path.join(mroot, "blobs"), 20000)
    return {"present": True, "count": len(models), "total_gb": round(sum(m["gb"] for m in models), 2),
            "models": sorted(models, key=lambda m: -m["gb"])[:30], "blobs": len(blobs) if blobs is not None else None,
            "manifests_mtime": _mtime(man)}


@probe(id="ollama.app_db", level="L2", family=AI, tier="T1", collect="extended", gate="ollama.present")
def ollama_app_db(h, facts):
    """Ollama app db.sqlite: chat/message counts, signed-in flag (users row count only), first-run flags."""
    db = os.path.join(_LA(h), "Ollama", "db.sqlite")
    if not os.path.exists(db):
        return None
    dst = _copy_db(h, db, "ollama")
    if not dst:
        return None
    try:
        out = {"present": True}
        for t in ("chats", "messages"):
            try:
                out[t] = _q(dst, f"select count(*) from {t}")[0][0]
            except Exception:
                out[t] = None
        try:
            out["signed_in"] = _q(dst, "select count(*) from users")[0][0] > 0
        except Exception:
            out["signed_in"] = None
        cs = _cols(dst, "settings")
        want = [c for c in ("has_completed_first_run", "claude_desktop_used", "codex_desktop_used", "turbo_enabled",
                            "websearch_enabled", "airplane_mode", "expose") if c in cs]
        if want:
            row = _q(dst, "select %s from settings limit 1" % ",".join(want))
            out["settings"] = dict(zip(want, row[0])) if row else None
        return out
    except Exception as e:
        return {"present": True, "error": type(e).__name__}
    finally:
        _rmdb(dst)


@probe(id="ollama.logs", level="L2", family=AI, tier="T1", collect="extended", gate="ollama.present")
def ollama_logs(h, facts):
    """Ollama server/app logs: API call counts by path, active days, first/last; decides installed-but-unused."""
    ld = os.path.join(_LA(h), "Ollama")
    files = glob.glob(os.path.join(ld, "*.log"))[:20]
    if not files:
        return None
    rx = re.compile(r'\[GIN\] (\d{4}/\d\d/\d\d) - (\d\d:\d\d:\d\d) \|.*\|\s+(GET|POST|HEAD|DELETE)\s+"([^"?]+)')
    api, days, ver = collections.Counter(), set(), set()
    first = last = None
    for lf in files:
        try:
            with open(lf, encoding="utf-8", errors="replace") as fh:
                for n, line in enumerate(fh):
                    if n > 500_000:
                        break
                    m = rx.search(line)
                    if m:
                        d = m.group(1) + " " + m.group(2)
                        days.add(m.group(1))
                        first = d if first is None or d < first else first
                        last = d if last is None or d > last else last
                        api[m.group(3) + " " + m.group(4)] += 1
                    v = re.search(r"version=(\d+\.\d+\.\d+)", line)
                    if v:
                        ver.add(v.group(1))
        except OSError:
            continue
    infer = sum(v for k, v in api.items() if re.search(r"/api/(chat|generate|embed)|/v1/(chat|completions)", k))
    return {"present": True, "files": len(files), "api_calls": _top(api, 12), "inference_calls": infer,
            "first": first, "last": last, "active_days": len(days), "versions_logged": sorted(ver)}
