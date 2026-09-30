"""L3 insights: persona derivations (reports/14-triage.md section 3), the fresh-vs-old classifier
(section 4) and the wave-2 derivations from reports/13-gap-hunt.md.

Every rule is pure: it reads only the facts dict ({probe_id: value} for probes with status ok)
and returns {"claim", "strength", "value"} or None. Probe value shapes differ between spec modules,
so the accessors below accept several key spellings and fall back to counting lists.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone

from ..registry import insight

OPERATOR_TOKENS = ("hn-e2e", "ns960", "ns923", "\\lhm", "/lhm", "\\shots", "user-insights-lab",
                   "\\hermes-", "/hermes-", ".hermes-test")

GAME_EXE_HINTS = ("steam", "game", "cs2", "cyberpunk", "f1_", "forza", "apex", "valorant", "fortnite",
                  "eldenring", "gta", "rdr2", "arc", "battlenet", "epicgames", "riot")


# ---------------------------------------------------------------- accessors

def _txt(v) -> str:
    try:
        return json.dumps(v, default=str).lower()
    except Exception:
        return str(v).lower()


def _get(v, *keys, default=None):
    """First present key at the top level, then one level down."""
    if not isinstance(v, dict):
        return default
    for k in keys:
        if k in v and v[k] not in (None, ""):
            return v[k]
    for sub in v.values():
        if isinstance(sub, dict):
            for k in keys:
                if k in sub and sub[k] not in (None, ""):
                    return sub[k]
    return default


def _num(x, default=None):
    if x is None or isinstance(x, bool):
        return default if x is None else int(x)
    if isinstance(x, (int, float)):
        return x
    if isinstance(x, str):
        m = re.search(r"-?\d+(?:\.\d+)?", x.replace(",", ""))
        return float(m.group()) if m else default
    if isinstance(x, (list, tuple, set)):
        return len(x)
    if isinstance(x, dict):
        for k in ("count", "n", "total", "value", "len"):
            if k in x and isinstance(x[k], (int, float)) and not isinstance(x[k], bool):
                return x[k]
        return len(x)
    return default


def _cnt(v, *keys, default=None):
    """Numeric value under any of keys (a list counts as its length)."""
    if v is None:
        return default
    if keys:
        got = _get(v, *keys)
        if got is not None:
            n = _num(got)
            return default if n is None else n
        return default
    return _num(v, default)


def _truth(v, *keys) -> bool:
    got = _get(v, *keys)
    if isinstance(got, str):
        return got.strip().lower() not in ("", "0", "false", "no", "none", "off", "disabled", "absent")
    if isinstance(got, (list, dict)):
        return len(got) > 0
    return bool(got)


def _names(v, *keys) -> list:
    """Flatten a list of names or dicts with name-like fields into lower-case strings."""
    src = _get(v, *keys) if keys else v
    if src is None or isinstance(src, (bool, int, float)):
        return []
    if isinstance(src, dict):
        items = list(src.keys()) if all(not isinstance(x, (dict, list)) for x in src.values()) else list(src.values())
    elif isinstance(src, (list, tuple)):
        items = list(src)
    else:
        items = [src]
    out = []
    for it in items:
        if isinstance(it, dict):
            n = it.get("name") or it.get("app") or it.get("DisplayName") or it.get("display_name") or it.get("id") \
                or it.get("exe") or it.get("path") or it.get("product") or it.get("title")
            if n:
                out.append(str(n).lower())
        elif it is not None:
            out.append(str(it).lower())
    return out


_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _dt(x):
    """Parse ISO text, /Date(ms)/, MM/DD/YYYY, epoch s/ms, or FILETIME into an aware datetime."""
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, datetime):
        return x if x.tzinfo else x.replace(tzinfo=timezone.utc)
    if isinstance(x, dict):
        for k in ("DateTime", "value", "date", "iso", "utc"):
            if k in x:
                return _dt(x[k])
        return None
    if isinstance(x, (int, float)):
        n = float(x)
        try:
            if n > 1e16:
                return _EPOCH + timedelta(microseconds=(n - 116444736000000000) / 10)
            if n > 1e11:
                return _EPOCH + timedelta(milliseconds=n)
            if n > 1e8:
                return _EPOCH + timedelta(seconds=n)
        except (OverflowError, ValueError):
            return None
        return None
    s = str(x).strip()
    m = re.search(r"/Date\((-?\d+)", s)
    if m:
        return _dt(int(m.group(1)))
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", s)
    if m:
        try:
            return datetime(int(m.group(3)), int(m.group(1)), int(m.group(2)), tzinfo=timezone.utc)
        except ValueError:
            return None
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2}))?)?", s)
    if m:
        try:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            hh, mm, ss = int(m.group(4) or 0), int(m.group(5) or 0), int(m.group(6) or 0)
            return datetime(y, mo, d, hh, mm, ss, tzinfo=timezone.utc)
        except ValueError:
            return None
    if re.fullmatch(r"\d{8}", s):
        try:
            return datetime(int(s[:4]), int(s[4:6]), int(s[6:]), tzinfo=timezone.utc)
        except ValueError:
            return None
    if re.fullmatch(r"\d{9,18}", s):
        return _dt(int(s))
    return None


def _date_of(v, *keys):
    if v is None:
        return None
    if not isinstance(v, dict):
        return _dt(v)
    return _dt(_get(v, *keys))


def _now():
    return datetime.now(timezone.utc)


def _days(d):
    return None if d is None else (_now() - d).total_seconds() / 86400.0


def _iso(d):
    return d.date().isoformat() if d else None


def _operator(s: str) -> bool:
    s = s.lower()
    return any(t in s for t in OPERATOR_TOKENS)


def _sessions(v):
    return _cnt(v, "sessions", "session_count", "n_sessions", "sessions_total", "total_sessions",
                "conversations", "threads", "count")


def _sessions_30d(v):
    return _cnt(v, "sessions_30d", "sessions_last_30d", "last_30d", "recent_30d", "n_30d", "sessions_recent")


def _share(v, *cats):
    """Sum of category shares (0..1) from l3.browser_category_share-like values."""
    if not isinstance(v, dict):
        return None
    shares = _get(v, "shares", "share", "category_share", "categories") or v
    if not isinstance(shares, dict):
        return None
    tot = 0.0
    seen = False
    for c in cats:
        x = shares.get(c)
        if isinstance(x, dict):
            x = x.get("share", x.get("pct"))
        n = _num(x)
        if n is not None:
            seen = True
            tot += n / 100.0 if n > 1 else n
    return tot if seen else None


def _res(claim, strength="weak", value=None):
    return {"claim": claim, "strength": strength, "value": value}


_VIRTUAL = ("remote", "basic", "virtual", "indirect", "parsec", "meta")


def _hw_changes(f):
    """(gpu_changes, monitor_changes, image_devices) from PnP first/last dates.
    change = device first seen > 2 d after the tenure anchor; image device = gone and last seen before the OS install."""
    a, _ = _tenure_anchor(f)
    inst = _date_of(f.get("os.install_date"), "install_date", "InstallDate", "date", "value")
    out = [0, 0, 0]
    for i, (pid, key, name) in enumerate((("hw.gpu_history", "adapters", "name"), ("hw.monitor_history", "monitors", "pnp"))):
        v = f.get(pid)
        rows = _get(v, key) if isinstance(v, dict) else None
        if not isinstance(rows, list):
            continue
        for r in rows:
            if not isinstance(r, dict) or any(w in str(r.get(name, "")).lower() for w in _VIRTUAL):
                continue
            first, last = _dt(r.get("first")), _dt(r.get("last"))
            if inst and last and not r.get("present") and last < inst - timedelta(days=1):
                out[2] += 1
            elif a and first and first > a + timedelta(days=2) and (_days(first) or 0) <= 183:
                out[i] += 1
    return tuple(out)


# ---------------------------------------------------------------- 1 managed_vs_personal

@insight(id="persona.managed_vs_personal",
         inputs=["work.join_state", "work.mdm", "work.security_agents", "work.policies", "acct.account_type",
                 "onedrive.accounts", "comms.teams_launched", "work.edition_org"])
def managed_vs_personal(f):
    """Employer-managed vs personal machine. Edition alone is not a signal."""
    if not any(k in f for k in ("work.join_state", "work.mdm", "work.security_agents", "onedrive.accounts")):
        return None
    score, why = 0, []
    js = f.get("work.join_state")
    if js is not None:
        t = _txt(js)
        joined = _truth(js, "domain_joined", "DomainJoined", "azure_ad_joined", "AzureAdJoined", "aad_joined",
                        "entra_joined", "joined")
        state = str(_get(js, "state", "join_type", "type") or "").lower()
        if state and any(w in state for w in ("domain", "azuread", "entra", "hybrid")) and "workplace" not in state:
            joined = True
        if joined and "workgroup" not in state:
            score += 3
            why.append("domain/Entra joined")
        elif '"workplace' in t:
            why.append("workplace-registered only (not scored)")
    mdm = f.get("work.mdm")
    if mdm is not None:
        n = _cnt(mdm, "real_mdm", "real_enrollments", "mdm_enrollments", "enrolled_count", "enrollments", "count")
        if (n or 0) > 0 or _truth(mdm, "mdm_enrolled", "enrolled", "managed"):
            score += 3
            why.append("MDM enrolled")
    ag = f.get("work.security_agents")
    if ag is not None:
        names = [n for n in _names(ag, "agents", "found", "products", "names", "items") if "warp" not in n
                 and "tailscale" not in n]
        n = len(names) if names else (_cnt(ag, "count", "n") or 0)
        if n > 0:
            score += 2
            why.append(f"{int(n)} EDR/ZTNA agent(s)")
    od = f.get("onedrive.accounts")
    if od is not None:
        b = _cnt(od, "business", "business_accounts", "n_business", "tenants")
        if (b or 0) > 0 or '"business1"' in _txt(od):
            score += 2
            why.append("OneDrive work tenant")
    tm = f.get("comms.teams_launched")
    if tm is not None and _truth(tm, "work_profile", "work", "work_launched", "tenant"):
        score += 2
        why.append("Teams work profile")
    managed = score >= 3
    ed = str(_get(f.get("work.edition_org") or {}, "edition", "EditionID") or "")
    claim = ("Employer-managed machine" if managed else "Personal machine (no join, MDM, work tenant or EDR)")
    if not managed and "enterprise" in ed.lower():
        claim += "; Enterprise edition is not treated as management"
    return _res(claim, "strong", {"managed": managed, "score": score, "reasons": why})


# ---------------------------------------------------------------- 2 sophistication

@insight(id="persona.sophistication",
         inputs=["dev.devmode_sudo_longpaths", "shell.explorer_prefs", "security.openssh_server", "security.rdp",
                 "net.tailscale", "dev.pwsh7", "apps.pkg_managers", "apps.winget_db", "dev.uv_tools",
                 "kbd.scancode_map", "security.telemetry"])
def sophistication(f):
    """Additive rubric from report 11 derive(): novice / power-user / expert."""
    score, why = 0, []

    def add(n, r):
        nonlocal score
        score += n
        why.append(f"+{n} {r}")

    ds = f.get("dev.devmode_sudo_longpaths")
    if ds is not None:
        if _truth(ds, "devmode", "developer_mode", "AllowDevelopmentWithoutDevLicense", "dev_mode"):
            add(2, "Developer Mode on")
        if _truth(ds, "sudo", "sudo_enabled", "sudo_Enabled"):
            add(2, "sudo enabled")
        if _truth(ds, "long_paths", "longpaths", "LongPathsEnabled", "long_paths_enabled"):
            add(1, "LongPathsEnabled")
    ex = f.get("shell.explorer_prefs")
    if ex is not None:
        hide = _get(ex, "HideFileExt", "hide_file_ext", "hide_extensions")
        if hide in (0, "0", False) or _truth(ex, "show_extensions", "file_extensions_shown"):
            add(1, "file extensions shown")
    ssh = f.get("security.openssh_server")
    if ssh is not None and (_truth(ssh, "installed", "service", "sshd", "present", "start") or ssh is True):
        add(2, "OpenSSH server installed")
    rdp = f.get("security.rdp")
    if rdp is not None:
        deny = _get(rdp, "fDenyTSConnections", "deny")
        if deny in (0, "0") or _truth(rdp, "rdp_on", "enabled", "rdp_enabled", "accepting"):
            add(1, "RDP enabled")
    if f.get("net.tailscale") is not None:
        add(2, "Tailscale")
    if f.get("dev.pwsh7") is not None:
        add(1, "PowerShell 7")
    pm = f.get("apps.pkg_managers")
    if pm is not None:
        names = [k for k in ("scoop", "choco", "pipx", "pnpm", "bun") if pm.get(k) is True] if isinstance(pm, dict) else []
        names += [n for n in _names(pm, "managers", "found", "names") if "winget" not in n]
        if names:
            add(1, "non-inbox package manager (" + ", ".join(names[:3]) + ")")
    wg = f.get("apps.winget_db")
    if wg is not None and (_cnt(wg, "user_installs", "installs", "packages", "count", "n") or 0) >= 5:
        add(1, "winget used for installs")
    if f.get("dev.uv_tools") is not None:
        add(1, "uv tools")
    sc = f.get("kbd.scancode_map")
    if sc is not None and (_truth(sc, "present", "remapped", "mappings", "entries") or sc is True):
        add(1, "keyboard scancode remap")
    tel = f.get("security.telemetry")
    if tel is not None:
        lvl = _num(_get(tel, "allow_telemetry", "AllowTelemetry", "level"))
        reduced = _truth(tel, "reduced", "non_default", "tailored_off", "user_changed")
        if reduced or (lvl is not None and lvl <= 1):
            add(1, "telemetry reduced by hand")
    if not why and not any(k in f for k in ("dev.devmode_sudo_longpaths", "shell.explorer_prefs")):
        return None
    tier = "expert" if score >= 10 else "power-user" if score >= 5 else "novice"
    return _res(f"Technical sophistication: {tier} (score {score})", "strong" if len(why) >= 3 else "weak",
                {"tier": tier, "score": score, "reasons": why})


# ---------------------------------------------------------------- 3 developer

def _dev_app_count(f):
    tx = f.get("apps.taxonomy_user_added")
    if tx is None:
        return None
    c = _get(tx, "counts", "by_category", "categories") or tx
    if isinstance(c, dict):
        for k in ("dev", "developer", "development", "dev_tools"):
            if k in c:
                return _num(c[k])
    return None


def _own_commit_repos(f):
    mc = f.get("dev.repos.my_commits")
    if mc is None:
        return None
    return _cnt(mc, "repos_with_own_commits", "user_repos_with_commits", "repos", "n_repos", "count", default=0)


@insight(id="persona.developer",
         inputs=["apps.taxonomy_user_added", "dev.git", "dev.toolchain_presence", "editor.vscode",
                 "l3.browser_category_share", "browser.history.localhost", "dev.repos", "dev.repos.my_commits"])
def developer(f):
    """Developer if >= 3 of six conditions hold."""
    hits = {}
    if f.get("dev.git") is not None:
        hits["git"] = True
    tc = f.get("dev.toolchain_presence")
    if tc is not None:
        names = _names(tc, "on_path", "toolchains", "found", "names")
        n = len(names) if names else _cnt(tc, "count", "n", default=1)
        if n:
            hits["toolchain"] = names[:6] or int(n)
    dn = _dev_app_count(f)
    if dn is not None and dn >= 5:
        hits["dev_apps"] = dn
    sh = _share(f.get("l3.browser_category_share"), "dev", "ai", "developer", "coding")
    if sh is not None and sh >= 0.20:
        hits["dev_ai_share"] = round(sh, 3)
    lh = f.get("browser.history.localhost")
    if lh is not None and (_cnt(lh, "visits", "count", "n", "localhost_visits", default=1) or 0) > 0:
        hits["localhost"] = True
    oc = _own_commit_repos(f)
    if oc:
        hits["own_repo"] = oc
    if not hits:
        return None
    dev = len(hits) >= 3
    return _res(("Developer" if dev else "Not a developer by the >= 3 rule") + f" ({len(hits)}/6: {', '.join(hits)})",
                "strong" if dev or len(hits) == 0 else "weak", {"developer": dev, "hits": hits})


# ---------------------------------------------------------------- 4 primary_dev_machine

@insight(id="persona.primary_dev_machine",
         inputs=["editor.vscode", "dev.git_global_config", "dev.repos", "dev.repos.my_commits", "dev.shell_history",
                 "dev.ssh_config"])
def primary_dev_machine(f):
    """Primary only if editor customised AND git identity set AND own-commit repo in the user area."""
    ed = f.get("editor.vscode")
    gc = f.get("dev.git_global_config")
    if ed is None and gc is None and "dev.repos.my_commits" not in f:
        return None
    customised = False
    if ed is not None:
        ext = _cnt(ed, "extensions", "n_extensions", "extension_count", default=0) or 0
        st = _cnt(ed, "settings_keys", "n_settings", "settings", "user_settings", default=0) or 0
        customised = ext > 0 or st > 0 or _truth(ed, "customised", "customized")
    ident = gc is not None and _truth(gc, "identity_set", "user_name_set", "has_identity", "user.name", "user_name",
                                      "name_set", "email_set")
    own = bool(_own_commit_repos(f))
    primary = customised and ident and own
    hist = _cnt(f.get("dev.shell_history"), "lines", "line_count", "count")
    claim = ("Primary dev machine" if primary else
             "Tool box / test target, not the primary dev machine; ask where the user codes")
    # The deciding input only exists when the repo walk ran. Without it, "not primary" is
    # an absence of evidence, not evidence of absence: say so weakly and let the agent ask.
    decided = "dev.repos.my_commits" in f or own
    strength = "strong" if decided and (primary or (customised and ident)) else "weak"
    if not decided and not primary:
        claim += " (own-commit scan not collected)"
    return _res(claim, strength, {"primary": primary, "editor_customised": customised, "git_identity": ident,
                                  "own_commit_repo": own, "shell_history_lines": hist})


# ---------------------------------------------------------------- 5 ai_power_user / 6 primary_agent

_AGENTS = {"claude_code": "claude_code.sessions", "codex": "codex.usage", "hermes": "hermes.usage"}


def _agent_sessions(f, recent=False):
    out = {}
    for name, pid in _AGENTS.items():
        v = f.get(pid)
        if v is None:
            continue
        n = _sessions_30d(v) if recent else None
        if n is None:
            n = _sessions(v)
        if n is not None:
            out[name] = n
    return out


@insight(id="persona.ai_power_user",
         inputs=["claude_code.sessions", "codex.usage", "hermes.usage", "codex.chatgpt_catalog", "ollama.models",
                 "ollama.logs", "browser.automation", "cua_driver.present", "l3.mcp_inventory"])
def ai_power_user(f):
    """Usage-weighted. Installed-but-unused tools score 0."""
    s = _agent_sessions(f)
    chat = _cnt(f.get("codex.chatgpt_catalog"), "conversations", "count", "n", "total")
    models = _cnt(f.get("ollama.models"), "models", "count", "n")
    if not s and chat is None and models is None:
        return None
    used = {k: v for k, v in s.items() if v and v >= 3}
    power = any(v >= 20 for v in s.values()) or len(used) >= 2 or (chat or 0) > 50
    level = "power_user" if power else "user" if any(v for v in s.values()) or (models or 0) > 0 else "tools_only"
    extras = [k for k in ("browser.automation", "cua_driver.present", "l3.mcp_inventory") if k in f]
    return _res(f"AI usage: {level} (sessions {s or 'none'}; ChatGPT conversations {chat})",
                "strong" if s else "weak",
                {"level": level, "sessions": s, "chatgpt_conversations": chat, "ollama_models": models,
                 "automation_tools": extras})


@insight(id="persona.primary_agent", inputs=["hermes.usage", "codex.usage", "claude_code.sessions"])
def primary_agent(f):
    """Agent with the most sessions in the last 30 days."""
    s = _agent_sessions(f, recent=True)
    s = {k: v for k, v in s.items() if v}
    if not s:
        return None
    top = max(s, key=s.get)
    return _res(f"Primary agent on this host: {top} ({int(s[top])} sessions)", "strong" if s[top] >= 3 else "weak",
                {"agent": top, "sessions": s})


# ---------------------------------------------------------------- 7 hermes_developer

@insight(id="persona.hermes_developer",
         inputs=["dev.repos.my_commits", "dev.repos", "apps.appx_registry", "filter.operator_noise"])
def hermes_developer(f):
    """Only user-authored commits in a non-operator hermes-agent clone. Side homes, lab tasks and commit-named
    bundle builds (HermesBundledCommit*) are operator test artifacts and never fire it."""
    mc = f.get("dev.repos.my_commits")
    repos = f.get("dev.repos")
    if mc is None and repos is None:
        return None
    t = _txt(mc)
    clone = [r for r in (_get(repos or {}, "repos") or []) if isinstance(r, dict)
             and "hermes-agent" in str(r.get("name", "")).lower() and r.get("class") not in ("operator", "tool_managed")]
    own = _own_commit_repos(f) or 0
    if "hermes-agent" in t or (clone and own):
        return _res("Contributes to hermes-agent (own commits in a user-area clone)", "strong",
                    {"hermes_developer": True, "clones": len(clone), "repos_with_own_commits": own})
    if clone:
        return _res("hermes-agent clone present without own commits (reader, not contributor)", "weak",
                    {"hermes_developer": False, "clones": len(clone)})
    return None


# ---------------------------------------------------------------- 8 gamer_tier / 9 genres

def _steam_hours(f):
    pt = f.get("steam.playtime")
    if pt is None:
        return None, None
    hours = _cnt(pt, "total_hours", "hours_total", "account_hours", "hours", "all_hours")
    if hours is None:
        mins = _cnt(pt, "total_minutes", "minutes")
        hours = mins / 60.0 if mins else None
    recent = _cnt(pt, "played_last_30d", "games_30d", "recent_games", "last_30d")
    return hours, recent


def _local_hours(f):
    ls = f.get("steam.local_sessions")
    if ls is None:
        return None, None
    return (_cnt(ls, "hours", "total_hours", "local_hours", "hours_total"),
            _cnt(ls, "hours_30d", "last_30d_hours", "hours_last_30d"))


def _launchers(f):
    out = []
    if "steam.present" in f or "steam.installed" in f or "steam.playtime" in f:
        out.append("steam")
    if "epic.present" in f or "epic.installs" in f:
        out.append("epic")
    lo = f.get("launchers.other")
    if lo is not None:
        out += [n for n in _names(lo, "launchers", "found", "names", "items") if n not in out]
    return out


@insight(id="persona.gamer_tier",
         inputs=["steam.playtime", "steam.local_sessions", "steam.installed", "epic.installs", "launchers.other",
                 "gaming.anticheat", "gcs.game_history", "nvidia.recommendations", "tuning.afterburner",
                 "hw.bluetooth", "consent.microphone", "srum.app_timeline"])
def gamer_tier(f):
    """Report 07 score: account hours, 30-day recency, installed GB, launchers, Afterburner."""
    if not any(k in f for k in ("steam.playtime", "steam.installed", "epic.installs", "launchers.other",
                                "gcs.game_history", "steam.present")):
        return None
    hours, recent = _steam_hours(f)
    gcs = f.get("gcs.game_history")
    gcs30 = _cnt(gcs, "played_30d", "last_30d", "recent_30d", default=0) or 0
    inst_gb = 0.0
    for pid in ("steam.installed", "epic.installs"):
        g = _cnt(f.get(pid), "size_gb", "total_gb", "gb", "installed_gb")
        if g:
            inst_gb += g
    for row in (_get(f.get("epic.installs") or {}, "manifests") or []):
        if isinstance(row, (list, tuple)) and len(row) > 1 and isinstance(row[1], (int, float)):
            inst_gb += row[1]
    launchers = _launchers(f)
    score = 0
    score += 2 if (hours or 0) >= 500 else 1 if (hours or 0) >= 50 else 0
    r = max(gcs30, recent or 0)
    score += 2 if r >= 3 else 1 if r else 0
    score += 1 if inst_gb >= 100 else 0
    score += 1 if len(launchers) >= 3 else 0
    score += 1 if "tuning.afterburner" in f else 0
    tier = "core" if score >= 5 else "regular" if score >= 3 else "casual" if score >= 1 else "none"
    lh, lh30 = _local_hours(f)
    return _res(f"Gamer tier: {tier} (score {score})", "strong",
                {"tier": tier, "score": score, "account_hours": hours, "local_hours": lh, "local_hours_30d": lh30,
                 "installed_gb": round(inst_gb, 1), "launchers": launchers,
                 "anticheat": "gaming.anticheat" in f})


@insight(id="persona.genres", inputs=["steam.playtime", "steam.appinfo_genres"])
def genres(f):
    """Genre hours from Steam genre ids."""
    g = f.get("steam.appinfo_genres")
    if g is None:
        return None
    gh = _get(g, "genre_hours", "hours_by_genre", "genres") if isinstance(g, dict) else None
    per_acc = _get(g, "per_account") if isinstance(g, dict) else None
    if not isinstance(gh, dict) and isinstance(per_acc, list):
        gh = {}
        for acc in per_acc:
            for row in (acc.get("genre_h") or []) if isinstance(acc, dict) else []:
                if isinstance(row, (list, tuple)) and len(row) == 2:
                    gh[row[0]] = gh.get(row[0], 0) + (_num(row[1]) or 0)
    if not isinstance(gh, dict):
        pt = f.get("steam.playtime") or {}
        per_app = _get(pt, "per_app", "apps", "games") if isinstance(pt, dict) else None
        app_genres = _get(g, "app_genres", "by_app", "apps") if isinstance(g, dict) else None
        if not isinstance(per_app, (dict, list)) or not isinstance(app_genres, dict):
            return None
        gh = {}
        rows = per_app.items() if isinstance(per_app, dict) else [(str(x.get("appid")), x) for x in per_app
                                                                   if isinstance(x, dict)]
        for appid, row in rows:
            h = _num(row.get("hours") if isinstance(row, dict) else row) or 0
            for gn in app_genres.get(str(appid), []) or []:
                gh[gn] = gh.get(gn, 0) + h
    gh = {k: round(_num(v) or 0, 1) for k, v in gh.items() if k.lower() not in ("free to play", "early access")}
    if not gh:
        return None
    top = sorted(gh.items(), key=lambda kv: -kv[1])[:4]
    return _res("Top genres by hours: " + ", ".join(f"{k} {v}" for k, v in top), "strong", {"genre_hours": dict(top)})


# ---------------------------------------------------------------- 10 machine_role

@insight(id="persona.machine_role",
         inputs=["steam.local_sessions", "steam.playtime", "l3.browser_category_share", "files.composition",
                 "userassist.focus", "l3.top_apps_by_time"])
def machine_role(f):
    """Gaming rig / work laptop / media / mixed from the largest share."""
    scores = {"gaming": 0.0, "work": 0.0, "media": 0.0}
    lh, lh30 = _local_hours(f)
    if lh is not None:
        scores["gaming"] += min(1.0, (lh or 0) / 100.0)
    sh = f.get("l3.browser_category_share")
    if sh is not None:
        scores["work"] += _share(sh, "dev", "ai", "work", "docs", "productivity") or 0
        scores["media"] += _share(sh, "video", "streaming", "music", "media") or 0
        scores["gaming"] += _share(sh, "gaming", "games") or 0
    comp = f.get("files.composition")
    if comp is not None:
        vid = _share(_get(comp, "byte_share", "share_by_kind", "shares") or {}, "video", "videos")
        if vid:
            scores["media"] += vid * 0.5
    ua = f.get("l3.top_apps_by_time") or f.get("userassist.focus")
    if ua is not None:
        t = _txt(ua)
        for k in ("steam", "game", "epic"):
            if k in t:
                scores["gaming"] += 0.2
                break
        for k in ("code", "terminal", "pwsh", "powershell", "codex", "hermes"):
            if k in t:
                scores["work"] += 0.2
                break
        for k in ("vlc", "media player", "spotify", "netflix"):
            if k in t:
                scores["media"] += 0.2
                break
    if not any(scores.values()):
        return None
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])
    role = ordered[0][0] if ordered[0][1] >= 1.5 * max(ordered[1][1], 0.01) else f"mixed ({ordered[0][0]}+{ordered[1][0]})"
    return _res(f"Machine role: {role}", "strong" if lh is not None and sh is not None else "weak",
                {"role": role, "scores": {k: round(v, 2) for k, v in scores.items()}, "local_steam_hours": lh})


# ---------------------------------------------------------------- 11 tinkerer_tuner

@insight(id="persona.tinkerer_tuner",
         inputs=["tuning.afterburner", "tuning.rtss", "tasks.nonms", "periph.rgb_suites", "hw.power_plan",
                 "hw.gpu_history", "hw.monitor_history"])
def tinkerer_tuner(f):
    """>= 2 of: custom fan/VF curve, power-limit task, RGB suite, Ultimate Performance plan, hardware swap."""
    hits = []
    ab = f.get("tuning.afterburner")
    if ab is not None and (_truth(ab, "custom_curve", "fan_curve", "vf_curve", "profiles", "custom_fan") or
                           "curve" in _txt(ab)):
        hits.append("Afterburner custom curve")
    elif ab is not None:
        hits.append("Afterburner installed")
    tk = f.get("tasks.nonms")
    if tk is not None and ("nvidia-smi" in _txt(tk) or "power_limit" in _txt(tk)):
        hits.append("GPU power-limit task")
    if f.get("periph.rgb_suites") is not None:
        hits.append("RGB suite")
    pp = f.get("hw.power_plan")
    if pp is not None and ("ultimate" in _txt(pp) or "e9a42b02" in _txt(pp)):
        hits.append("Ultimate Performance plan")
    g, m, _img = _hw_changes(f)
    if g:
        hits.append(f"GPU swap ({g})")
    if m:
        hits.append(f"monitor change ({m})")
    if not hits:
        return None
    yes = len(hits) >= 2
    return _res(("Hardware tinkerer/tuner: " if yes else "Some tuning: ") + ", ".join(hits),
                "strong" if yes else "weak", {"tinkerer": yes, "hits": hits})


# ---------------------------------------------------------------- 12 creator / 13 streamer_recorder

CREATIVE = ("blender", "gimp", "obs", "davinci", "resolve", "premiere", "photoshop", "krita", "inkscape",
            "sketchup", "audacity", "reaper", "clipchamp", "affinity", "creative cloud", "lightroom", "kdenlive",
            "shotcut", "fl studio", "ableton", "maya", "freecad", "unreal", "unity")


@insight(id="persona.creator",
         inputs=["apps.taxonomy_user_added", "media.obs", "userassist.focus", "consent.microphone", "consent.webcam",
                 "files.composition"])
def creator(f):
    """Creative apps installed AND launched by the user; agent-container config does not count."""
    tx = f.get("apps.taxonomy_user_added")
    installed = []
    if tx is not None:
        c = _get(tx, "counts", "by_category", "categories") or {}
        n = _num(c.get("creative")) if isinstance(c, dict) else None
        installed = [x for x in CREATIVE if x in _txt(tx)]
        if n and not installed:
            installed = [f"{int(n)} creative apps"]
    obs = f.get("media.obs")
    if obs is not None:
        installed.append("obs")
    if not installed:
        return None
    launched = []
    ua = _txt(f.get("userassist.focus"))
    launched += [x for x in CREATIVE if x in ua]
    if obs is not None:
        t = _txt(obs)
        if "openai.codex" not in t and (_truth(obs, "appdata_config", "config_dir", "profiles", "scenes") or
                                        (_cnt(obs, "recordings", "scene_collections", default=0) or 0) > 0):
            launched.append("obs(config)")
    mic = _txt(f.get("consent.microphone")) + _txt(f.get("consent.webcam"))
    launched += [x for x in ("obs64", "blender", "resolve") if x in mic]
    yes = bool(launched)
    return _res(("Creator: creative apps launched by the user (" + ", ".join(sorted(set(launched))) + ")") if yes
                else "Creative apps installed but no user-launch evidence (may be agent-installed)", "weak",
                {"creator": yes, "installed": sorted(set(installed))[:10], "launched": sorted(set(launched))})


@insight(id="persona.streamer_recorder",
         inputs=["media.obs", "nvidia.broadcast", "nvidia.shadowplay", "captures.gamebar", "captures.nvidia",
                 "consent.screen_capture"])
def streamer_recorder(f):
    """Recording if captures exist or OBS used mic/camera."""
    caps = {}
    for pid in ("captures.gamebar", "captures.nvidia"):
        n = _cnt(f.get(pid), "count", "files", "n", "captures", "videos")
        if n:
            caps[pid.split(".")[1]] = n
    tools = [k for k in ("media.obs", "nvidia.broadcast", "nvidia.shadowplay") if k in f]
    if not caps and not tools:
        return None
    yes = bool(caps)
    return _res(("Records gameplay/screen (captures: " + ", ".join(f"{k} {int(v)}" for k, v in caps.items()) + ")")
                if yes else "Capture tools present, no captures found", "weak",
                {"records": yes, "captures": caps, "tools": tools})


# ---------------------------------------------------------------- 14 voice_chat_gamer

@insight(id="persona.voice_chat_gamer", inputs=["consent.microphone", "discord.usage"])
def voice_chat_gamer(f):
    """Mic use by game exes or Discord voice days >= 3."""
    mic = f.get("consent.microphone")
    dc = f.get("discord.usage")
    if mic is None and dc is None:
        return None
    game_mic = [n for n in _names(mic, "recent", "items", "exes", "top") if any(g in n for g in GAME_EXE_HINTS)
                and "steamwebhelper" not in n] if mic is not None else []
    rtc = _cnt(dc, "rtc_days", "voice_days", "rtc_days_90d", "voice_days_90d") if dc is not None else None
    discord_mic = "discord" in _txt(mic)
    yes = bool(game_mic) or (rtc or 0) >= 3
    if not yes and not discord_mic:
        return None
    return _res("Uses voice chat while gaming" if yes else "Discord has mic access; voice use not established",
                "weak", {"voice_chat": yes, "game_mic_apps": len(game_mic), "discord_rtc_days": rtc})


# ---------------------------------------------------------------- 15 hands_on_vs_remote

@insight(id="persona.hands_on_vs_remote",
         inputs=["srum.app_timeline", "security.openssh_server", "net.tailscale", "acct.hello", "bam.last_run"])
def hands_on_vs_remote(f):
    """Deep: input/focus ratio. Default: sshd + Tailscale = remotely administered; Hello use = person present."""
    st = f.get("srum.app_timeline")
    if st is not None:
        r = _num(_get(st, "input_focus_ratio", "input_over_focus", "ratio"))
        if r is not None:
            mode = "remote-driven" if r < 0.05 else "hands-on" if r > 0.3 else "mixed"
            return _res(f"Use is {mode} (input/focus {r:.2f}, 7-day window)", "strong", {"mode": mode, "ratio": r})
    ssh = "security.openssh_server" in f
    ts = "net.tailscale" in f
    hello = f.get("acct.hello")
    face = isinstance(hello, dict) and (hello.get("face") is True or
                                        str(hello.get("last_provider") or "").lower() == "face")
    if not (ssh or ts or face):
        return None
    remote = ssh and ts
    mode = ("remotely administered, person present at times" if remote and face else
            "remotely administered" if remote else "hands-on (no remote-admin path)")
    return _res(f"Access pattern: {mode}", "weak", {"sshd": ssh, "tailscale": ts, "hello_face": face})


# ---------------------------------------------------------------- 16 activity_hours

def _hour_hist(v):
    if v is None:
        return None
    h = _get(v, "hours", "hour", "by_hour", "hour_hist", "hours_local", "start_hours") if isinstance(v, dict) else v
    if isinstance(h, dict):
        try:
            arr = [0] * 24
            for k, n in h.items():
                arr[int(k) % 24] += _num(n) or 0
            h = arr
        except (ValueError, TypeError):
            return None
    if isinstance(h, list) and len(h) == 24 and all(isinstance(x, (int, float)) for x in h):
        return h
    return None


@insight(id="persona.activity_hours",
         inputs=["browser.history.hour_weekday", "steam.local_sessions", "dev.repos.commit_hours", "files.screenshots",
                 "wu.active_hours", "srum.app_resource"])
def activity_hours(f):
    """Peak hours per host; 'night' only if >= 40% of events fall 00-05 in >= 2 sources."""
    srcs = {}
    for pid in ("browser.history.hour_weekday", "steam.local_sessions", "dev.repos.commit_hours", "files.screenshots",
                "srum.app_resource"):
        h = _hour_hist(f.get(pid))
        if h and sum(h) > 0:
            srcs[pid] = h
    if not srcs:
        return None
    total = [sum(h[i] for h in srcs.values()) for i in range(24)]
    s = sum(total)
    peak = sorted(range(24), key=lambda i: -total[i])[:3]
    night_srcs = [p for p, h in srcs.items() if sum(h[0:6]) / sum(h) >= 0.40]
    night = len(night_srcs) >= 2
    shares = {p: round(sum(h[0:6]) / sum(h), 2) for p, h in srcs.items()}
    label = "night" if night else ("afternoon-evening" if sum(total[12:24]) / s >= 0.6 else "daytime" if
                                   sum(total[8:18]) / s >= 0.6 else "spread")
    return _res(f"Active hours: {label}; peak hours {sorted(peak)}", "strong" if len(srcs) >= 2 else "weak",
                {"label": label, "peak_hours": sorted(peak), "night_share_by_source": shares, "sources": list(srcs)})


# ---------------------------------------------------------------- 17 power_habit

def _power_split(f):
    ps = f.get("l3.power_event_split")
    if isinstance(ps, dict):
        return {k: _cnt(ps, k, default=0) for k in ("crash", "button_held", "power_removed")}
    ph = f.get("eventlog.power_history")
    if not isinstance(ph, dict):
        return None
    got = {k: _cnt(ph, k, f"kp41_{k}", default=None) for k in ("crash", "button_held", "power_removed")}
    if any(v is not None for v in got.values()):
        return {k: v or 0 for k, v in got.items()}
    return None


@insight(id="persona.power_habit", inputs=["boot.uptime", "eventlog.power_history", "l3.power_event_split"])
def power_habit(f):
    """Power-cycles daily vs sleeps; KP41 power_removed counts toward the power-off habit, not health."""
    up = f.get("boot.uptime")
    ph = f.get("eventlog.power_history")
    if up is None and ph is None:
        return None
    uptime_h = _cnt(up, "uptime_h", "uptime_hours", "hours")
    if uptime_h is None:
        d = _cnt(up, "uptime_days", "days")
        uptime_h = d * 24 if d is not None else None
    boots = _cnt(ph, "boots_30d", "boots", "n_boots", "startups")
    resumes = _cnt(ph, "resumes_30d", "resumes", "sleep_resumes", "wakes")
    split = _power_split(f) or {}
    removed = split.get("power_removed", 0) or 0
    habit = "unknown"
    if boots is not None and boots >= 20 and (resumes or 0) <= 3:
        habit = "power_cycles_daily"
    elif uptime_h is not None and uptime_h > 72 and (resumes or 0) > 2 * (boots or 0):
        habit = "sleeps"
    elif uptime_h is not None and uptime_h > 72:
        habit = "stays_on"
    elif boots is not None:
        habit = "mixed"
    claim = f"Power habit: {habit}" + (" (no boot history)" if habit == "unknown" else "")
    if removed:
        claim += f"; {int(removed)} hard power-offs (switch or outage)"
    if habit == "power_cycles_daily":
        claim += "; scheduled jobs need missed-run catch-up"
    return _res(claim, "strong" if boots is not None else "weak",
                {"habit": habit, "uptime_h": uptime_h, "boots": boots, "resumes": resumes, "power_removed": removed})


# ---------------------------------------------------------------- 18 mobility

@insight(id="persona.mobility", inputs=["hw.battery", "net.history", "net.wifi_profiles", "hw.sleep_study"])
def mobility(f):
    """Laptop that moves if battery AND >= 3 wireless networks."""
    bat = f.get("hw.battery")
    if bat is None:
        if "net.history" in f or "net.wifi_profiles" in f:
            return _res("Desktop (no battery): no mobility defaults", "weak", {"battery": False, "moves": False})
        return None
    wl = _cnt(f.get("net.history"), "wireless", "n_wireless", "wifi", "wireless_count")
    if wl is None:
        wl = _cnt(f.get("net.wifi_profiles"), "count", "profiles", "n")
    moves = (wl or 0) >= 3
    return _res(("Laptop that moves between networks" if moves else "Laptop, mostly stationary") +
                f" ({wl} wireless networks); use battery-aware defaults", "weak",
                {"battery": True, "wireless_networks": wl, "moves": moves})


# ---------------------------------------------------------------- 19 local_llm_capability

@insight(id="persona.local_llm_capability", inputs=["hw.gpu", "hw.ram", "host.native_arch", "hw.nvidia_smi"])
def local_llm_capability(f):
    """VRAM >= 24 GB: large local models; 8-16 GB: small/quantised; ARM64 warns about x64-only CUDA tooling."""
    g = f.get("hw.gpu")
    ns = f.get("hw.nvidia_smi")
    if g is None and ns is None:
        return None
    vram = _cnt(ns, "vram_gb", "memory_total_gb", "vram") if ns is not None else None
    if vram is None and ns is not None:
        mib = _cnt(ns, "memory_total_mib", "memory_total_mb", "memory.total")
        vram = mib / 1024.0 if mib else None
    if vram is None and g is not None:
        vram = _cnt(g, "vram_gb", "max_vram_gb", "dedicated_gb", "vram")
        if vram is None:
            b = _cnt(g, "vram_bytes", "qwMemorySize", "adapter_ram")
            vram = b / 2 ** 30 if b else None
        if vram is not None and vram > 4096:
            vram = vram / 1024.0
    ram = _cnt(f.get("hw.ram"), "total_gb", "gb", "ram_gb", "installed_gb")
    if ram is not None and ram > 4096:
        ram = ram / 2 ** 30
    arch = str(_get(f.get("host.native_arch") or {}, "arch", "native", "machine", "value") or f.get("host.native_arch")
               or "").lower()
    arm = "arm" in arch
    if vram is None:
        cap = "unknown_vram"
    elif vram >= 24:
        cap = "large_models"
    elif vram >= 7.5:
        cap = "small_quantised"
    else:
        cap = "cpu_or_tiny"
    claim = f"Local LLM capability: {cap} (VRAM {round(vram, 1) if vram else '?'} GB, RAM {round(ram) if ram else '?'} GB)"
    if arm:
        claim += "; ARM64 host: x64-only CUDA tooling may not run"
    return _res(claim, "strong" if vram is not None else "weak",
                {"capability": cap, "vram_gb": vram, "ram_gb": ram, "arm64": arm})


# ---------------------------------------------------------------- 20 health_flags

@insight(id="persona.health_flags",
         inputs=["hw.disks", "hw.ram", "l3.power_event_split", "health.whea_gpu", "health.wer", "health.dumps",
                 "eventlog.power_history"])
def health_flags(f):
    """Disk free < 15%, RAM below rated speed or single channel, bugchecks/WHEA in 30 d, repeated app crashes."""
    flags = []
    d = f.get("hw.disks")
    if d is not None:
        vols = _get(d, "volumes", "disks", "drives") if isinstance(d, dict) else d
        vols = vols if isinstance(vols, list) else [d]
        for v in vols:
            if not isinstance(v, dict):
                continue
            pct = _num(v.get("free_pct") or v.get("pct_free"))
            if pct is None and v.get("free_gb") is not None and v.get("size_gb"):
                pct = 100.0 * _num(v["free_gb"]) / _num(v["size_gb"])
            if pct is not None and pct < 15:
                flags.append(f"disk {v.get('letter') or v.get('name') or '?'} {pct:.0f}% free")
    r = f.get("hw.ram")
    if r is not None:
        sp = _num(_get(r, "configured_mhz", "speed_mhz", "configured_speed"))
        rated = _num(_get(r, "rated_mhz", "max_mhz", "speed_rated"))
        for part in (_get(r, "part") or []) if isinstance(_get(r, "part"), list) else []:
            m = re.search(r"-(\d{4})$", str(part))
            if m and int(m.group(1)) > (rated or 0):
                rated = int(m.group(1))
        if sp and rated and sp < rated * 0.9:
            flags.append(f"RAM at {int(sp)} of rated {int(rated)} MT/s")
        mods, slots = _num(_get(r, "modules", "dimms")), _num(_get(r, "slots"))
        lpddr = any(t in (30, 35) for t in (_get(r, "smbios_type") or []) if isinstance(t, int))
        if mods == 1 and (slots or 0) >= 2 and not lpddr:
            flags.append(f"single-channel RAM (1 of {int(slots)} slots)")
    split = _power_split(f) or {}
    ps = f.get("l3.power_event_split")
    crash30 = None
    if isinstance(ps, dict):
        crash30 = _num(ps.get("crash_30d"))
        if crash30 is None and "crash_30d" in ps:
            crash30 = 0
    if crash30 is None and not isinstance(ps, dict):
        crash30 = split.get("crash") or 0
    if crash30:
        flags.append(f"{int(crash30)} bugcheck crash(es) in 30 d")
    wh = _cnt(f.get("health.whea_gpu"), "count_30d", "events_30d")
    if wh:
        flags.append(f"{int(wh)} WHEA/GPU error event(s) in 30 d")
    wer = f.get("health.wer")
    if wer is not None:
        n = _cnt(wer, "max_repeat", "top_count")
        top = (_get(wer, "crashes_by_app") or [[None]])[0]
        if n and n >= 5:
            flags.append(f"repeated app crashes ({top[0] if isinstance(top, list) else '?'} x{int(n)})")
    dm = _cnt(f.get("health.dumps"), "count_30d", "minidumps_30d")
    if dm:
        flags.append(f"{int(dm)} crash dump(s) in 30 d")
    if not any(k in f for k in ("hw.disks", "hw.ram", "health.wer", "health.whea_gpu", "eventlog.power_history")):
        return None
    return _res(("Health flags: " + "; ".join(flags)) if flags else "No health flags", "strong",
                {"flags": flags})


# ---------------------------------------------------------------- 21-24

@insight(id="persona.locale_format", inputs=["locale.user", "locale.geo", "kbd.layouts", "tz.zone", "lang.ui"])
def locale_format(f):
    """Reply language = UI language; date/currency from the user locale."""
    ui = _get(f.get("lang.ui") or {}, "ui_language", "language", "lang", "value") or f.get("lang.ui")
    loc = _get(f.get("locale.user") or {}, "locale", "LocaleName", "name", "value") or f.get("locale.user")
    tz = _get(f.get("tz.zone") or {}, "zone", "TimeZoneKeyName", "name", "value") or f.get("tz.zone")
    geo = _get(f.get("locale.geo") or {}, "name", "geo", "Name", "value") or f.get("locale.geo")
    if not any(isinstance(x, str) for x in (ui, loc, tz, geo)):
        return None
    s = lambda x: x if isinstance(x, str) else None
    fmt = _get(f.get("locale.user") or {}, "short_date", "sShortDate", "date_format")
    return _res(f"Reply language {s(ui)}; formats from {s(loc)}; time zone {s(tz)}", "strong",
                {"ui_language": s(ui), "locale": s(loc), "geo": s(geo), "tz": s(tz), "short_date": fmt,
                 "keyboards": _names(f.get("kbd.layouts"), "layouts", "names", "items")[:5]})


@insight(id="persona.ui_theme", inputs=["theme.dark"])
def ui_theme(f):
    """Match dark UI."""
    t = f.get("theme.dark")
    if t is None:
        return None
    apps = _get(t, "apps_dark", "AppsUseLightTheme", "apps", "dark") if isinstance(t, dict) else t
    dark = (apps in (0, "0") if isinstance(t, dict) and "AppsUseLightTheme" in t else bool(apps))
    if isinstance(t, dict) and "AppsUseLightTheme" in t:
        dark = t["AppsUseLightTheme"] in (0, "0", False)
    return _res("Dark UI" if dark else "Light UI", "strong", {"dark": dark})


@insight(id="persona.browser_target",
         inputs=["browser.default", "l3.primary_browser", "browser.profile_prefs", "browser.bookmarks",
                 "browser.extensions"])
def browser_target(f):
    """Primary browser; skip import when bookmark/extension counts are 0; no sync unless signed in."""
    pb = f.get("l3.primary_browser")
    primary = _get(pb or {}, "primary", "browser", "name") if isinstance(pb, dict) else pb
    default = _get(f.get("browser.default") or {}, "prog_id", "browser", "default", "name", "ProgId") \
        if isinstance(f.get("browser.default"), dict) else f.get("browser.default")
    if not primary and not default:
        return None
    bm = _cnt(f.get("browser.bookmarks"), "total", "count", "bookmarks", default=0)
    ext = _cnt(f.get("browser.extensions"), "user_installed", "count", "total", "extensions", default=0)
    signed = _truth(f.get("browser.profile_prefs") or {}, "signed_in", "signed_in_profiles", "sync")
    return _res(f"Use {primary or default}; bookmarks {bm}, extensions {ext}; signed-in profile: {signed}", "strong",
                {"primary": primary, "default": default, "offer_import": bool(bm or ext), "signed_in": signed})


PM_WORDS = ("bitwarden", "1password", "lastpass", "keepass", "dashlane", "proton pass", "nordpass", "enpass")


@insight(id="persona.password_manager", inputs=["pm.desktop", "browser.extensions", "l3.browser_category_share"])
def password_manager(f):
    """Desktop app, extension, or >= 10 visits to a PM web vault."""
    found = []
    pm = f.get("pm.desktop")
    if pm is not None and _names(pm, "managers", "found", "names"):
        found += [f"{w} (desktop)" for w in PM_WORDS if w in _txt(pm)] or ["desktop app"]
    ex = _txt(f.get("browser.extensions"))
    found += [f"{w} (extension)" for w in PM_WORDS if w in ex]
    sh = f.get("l3.browser_category_share")
    if sh is not None:
        vis = _cnt(sh, "password_manager_visits", "pm_vault_visits")
        if vis is None:
            s = _get(sh, "counts", "visits") or {}
            vis = _num(s.get("password_manager")) if isinstance(s, dict) else None
        if vis and vis >= 10:
            found.append(f"web vault ({int(vis)} visits)")
    if not found:
        return None
    return _res("Uses a password manager: " + ", ".join(found), "weak", {"password_manager": found})


# ---------------------------------------------------------------- 25-28

@insight(id="persona.comms_surface",
         inputs=["comms.native_apps", "comms.teams_launched", "discord.usage", "browser.history.workspaces"])
def comms_surface(f):
    """Native if the app shows recent use; package presence is not use."""
    out = {}
    na = f.get("comms.native_apps")
    if na is not None and isinstance(na, dict):
        for app, v in na.items():
            if app in ("present", "count"):
                continue
            used = _truth(v, "recent_use", "used", "launched", "last_used") if isinstance(v, dict) else False
            out[app] = "native" if used else "installed_unused"
    if "discord.usage" in f:
        out["discord"] = "native"
    if f.get("comms.teams_launched") is not None:
        out["teams"] = "native"
    ws = f.get("browser.history.workspaces")
    if isinstance(ws, dict):
        for k, v in ws.items():
            if (_num(v) or 0) > 0 and k not in out:
                out[k] = "web"
    if not out:
        return None
    return _res("Comms: " + ", ".join(f"{k}={v}" for k, v in out.items()), "strong", {"surfaces": out})


@insight(id="persona.storage_locality", inputs=["onedrive.accounts", "onedrive.kfm", "sync.other"])
def storage_locality(f):
    """Local-only if no signed-in sync client and no known-folder redirection."""
    if not any(k in f for k in ("onedrive.accounts", "onedrive.kfm", "sync.other")):
        return _res("Local-only storage (no signed-in sync client, no folder redirection)", "weak",
                    {"local_only": True})
    od = f.get("onedrive.accounts")
    signed = od is not None and (_cnt(od, "signed_in", "accounts", "count", "n", default=1) or 0) > 0
    kfm = f.get("onedrive.kfm")
    redirected = kfm is not None and _truth(kfm, "redirected", "any", "folders", "kfm")
    other = _names(f.get("sync.other"), "clients", "found", "names") if "sync.other" in f else []
    local = not signed and not redirected and not other
    return _res("Local-only storage" if local else "Cloud-synced storage (" + ", ".join(
        [x for x, y in (("OneDrive", signed), ("KFM", redirected)) if y] + other) + ")", "strong",
        {"local_only": local, "onedrive_signed_in": signed, "kfm": redirected, "other": other})


@insight(id="persona.notes_location", inputs=["productivity.notes_apps"])
def notes_location(f):
    """No notes app: ask where notes live."""
    n = f.get("productivity.notes_apps")
    names = _names(n, "apps", "found", "names") if n is not None else []
    if not names:
        return _res("No notes app found; ask where notes live", "weak", {"notes_apps": []})
    return _res("Notes app(s): " + ", ".join(names[:4]), "weak", {"notes_apps": names})


@insight(id="persona.media_consumer", inputs=["media.players", "media.libraries", "files.composition"])
def media_consumer(f):
    """No music player and empty Music folder: do not lead with music integrations."""
    pl = f.get("media.players")
    lib = f.get("media.libraries")
    if pl is None and lib is None and "files.composition" not in f:
        return None
    players = _names(pl, "players", "found", "names") if pl is not None else []
    music = _cnt((lib or {}).get("Music"), "audio", default=0) if isinstance(lib, dict) else 0
    video = _cnt((lib or {}).get("Videos"), "video", default=0) if isinstance(lib, dict) else 0
    music_player = any(w in " ".join(players) for w in ("spotify", "foobar", "musicbee", "itunes", "winamp", "aimp",
                                                         "tidal", "apple music"))
    if not music_player and not music:
        claim = "Not a music listener on this box: skip music integrations"
    else:
        claim = "Listens to music locally"
    return _res(claim, "strong", {"players": players, "music_files": music, "video_files": video,
                                  "music_player": music_player})


# ---------------------------------------------------------------- 29-31

@insight(id="persona.multi_machine_owner",
         inputs=["steam.remote_clients", "net.tailscale", "steam.login_users", "apps.install_timeline"])
def multi_machine_owner(f):
    """Several own machines if Tailscale peers >= 3 or Steam remote peers >= 1."""
    peers = _cnt(f.get("net.tailscale"), "peers", "n_peers", "peer_count", "online_peers")
    remote = _cnt(f.get("steam.remote_clients"), "count", "clients", "n")
    if peers is None and remote is None:
        return None
    yes = (peers or 0) >= 3 or (remote or 0) >= 1
    return _res(("Owns several machines" if yes else "Single machine evidence only") +
                f" (Tailscale peers {peers}, Steam remote clients {remote}); offer cross-machine sync" * yes,
                "strong", {"multi_machine": yes, "tailscale_peers": peers, "steam_remote_clients": remote})


def _defaults_untouched(f):
    out = []
    hn = f.get("host.name_default")
    if hn is not None and (hn is True or _truth(hn, "default", "is_default", "matches")):
        out.append("hostname")
    elif hn is None and isinstance(f.get("host.name"), dict):
        if re.fullmatch(r"(DESKTOP|LAPTOP)-[A-Z0-9]{7,8}", str(f["host.name"].get("name") or "").upper()):
            out.append("hostname")
    un = f.get("profile.username_default")
    if un is not None and (un is True or _truth(un, "default", "is_default")):
        out.append("username")
    sp = f.get("theme.spotlight_suggestions")
    if sp is not None and _truth(sp, "suggestions_on", "on", "enabled", "default"):
        out.append("spotlight/suggestions")
    pb = f.get("pins.taskband_oem")
    if pb is not None and (_cnt(pb, "oem_pins", "count", "n", default=1) or 0) > 0:
        out.append("OEM taskbar pins")
    wp = f.get("theme.wallpaper")
    if wp is not None and _truth(wp, "default", "is_default", "stock"):
        out.append("wallpaper")
    return out


@insight(id="persona.setup_defaults_untouched",
         inputs=["theme.wallpaper", "theme.spotlight_suggestions", "host.name_default", "host.name", "pins.taskband_oem"])
def setup_defaults_untouched(f):
    """Low cosmetic customisation when >= 3 defaults remain. Tone hint only."""
    d = _defaults_untouched(f)
    if not any(k in f for k in ("theme.wallpaper", "theme.spotlight_suggestions", "host.name_default", "host.name",
                                "pins.taskband_oem")):
        return None
    low = len(d) >= 3
    return _res(("Low cosmetic customisation: " if low else "Defaults left: ") + (", ".join(d) or "none"), "weak",
                {"low_customisation": low, "defaults": d})


@insight(id="persona.terse_user", inputs=["browser.history.search_terms", "files.clutter"])
def terse_user(f):
    """Short queries and a clean desktop: short replies."""
    st = f.get("browser.history.search_terms")
    cl = f.get("files.clutter")
    words = _num(_get(st or {}, "avg_words", "mean_words", "mean_len_words")) if st is not None else None
    desk = _cnt(cl, "desktop_items", "desktop", "desktop_files") if cl is not None else None
    if words is None and desk is None:
        return None
    terse = (words is not None and words <= 3) and (desk is None or desk <= 10)
    if not terse:
        return None
    return _res("Terse: prefer short replies", "weak",
                {"terse": terse, "mean_query_words": words, "desktop_items": desk})


# ================================================================ section 4: fresh vs old

def _tenure_anchor(f):
    pc = _date_of(f.get("profile.created"), "created", "ctime", "date", "created_utc", "value")
    ob = _date_of(f.get("setup.oobe_done"), "date", "last_write", "oobe_done", "time", "value", "created")
    anchors = [d for d in (pc, ob) if d]
    if anchors:
        return max(anchors), {"profile_created": _iso(pc), "oobe_done": _iso(ob)}
    inst = _date_of(f.get("os.install_date"), "install_date", "InstallDate", "date", "value")
    return inst, {"os_install_date": _iso(inst)}


@insight(id="install.user_tenure_days",
         inputs=["profile.created", "setup.oobe_done", "os.install_date", "net.history", "steam.local_sessions",
                 "l3.browser_install_age", "browser.profile_prefs", "prefetch.stat"])
def user_tenure_days(f):
    """Days since the current profile took ownership: max(profile.created, setup.oobe_done)."""
    a, src = _tenure_anchor(f)
    if a is None:
        return None
    days = int(_days(a))
    first_net = _date_of(f.get("net.history"), "first_created", "first_date", "first_seen", "oldest", "first")
    val = {"tenure_days": days, "anchor": _iso(a), **src}
    claim = f"User has owned this install for {days} days (since {_iso(a)})"
    if first_net and (first_net - a).days > 3:
        val["first_online"] = _iso(first_net)
        claim += f"; first network {_iso(first_net)}"
    return _res(claim, "strong" if "profile_created" in src or "oobe_done" in src else "weak", val)


@insight(id="install.os_lineage",
         inputs=["setup.sysreset", "hw.gpu_history", "hw.monitor_history", "setup.oem", "setup.image_date", "os.install_date",
                 "setup.source_os_lineage", "setup.oldest_lineage_date", "setup.windows_old",
                 "pca.generaldb", "amcache.inventory_app", "wdi.startupinfo"])
def os_lineage(f):
    """reset > redeployed_image > oem_image > in_place_upgrade > clean, plus lineage age in years."""
    inst = _date_of(f.get("os.install_date"), "install_date", "InstallDate", "date", "value")
    oldest = _date_of(f.get("setup.oldest_lineage_date"), "oldest", "date", "oldest_date", "value")
    img = _date_of(f.get("setup.image_date"), "image_date", "date", "clone_date", "oldest", "value")
    lineage = f.get("setup.source_os_lineage")
    gh = f.get("hw.gpu_history")
    if inst is None and not any(k in f for k in ("setup.sysreset", "setup.oem", "setup.source_os_lineage")):
        return None
    img_devices = _hw_changes(f)[2]
    redeployed = img_devices > 0 or (gh is not None and _truth(gh, "image_devices_absent", "redeployed"))
    oem = _truth(f.get("setup.oem") or {}, "oem_image")
    reset = _truth(f.get("setup.sysreset") or {}, "reset")
    wold = _truth(f.get("setup.windows_old") or {}, "windows_old", "windows_bt")
    if oldest is None:
        oldest = _date_of(lineage, "oldest")
    if reset:
        kind = "reset"
    elif redeployed:
        kind = "redeployed_image"
    elif oem and img and inst and img < inst - timedelta(days=30):
        kind = "oem_image"
    elif wold:
        kind = "in_place_upgrade"
    elif lineage is not None and (_cnt(lineage, "count", default=0) or 0) > 0:
        kind = "in_place_upgrade"
    else:
        kind = "clean"
    first = min([d for d in (oldest, inst, img) if d], default=None)
    years = round(_days(first) / 365.25, 2) if first else None
    val = {"os_lineage": kind, "install_date": _iso(inst), "oldest_lineage_date": _iso(oldest),
           "image_date": _iso(img), "lineage_years": years, "oem": oem, "image_devices_gone": img_devices}
    claim = f"OS lineage: {kind}" + (f", {years} years old (from {_iso(first)})" if years is not None else "")
    return _res(claim, "strong" if inst else "weak", val)


def _lived_in(f):
    pts, why = 0.0, []

    def add(n, r):
        nonlocal pts
        pts += n
        why.append(f"+{n} {r}")
    fp = f.get("age.footprint_counts")
    pf = _cnt(f.get("prefetch.stat"), "files", "count", "n")
    if pf is None:
        pf = _cnt(fp, "prefetch", "prefetch_count", "Prefetch")
    if pf is not None and pf >= 400:
        add(1, f"Prefetch {int(pf)}")
    rc = _cnt(fp, "recent", "recent_lnk", "recent_count", "Recent")
    if rc is None:
        rc = _cnt(f.get("files.recent_lnk"), "count", "n", "lnk")
    if rc is not None and rc >= 150:
        add(1, f"Recent .lnk {int(rc)}")
    ua = _cnt(f.get("userassist.focus"), "total_focus_h", "focus_hours", "total_hours", "focus_h")
    if ua is not None and ua >= 50:
        add(1, f"UserAssist focus {ua:.0f} h")
    fs = _cnt(f.get("featureusage.appswitched"), "total", "switches", "count", "sum")
    if fs is not None and fs >= 1000:
        add(1, f"FeatureUsage switches {int(fs)}")
    bv = _cnt(f.get("browser.history.stats"), "visits", "total_visits", "visit_count", "visits_total")
    if bv is not None and bv >= 1000:
        add(1, f"browser visits {int(bv)}")
    s = _agent_sessions(f)
    if any((v or 0) >= 20 for v in s.values()):
        add(1, "local agent sessions >= 20")
    lh, _ = _local_hours(f)
    if lh is not None and lh >= 50:
        add(1, f"local Steam {lh:.0f} h")
    gu = _cnt(f.get("gcs.game_history"), "uninstalled_since", "uninstalled", "n_uninstalled")
    if gu is not None and gu >= 5:
        add(1, f"GCS uninstalled games {int(gu)}")
    comp = f.get("files.composition")
    dl = _num(_get(comp or {}, "downloads_older_90d_share", "downloads_old_share", "older_90d_share"))
    if dl is not None and (dl / 100.0 if dl > 1 else dl) > 0.25:
        add(1, "Downloads > 25% older than 90 d")
    cl = f.get("files.clutter")
    if (_cnt(cl, "temp_files", "temp", "TEMP") or 0) > 1000:
        add(0.5, "TEMP > 1000 files")
    pw = f.get("files.profile_size_walk")
    if (_cnt(pw, "custom_root_dirs", "custom_dirs", "nonstandard_dirs") or 0) >= 10:
        add(0.5, "custom profile-root dirs >= 10")
    g, m, _img = _hw_changes(f)
    hw = int(bool(g)) + int(bool(m))
    if hw:
        add(min(hw, 2), "device changes after install")
    ua_n = None
    tx = f.get("apps.taxonomy_user_added")
    if tx is not None:
        ua_n = _cnt(tx, "user_added_total", "user_added", "total", "count", "n")
    if ua_n is None:
        ua_n = _cnt(f.get("apps.preinstalled_split"), "user_added", "n_user_added", "user")
    if ua_n:
        n = min(3, int(ua_n // 20))
        if n:
            add(n, f"user-added apps {int(ua_n)}")
    it = f.get("apps.install_timeline")
    months = _cnt(it, "months_spread", "distinct_months", "months_active") if it is not None else None
    if months is None and tx is not None:
        bm = _get(tx, "by_month")
        months = len([m for m, n in bm.items() if (_num(n) or 0) > 0]) if isinstance(bm, dict) else None
    if (months or 0) >= 3:
        add(1, f"installs spread over {months} months")
    pts = min(pts, 10.0)
    d = _defaults_untouched(f)
    pen = 0.5 * len(d)
    return max(0.0, pts - pen), why, d


@insight(id="install.lived_in_score",
         inputs=["prefetch.stat", "age.footprint_counts", "userassist.focus", "featureusage.appswitched",
                 "browser.history.stats", "hermes.usage", "codex.usage", "claude_code.sessions",
                 "steam.local_sessions", "gcs.game_history", "files.composition", "files.clutter",
                 "files.profile_size_walk", "hw.gpu_history", "hw.monitor_history", "apps.taxonomy_user_added",
                 "apps.preinstalled_split", "apps.install_timeline", "host.name_default", "host.name", "profile.username_default",
                 "theme.spotlight_suggestions", "pins.taskband_oem"])
def lived_in_score(f):
    """Sum of accumulated-use points (cap 10) minus 0.5 per untouched default (min 0)."""
    score, why, d = _lived_in(f)
    if not why and not d:
        return None
    return _res(f"Lived-in score {score:g}/10", "strong" if len(why) >= 3 else "weak",
                {"lived_in": score, "points": why, "defaults_untouched": d})


@insight(id="install.user_history_elsewhere",
         inputs=["files.age_hist", "files.pre_install_mtime", "files.composition", "steam.playtime", "codex.chatgpt_catalog",
                 "hermes.usage", "os.install_date", "profile.created"])
def user_history_elsewhere(f):
    """Account/file history older than this install. Says the person is not new; never makes the install old."""
    a, _ = _tenure_anchor(f)
    ev = {}
    pim = _cnt(f.get("files.pre_install_mtime"), "count", "files", "n", "older_than_install")
    migrated = bool(pim)
    if pim:
        ev["files_older_than_install"] = pim
    ah = f.get("files.age_hist")
    oldest_file = _date_of(ah, "oldest", "oldest_mtime", "min") if ah is not None else None
    if oldest_file and a and oldest_file < a - timedelta(days=30):
        migrated = True
        ev["oldest_file"] = _iso(oldest_file)
    comp = f.get("files.composition")
    if isinstance(comp, dict) and isinstance(comp.get("folders"), dict) and a is not None:
        old = {k: v.get("oldest_year") for k, v in comp["folders"].items()
               if isinstance(v, dict) and isinstance(v.get("oldest_year"), int) and v["oldest_year"] < a.year - 1
               and not _operator(k)}
        if old:
            migrated = True
            ev["files_older_than_install"] = old
    pt = f.get("steam.playtime")
    lp = _date_of(pt, "oldest_last_played", "first_last_played", "oldest") if pt is not None else None
    if lp is None and isinstance(pt, dict):
        dates = [_dt(row[2]) for acc in (pt.get("per_account") or []) if isinstance(acc, dict)
                 for row in (acc.get("top") or []) if isinstance(row, (list, tuple)) and len(row) > 2]
        lp = min([d for d in dates if d], default=None)
    if lp and a and lp < a:
        ev["steam_since"] = _iso(lp)
    elif pt is not None and (_steam_hours(f)[0] or 0) > 0:
        ev["steam_account_hours"] = _steam_hours(f)[0]
    cg = f.get("codex.chatgpt_catalog")
    cgd = _date_of(cg, "oldest", "oldest_date", "first", "since") if cg is not None else None
    if cgd and (a is None or cgd < a):
        ev["chatgpt_since"] = _iso(cgd)
    hu = f.get("hermes.usage")
    hd = _date_of(hu, "first", "first_session", "oldest", "since") if hu is not None else None
    if hd and a and hd < a - timedelta(days=1):
        ev["hermes_since"] = _iso(hd)
    if not ev:
        if not any(k in f for k in ("files.age_hist", "files.pre_install_mtime", "steam.playtime",
                                    "codex.chatgpt_catalog")):
            return None
        return _res("No history older than this install", "weak",
                    {"user_history_elsewhere": False, "migrated_user": False})
    return _res("User has history that predates this install (" + ", ".join(ev) + ")", "strong",
                {"user_history_elsewhere": True, "migrated_user": migrated, "evidence": ev})


@insight(id="install.install_state",
         inputs=["profile.created", "setup.oobe_done", "os.install_date", "prefetch.stat", "age.footprint_counts",
                 "browser.history.stats", "userassist.focus", "apps.taxonomy_user_added"])
def install_state(f):
    """fresh: tenure < 30 d and lived_in < 3; settling: tenure < 120 d and lived_in < 6; else established."""
    a, _ = _tenure_anchor(f)
    if a is None:
        return None
    t = _days(a)
    li, _, _ = _lived_in(f)
    state = "fresh" if t < 30 and li < 3 else "settling" if t < 120 and li < 6 else "established"
    return _res(f"Install state: {state} (tenure {int(t)} d, lived-in {li:g})", "strong",
                {"install_state": state, "tenure_days": int(t), "lived_in": li})


# ================================================================ report 13 derivations

_PHONE_VENDORS = ("samsung", "google", "pixel", "xiaomi", "oneplus", "motorola", "huawei", "oppo", "vivo", "apple",
                  "iphone", "nothing", "realme")


@insight(id="phone.wu_driver_hint", inputs=["wu.history", "phone.link", "hw.usb_history"])
def wu_driver_hint(f):
    """Phone vendor from Windows Update driver titles (modem/USB/ADB drivers). Model is not recorded."""
    wu = f.get("wu.history")
    if wu is None:
        return None
    titles = _names(wu, "driver_titles", "drivers", "titles") or [_txt(wu)]
    hits = set()
    for t in titles:
        if not re.search(r"\b(modem|usb|mtp|adb|android|mobile)\b", t):
            continue
        for v in _PHONE_VENDORS:
            if re.search(rf"\b{v}\b", t):
                hits.add(v)
    if not hits:
        return None
    paired = f.get("phone.link")
    return _res("A " + "/".join(sorted(hits)) + " phone was connected by USB at least once (model unknown)", "weak",
                {"phone_vendor": sorted(hits), "phone_link_paired": _truth(paired or {}, "paired",
                                                                           "HasPreviouslyPaired")})


@insight(id="l3.image_origin",
         inputs=["pca.generaldb", "wdi.startupinfo", "amcache.inventory_app", "lsm.sessions", "os.install_date",
                 "profile.created", "setup.oem", "setup.image_date", "setup.system_log_oldest",
                 "eventlog.power_history"])
def image_origin(f):
    """Earliest record per source vs the profile creation: >= 2 sources more than 30 d older = OEM-preloaded image."""
    a, _ = _tenure_anchor(f)
    earliest = {}
    for pid, keys in (("pca.generaldb", ("first", "first_date", "oldest", "min_date", "start")),
                      ("wdi.startupinfo", ("oldest_other_sid", "oldest", "first", "min_date")),
                      ("amcache.inventory_app", ("oldest_install", "oldest", "first", "min_install_date")),
                      ("lsm.sessions", ("first", "oldest", "since", "first_event")),
                      ("setup.image_date", ("setupapi_first", "image_date")),
                      ("setup.system_log_oldest", ("oldest", "system_log_oldest", "date", "value")),
                      ("eventlog.power_history", ("system_log_oldest",))):
        d = _date_of(f.get(pid), *keys) if pid in f else None
        if d is None and pid == "setup.image_date" and isinstance(f.get(pid), dict):
            d = _dt(str(f[pid].get("setupapi_first") or "").replace("/", "-"))
        if d:
            earliest[pid] = d
    if not earliest or a is None:
        return None
    pre = {k: _iso(v) for k, v in earliest.items() if v < a - timedelta(days=30)}
    if len(pre) >= 2:
        oem = _truth(f.get("setup.oem") or {}, "oem_image")
        origin = "oem_image" if oem else "pre_staged_image"
        claim = (f"{'OEM-preloaded' if oem else 'Pre-staged'} image: {len(pre)} sources have records from more than "
                 f"30 days before the profile ({_iso(a)})")
        return _res(claim, "strong" if len(pre) >= 3 or oem else "weak",
                    {"origin": origin, "pre_profile_records": pre, "profile": _iso(a)})
    return _res(f"Clean install: earliest records are within 30 days of the profile ({_iso(a)})",
                "strong" if len(earliest) >= 2 else "weak",
                {"origin": "clean", "earliest": {k: _iso(v) for k, v in earliest.items()}})


RE_TOOLS = ("ghidra", "dnspy", "ida", "x64dbg", "ilspy", "cheat engine", "wireshark", "procmon", "jdk")


@insight(id="l3.persona_additions",
         inputs=["explorer.muicache", "msi.install_events", "search.windows_db", "gpu.per_app_prefs",
                 "boot.diag_perf"])
def persona_additions(f):
    """Launched-app and install-event evidence: AAA gamer, creator, reverse-engineer."""
    mui = f.get("explorer.muicache")
    msi = f.get("msi.install_events")
    sdb = f.get("search.windows_db")
    gpu = f.get("gpu.per_app_prefs")
    if not any((mui, msi, sdb, gpu)):
        return None
    blob = " ".join(_txt(x) for x in (mui, msi, sdb) if x is not None)
    blob = " ".join(w for w in blob.split() if not _operator(w))
    tags = {}
    games = _cnt(mui, "games", "n_games", "game_names")
    hdr = _cnt(gpu, "autohdr_games", "autohdr", "n_autohdr")
    if (games or 0) >= 10 or (hdr or 0) >= 2:
        tags["aaa_gamer"] = {"launched_games": games, "autohdr_titles": hdr}
    cr = sorted({w for w in CREATIVE if w in blob})
    if len(cr) >= 3:
        tags["creator"] = cr
    rt = sorted({w for w in RE_TOOLS if w in blob and w != "jdk"})
    if rt:
        tags["reverse_engineering"] = rt
    if not tags:
        return None
    return _res("Launched/installed-app evidence: " + ", ".join(tags), "strong" if len(tags) >= 1 and mui else "weak",
                tags)


@insight(id="persona.dense_ui", inputs=["display.per_monitor_dpi", "shell.taskbar_dev_settings"])
def dense_ui(f):
    """Display scaled below recommended plus the taskbar End-task toggle: prefer compact output."""
    dpi = f.get("display.per_monitor_dpi")
    tb = f.get("shell.taskbar_dev_settings")
    below = dpi is not None and ((_num(_get(dpi, "min_offset", "dpi_offset", "DpiValue")) or 0) < 0 or
                                 _truth(dpi, "below_recommended", "scaled_down"))
    endtask = tb is not None and _truth(tb, "TaskbarEndTask", "end_task", "endtask")
    if not (below or endtask):
        return None
    return _res("Dense-UI power user: prefer compact output", "weak", {"scaled_below": below, "end_task": endtask})


@insight(id="persona.clipboard_flows", inputs=["input.typing_clipboard"])
def clipboard_flows(f):
    """Clipboard history on and Win+V used."""
    v = f.get("input.typing_clipboard")
    if v is None:
        return None
    on = _truth(v, "clipboard_history", "EnableClipboardHistory", "history_on")
    used = _truth(v, "win_v_used", "ShellHotKeyUsed", "hotkey_used")
    if not on:
        return None
    return _res("Clipboard history in use" + (" (Win+V used)" if used else ""), "weak",
                {"clipboard_history": on, "win_v_used": used})


@insight(id="persona.windows_ai_off", inputs=["winai.surfaces"])
def windows_ai_off(f):
    """No Recall, Copilot disabled, no Copilot app: do not suggest Windows Copilot features."""
    v = f.get("winai.surfaces")
    if v is None:
        return None
    t = _txt(v)
    recall = _truth(v, "recall", "recall_present", "aix")
    copilot = _truth(v, "copilot_app", "copilot_enabled") and "featureisdisabled" not in t
    if recall or copilot:
        return _res("Windows AI surfaces present", "weak", {"recall": recall, "copilot": copilot})
    return _res("Windows Copilot/Recall off: do not suggest them", "weak", {"recall": False, "copilot": False})


@insight(id="persona.skip_print_phone", inputs=["print.printers_scanners", "phone.link"])
def skip_print_phone(f):
    """No physical printer/scanner and no paired phone: skip those onboarding steps."""
    pr = f.get("print.printers_scanners")
    ph = f.get("phone.link")
    if pr is None and ph is None:
        return None
    printers = _cnt(pr, "physical", "physical_printers", "n_physical", "real_printers", default=0) if pr is not None else None
    scanners = _cnt(pr, "scanners", "n_scanners", default=0) if pr is not None else None
    paired = _truth(ph or {}, "paired", "HasPreviouslyPaired", "has_paired")
    skip = []
    if pr is not None and not printers and not scanners:
        skip.append("print/scan")
    if ph is not None and not paired:
        skip.append("phone")
    if not skip:
        return None
    return _res("Skip onboarding for: " + ", ".join(skip), "weak",
                {"physical_printers": printers, "scanners": scanners, "phone_paired": paired})
