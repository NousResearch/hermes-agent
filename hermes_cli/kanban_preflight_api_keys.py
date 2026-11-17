#!/usr/bin/env python3
"""
kanban_preflight_api_keys.py — Pre-Flight-Check für fehlende API-Keys

Scannt alle `ready` Kanban-Tasks, ermittelt aus Assignee-Profil + Task-Body
benötigte API-Keys, und blockt Tasks sofort, wenn mindestens ein Key fehlt.

PFLICHT-GATE (seit t_c13ed04e, nicht überspringbar):
    check_single_task wird von kanban_db.claim_task UND kanban_db.claim_review_task
    aufgerufen — jeder Task-Dispatch (ready -> running UND review -> running)
    läuft damit zwingend durch den Check; es gibt keinen Claim-Pfad, der ihn
    umgeht. Fehlende Keys: Task -> status='blocked', block_kind='missing-api-keys',
    Events 'blocked' + 'preflight_block' mit profil-spezifischem Grund nach
    dem Muster:
        "missing-api-keys: profile '<assignee>' lacks required API key(s)
         for toolsets [...]: one of [KEY_A, KEY_B]"
    Lesart: fehlender Key pro Profil namentlich genannt -> betroffenes Profil
    zieht den Key nach (.env / Keychain) und hebt den Block auf. Gate-interner
    Fehler: fail-open mit ERROR-Log (niemals still).

Nutzung:
    python3 kanban_preflight_api_keys.py [--dry-run] [--board PATH]

Exit-Codes:
    0 = keine Aktion nötig oder erfolgreich verarbeitet
    1 = mindestens ein Task wurde geblockt (oder würde geblockt werden)

Constraints:
    - Keine Live-API-Calls.
    - Keine hardcoded Secrets.
    - Prüft os.environ + pass-store.
"""

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

DEFAULT_BOARD = Path.home() / ".hermes" / "kanban.db"
PASS_CACHE_TTL_SECONDS = 60

# Toolset -> Liste von Key-Gruppen. Eine Gruppe ist erfüllt, wenn mindestens
# ein Key aus der Gruppe vorhanden ist. Das Toolset ist erfüllt, wenn alle
# Gruppen erfüllt sind.
TOOLSET_REQUIRED_KEYS: dict[str, list[list[str]]] = {
    "web": [
        ["BRAVE_SEARCH_API_KEY", "TAVILY_API_KEY", "GOOGLE_API_KEY",
         "PERPLEXITY_API_KEY", "FIRECRAWL_API_KEY"],
    ],
    "search": [
        ["BRAVE_SEARCH_API_KEY", "TAVILY_API_KEY", "GOOGLE_API_KEY",
         "PERPLEXITY_API_KEY", "FIRECRAWL_API_KEY"],
    ],
    "browser": [
        ["BROWSERBASE_API_KEY", "CAMOFOX_API_KEY"],
    ],
    "x_search": [
        ["XAI_API_KEY"],
    ],
    "image_gen": [
        ["FAL_KEY"],
    ],
    "video_gen": [
        ["FAL_KEY", "XAI_API_KEY"],
    ],
    "tts": [
        ["ELEVENLABS_API_KEY", "OPENAI_API_KEY", "VOICE_TOOLS_OPENAI_KEY",
         "MINIMAX_API_KEY", "MISTRAL_API_KEY", "XAI_API_KEY"],
    ],
    "discord": [
        ["DISCORD_BOT_TOKEN"],
    ],
    "discord_admin": [
        ["DISCORD_BOT_TOKEN"],
    ],
    "homeassistant": [
        ["HASS_TOKEN"],
    ],
    "spotify": [
        ["SPOTIFY_CLIENT_ID"],
        ["SPOTIFY_CLIENT_SECRET"],
    ],
    # Obsidian ist ein Skill-Toolset, das den Obsidian-Vault anbindet.
    "obsidian": [
        ["OBSIDIAN_API_KEY"],
    ],
}

# xAI kann alternativ via OAuth autorisiert sein (Provider "xai-oauth", z.B.
# TTS und x_search). OAuth existiert dabei NICHT als env-var — Hermes hält
# die Tokens im Auth-Store: ~/.hermes/profiles/<profil>/auth.json, Fallback
# ~/.hermes/auth.json (Quelle: tools/xai_http.py::has_xai_credentials,
# hermes_cli/auth.py). Der Waiver prüft daher Auth-Store + config.yaml statt
# erfundener Env-var-Namen. Gedeckte Keys:
XAI_OAUTH_WAIVED_KEYS = frozenset({"XAI_API_KEY"})
OAUTH_AWARE_ENV = "CRON_PREFLIGHT_OAUTH_AWARE"
XAI_OAUTH_PROVIDER = "xai-oauth"
_OAUTH_CACHE_TTL_SECONDS = 300

# Schlüsselwörter im Task-Body, die auf ein konkretes Toolset/Key hindeuten.
BODY_TOOLSET_HINTS: dict[str, list[str]] = {
    "web": ["web_search", "web_extract", "brave", "tavily", "perplexity",
            "firecrawl", "google search"],
    "x_search": ["x_search", "twitter", "x.com", "tweet", "post on x"],
    "image_gen": ["image_generate", "image_gen", "fal.ai", "fal_key"],
    "video_gen": ["video_generate", "video_gen"],
    "tts": ["text_to_speech", "tts", "elevenlabs", "voice_gen"],
    "discord": ["discord bot", "discord kanal", "discord webhook",
                "discord_server", "discord bot token"],
    "homeassistant": ["homeassistant", "home assistant", "hass"],
    "obsidian": ["obsidian", "vault"],
}

# Provider-abhängige Body-Hinweise, die zusätzliche Keys erzwingen.
# Hinweis: Einträge werden als Substrings geprüft. Nur Begriffe verwenden,
# die tatsächlich API-Nutzung signalisieren, nicht bloße Namensnennungen.
BODY_EXTRA_KEY_HINTS: list[tuple[list[str], list[str]]] = [
    # ([Wörter im Body], [benötigte Keys — eine Gruppe = eine OR-Liste])
    # Falsifiziert t_18b97ae4: 'anthropic model' / 'Claude' in einer
    # Tool-Namen-Liste (Vault-Kuratierung) ist KEINE API-Nutzung. Nur
    # explizite API-/Key-Signalwörter erzwingen den Key.
    (["anthropic api", "anthropic key", "anthropic_key", "anthropic token",
      "anthropic sdk", "claude api"], ["ANTHROPIC_API_KEY"]),
    (["codex"], ["OPENAI_API_KEY"]),
    # Falsifiziert t_18b97ae4/t_4e803114: 'Copilot' als Tool-Name bzw.
    # Dateiname 'recherche-copilot-...' ist keine Nutzung.
    (["copilot api", "copilot token", "copilot key", "github copilot cli",
      "copilot_chat", "copilot-chat"], ["COPILOT_GITHUB_TOKEN"]),
    (["xai api", "grok api", "call grok", "use grok", "xai_key", "xai token"], ["XAI_API_KEY"]),
    # "fal" allein matched deutsche Wörter (Falsch-Positive, falsifiziert,
    # Fallbacks). Nur die markenrechtlichen Schreibweisen erzwingen den Key.
    (["fal.ai", "fal_key", "fal api"], ["FAL_KEY"]),
    (["elevenlabs", "eleven labs"], ["ELEVENLABS_API_KEY"]),
    # Falsifiziert t_e62da05c: eine reine Provider-ID bzw. Modell-Nennung in
    # einer Arbeitsroute ist KEINE Zugangsdaten-Nutzung. Nur explizite
    # Signal-Phrasen erzwingen den Key - analog zur Anthropic-/Copilot-/xAI-
    # Gruppe weiter oben.
    (["openai api", "openai key", "openai_key", "openai token", "openai_token",
      "openai sdk", "openai base url", "openai credentials",
      "gpt-4 api", "gpt-4 key", "gpt-4 token",
      "gpt4 api", "gpt4 key", "gpt4 token"], ["OPENAI_API_KEY"]),
    (["perplexity"], ["PERPLEXITY_API_KEY"]),
    # GitHub: bloße Erwähnung von "github", "github.com" oder ein Repo-Name
    # signalisieren noch keine API-Nutzung. Nur konkrete CLI/API-Aktionen
    # erzwingen GITHUB_TOKEN/GH_TOKEN.
    (["gh ", "git push", "git clone", "pull request", "github rest api",
      "github graphql", "call github api", "github token", "GITHUB_TOKEN",
      "create issue", "create pr", "gh pr", "gh repo", "gh auth",
      "gh issue", "gh release"],
     ["GITHUB_TOKEN", "GH_TOKEN"]),
]

PROFILE_CONFIG_TOOLSETS: dict[str, list[str]] = {
    # Defaults, falls kein config.yaml gelesen werden kann.
    "operator": ["web", "terminal", "file", "skills", "todo", "memory",
                 "session_search", "cronjob", "kanban", "execute_code",
                 "delegate_task"],
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class KeyCheck:
    name: str
    found: bool
    source: str = ""  # env | pass | "" (missing)


@dataclass
class TaskCheck:
    task_id: str
    title: str
    assignee: str
    toolsets: list[str]
    missing_groups: list[list[str]] = field(default_factory=list)
    checked_keys: list[KeyCheck] = field(default_factory=list)
    already_blocked: bool = False
    claimed: bool = False


@dataclass
class ScanResult:
    tasks_checked: int = 0
    already_blocked: int = 0
    claimed_skipped: int = 0
    newly_blocked: int = 0
    checks: list[TaskCheck] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pass-Store Unterstützung
# ---------------------------------------------------------------------------

class PassStoreCache:
    def __init__(self) -> None:
        self._tree: Optional[list[str]] = None
        self._last_fetch: float = 0.0

    def _fetch_tree(self) -> list[str]:
        now = time.time()
        if self._tree is not None and (now - self._last_fetch) < PASS_CACHE_TTL_SECONDS:
            return self._tree
        try:
            result = subprocess.run(
                ["pass", "ls"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                self._tree = []
                self._last_fetch = now
                return self._tree
            paths = []
            for line in result.stdout.splitlines():
                # Entferne Baum-Zeichen und Einrückung
                clean = re.sub(r"^[├└│─\s]+", "", line).strip()
                if clean and not clean.endswith("/"):
                    paths.append(clean)
            self._tree = paths
            self._last_fetch = now
            return paths
        except Exception:
            self._tree = []
            self._last_fetch = now
            return self._tree

    def has_key(self, key_name: str) -> bool:
        # 1. Exakter Eintrag probieren
        try:
            result = subprocess.run(
                ["pass", "show", key_name],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return True
        except Exception:
            pass

        # 2. Pass-Baum nach Eintrag durchsuchen, der auf key_name endet
        key_lower = key_name.lower()
        for path in self._fetch_tree():
            if path.lower().endswith(f"/{key_lower}") or path.lower() == key_lower:
                try:
                    result = subprocess.run(
                        ["pass", "show", path],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        return True
                except Exception:
                    continue
        return False


_PASS_CACHE = PassStoreCache()


# ---------------------------------------------------------------------------
# Key-Auflösung
# ---------------------------------------------------------------------------

def _env_has_key(key_name: str) -> bool:
    value = os.environ.get(key_name, "").strip()
    return bool(value) and value.lower() not in ("", "none", "null", "undefined")


def _profile_env_has_key(profile: str, key_name: str) -> bool:
    env_path = Path.home() / ".hermes" / "profiles" / profile / ".env"
    if not env_path.exists():
        return False
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == key_name and v.strip():
                    return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# xAI-OAuth-Waiver (Auth-Store + config.yaml, siehe XAI_OAUTH_WAIVED_KEYS)
# ---------------------------------------------------------------------------

def _oauth_aware_enabled() -> bool:
    """CRON_PREFLIGHT_OAUTH_AWARE: default on; '0'/'false'/'no' schaltet den
    alten Env-/pass-only-Workflow wieder ein."""
    raw = os.environ.get(OAUTH_AWARE_ENV, "").strip().lower()
    return raw not in ("0", "false", "no")


def _pool_entry_healthy(entry: dict) -> bool:
    """Static-Pendant zum Pool-Refresh-Verhalten (kein Netzwerk-I/O):
    Ein Pool-Eintrag mit last_status='dead' und 'invalid_grant' im
    Refresh-Fehler ist dauerhaft revoked (Refresh wird nicht wieder
    erfolgreich); nur ein spaeterer Reset-Marker (status_cleared_at nach
    last_status_at) belebt ihn wieder."""
    if entry.get("last_status") != "dead":
        return True
    err = str(entry.get("last_error_message") or "") + str(entry.get("last_error_reason") or "")
    if "invalid_grant" not in err:
        return True
    try:
        cleared = float(entry.get("status_cleared_at") or 0)
        status_at = float(entry.get("last_status_at") or 0)
        return bool(cleared and cleared > status_at)
    except (TypeError, ValueError):
        return False


def _auth_store_authorized(profile: str) -> bool:
    """Static-Pendant zu hermes_cli/auth.py: profil-first, global-root-Fallback
    JE Provider (Singleton-Tokens) bzw. Pool-Fallback nur wenn das Profil
    selbst KEINE Pool-Einträge hat (read_credential_pool-Semantik)."""
    for auth_path in (
        Path.home() / ".hermes" / "profiles" / profile / "auth.json",
        Path.home() / ".hermes" / "auth.json",
    ):
        if not auth_path.exists():
            continue
        try:
            with open(auth_path, "r", encoding="utf-8-sig") as f:
                store = json.load(f)
        except Exception:
            continue
        tokens = ((store.get("providers") or {}).get(XAI_OAUTH_PROVIDER) or {}).get("tokens") or {}
        if str(tokens.get("access_token") or tokens.get("refresh_token") or "").strip():
            return True
        entries = (store.get("credential_pool") or {}).get(XAI_OAUTH_PROVIDER) or []
        if isinstance(entries, list) and any(
            isinstance(e, dict) and str(e.get("access_token") or "").strip() and _pool_entry_healthy(e)
            for e in entries
        ):
            return True
    return False


def _profile_is_xai_oauth_authorized(profile: str) -> bool:
    """True, wenn dem Profil xAI-OAuth-Credential-Material statisch zur
    Verfügung steht: config.yaml model.provider=xai-oauth ODER non-empty
    Tokens im Auth-Store (Profil, global-root-Fallback je Provider). Nur
    Present-Checks, keine Werte, keine Netzwerk-I/O."""
    config_path = Path.home() / ".hermes" / "profiles" / profile / "config.yaml"
    try:
        import yaml
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            if ((cfg.get("model") or {}).get("provider")) == XAI_OAUTH_PROVIDER:
                return True
    except Exception:
        pass

    now = time.time()
    cached = _OAUTH_CHECK_CACHE.get(profile)
    if cached and (now - cached[1]) < _OAUTH_CACHE_TTL_SECONDS:
        return cached[0]
    authorized = _auth_store_authorized(profile)
    _OAUTH_CHECK_CACHE[profile] = (authorized, now)
    return authorized


_OAUTH_CHECK_CACHE: dict[str, tuple[bool, float]] = {}


# TTS-Provider -> Key-Gruppen (Quelle: docs/user-guide/features/tts.md).
# Freie/lokale Provider brauchen keinen Key. Unbekannter/fehlender Provider ->
# None -> legacy volle Gruppe (konservativ).
TTS_PROVIDER_KEY_GROUPS: dict[str, Optional[list[list[str]]]] = {
    "edge": [],
    "neutts": [],
    "kittentts": [],
    "piper": [],
    "nous": [],
    "xai": [["XAI_API_KEY"]],
    "elevenlabs": [["ELEVENLABS_API_KEY"]],
    "openai": [["OPENAI_API_KEY"]],
    "minimax": [["MINIMAX_API_KEY"]],
    "mistral": [["MISTRAL_API_KEY"]],
}


def _profile_tts_groups(profile: str) -> Optional[list[list[str]]]:
    config_path = Path.home() / ".hermes" / "profiles" / profile / "config.yaml"
    try:
        import yaml
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            tts_cfg = cfg.get("tts") or {}
            provider = str(tts_cfg.get("provider") or "").strip().lower()
            if provider:
                return TTS_PROVIDER_KEY_GROUPS.get(provider)
    except Exception:
        pass
    return None


# Bild-/Video-Provider -> Key-Gruppen. Provider-aware Pendant zu
# TTS_PROVIDER_KEY_GROUPS: ein Toolset braucht den Key seines konfigurierten
# Providers, nicht unbedingt FAL_KEY. "xai" deckt xAI ab, wobei XAI_API_KEY
# über den OAuth-Waiver (XAI_OAUTH_WAIVED_KEYS, Auth-Store) alternativ erfüllt
# wird. "nous" = managed Nous-Gateway (Credentials serverseitig) -> kein Key.
# Unbekannter/fehlender Provider -> None -> legacy volle Gruppe (konservativ,
# fail-closed).
MEDIA_PROVIDER_KEY_GROUPS: dict[str, dict[str, Optional[list[list[str]]]]] = {
    "image_gen": {
        "fal": [["FAL_KEY"]],
        "xai": [["XAI_API_KEY"]],
        "xai-oauth": [["XAI_API_KEY"]],
        "nous": [],
    },
    "video_gen": {
        # fal: legacy-OR-Gruppe unverändert (FAL_KEY ODER XAI_API_KEY) —
        # bestehendes Verhalten nicht verschärfen.
        "fal": [["FAL_KEY", "XAI_API_KEY"]],
        "xai": [["XAI_API_KEY"]],
        "xai-oauth": [["XAI_API_KEY"]],
        "nous": [],
    },
}

# Dedupe-Marker je Media-Toolset (analog "__tts__"), damit die legacy-Gruppe
# nicht doppelt landet, wenn der Provider-Key-Override greift.
_MEDIA_TOOLSET_SECTIONS = {
    "image_gen": ("__image__", "image_gen"),
    "video_gen": ("__video__", "video_gen"),
}


def _profile_media_groups(profile: str, toolset: str) -> Optional[list[list[str]]]:
    """Key-Gruppen für ein Media-Toolset aus dem konfigurierten Provider des
    Profils (config.yaml <toolset>.provider), analog _profile_tts_groups.
    None = kein bekannter Provider -> legacy volle Toolset-Gruppe."""
    section = _MEDIA_TOOLSET_SECTIONS[toolset][1]
    provider_map = MEDIA_PROVIDER_KEY_GROUPS[toolset]
    config_path = Path.home() / ".hermes" / "profiles" / profile / "config.yaml"
    try:
        import yaml
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            media_cfg = cfg.get(section) or {}
            provider = str(media_cfg.get("provider") or "").strip().lower()
            if provider:
                return provider_map.get(provider)
    except Exception:
        pass
    return None


def _check_key(key_name: str, profile: str) -> KeyCheck:
    if _oauth_aware_enabled() and key_name in XAI_OAUTH_WAIVED_KEYS \
            and _profile_is_xai_oauth_authorized(profile):
        return KeyCheck(name=key_name, found=True, source="xai-oauth")
    if _env_has_key(key_name):
        return KeyCheck(name=key_name, found=True, source="env")
    if _profile_env_has_key(profile, key_name):
        return KeyCheck(name=key_name, found=True, source="profile_env")
    if _PASS_CACHE.has_key(key_name):
        return KeyCheck(name=key_name, found=True, source="pass")
    return KeyCheck(name=key_name, found=False, source="")


def _check_group(keys: Iterable[str], profile: str) -> tuple[bool, list[KeyCheck]]:
    """Gruppe erfüllt, wenn mindestens ein Key vorhanden ist."""
    checks = [_check_key(k, profile) for k in keys]
    satisfied = any(c.found for c in checks)
    return satisfied, checks


# ---------------------------------------------------------------------------
# Toolset-Ableitung
# ---------------------------------------------------------------------------

def _load_profile_default_toolsets(profile: str) -> set[str]:
    """Liest toolsets aus der Profil-config.yaml, falls vorhanden."""
    config_path = Path.home() / ".hermes" / "profiles" / profile / "config.yaml"
    if not config_path.exists():
        return set(PROFILE_CONFIG_TOOLSETS.get(profile, []))
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        toolsets = cfg.get("toolsets", [])
        if isinstance(toolsets, list):
            return set(toolsets)
    except Exception:
        pass
    return set(PROFILE_CONFIG_TOOLSETS.get(profile, []))


def _hint_matches(hint: str, body_lower: str) -> bool:
    """Wortgrenzen-Match statt blindem Substring.

    Blindes Substring-Matching erzeugte falsche API-Key-Blocks:
    'tts' in 'Trinity-Schnit*tts*telle', 'gh ' in 'enou*gh *that',
    'openai' in 'openAI-*kompatibel*'. Ein hint mit Wortzeichen an den
    Raendern braucht Boundaries; reine Satzzeichen-/Symbol-Hints
    ('gh pr', 'fal.ai', 'gpt-4') bleiben als Phrase-Substring matchbar.
    """
    pattern = r"(?<![a-z0-9_-])" + re.escape(hint) + r"(?![a-z0-9_-])"
    return re.search(pattern, body_lower) is not None


def _infer_toolsets_from_body(body: str) -> set[str]:
    body_lower = (body or "").lower()
    found: set[str] = set()
    for toolset, hints in BODY_TOOLSET_HINTS.items():
        for hint in hints:
            if _hint_matches(hint.lower(), body_lower):
                found.add(toolset)
                break
    return found


def _infer_extra_required_keys(body: str) -> list[list[str]]:
    """Body-Hinweise, die zusätzliche Key-Gruppen erzwingen."""
    body_lower = (body or "").lower()
    groups: list[list[str]] = []
    for hints, keys in BODY_EXTRA_KEY_HINTS:
        if any(_hint_matches(h.lower(), body_lower) for h in hints):
            groups.append(keys)
    return groups


def _collect_required_keys(toolsets: Iterable[str], body: str = "",
                           profile: str = "") -> list[list[str]]:
    groups: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    profile_tts_groups = _profile_tts_groups(profile) if profile else None
    profile_media_groups: dict[str, Optional[list[list[str]]]] = {
        ts: _profile_media_groups(profile, ts) if profile else None
        for ts in _MEDIA_TOOLSET_SECTIONS
    } if profile else {}
    for ts in toolsets:
        for group in TOOLSET_REQUIRED_KEYS.get(ts, []):
            override: Optional[list[list[str]]] = None
            marker: Optional[tuple[str, ...]] = None
            if ts == "tts" and profile_tts_groups is not None:
                override, marker = profile_tts_groups, ("__tts__",)
            elif ts in _MEDIA_TOOLSET_SECTIONS and profile_media_groups.get(ts) is not None:
                marker = _MEDIA_TOOLSET_SECTIONS[ts][0]
                override = profile_media_groups[ts]
            key = marker if marker is not None else tuple(sorted(group))
            if key not in seen:
                seen.add(key)
                if override is not None:
                    groups.extend(override)
                else:
                    groups.append(list(group))
    for group in _infer_extra_required_keys(body):
        key = tuple(sorted(group))
        if key not in seen:
            seen.add(key)
            groups.append(list(group))
    return groups


# ---------------------------------------------------------------------------
# Kanban-DB Zugriff
# ---------------------------------------------------------------------------

def _get_ready_tasks(conn: sqlite3.Connection) -> list[dict]:
    now = int(time.time())
    cursor = conn.execute(
        "SELECT id, title, body, assignee, status, skills, block_kind, "
        "       claim_lock, claim_expires "
        "FROM tasks "
        "WHERE status = 'ready'"
    )
    rows = []
    for row in cursor.fetchall():
        rows.append(dict(row))
    return rows


def _block_task(conn: sqlite3.Connection, task_id: str, reason: str) -> None:
    now = int(time.time())
    with conn:
        conn.execute(
            "UPDATE tasks SET status = 'blocked', block_kind = ? "
            "WHERE id = ? AND status = 'ready'",
            ("missing-api-keys", task_id),
        )
        payload = json.dumps({"reason": reason}, ensure_ascii=False)
        conn.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
            "VALUES (?, NULL, 'blocked', ?, ?)",
            (task_id, payload, now),
        )
        conn.execute(
            "INSERT INTO task_comments (task_id, author, body, created_at) "
            "VALUES (?, ?, ?, ?)",
            (task_id, "preflight-api-keys", reason, now),
        )


# ---------------------------------------------------------------------------
# Hauptlogik
# ---------------------------------------------------------------------------

def _run_scan(board_path: Path, dry_run: bool = False) -> ScanResult:
    result = ScanResult()
    conn = sqlite3.connect(board_path)
    conn.row_factory = sqlite3.Row

    try:
        ready_tasks = _get_ready_tasks(conn)
        for task in ready_tasks:
            result.tasks_checked += 1
            task_id = task["id"]
            title = task["title"] or ""
            body = task["body"] or ""
            assignee = task["assignee"] or "operator"

            check = TaskCheck(
                task_id=task_id,
                title=title,
                assignee=assignee,
                toolsets=[],
            )

            # Bereits geblockt mit diesem Grund -> überspringen
            if task.get("status") == "blocked" and task.get("block_kind") == "missing-api-keys":
                check.already_blocked = True
                result.already_blocked += 1
                result.checks.append(check)
                continue

            # Gerade geclaimt -> nicht blocken, Worker startet ohnehin gleich
            claim_expires = task.get("claim_expires") or 0
            if task.get("claim_lock") and claim_expires > int(time.time()):
                check.claimed = True
                result.claimed_skipped += 1
                result.checks.append(check)
                continue

            # Toolsets ableiten
            body_toolsets = _infer_toolsets_from_body(body)
            toolsets = _load_profile_default_toolsets(assignee)
            # Phantom-blocker guard: do not treat a toolset as required just
            # because a profile lists it in config.yaml. Only require API keys
            # for body-driven toolsets or when the task body explicitly signals
            # that toolset's work. This prevents false blocks from profile defaults
            # (e.g. x_search, image_gen, obsidian) on tasks that never use them.
            profile_only_toolsets = {
                "web", "search", "x_search", "image_gen", "video_gen", "tts",
                "discord", "discord_admin", "homeassistant", "obsidian",
            }
            for ts in profile_only_toolsets:
                if ts in toolsets and ts not in body_toolsets:
                    toolsets.discard(ts)
            toolsets |= body_toolsets

            # Task skills können Toolset-Hinweise enthalten (JSON-Array)
            try:
                skills = json.loads(task.get("skills") or "[]") or []
                if isinstance(skills, list):
                    for skill in skills:
                        if isinstance(skill, str):
                            toolsets |= _infer_toolsets_from_body(skill)
            except Exception:
                pass

            check.toolsets = sorted(toolsets)

            # Benötigte Keys prüfen
            required_groups = _collect_required_keys(toolsets, body, assignee)
            all_checked: list[KeyCheck] = []
            missing_groups: list[list[str]] = []
            for group in required_groups:
                satisfied, checks = _check_group(group, assignee)
                all_checked.extend(checks)
                if not satisfied:
                    missing_groups.append(group)

            check.checked_keys = all_checked
            check.missing_groups = missing_groups

            if missing_groups:
                missing_str = "; ".join(
                    f"one of [{', '.join(g)}]" for g in missing_groups
                )
                reason = (
                    f"missing-api-keys: profile '{assignee}' lacks required "
                    f"API key(s) for toolsets {check.toolsets}: {missing_str}"
                )
                if dry_run:
                    print(f"[DRY-RUN] would block {task_id}: {reason}")
                else:
                    _block_task(conn, task_id, reason)
                result.newly_blocked += 1

            result.checks.append(check)
    finally:
        conn.close()

    return result


def _format_report(result: ScanResult, dry_run: bool) -> str:
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("KANBAN API-KEY PRE-FLIGHT REPORT")
    lines.append("=" * 60)
    lines.append(f"Mode:          {'DRY-RUN' if dry_run else 'LIVE'}")
    lines.append(f"Tasks checked: {result.tasks_checked}")
    lines.append(f"Already blocked (missing-api-keys): {result.already_blocked}")
    lines.append(f"Skipped (claimed): {result.claimed_skipped}")
    lines.append(f"Newly blocked:   {result.newly_blocked}")
    lines.append("")

    for check in result.checks:
        if check.already_blocked or check.claimed:
            continue
        if not check.missing_groups:
            continue
        lines.append(f"  Task: {check.task_id} — {check.title}")
        lines.append(f"    Assignee: {check.assignee}")
        lines.append(f"    Toolsets: {', '.join(check.toolsets)}")
        for group in check.missing_groups:
            lines.append(f"    Missing key group: one of [{', '.join(group)}]")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Pre-Flight-Check für fehlende API-Keys in ready Kanban-Tasks."
    )
    parser.add_argument(
        "--board",
        type=Path,
        default=DEFAULT_BOARD,
        help=f"Pfad zur kanban.db (default: {DEFAULT_BOARD})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Nur Report erstellen, keine Tasks blocken.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Immer Report ausgeben, auch wenn keine Tasks geblockt wurden.",
    )
    args = parser.parse_args(argv)

    if not args.board.exists():
        print(f"Board not found: {args.board}", file=sys.stderr)
        return 2

    result = _run_scan(args.board, dry_run=args.dry_run)

    # Silent unless there is actionable work. Cron no-agent jobs deliver
    # stdout verbatim, so empty stdout means no notification.
    if result.newly_blocked > 0 or args.verbose:
        print(_format_report(result, dry_run=args.dry_run))

    return 1 if result.newly_blocked > 0 else 0


def check_single_task(
    board_path: Path, task_id: str, status: str = "ready",
) -> tuple[bool, str]:
    """Pruefe ein einzelnes Task auf fehlende API-Keys.

    Args:
        status: Claim-Quellstatus, der geprüft wird (``ready`` für den
            Implementer-Claim, ``review`` für den Reviewer-Claim — beide
            Dispatch-Pfade müssen durch dasselbe Gate, sonst wäre die
            Review-Lane ein Bypass).

    Returns:
        (should_block, reason) — reason ist leer wenn should_block False.
    """
    conn = sqlite3.connect(board_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT id, title, body, assignee, status, skills FROM tasks "
            "WHERE id = ? AND status = ?",
            (task_id, status),
        ).fetchone()
        if not row:
            return False, ""

        body = row["body"] or ""
        assignee = row["assignee"] or "operator"

        # Toolsets ableiten (gleiche Logik wie _run_scan)
        body_toolsets = _infer_toolsets_from_body(body)
        toolsets = _load_profile_default_toolsets(assignee)
        profile_only_toolsets = {
            "web", "search", "x_search", "image_gen", "video_gen", "tts",
            "discord", "discord_admin", "homeassistant", "obsidian",
        }
        for ts in profile_only_toolsets:
            if ts in toolsets and ts not in body_toolsets:
                toolsets.discard(ts)
        toolsets |= body_toolsets

        try:
            skills = json.loads(row["skills"] or "[]") or []
            if isinstance(skills, list):
                for skill in skills:
                    if isinstance(skill, str):
                        toolsets |= _infer_toolsets_from_body(skill)
        except Exception:
            pass

        # Keys pruefen
        required_groups = _collect_required_keys(toolsets, body, assignee)
        missing_groups: list[list[str]] = []
        for group in required_groups:
            satisfied, _ = _check_group(group, assignee)
            if not satisfied:
                missing_groups.append(group)

        if missing_groups:
            missing_str = "; ".join(
                f"one of [{', '.join(g)}]" for g in missing_groups
            )
            reason = (
                f"missing-api-keys: profile '{assignee}' lacks required "
                f"API key(s) for toolsets {sorted(toolsets)}: {missing_str}"
            )
            return True, reason

        return False, ""
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
