#!/usr/bin/env python3
"""Discord-Brücke für Claude Code (PermissionRequest-Hook).

Leitet genau die Nachfragen, die Claude Code selbst stellen würde, an die
Hermes-Discord-Bridge weiter — aber nur, wenn Claude Code von Hermes
gestartet wurde (z. B. ``claude -p`` aus dem Terminal-Werkzeug). In einer
normalen interaktiven Sitzung tut der Hook nichts, Claude Code fragt wie
gewohnt im Terminal.

Was weitergeleitet wird:

- Berechtigungsanfragen („Darf ich ausführen?“) für jedes Werkzeug
  (Bash, Edit, Write, WebFetch, MCP-Werkzeuge …), und zwar nur dort, wo
  Claude Code ohne den Hook nachfragen würde.
- ``AskUserQuestion`` (Multiple-Choice-Rückfragen): Fragen und Optionen
  erscheinen als Buttons in Discord, die Antworten gehen per
  ``updatedInput`` zurück an Claude Code (nur in interaktiven Sitzungen —
  im ``-p``-Modus bietet Claude Code das Werkzeug gar nicht an).

Sitzungs-Threads: ``UserPromptSubmit`` meldet den Prompt (der erste einer
Sitzung öffnet den Thread in Discord), ``Stop`` meldet die fertige Antwort
der Runde. Beide sind reine Meldungen — sie legen eine Datei ab und warten
auf nichts, blockieren die Sitzung also nie. Die Rückfragen derselben
Sitzung stellt die Bridge dann in diesem Thread.

Drei Hook-Ereignisarten, ein Skript:

- ``PermissionRequest`` feuert genau dann, wenn Claude Code seinen
  Rückfrage-Dialog zeigen würde (interaktive Sitzungen, etwa in tmux).
- ``PreToolUse`` ist für ``claude -p`` nötig: dort gibt es keinen Dialog,
  Werkzeuge ohne Freigabe werden still abgelehnt. Der Hook bildet deshalb
  Claude Codes Regelwerk nach (``permissions.allow/deny`` aus den
  settings.json-Dateien, ``--allowedTools``/``--disallowedTools`` aus der
  Kommandozeile, ``permission_mode``, nie fragende Werkzeuge wie Read/Glob)
  und fragt nur bei Werkzeugen nach, die sonst abgelehnt würden. Wird das
  Werkzeug per PreToolUse erlaubt, gibt es keinen zweiten Dialog mehr.

Erkennung „von Hermes gestartet“ (eine Bedingung genügt):

- ``HERMES_CLAUDE_BRIDGE=1`` (oder ``an``/``on``) in der Umgebung — erzwingt
  die Weiterleitung, auch aus einem normalen Terminal;
- ``HERMES_HOME`` in der Umgebung — das Gateway setzt sie, Kindprozesse
  erben sie;
- ein Vorfahrenprozess gehört zu Hermes (``hermes_cli``, ``hermes-agent``).

Ausschalten, ohne die settings.json anzufassen: Datei
``~/.claude/discord-bridge.aus`` anlegen (löschen schaltet wieder ein) oder
``HERMES_CLAUDE_BRIDGE=aus`` setzen. Bei Ablehnung, Timeout, nicht laufendem
Gateway oder unklarer Antwort wird abgelehnt; Dauerfreigaben gibt es nicht.

Einrichtung in ``~/.claude/settings.json`` (Hook-Timeout über der Wartezeit,
Standard 330 s)::

    "hooks": {
      "PermissionRequest": [
        {"hooks": [{"type": "command",
                    "command": "python3 ~/.claude/hooks/discord-bridge.py",
                    "timeout": 360}]}
      ],
      "PreToolUse": [
        {"hooks": [{"type": "command",
                    "command": "python3 ~/.claude/hooks/discord-bridge.py",
                    "timeout": 360}]}
      ],
      "UserPromptSubmit": [
        {"hooks": [{"type": "command",
                    "command": "python3 ~/.claude/hooks/discord-bridge.py",
                    "timeout": 15}]}
      ],
      "Stop": [
        {"hooks": [{"type": "command",
                    "command": "python3 ~/.claude/hooks/discord-bridge.py",
                    "timeout": 15}]}
      ]
    }
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shlex
import subprocess
import sys
import time
import uuid

HERMES_HOME = os.environ.get("HERMES_HOME") or os.path.join(os.path.expanduser("~"), ".hermes")
ABLAGE = os.path.join(HERMES_HOME, "opencode_bridge")
WARTEZEIT_S = float(os.environ.get("BEFEHLSWAECHTER_WARTEZEIT_S") or 330)
ABFRAGE_S = 0.5
HEARTBEAT_MAX_ALTER_S = 120
AUS_DATEI = os.path.join(os.path.expanduser("~"), ".claude", "discord-bridge.aus")
# Neben threads.json der Bridge, damit der Sitzungszustand demselben
# HERMES_HOME folgt wie die Ablage (und Tests ihn sauber umlenken können).
SITZUNGEN_DATEI = os.path.join(ABLAGE, "claude_sessions.json")
SITZUNG_TTL_S = 7 * 24 * 3600
NOTIZ_TEXT_MAX = 1700
TRANSCRIPT_TAIL_BYTES = 4_000_000  # nur das Ende lesen; Transkripte werden groß
SCHALTER_ENV = "HERMES_CLAUDE_BRIDGE"
DETAILS_MAX = 900

LESEND_WERKZEUGE = {"Read", "Glob", "Grep", "LS", "NotebookRead"}
SCHREIBEND_WERKZEUGE = {"Edit", "Write", "MultiEdit", "NotebookEdit"}
NETZ_WERKZEUGE = {"WebFetch", "WebSearch"}
# Werkzeuge, für die Claude Code nie einen Rechte-Dialog zeigt.
NIE_NACHFRAGEN = LESEND_WERKZEUGE | {
    "TodoWrite", "TodoRead", "Task", "Agent", "ToolSearch", "ListMcpResourcesTool",
    "ReadMcpResourceTool", "BashOutput", "KillShell", "TaskOutput", "TaskStop",
    "AskUserQuestion", "EnterPlanMode", "ExitPlanMode", "Skill", "SendUserFile",
    "Monitor", "ScheduleWakeup", "ReportFindings",
}
# Modi, in denen Claude Code selbst entscheidet (kein Dialog, den wir auslagern könnten).
MODI_OHNE_DIALOG = {"bypassPermissions", "dontAsk", "auto", "plan"}


# ---------------------------------------------------------------------------
# Schalter und Hermes-Erkennung
# ---------------------------------------------------------------------------


def ausgeschaltet(env: dict | None = None) -> bool:
    env = os.environ if env is None else env
    if os.path.exists(AUS_DATEI):
        return True
    return (env.get(SCHALTER_ENV) or "").strip().lower() in {"aus", "off", "0", "false"}


def hermes_vorfahr(pid: int | None = None, *, max_tiefe: int = 15) -> bool:
    """Gehört ein Elternprozess zu Hermes? (Gateway, CLI oder deren Kinder)"""
    pid = os.getppid() if pid is None else pid
    muster = re.compile(r"hermes_cli|hermes-agent|hermes\s+gateway|/\.hermes/")
    for _ in range(max_tiefe):
        if pid <= 1:
            return False
        try:
            out = subprocess.run(
                ["ps", "-o", "ppid=,command=", "-p", str(pid)],
                capture_output=True, text=True, timeout=3,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return False
        if not out:
            return False
        ppid_text, _, befehl = out.partition(" ")
        if muster.search(befehl):
            return True
        try:
            pid = int(ppid_text)
        except ValueError:
            return False
    return False


def von_hermes_gestartet(env: dict | None = None, *, vorfahren_pruefen: bool = True) -> bool:
    env = os.environ if env is None else env
    if (env.get(SCHALTER_ENV) or "").strip().lower() in {"1", "an", "on", "true"}:
        return True
    if env.get("HERMES_HOME"):
        return True
    return vorfahren_pruefen and hermes_vorfahr()


def aktiv(env: dict | None = None) -> bool:
    return not ausgeschaltet(env) and von_hermes_gestartet(env)


# ---------------------------------------------------------------------------
# Regelwerk nachbilden: würde Claude Code für dieses Werkzeug nachfragen?
# ---------------------------------------------------------------------------


def _settings_regeln(pfad: str) -> dict:
    try:
        with open(pfad, encoding="utf-8") as fh:
            daten = json.load(fh)
    except (OSError, ValueError):
        return {"allow": [], "deny": [], "ask": []}
    perms = daten.get("permissions") if isinstance(daten, dict) else None
    perms = perms if isinstance(perms, dict) else {}
    return {
        art: [str(r) for r in perms.get(art, []) if isinstance(r, str)]
        for art in ("allow", "deny", "ask")
    }


def regeln_laden(cwd: str) -> dict:
    """allow/deny/ask aus Nutzer-, Projekt- und lokalen Einstellungen."""
    gesamt = {"allow": [], "deny": [], "ask": []}
    for pfad in (
        os.path.join(os.path.expanduser("~"), ".claude", "settings.json"),
        os.path.join(cwd, ".claude", "settings.json"),
        os.path.join(cwd, ".claude", "settings.local.json"),
    ):
        for art, regeln in _settings_regeln(pfad).items():
            gesamt[art].extend(regeln)
    return gesamt


def _argv_werte(argv: list[str], namen: set[str]) -> list[str]:
    """Werte hinter --allowedTools & Co. einsammeln (Komma- oder Leerzeichen-getrennt)."""
    werte: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        name, _, inline = tok.partition("=")
        if name in namen:
            if inline:
                werte.extend(t.strip() for t in inline.split(",") if t.strip())
            else:
                i += 1
                while i < len(argv) and not argv[i].startswith("-"):
                    werte.extend(t.strip() for t in argv[i].split(",") if t.strip())
                    i += 1
                continue
        i += 1
    # ps trennt "Bash(git *)" in zwei Wörter — Klammern wieder zusammensetzen
    zusammen: list[str] = []
    for w in werte:
        if zusammen and zusammen[-1].count("(") > zusammen[-1].count(")"):
            zusammen[-1] += " " + w
        else:
            zusammen.append(w)
    return zusammen


def cli_regeln(ppid: int | None = None) -> dict:
    """--allowedTools/--disallowedTools des aufrufenden claude-Prozesses."""
    leer = {"allow": [], "deny": []}
    pid = os.getppid() if ppid is None else ppid
    try:
        out = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True, timeout=3).stdout.strip()
        argv = shlex.split(out)
    except (OSError, subprocess.SubprocessError, ValueError):
        return leer
    if not any(os.path.basename(a) == "claude" for a in argv[:2]):
        return leer
    return {
        "allow": _argv_werte(argv, {"--allowedTools", "--allowed-tools"}),
        "deny": _argv_werte(argv, {"--disallowedTools", "--disallowed-tools"}),
    }


def _pfad_passt(muster: str, pfad: str, cwd: str) -> bool:
    if not pfad:
        return False
    absolut = os.path.normpath(os.path.join(cwd, os.path.expanduser(pfad)))
    if muster.startswith("//"):
        return fnmatch.fnmatch(absolut, muster[1:])
    if muster.startswith("~/"):
        return fnmatch.fnmatch(absolut, os.path.expanduser(muster))
    if muster.startswith("/"):
        return fnmatch.fnmatch(absolut, os.path.normpath(os.path.join(cwd, muster.lstrip("/"))))
    rel = os.path.relpath(absolut, cwd)
    return fnmatch.fnmatch(rel, muster) or fnmatch.fnmatch(absolut, muster) or fnmatch.fnmatch(rel, os.path.join("**", muster))


def regel_passt(regel: str, werkzeug: str, eingabe: dict, cwd: str) -> bool:
    """Eine Claude-Code-Berechtigungsregel gegen einen Werkzeugaufruf prüfen."""
    m = re.match(r"^([^(]+)(?:\((.*)\))?$", regel.strip(), re.S)
    if not m:
        return False
    name, muster = m.group(1).strip(), m.group(2)
    if name != werkzeug and not (name.startswith("mcp__") and werkzeug.startswith(name + "__")):
        return False
    if muster is None or muster.strip() in ("", "*"):
        return True
    muster = muster.strip()
    if werkzeug == "Bash":
        befehl = str(eingabe.get("command") or "").strip()
        if muster.endswith(":*"):
            praefix = muster[:-2]
            return befehl == praefix or befehl.startswith(praefix + " ") or befehl.startswith(praefix)
        if muster.endswith(" *"):
            praefix = muster[:-2]
            return befehl == praefix or befehl.startswith(praefix + " ")
        if muster.endswith("*"):
            return befehl.startswith(muster[:-1])
        return befehl == muster
    if werkzeug in SCHREIBEND_WERKZEUGE | LESEND_WERKZEUGE:
        pfad = str(eingabe.get("file_path") or eingabe.get("notebook_path") or eingabe.get("path") or "")
        return _pfad_passt(muster, pfad, cwd)
    if werkzeug in NETZ_WERKZEUGE:
        ziel = str(eingabe.get("url") or eingabe.get("query") or "")
        if muster.startswith("domain:"):
            host = re.sub(r"^[a-z]+://", "", ziel).split("/")[0].lower()
            return host == muster[7:].lower() or host.endswith("." + muster[7:].lower())
        return fnmatch.fnmatch(ziel, muster)
    return fnmatch.fnmatch(json.dumps(eingabe, sort_keys=True), muster)


def braucht_erlaubnis(werkzeug: str, eingabe: dict, permission_mode: str, cwd: str, regeln: dict) -> bool:
    """True, wenn Claude Code diesen Aufruf ohne Freigabe nicht ausführen würde."""
    if werkzeug in NIE_NACHFRAGEN:
        return False
    if permission_mode in MODI_OHNE_DIALOG:
        return False
    if permission_mode == "acceptEdits" and werkzeug in SCHREIBEND_WERKZEUGE:
        return False
    for regel in regeln.get("deny", []):
        if regel_passt(regel, werkzeug, eingabe, cwd):
            return False  # Claude Code lehnt selbst ab — nichts zu fragen
    for regel in regeln.get("allow", []):
        if regel_passt(regel, werkzeug, eingabe, cwd):
            return False
    return True


# ---------------------------------------------------------------------------
# Anfrage aus dem Hook-Input bauen
# ---------------------------------------------------------------------------


def _kurz(text: str, maximum: int = DETAILS_MAX) -> str:
    text = str(text or "")
    return text if len(text) <= maximum else text[: maximum - 1] + "…"


def anfrage_aus_hook(daten: dict) -> dict:
    """Übersetzt ein PermissionRequest-Ereignis in eine Bridge-Anfrage."""
    werkzeug = str(daten.get("tool_name") or "?")
    eingabe = daten.get("tool_input") or {}
    if not isinstance(eingabe, dict):
        eingabe = {"input": eingabe}
    projekt = str(daten.get("cwd") or os.getcwd())

    if werkzeug == "AskUserQuestion":
        fragen = []
        for frage in eingabe.get("questions") or []:
            if not isinstance(frage, dict):
                continue
            optionen = [
                {"label": str(o.get("label", "")), "description": str(o.get("description", ""))}
                for o in (frage.get("options") or []) if isinstance(o, dict)
            ]
            fragen.append({
                "question": str(frage.get("question", "")),
                "header": str(frage.get("header", "")),
                "options": optionen,
                "multiSelect": bool(frage.get("multiSelect", False)),
            })
        return {"kind": "question", "questions": fragen, "project": projekt, "tool": werkzeug}

    pfad, zugriff, details = "-", "unklar", ""
    if werkzeug == "Bash":
        befehl = str(eingabe.get("command") or "").strip() or "(leer)"
        details = str(eingabe.get("description") or "")
        zugriff = "ausführen"
    elif werkzeug in SCHREIBEND_WERKZEUGE:
        pfad = str(eingabe.get("file_path") or eingabe.get("notebook_path") or "-")
        befehl = f"{werkzeug} {pfad}"
        zugriff = "schreiben"
        if werkzeug == "Write":
            details = _kurz(str(eingabe.get("content") or ""), 400)
        elif werkzeug == "Edit":
            details = "alt: " + _kurz(str(eingabe.get("old_string") or ""), 300) + "\nneu: " + _kurz(str(eingabe.get("new_string") or ""), 300)
    elif werkzeug in LESEND_WERKZEUGE:
        pfad = str(eingabe.get("file_path") or eingabe.get("path") or "-")
        ziel = pfad if pfad != "-" else str(eingabe.get("pattern") or "")
        befehl = f"{werkzeug} {ziel}".strip()
        zugriff = "lesen"
    elif werkzeug in NETZ_WERKZEUGE:
        ziel = str(eingabe.get("url") or eingabe.get("query") or "")
        befehl = f"{werkzeug} {ziel}".strip()
        zugriff = "netz"
        details = str(eingabe.get("prompt") or "")
    else:
        befehl = werkzeug
        try:
            details = json.dumps(eingabe, ensure_ascii=False)
        except (TypeError, ValueError):
            details = str(eingabe)
    return {
        "kind": "permission",
        "tool": werkzeug,
        "command": befehl,
        "path": pfad or "-",
        "access": zugriff,
        "details": _kurz(details),
        "project": projekt,
    }


# ---------------------------------------------------------------------------
# Bridge-Ablage
# ---------------------------------------------------------------------------


def bruecke_ausgefallen(hermes_home: str = HERMES_HOME) -> str | None:
    try:
        with open(os.path.join(hermes_home, "state", "gateway.heartbeat"), encoding="utf-8") as fh:
            heartbeat = json.load(fh)
    except (OSError, ValueError):
        return "kein Gateway-Heartbeat gefunden"
    updated = str(heartbeat.get("updated_at") or "")
    try:
        from datetime import datetime

        alter = time.time() - datetime.fromisoformat(updated.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return "Gateway-Heartbeat ohne Zeitstempel"
    if not alter < HEARTBEAT_MAX_ALTER_S:
        return f"Gateway-Heartbeat ist {round(alter)} s alt"
    try:
        with open(os.path.join(hermes_home, "gateway_state.json"), encoding="utf-8") as fh:
            zustand = json.load(fh)
    except (OSError, ValueError):
        return "gateway_state.json nicht lesbar"
    if zustand.get("gateway_state") != "running":
        return f"Gateway-Zustand: {zustand.get('gateway_state') or 'unbekannt'}"
    discord = (zustand.get("platforms") or {}).get("discord", {}).get("state")
    if discord != "connected":
        return f"Discord ist {discord or 'nicht konfiguriert'}"
    return None


def bruecke_fragen(
    anfrage: dict,
    *,
    session_id: str,
    ablage: str = ABLAGE,
    wartezeit_s: float = WARTEZEIT_S,
    abfrage_s: float = ABFRAGE_S,
    ausfallpruefung: bool = True,
    hermes_home: str = HERMES_HOME,
) -> dict:
    """Anfrage ablegen, auf die Entscheidung warten.

    Rückgabe: ``{"erlaubt": bool, "grund": str, "answers": dict | None}``.
    Alles außer einer sauberen once-/answer-Antwort mit passender ID zählt
    als Ablehnung.
    """
    if ausfallpruefung:
        ausfall = bruecke_ausgefallen(hermes_home)
        if ausfall:
            return {"erlaubt": False, "grund": f"Discord-Bridge nicht erreichbar: {ausfall}", "answers": None}
    anfragen = os.path.join(ablage, "requests")
    antworten = os.path.join(ablage, "decisions")
    request_id = str(uuid.uuid4())
    jetzt = time.time()
    nutzlast = {
        "version": 1,
        "id": request_id,
        "agent": "claude-code",
        "created_at": jetzt,
        "expires_at": jetzt + wartezeit_s,
        "session_id": session_id,
        **anfrage,
    }
    anfrage_datei = os.path.join(anfragen, f"{request_id}.json")
    antwort_datei = os.path.join(antworten, f"{request_id}.json")
    try:
        for d in (ablage, anfragen, antworten):
            os.makedirs(d, mode=0o700, exist_ok=True)
        tmp = os.path.join(anfragen, f".tmp-{request_id}")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(nutzlast, fh, ensure_ascii=False)
        os.chmod(tmp, 0o600)
        os.replace(tmp, anfrage_datei)
    except OSError as exc:
        return {"erlaubt": False, "grund": f"Anfrage konnte nicht abgelegt werden ({exc})", "answers": None}

    ende = time.monotonic() + wartezeit_s
    try:
        while time.monotonic() < ende:
            try:
                with open(antwort_datei, encoding="utf-8") as fh:
                    roh = fh.read()
            except OSError:
                time.sleep(abfrage_s)
                continue
            try:
                antwort = json.loads(roh)
            except ValueError:
                return {"erlaubt": False, "grund": "unlesbare Antwort der Bridge", "answers": None}
            if not isinstance(antwort, dict) or antwort.get("version") != 1 or antwort.get("id") != request_id:
                return {"erlaubt": False, "grund": "Antwort der Bridge passt nicht zur Anfrage", "answers": None}
            entscheidung = antwort.get("decision")
            if entscheidung == "once" and anfrage.get("kind") == "permission":
                return {"erlaubt": True, "grund": "in Discord einmalig erlaubt", "answers": None}
            if entscheidung == "answer" and anfrage.get("kind") == "question" and isinstance(antwort.get("answers"), dict):
                return {"erlaubt": True, "grund": "in Discord beantwortet", "answers": antwort["answers"]}
            if entscheidung == "reject":
                if antwort.get("source") == "timeout":
                    return {"erlaubt": False, "grund": "in Discord nicht rechtzeitig beantwortet", "answers": None}
                return {"erlaubt": False, "grund": "in Discord abgelehnt", "answers": None}
            return {"erlaubt": False, "grund": f"unpassende Entscheidung „{entscheidung}“", "answers": None}
        return {"erlaubt": False, "grund": f"keine Antwort innerhalb von {int(wartezeit_s)} s", "answers": None}
    finally:
        for p in (anfrage_datei, antwort_datei):
            try:
                os.unlink(p)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Sitzungs-Meldungen (Thread je Claude-Code-Sitzung)
# ---------------------------------------------------------------------------


def _sitzungen_laden() -> dict:
    try:
        with open(SITZUNGEN_DATEI, encoding="utf-8") as fh:
            daten = json.load(fh)
    except (OSError, ValueError):
        return {}
    return daten if isinstance(daten, dict) else {}


def _sitzungen_speichern(daten: dict) -> None:
    jetzt = time.time()
    daten = {
        sid: eintrag for sid, eintrag in daten.items()
        if isinstance(eintrag, dict) and jetzt - float(eintrag.get("ts") or 0) < SITZUNG_TTL_S
    }
    try:
        os.makedirs(os.path.dirname(SITZUNGEN_DATEI), mode=0o700, exist_ok=True)
        tmp = SITZUNGEN_DATEI + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(daten, fh)
        os.chmod(tmp, 0o600)
        os.replace(tmp, SITZUNGEN_DATEI)
    except OSError:
        pass  # Meldungen dürfen die Sitzung nie anhalten


def notiz_melden(notice: str, *, session_id: str, project: str, text: str,
                 ablage: str = ABLAGE) -> bool:
    """Legt eine Sitzungs-Meldung ab (ohne auf eine Antwort zu warten)."""
    if notice not in ("start", "prompt", "result"):
        return False
    if not session_id:
        return False
    anfragen = os.path.join(ablage, "requests")
    request_id = str(uuid.uuid4())
    jetzt = time.time()
    nutzlast = {
        "version": 1,
        "id": request_id,
        "agent": "claude-code",
        "kind": "notice",
        "notice": notice,
        "created_at": jetzt,
        "expires_at": jetzt + 3600,
        "started_at": jetzt,
        "session_id": session_id,
        "project": project,
        "text": _kurz(text, NOTIZ_TEXT_MAX),
    }
    try:
        os.makedirs(anfragen, mode=0o700, exist_ok=True)
        tmp = os.path.join(anfragen, f".tmp-{request_id}")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(nutzlast, fh, ensure_ascii=False)
        os.chmod(tmp, 0o600)
        os.replace(tmp, os.path.join(anfragen, f"{request_id}.json"))
        return True
    except OSError:
        return False


def letzte_antwort(transcript_path: str) -> tuple[str, str]:
    """Letzte Assistenten-Textantwort aus dem Transkript → (id, text)."""
    if not transcript_path or not os.path.isfile(transcript_path):
        return "", ""
    try:
        groesse = os.path.getsize(transcript_path)
        with open(transcript_path, "rb") as fh:
            if groesse > TRANSCRIPT_TAIL_BYTES:
                fh.seek(groesse - TRANSCRIPT_TAIL_BYTES)
                fh.readline()  # angebrochene Zeile verwerfen
            roh = fh.read().decode("utf-8", "replace")
    except OSError:
        return "", ""
    letzte_id, letzter_text = "", ""
    for zeile in roh.splitlines():
        zeile = zeile.strip()
        if not zeile:
            continue
        try:
            eintrag = json.loads(zeile)
        except ValueError:
            continue
        if not isinstance(eintrag, dict) or eintrag.get("type") != "assistant":
            continue
        nachricht = eintrag.get("message")
        if not isinstance(nachricht, dict):
            continue
        teile = nachricht.get("content")
        if not isinstance(teile, list):
            continue
        texte = [
            t.get("text", "") for t in teile
            if isinstance(t, dict) and t.get("type") == "text" and str(t.get("text") or "").strip()
        ]
        if texte:
            letzte_id = str(nachricht.get("id") or eintrag.get("uuid") or "")
            letzter_text = "\n".join(texte).strip()
    return letzte_id, letzter_text


def sitzung_prompt(daten: dict) -> bool:
    """UserPromptSubmit: erster Prompt öffnet den Thread, weitere melden sich."""
    session_id = str(daten.get("session_id") or "")
    prompt = str(daten.get("prompt") or "").strip()
    if not session_id or not prompt:
        return False
    sitzungen = _sitzungen_laden()
    eintrag = sitzungen.get(session_id) or {}
    notice = "prompt" if eintrag.get("gestartet") else "start"
    ok = notiz_melden(notice, session_id=session_id,
                      project=str(daten.get("cwd") or os.getcwd()), text=prompt)
    if ok:
        eintrag.update({"gestartet": True, "ts": time.time()})
        sitzungen[session_id] = eintrag
        _sitzungen_speichern(sitzungen)
    return ok


def sitzung_ergebnis(daten: dict) -> bool:
    """Stop: die fertige Antwort der Runde melden (einmal je Nachricht)."""
    session_id = str(daten.get("session_id") or "")
    if not session_id:
        return False
    sitzungen = _sitzungen_laden()
    eintrag = sitzungen.get(session_id) or {}
    if not eintrag.get("gestartet"):
        return False  # ohne Thread keine Antwort melden
    antwort_id, text = letzte_antwort(str(daten.get("transcript_path") or ""))
    if not text or eintrag.get("gemeldet") == (antwort_id or text[:80]):
        return False
    ok = notiz_melden("result", session_id=session_id,
                      project=str(daten.get("cwd") or os.getcwd()), text=text)
    if ok:
        eintrag.update({"gemeldet": antwort_id or text[:80], "ts": time.time()})
        sitzungen[session_id] = eintrag
        _sitzungen_speichern(sitzungen)
    return ok


# ---------------------------------------------------------------------------
# Hook-Ein- und -Ausgabe
# ---------------------------------------------------------------------------


def pretooluse_ausgabe(ergebnis: dict) -> dict:
    """Baut das PreToolUse-Ausgabe-JSON aus dem Bridge-Ergebnis."""
    if ergebnis["erlaubt"]:
        entscheidung, grund = "allow", f"Vom Nutzer über Discord {ergebnis['grund']}"
    else:
        entscheidung = "deny"
        grund = f"Vom Nutzer über Discord nicht erlaubt ({ergebnis['grund']}). Nicht erneut versuchen."
    return {"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": entscheidung,
        "permissionDecisionReason": grund,
    }}


def entscheidung_ausgabe(ergebnis: dict, daten: dict) -> dict:
    """Baut das PermissionRequest-Ausgabe-JSON aus dem Bridge-Ergebnis."""
    werkzeug = str(daten.get("tool_name") or "")
    if ergebnis["erlaubt"]:
        decision: dict = {"behavior": "allow"}
        if werkzeug == "AskUserQuestion" and isinstance(ergebnis.get("answers"), dict):
            eingabe = daten.get("tool_input") or {}
            eingabe = eingabe if isinstance(eingabe, dict) else {}
            decision["updatedInput"] = {**eingabe, "answers": ergebnis["answers"]}
    else:
        decision = {
            "behavior": "deny",
            "message": (
                f"Über Discord nicht beantwortet ({ergebnis['grund']}). Nicht erneut versuchen; "
                "falls die Aktion nötig ist, den Nutzer direkt fragen."
                if werkzeug == "AskUserQuestion"
                else f"Vom Nutzer über Discord nicht erlaubt ({ergebnis['grund']}). Nicht erneut versuchen."
            ),
        }
    return {"hookSpecificOutput": {"hookEventName": "PermissionRequest", "decision": decision}}


def main() -> int:
    if not aktiv():
        return 0
    try:
        daten = json.load(sys.stdin)
    except ValueError:
        return 0
    if not isinstance(daten, dict):
        return 0
    ereignis = daten.get("hook_event_name") or ("PermissionRequest" if daten.get("tool_name") else "")
    if ereignis == "UserPromptSubmit":
        sitzung_prompt(daten)
        return 0  # keine Ausgabe: der Prompt läuft unverändert weiter
    if ereignis == "Stop":
        sitzung_ergebnis(daten)
        return 0
    if not daten.get("tool_name"):
        return 0
    if ereignis not in ("PermissionRequest", "PreToolUse"):
        return 0
    werkzeug = str(daten["tool_name"])
    eingabe = daten.get("tool_input") or {}
    eingabe = eingabe if isinstance(eingabe, dict) else {}
    cwd = str(daten.get("cwd") or os.getcwd())
    if ereignis == "PreToolUse":
        regeln = regeln_laden(cwd)
        cli = cli_regeln()
        regeln["allow"] += cli["allow"]
        regeln["deny"] += cli["deny"]
        if not braucht_erlaubnis(werkzeug, eingabe, str(daten.get("permission_mode") or "default"), cwd, regeln):
            return 0  # Claude Code entscheidet wie gewohnt selbst
    anfrage = anfrage_aus_hook(daten)
    if anfrage["kind"] == "question" and not anfrage["questions"]:
        return 0  # nichts Sinnvolles weiterzuleiten — Claude Code entscheidet selbst
    ergebnis = bruecke_fragen(anfrage, session_id=str(daten.get("session_id") or ""))
    if ereignis == "PreToolUse":
        print(json.dumps(pretooluse_ausgabe(ergebnis), ensure_ascii=False))
    else:
        print(json.dumps(entscheidung_ausgabe(ergebnis, daten), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
