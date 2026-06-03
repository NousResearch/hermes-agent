#!/usr/bin/env python3
import json
import os
import re
import sqlite3
import unicodedata
import urllib.request
import urllib.parse
import hashlib
import hmac
import secrets
import time
from datetime import datetime
from html import unescape
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "doni_recall.db"
API_KEY = os.getenv("DONI_RECALL_API_KEY", "").strip()
ALLOWED_IMPORT_SUFFIXES = {".txt", ".md", ".markdown", ".log"}
OBSIDIAN_VAULT = Path(os.getenv("OBSIDIAN_VAULT_PATH", str(Path.home() / "Documents" / "Obsidian Vault"))).expanduser()
OBSIDIAN_REKOL_DIR = os.getenv("DONI_REKOL_OBSIDIAN_DIR", "Inbox/Rekol").strip() or "Inbox/Rekol"

# Optional OAuth2/OIDC mode
OAUTH_ENABLED = os.getenv("DONI_RECALL_OAUTH_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
OAUTH_CLIENT_ID = os.getenv("DONI_RECALL_OAUTH_CLIENT_ID", "").strip()
OAUTH_CLIENT_SECRET = os.getenv("DONI_RECALL_OAUTH_CLIENT_SECRET", "").strip()
OAUTH_AUTHORIZE_URL = os.getenv("DONI_RECALL_OAUTH_AUTHORIZE_URL", "").strip()
OAUTH_TOKEN_URL = os.getenv("DONI_RECALL_OAUTH_TOKEN_URL", "").strip()
OAUTH_USERINFO_URL = os.getenv("DONI_RECALL_OAUTH_USERINFO_URL", "").strip()
OAUTH_SCOPE = os.getenv("DONI_RECALL_OAUTH_SCOPE", "openid email profile").strip()
OAUTH_REDIRECT_URI = os.getenv("DONI_RECALL_OAUTH_REDIRECT_URI", "http://127.0.0.1:8099/oauth/callback").strip()
OAUTH_ALLOWED_EMAILS = {
    e.strip().lower() for e in os.getenv("DONI_RECALL_OAUTH_ALLOWED_EMAILS", "").split(",") if e.strip()
}
SESSION_SECRET = os.getenv("DONI_RECALL_SESSION_SECRET", "change-me-session-secret").strip().encode("utf-8")
SESSION_TTL_SECONDS = int(os.getenv("DONI_RECALL_SESSION_TTL_SECONDS", "28800"))

_OAUTH_STATE_STORE: dict[str, float] = {}
_SESSIONS: dict[str, dict] = {}


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS notes (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          title TEXT NOT NULL,
          source_type TEXT NOT NULL,
          source_ref TEXT,
          content TEXT NOT NULL,
          summary TEXT,
          created_at TEXT NOT NULL
        );
        """
    )
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts
        USING fts5(title, content, content='notes', content_rowid='id');
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS notes_ai AFTER INSERT ON notes BEGIN
          INSERT INTO notes_fts(rowid, title, content) VALUES (new.id, new.title, new.content);
        END;
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS notes_ad AFTER DELETE ON notes BEGIN
          INSERT INTO notes_fts(notes_fts, rowid, title, content) VALUES('delete', old.id, old.title, old.content);
        END;
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS notes_au AFTER UPDATE ON notes BEGIN
          INSERT INTO notes_fts(notes_fts, rowid, title, content) VALUES('delete', old.id, old.title, old.content);
          INSERT INTO notes_fts(rowid, title, content) VALUES (new.id, new.title, new.content);
        END;
        """
    )
    cols = {r[1] for r in conn.execute("PRAGMA table_info(notes)").fetchall()}
    if "obsidian_path" not in cols:
        conn.execute("ALTER TABLE notes ADD COLUMN obsidian_path TEXT")
    conn.commit()
    conn.close()


def _safe_slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9\-\s_]", "", s)
    s = re.sub(r"[\s_]+", "-", s).strip("-")
    return s or "note"


def _obsidian_target_path(note_id: int, title: str, created_at: str) -> Path:
    day = (created_at or datetime.utcnow().isoformat() + "Z")[:10]
    fname = f"{day}--{note_id}-{_safe_slug(title)[:80]}.md"
    return OBSIDIAN_VAULT / OBSIDIAN_REKOL_DIR / fname


def export_note_to_obsidian(note_id: int) -> str | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT id,title,source_type,source_ref,summary,content,created_at FROM notes WHERE id = ?",
        (note_id,),
    ).fetchone()
    if not row:
        conn.close()
        return None

    md = [
        f"# {row['title']}",
        "",
        f"- id: {row['id']}",
        f"- source_type: {row['source_type']}",
        f"- source_ref: {row['source_ref'] or ''}",
        f"- created_at: {row['created_at']}",
        "",
        "## Sažetak",
        row["summary"] or "",
        "",
        "## Sadržaj",
        row["content"] or "",
        "",
    ]

    target = _obsidian_target_path(int(row["id"]), str(row["title"]), str(row["created_at"]))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(md), encoding="utf-8")
    conn.execute("UPDATE notes SET obsidian_path = ? WHERE id = ?", (str(target), int(row["id"])))
    conn.commit()
    conn.close()
    return str(target)


def is_youtube_url(url: str) -> bool:
    u = (url or "").lower()
    return ("youtube.com/" in u) or ("youtu.be/" in u)



def summarize_text(text: str, max_sentences: int = 3) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", clean)
    return " ".join(sentences[:max_sentences])[:800]


def clean_html_to_text(html: str) -> str:
    html = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    html = re.sub(r"(?is)<style.*?>.*?</style>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_url_text(url: str) -> tuple[str, str]:
    req = urllib.request.Request(url, headers={"User-Agent": "DoniRecallMVP/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
    title_match = re.search(r"(?is)<title[^>]*>(.*?)</title>", raw)
    title = title_match.group(1).strip() if title_match else url
    text = clean_html_to_text(raw)
    return title[:200], text


def add_note(title: str, content: str, source_type: str, source_ref: str | None = None) -> int:
    conn = get_conn()
    now = datetime.utcnow().isoformat() + "Z"
    summary = summarize_text(content)
    cur = conn.execute(
        "INSERT INTO notes(title, source_type, source_ref, content, summary, created_at) VALUES(?,?,?,?,?,?)",
        (title[:200], source_type, source_ref, content, summary, now),
    )
    conn.commit()
    note_id = cur.lastrowid
    conn.close()
    try:
        export_note_to_obsidian(int(note_id))
    except Exception:
        pass
    return note_id


def list_notes(limit: int = 20):
    conn = get_conn()
    rows = conn.execute(
        "SELECT id,title,source_type,source_ref,summary,created_at FROM notes ORDER BY id DESC LIMIT ?",
        (max(1, min(limit, 100)),),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats():
    conn = get_conn()
    total_notes = conn.execute("SELECT COUNT(*) AS c FROM notes").fetchone()["c"]
    per_source = conn.execute(
        "SELECT source_type, COUNT(*) AS c FROM notes GROUP BY source_type ORDER BY c DESC"
    ).fetchall()
    latest = conn.execute(
        "SELECT id,title,created_at FROM notes ORDER BY id DESC LIMIT 5"
    ).fetchall()
    conn.close()
    return {
        "total_notes": total_notes,
        "source_breakdown": [dict(r) for r in per_source],
        "latest": [dict(r) for r in latest],
    }


def import_local_file(path_str: str, title: str | None = None) -> int:
    p = Path(path_str).expanduser().resolve()
    if not p.exists() or not p.is_file():
        raise ValueError("file not found")
    if p.suffix.lower() not in ALLOWED_IMPORT_SUFFIXES:
        raise ValueError(f"unsupported file type: {p.suffix}")
    content = p.read_text(encoding="utf-8", errors="ignore").strip()
    if len(content) < 30:
        raise ValueError("file has too little text")
    note_title = (title or p.stem or "Imported File").strip()
    return add_note(note_title, content, "file", source_ref=str(p))


def safe_fts_query(q: str) -> str:
    tokens = [t for t in re.findall(r"\w+", q.lower(), flags=re.UNICODE) if len(t) > 1]
    return " OR ".join(tokens[:8])


def _norm_text(s: str) -> str:
    s = (s or "").lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", s).strip()


def search_notes(q: str, limit: int = 10):
    cap = max(1, min(limit, 20))
    fts_q = safe_fts_query(q)
    conn = get_conn()

    rows = []
    if fts_q:
        rows = conn.execute(
            """
            SELECT n.id,n.title,n.summary,n.source_type,n.source_ref,n.created_at,
                   snippet(notes_fts, 1, '[', ']', ' ... ', 18) AS snippet
            FROM notes_fts
            JOIN notes n ON n.id = notes_fts.rowid
            WHERE notes_fts MATCH ?
            ORDER BY bm25(notes_fts)
            LIMIT ?
            """,
            (fts_q, cap),
        ).fetchall()

    # Fallback 1: SQL LIKE
    if not rows and q.strip():
        like_q = f"%{q.strip()}%"
        rows = conn.execute(
            """
            SELECT id,title,summary,source_type,source_ref,created_at,
                   substr(content, 1, 220) AS snippet
            FROM notes
            WHERE title LIKE ? OR content LIKE ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (like_q, like_q, cap),
        ).fetchall()

    # Fallback 2: unicode-normalized Python filter (diacritics tolerant)
    if not rows and q.strip():
        nq = _norm_text(q)
        all_rows = conn.execute(
            "SELECT id,title,summary,source_type,source_ref,created_at,content FROM notes ORDER BY id DESC LIMIT 500"
        ).fetchall()
        py_hits = []
        for r in all_rows:
            hay = _norm_text((r["title"] or "") + " " + (r["content"] or ""))
            if nq and nq in hay:
                py_hits.append(
                    {
                        "id": r["id"],
                        "title": r["title"],
                        "summary": r["summary"],
                        "source_type": r["source_type"],
                        "source_ref": r["source_ref"],
                        "created_at": r["created_at"],
                        "snippet": (r["content"] or "")[:220],
                    }
                )
            if len(py_hits) >= cap:
                break
        conn.close()
        return py_hits

    conn.close()
    return [dict(r) for r in rows]


def answer_question(question: str):
    hits = search_notes(question, limit=3)
    if not hits:
        latest = list_notes(limit=2)
        if not latest:
            return {
                "answer": "Nemam konteksta još. Dodaj barem jednu bilješku ili URL.",
                "sources": [],
            }
        context = "\n".join(f"- {n['title']}: {n.get('summary') or ''}" for n in latest)
        return {
            "answer": f"Nisam našao direktan match. Zadnje bilješke su:\n{context}",
            "sources": [n["id"] for n in latest],
        }

    bullets = []
    for h in hits:
        snippet = h.get("snippet") or h.get("summary") or ""
        bullets.append(f"- ({h['id']}) {h['title']}: {snippet}")

    return {
        "answer": "Evo najrelevantnijeg konteksta za tvoje pitanje:\n" + "\n".join(bullets),
        "sources": [h["id"] for h in hits],
    }


def oauth_configured() -> bool:
    return all([
        OAUTH_ENABLED,
        OAUTH_CLIENT_ID,
        OAUTH_CLIENT_SECRET,
        OAUTH_AUTHORIZE_URL,
        OAUTH_TOKEN_URL,
        OAUTH_USERINFO_URL,
        OAUTH_REDIRECT_URI,
    ])


def _cleanup_auth_stores():
    now = time.time()
    expired_states = [k for k, v in _OAUTH_STATE_STORE.items() if v < now]
    for k in expired_states:
        _OAUTH_STATE_STORE.pop(k, None)
    expired_sessions = [k for k, v in _SESSIONS.items() if v.get("exp", 0) < now]
    for k in expired_sessions:
        _SESSIONS.pop(k, None)


def _sign_value(value: str) -> str:
    return hmac.new(SESSION_SECRET, value.encode("utf-8"), hashlib.sha256).hexdigest()


def _issue_session(user: dict) -> str:
    _cleanup_auth_stores()
    sid = secrets.token_urlsafe(32)
    _SESSIONS[sid] = {
        "user": user,
        "exp": time.time() + SESSION_TTL_SECONDS,
    }
    return f"{sid}.{_sign_value(sid)}"


def _read_session(cookie_header: str | None) -> dict | None:
    _cleanup_auth_stores()
    if not cookie_header:
        return None
    parts = [p.strip() for p in cookie_header.split(";")]
    raw = None
    for p in parts:
        if p.startswith("doni_recall_session="):
            raw = p.split("=", 1)[1].strip()
            break
    if not raw or "." not in raw:
        return None
    sid, sig = raw.split(".", 1)
    expected = _sign_value(sid)
    if not hmac.compare_digest(sig, expected):
        return None
    sess = _SESSIONS.get(sid)
    if not sess:
        return None
    if sess.get("exp", 0) < time.time():
        _SESSIONS.pop(sid, None)
        return None
    return sess


def _create_oauth_state() -> str:
    _cleanup_auth_stores()
    state = secrets.token_urlsafe(24)
    _OAUTH_STATE_STORE[state] = time.time() + 600
    return state


def _consume_oauth_state(state: str) -> bool:
    _cleanup_auth_stores()
    exp = _OAUTH_STATE_STORE.pop(state, None)
    return bool(exp and exp >= time.time())


def _oauth_login_url(state: str) -> str:
    q = {
        "response_type": "code",
        "client_id": OAUTH_CLIENT_ID,
        "redirect_uri": OAUTH_REDIRECT_URI,
        "scope": OAUTH_SCOPE,
        "state": state,
    }
    return f"{OAUTH_AUTHORIZE_URL}?{urllib.parse.urlencode(q)}"


def _oauth_fetch_json(url: str, method: str = "GET", headers: dict | None = None, data: dict | None = None) -> dict:
    body = None
    req_headers = headers.copy() if headers else {}
    if data is not None:
        body = urllib.parse.urlencode(data).encode("utf-8")
        req_headers.setdefault("Content-Type", "application/x-www-form-urlencoded")
    req = urllib.request.Request(url, data=body, method=method, headers=req_headers)
    with urllib.request.urlopen(req, timeout=20) as resp:
        text = resp.read().decode("utf-8", errors="ignore")
    return json.loads(text)


def exchange_code_for_user(code: str) -> dict:
    token = _oauth_fetch_json(
        OAUTH_TOKEN_URL,
        method="POST",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": OAUTH_REDIRECT_URI,
            "client_id": OAUTH_CLIENT_ID,
            "client_secret": OAUTH_CLIENT_SECRET,
        },
    )
    access_token = token.get("access_token", "")
    if not access_token:
        raise ValueError("oauth token response missing access_token")

    user = _oauth_fetch_json(
        OAUTH_USERINFO_URL,
        method="GET",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    email = str(user.get("email", "")).strip().lower()
    if OAUTH_ALLOWED_EMAILS and email not in OAUTH_ALLOWED_EMAILS:
        raise PermissionError("email not allowed")
    return {
        "email": email,
        "name": str(user.get("name") or user.get("preferred_username") or email),
    }


class Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, data: dict):
        out = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def _redirect(self, location: str):
        self.send_response(302)
        self.send_header("Location", location)
        self.end_headers()

    def _auth_ok(self) -> bool:
        # API key auth (still supported)
        if API_KEY:
            provided = self.headers.get("X-API-Key", "").strip()
            if provided == API_KEY:
                return True

        # OAuth session auth
        if oauth_configured():
            sess = _read_session(self.headers.get("Cookie"))
            if sess:
                return True
            self._json(401, {"error": "unauthorized", "hint": "login via /login or send X-API-Key"})
            return False

        # Open mode if neither API key nor OAuth configured
        if not API_KEY and not oauth_configured():
            return True

        self._json(401, {"error": "unauthorized"})
        return False

    def _html(self, html: str):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/login":
            if not oauth_configured():
                self._json(400, {"error": "oauth not configured"})
                return
            state = _create_oauth_state()
            self._redirect(_oauth_login_url(state))
            return

        if parsed.path == "/oauth/callback":
            if not oauth_configured():
                self._json(400, {"error": "oauth not configured"})
                return
            q = parse_qs(parsed.query)
            code = (q.get("code", [""])[0] or "").strip()
            state = (q.get("state", [""])[0] or "").strip()
            if not code or not state or not _consume_oauth_state(state):
                self._json(400, {"error": "invalid oauth callback state/code"})
                return
            try:
                user = exchange_code_for_user(code)
                cookie_value = _issue_session(user)
                self.send_response(302)
                self.send_header("Location", "/")
                self.send_header(
                    "Set-Cookie",
                    f"doni_recall_session={cookie_value}; HttpOnly; SameSite=Lax; Path=/; Max-Age={SESSION_TTL_SECONDS}",
                )
                self.end_headers()
            except Exception as e:
                self._json(400, {"error": f"oauth callback failed: {e}"})
            return

        if parsed.path == "/logout":
            self.send_response(302)
            self.send_header("Location", "/")
            self.send_header("Set-Cookie", "doni_recall_session=; HttpOnly; SameSite=Lax; Path=/; Max-Age=0")
            self.end_headers()
            return

        if parsed.path == "/":
            if oauth_configured() and not _read_session(self.headers.get("Cookie")):
                self._redirect("/login")
                return
            self._html(self.render_ui())
            return

        if parsed.path.startswith("/api/") and not self._auth_ok():
            return
        if parsed.path == "/api/notes":
            q = parse_qs(parsed.query)
            limit = int(q.get("limit", [20])[0])
            self._json(200, {"notes": list_notes(limit=limit)})
            return
        if parsed.path == "/api/search":
            q = parse_qs(parsed.query)
            query = q.get("q", [""])[0]
            self._json(200, {"results": search_notes(query)})
            return
        if parsed.path == "/api/stats":
            self._json(200, get_stats())
            return

        self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path.startswith("/api/") and not self._auth_ok():
            return
        try:
            payload = self._read_json()
        except Exception:
            self._json(400, {"error": "invalid json"})
            return

        if self.path == "/api/add_text":
            title = (payload.get("title") or "Untitled").strip()
            text = (payload.get("text") or "").strip()
            if not text:
                self._json(400, {"error": "text is required"})
                return
            note_id = add_note(title, text, "text")
            self._json(200, {"ok": True, "id": note_id})
            return

        if self.path == "/api/add_youtube":
            url = (payload.get("url") or "").strip()
            if not url or not is_youtube_url(url):
                self._json(400, {"error": "youtube url is required"})
                return
            try:
                title, text = fetch_url_text(url)
                if len(text) < 80:
                    self._json(400, {"error": "fetch failed or page had too little text"})
                    return
                note_id = add_note(title, text, "youtube", source_ref=url)
                self._json(200, {"ok": True, "id": note_id, "title": title, "source_type": "youtube"})
            except Exception as e:
                self._json(500, {"error": f"fetch failed: {e}"})
            return

        if self.path == "/api/add_telegram":
            text = (payload.get("text") or "").strip()
            chat = (payload.get("chat") or "telegram").strip()
            author = (payload.get("author") or "").strip()
            title = (payload.get("title") or "Telegram unos").strip()
            if not text:
                self._json(400, {"error": "text is required"})
                return
            prefix = f"[chat={chat}]" + (f" [author={author}]" if author else "")
            note_id = add_note(title, f"{prefix}\n\n{text}", "telegram", source_ref=chat)
            self._json(200, {"ok": True, "id": note_id, "source_type": "telegram"})
            return

        if self.path == "/api/add_url":
            url = (payload.get("url") or "").strip()
            if not url:
                self._json(400, {"error": "url is required"})
                return
            try:
                title, text = fetch_url_text(url)
                if len(text) < 80:
                    self._json(400, {"error": "fetch failed or page had too little text"})
                    return
                source_type = "youtube" if is_youtube_url(url) else "url"
                note_id = add_note(title, text, source_type, source_ref=url)
                self._json(200, {"ok": True, "id": note_id, "title": title, "source_type": source_type})
            except Exception as e:
                self._json(500, {"error": f"fetch failed: {e}"})
            return

        if self.path == "/api/add_file":
            path_str = (payload.get("path") or "").strip()
            title = (payload.get("title") or "").strip() or None
            if not path_str:
                self._json(400, {"error": "path is required"})
                return
            try:
                note_id = import_local_file(path_str, title=title)
                self._json(200, {"ok": True, "id": note_id})
            except Exception as e:
                self._json(400, {"error": str(e)})
            return

        if self.path == "/api/ask":
            q = (payload.get("question") or "").strip()
            if not q:
                self._json(400, {"error": "question is required"})
                return
            self._json(200, answer_question(q))
            return

        self._json(404, {"error": "not found"})

    def render_ui(self):
        return """<!doctype html>
<html>
<head>
  <meta charset='utf-8'/>
  <meta name='viewport' content='width=device-width, initial-scale=1'/>
  <title>Doni Recall-like MVP</title>
  <style>
    :root {
      --bg: #f3f6fb;
      --card: #ffffff;
      --line: #d8dfeb;
      --text: #0f172a;
      --muted: #5b667a;
      --accent: #2563eb;
      --accent-2: #1d4ed8;
    }
    * { box-sizing: border-box; }
    body {
      font-family: Inter, Arial, sans-serif;
      margin: 0;
      padding: 14px;
      max-width: 1100px;
      color: var(--text);
      background: radial-gradient(circle at top right, #dbeafe 0%, #f8fafc 45%, #eef2ff 100%);
    }
    h1 { margin: 0 0 8px; font-size: clamp(24px, 4vw, 34px); }
    h3 { margin: 18px 0 8px; font-size: 18px; }
    p { line-height: 1.45; color: var(--muted); margin: 6px 0; }
    .row { display: grid; grid-template-columns: 1fr; gap: 14px; }
    .row > div {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
      box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
    }
    @media (min-width: 920px) { .row { grid-template-columns: 1fr 1fr; } }
    textarea, input {
      width: 100%;
      padding: 12px;
      margin: 6px 0;
      font-size: 16px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #f8fafc;
      color: var(--text);
    }
    button {
      width: 100%;
      padding: 11px 14px;
      min-height: 44px;
      border: 1px solid var(--accent);
      border-radius: 12px;
      background: linear-gradient(180deg, var(--accent), var(--accent-2));
      color: white;
      font-weight: 600;
      cursor: pointer;
    }
    button:hover { filter: brightness(1.04); }
    @media (min-width: 920px) { button { width: auto; } }
    pre {
      background: #0f172a;
      color: #e2e8f0;
      padding: 12px;
      border-radius: 12px;
      white-space: pre-wrap;
      border: 1px solid #1e293b;
    }
    .history-list { margin-top: 10px; display: grid; gap: 8px; }
    .history-item {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px 10px;
      background: #f8fafc;
    }
    .history-top { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
    .history-title { font-size: 14px; font-weight: 600; color: var(--text); }
    .history-meta { font-size: 12px; color: var(--muted); margin-top: 4px; }
    .badge {
      font-size: 11px;
      font-weight: 700;
      border-radius: 999px;
      padding: 2px 8px;
      border: 1px solid transparent;
      text-transform: uppercase;
      letter-spacing: .2px;
    }
    .badge-text { background: #ecfeff; color: #155e75; border-color: #a5f3fc; }
    .badge-url { background: #eff6ff; color: #1e40af; border-color: #bfdbfe; }
    .badge-file { background: #f5f3ff; color: #5b21b6; border-color: #ddd6fe; }
  </style>
</head>
<body>
  <h1>Doni Recall-like MVP</h1>
  <p>Capture → Summary → Search → Q&A (lokalno, SQLite FTS5)</p>
  <p><small>OAuth: koristi <a href='/login'>/login</a> i <a href='/logout'>/logout</a> kad je OAuth uključen.</small></p>
  <p><small>API key (optional): ako postaviš `DONI_RECALL_API_KEY`, upiši isti key ovdje.</small></p>
  <input id='apiKey' placeholder='X-API-Key (optional)' />

  <div class='row'>
    <div>
      <h3>Dodaj tekst</h3>
      <input id='title' placeholder='Naslov' />
      <textarea id='text' rows='6' placeholder='Sadržaj'></textarea>
      <button onclick='addText()'>Spremi</button>

      <h3>Dodaj URL</h3>
      <input id='url' placeholder='https://...' />
      <button onclick='addUrl()'>Fetch + spremi</button>

      <h3>Dodaj YouTube</h3>
      <input id='ytUrl' placeholder='https://www.youtube.com/watch?v=...' />
      <button onclick='addYouTube()'>Spremi YouTube</button>

      <h3>Dodaj Telegram tekst</h3>
      <input id='tgTitle' placeholder='Naslov (optional)' />
      <textarea id='tgText' rows='4' placeholder='Poruka/sažetak iz Telegrama'></textarea>
      <button onclick='addTelegram()'>Spremi Telegram</button>

      <h3>Dodaj lokalni file</h3>
      <input id='filePath' placeholder='/mnt/d/.../note.md' />
      <input id='fileTitle' placeholder='Naslov (optional)' />
      <button onclick='addFile()'>Import file</button>
    </div>

    <div>
      <h3>Search</h3>
      <input id='q' placeholder='upit...' />
      <button onclick='searchQ()'>Traži</button>

      <h3>Ask</h3>
      <input id='ask' placeholder='pitanje...' />
      <button onclick='askQ()'>Pitaj</button>

      <h3>Zadnje bilješke</h3>
      <div id='historyList' class='history-list'>
        <div class='history-item'>Učitavam...</div>
      </div>
    </div>
  </div>

  <h3>Output</h3>
  <pre id='out'>ready</pre>

<script>
const out = document.getElementById('out');
const historyList = document.getElementById('historyList');

function authHeaders() {
  const key = (apiKey.value || '').trim();
  return key ? {'X-API-Key': key} : {};
}

function sourceBadgeClass(sourceType) {
  if (sourceType === 'url') return 'badge badge-url';
  if (sourceType === 'file') return 'badge badge-file';
  return 'badge badge-text';
}

function renderHistory(notes) {
  historyList.innerHTML = '';
  if (!notes || !notes.length) {
    const empty = document.createElement('div');
    empty.className = 'history-item';
    empty.textContent = 'Nema bilješki još.';
    historyList.appendChild(empty);
    return;
  }

  notes.forEach((n) => {
    const item = document.createElement('div');
    item.className = 'history-item';

    const top = document.createElement('div');
    top.className = 'history-top';

    const title = document.createElement('div');
    title.className = 'history-title';
    title.textContent = `#${n.id} ${n.title || 'Untitled'}`;

    const badge = document.createElement('span');
    badge.className = sourceBadgeClass(n.source_type);
    badge.textContent = n.source_type || 'text';

    top.appendChild(title);
    top.appendChild(badge);

    const meta = document.createElement('div');
    meta.className = 'history-meta';
    meta.textContent = n.created_at || '';

    item.appendChild(top);
    item.appendChild(meta);
    historyList.appendChild(item);
  });
}

async function loadHistory() {
  try {
    const r = await fetch('/api/notes?limit=8', {headers: authHeaders()});
    const j = await r.json();
    renderHistory(j.notes || []);
  } catch (e) {
    historyList.innerHTML = '<div class="history-item">Greška pri učitavanju.</div>';
  }
}

async function post(url, payload) {
  const r = await fetch(url, {
    method:'POST',
    headers:{'Content-Type':'application/json', ...authHeaders()},
    body: JSON.stringify(payload)
  });
  const j = await r.json();
  out.textContent = JSON.stringify(j, null, 2);
  if (r.ok) await loadHistory();
}
function addText() { post('/api/add_text', {title: title.value, text: text.value}); }
function addUrl() { post('/api/add_url', {url: url.value}); }
function addYouTube() { post('/api/add_youtube', {url: ytUrl.value}); }
function addTelegram() { post('/api/add_telegram', {title: tgTitle.value, text: tgText.value, chat: 'telegram'}); }
function addFile() { post('/api/add_file', {path: filePath.value, title: fileTitle.value}); }
async function searchQ() {
  const r = await fetch('/api/search?q=' + encodeURIComponent(q.value), {headers: authHeaders()});
  out.textContent = JSON.stringify(await r.json(), null, 2);
}
function askQ() { post('/api/ask', {question: ask.value}); }
loadHistory();
</script>
</body>
</html>"""


def main():
    init_db()
    host = os.getenv("DONI_RECALL_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.getenv("DONI_RECALL_PORT", "8099"))
    server = HTTPServer((host, port), Handler)
    if host == "0.0.0.0":
        print(f"Doni Recall MVP running on http://0.0.0.0:{port} (LAN enabled)")
    else:
        print(f"Doni Recall MVP running on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
