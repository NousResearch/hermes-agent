#!/usr/bin/env python3
"""Working Grok Bot (Cursor/sand backend) chat client.

Route:  POST https://api2.cursor.sh/aiserver.v1.InferenceService/Stream
Body:   Connect streaming envelope + protobuf InferenceStreamRequest

Reverse-engineered from /Applications/Grok Bot.app:
  - dist/host/host-main.cjs  -> InferenceService/Stream, InferenceStreamRequest (UC)
  - dist/electron-main/main.cjs -> auth, checksum, PKCE login

Verified live 2026-08-29: model "grok-4.6" replied "PONG" to "Reply with exactly: PONG".

Usage:
    python3 grokbot_client.py chat "hello" [--model grok-4.6] [--tools]
    python3 grokbot_client.py models
    python3 grokbot_client.py probe          # try each model id, report which work
"""

from __future__ import annotations

import json
import re
import struct
import urllib.error
import urllib.request
import uuid

from agent.grokbot import login as _g

API_BASE = _g.API_BASE
CHAT_URL = f"{API_BASE}/aiserver.v1.InferenceService/Stream"
CT = "application/connect+proto"

# Model ids verified against this account (see `probe`).
DEFAULT_MODEL = "grok-4.6"
KNOWN_MODELS = ["grok-4.6", "grok-4.5", "default", "composer-2.5"]


# ------------------------------------------------------------ protobuf ------

def vi(n: int) -> bytes:
    """Varint."""
    b = bytearray()
    while True:
        x = n & 0x7F
        n >>= 7
        b.append(x | 0x80 if n else x)
        if not n:
            return bytes(b)


def tag(f: int, w: int) -> bytes:
    return vi((f << 3) | w)


def s(f: int, v: str) -> bytes:
    e = v.encode()
    return tag(f, 2) + vi(len(e)) + e


def m(f: int, p: bytes) -> bytes:
    return tag(f, 2) + vi(len(p)) + p


def envelope(proto: bytes) -> bytes:
    """Connect streaming envelope: 1 flag byte + 4-byte BE length + payload."""
    return b"\x00" + struct.pack(">I", len(proto)) + proto


# ---------------------------------------------------------- request ---------

# InferenceStreamRequest (UC):
#   1  messages        (vb)   repeated
#   2  tools           (TD)
#   5  model_id        string
#   6  invocation_id   string
#   7  requested_model (FQ)   required
#   8  conversation_id string
#
# message (vb):   1 role(enum) 2 text
# requested_model (FQ): 1 model_id  2 max_mode  4 built_in_model

ROLE_USER = 1
ROLE_ASSISTANT = 2
ROLE_TOOL = 3
ROLE_SYSTEM = 4


def _content_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        bits = []
        for p in content:
            if isinstance(p, dict):
                bits.append(p.get("text") or p.get("content") or "")
            else:
                bits.append(str(p))
        return "".join(bits)
    return str(content)


def openai_messages_to_history(msgs: list[dict]) -> tuple[str, list[tuple[int, str]]]:
    """Flatten OpenAI chat messages into proto (role, text) + last user prompt.

    Native proto tool-result encoding still errors. Tool results become extra
    user text. Assistant ``tool_calls`` are NOT serialized as ``Called tools:``
    (that leaked into the model transcript and got imitated).
    """
    history: list[tuple[int, str]] = []
    pending_user = ""
    for m in msgs or []:
        role = (m.get("role") or "user").lower()
        text = _content_text(m.get("content"))
        if role == "system":
            history.append((ROLE_USER, f"[SYSTEM]\n{text}"))
        elif role == "user":
            if pending_user:
                history.append((ROLE_USER, pending_user))
            pending_user = text
        elif role == "assistant":
            if pending_user:
                history.append((ROLE_USER, pending_user))
                pending_user = ""
            if text:
                history.append((ROLE_ASSISTANT, text))
        elif role == "tool":
            name = m.get("name") or "tool"
            pending_user = (
                (pending_user + "\n" if pending_user else "")
                + f"TOOL_RESULT {name}: {text}"
            )
    return pending_user, history


def message(role: int, text: str) -> bytes:
    return tag(1, 0) + vi(role) + s(2, text)


def encode_tool_def(name: str, description: str = "", parameters=None) -> bytes:
    """InferenceStreamRequest.tools item (live-verified 2026-08-29).

    field 1 name, field 2 description, field 5 parameters JSON schema.
    """
    body = s(1, name)
    if description:
        body += s(2, description)
    if parameters is not None:
        if not isinstance(parameters, str):
            parameters = json.dumps(parameters)
        body += s(5, parameters)
    return body


def build_request(prompt: str, model: str = DEFAULT_MODEL,
                  conversation_id: str | None = None,
                  history: list[tuple[int, str]] | None = None,
                  tools: list[dict] | None = None) -> bytes:
    msgs = list(history or []) + ([(ROLE_USER, prompt)] if prompt else [])
    body = b"".join(m(1, message(r, t)) for r, t in msgs)
    for t in tools or []:
        fn = t.get("function") if t.get("type") == "function" or "function" in t else t
        if not isinstance(fn, dict):
            continue
        name = fn.get("name") or t.get("name")
        if not name:
            continue
        body += m(2, encode_tool_def(name, fn.get("description") or "",
                                     fn.get("parameters") or t.get("parameters")))
    body += m(7, s(1, model))                                  # requested_model
    body += s(6, str(uuid.uuid4()))                            # invocation_id
    body += s(8, conversation_id or str(uuid.uuid4()))         # conversation_id
    return body


# ------------------------------------------------------------ response ------

def _walk(buf: bytes):
    """Minimal protobuf walker -> list of (field_no, wire_type, value)."""
    i, out = 0, []
    while i < len(buf):
        try:
            key, i = _read_varint(buf, i)
        except Exception:
            break
        f, w = key >> 3, key & 7
        if w == 0:
            v, i = _read_varint(buf, i)
            out.append((f, w, v))
        elif w == 2:
            ln, i = _read_varint(buf, i)
            out.append((f, w, buf[i:i + ln]))
            i += ln
        elif w == 5:
            out.append((f, w, buf[i:i + 4])); i += 4
        elif w == 1:
            out.append((f, w, buf[i:i + 8])); i += 8
        else:
            break
    return out


def _read_varint(buf: bytes, i: int) -> tuple[int, int]:
    n = shift = 0
    while True:
        b = buf[i]; i += 1
        n |= (b & 0x7F) << shift
        if not b & 0x80:
            return n, i
        shift += 7


_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_B64_RE = re.compile(r"^[A-Za-z0-9+/=_-]{32,}$")


def _looks_like_text(t: str) -> bool:
    if not t or len(t.strip()) < 1:
        return False
    if _UUID_RE.match(t):                      # request/conversation ids
        return False
    if _B64_RE.match(t):                       # opaque blobs
        return False
    if t.startswith("http") or t.startswith("{"):
        return False
    # must contain at least one letter or digit and mostly printable
    if not re.search(r"[A-Za-z0-9]", t):
        return False
    printable = sum(1 for ch in t if ch == "\n" or ch == "\t" or 32 <= ord(ch) < 0x110000)
    return printable / len(t) > 0.95


_LEAK_RE = re.compile(r"\\confidence\{\d+\}")


def _clean(t: str) -> str:
    """Strip leaked system-prompt scaffolding from model output."""
    t = _LEAK_RE.sub("", t)
    t = re.sub(r"^[\x00-\x1f]+", "", t)          # leading control byte = length prefix
    return t.strip()


def _parse_event(chunk: bytes) -> dict:
    """Parse one InferenceStreamResponse frame into a dict.

    Observed shape (verified live 2026-08-29):
      field 1 -> streamed text event  -> nested field 1 = text
      field 3 -> usage/telemetry
      field 4 -> model routing info   (nested: 1=?, 2=model_id, ...)
      field 5 -> timing
      field 7 -> request/conversation id (nested field 1 = uuid)
      field 9 -> opaque blob
    """
    out: dict = {}
    for f, w, v in _walk(chunk):
        if f == 1 and w == 2 and isinstance(v, bytes):
            sub = dict((sf, sv) for sf, sw, sv in _walk(v) if sw == 2)
            txt = sub.get(1)
            if isinstance(txt, bytes):
                try:
                    out.setdefault("text", "")
                    out["text"] += txt.decode("utf-8")
                except UnicodeDecodeError:
                    pass
        elif f == 2 and w == 2 and isinstance(v, bytes):
            # tool_call_part: 1=id, 2=name, 3=arguments JSON (streamed), 4=is_final
            inner = {}
            for sf, sw, sv in _walk(v):
                if sw == 2 and isinstance(sv, bytes):
                    inner[sf] = sv.decode("utf-8", "replace")
                elif sw == 0:
                    inner[sf] = sv
            out.setdefault("tool_parts", []).append(inner)
        elif f == 4 and w == 2 and isinstance(v, bytes):
            sub = dict((sf, sv) for sf, sw, sv in _walk(v) if sw == 2)
            mid = sub.get(2)
            if isinstance(mid, bytes):
                out.setdefault("model", mid.decode("utf-8", "replace"))
        elif f == 7 and w == 2 and isinstance(v, bytes):
            sub = dict((sf, sv) for sf, sw, sv in _walk(v) if sw == 2)
            rid = sub.get(1)
            if isinstance(rid, bytes):
                out.setdefault("request_id", rid.decode("utf-8", "replace"))
    return out


def _extract_text(blob: bytes) -> str:
    """Pull human-readable text out of a response chunk."""
    parts = []
    for f, w, v in _walk(blob):
        if w == 2 and isinstance(v, bytes):
            try:
                t = v.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if _looks_like_text(t):
                parts.append(t)
    return "".join(parts)


def _merge_tool_parts(parts: list[dict]) -> list[dict]:
    """Accumulate streamed tool_call_part frames into final OpenAI tool_calls."""
    by_id: dict[str, dict] = {}
    order: list[str] = []
    for p in parts:
        raw_id = str(p.get(1) or "")
        oid = raw_id.split("\n", 1)[0] or raw_id
        if oid not in by_id:
            by_id[oid] = {"id": oid or f"call_{uuid.uuid4().hex[:12]}",
                          "name": "", "arguments": ""}
            order.append(oid)
        rec = by_id[oid]
        if p.get(2):
            rec["name"] = p[2]
        if p.get(3):
            frag = str(p[3])
            cur = rec["arguments"]
            # Prefer a complete JSON object over concatenated stream fragments.
            def _ok(x):
                try:
                    json.loads(x); return True
                except Exception:
                    return False
            if not cur:
                rec["arguments"] = frag
            elif _ok(frag) and (not _ok(cur) or len(frag) >= len(cur)):
                rec["arguments"] = frag
            elif not _ok(cur):
                rec["arguments"] = cur + frag
    out = []
    for oid in order:
        rec = by_id[oid]
        args = rec["arguments"] or "{}"
        # streamed args sometimes arrive as '{' then '"k":"v"}' then full json
        try:
            json.loads(args)
        except json.JSONDecodeError:
            # keep last complete-looking object
            m = re.search(r"\{.*\}", args)
            args = m.group(0) if m else "{}"
        if not rec["name"]:
            continue
        out.append({
            "id": rec["id"] if rec["id"].startswith("call") else f"call_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {"name": rec["name"], "arguments": args},
        })
    return out


def infer(prompt: str, model: str = DEFAULT_MODEL, token: str | None = None,
          conversation_id: str | None = None, history=None,
          tools: list[dict] | None = None, on_chunk=None) -> dict:
    """Run InferenceService/Stream. Returns {text, tool_calls, model}."""
    routed_model = ""
    def _post(tok: str) -> bytes:
        body = envelope(build_request(prompt, model, conversation_id, history, tools=tools))
        h = _g.headers(tok)
        h["Content-Type"] = CT
        h["Accept"] = CT
        req = urllib.request.Request(CHAT_URL, data=body, headers=h, method="POST")
        try:
            resp = urllib.request.urlopen(req, timeout=180)
            return resp.read()
        except urllib.error.HTTPError as e:
            raw = e.read()
            raise RuntimeError(f"HTTP {e.code}: {raw[:400]!r}") from None

    tok = token or (_g._load() or {}).get("accessToken")
    if not tok:
        raise RuntimeError("No Grok Bot session. Run: python3 grokbot_login.py login")
    try:
        raw = _post(tok)
    except RuntimeError as e:
        if token is None and str(e).startswith("HTTP 401"):
            sess = _g.refresh_session()
            raw = _post(sess["accessToken"])
        else:
            raise

    text_parts = []
    tool_parts = []
    i = 0
    while i + 5 <= len(raw):
        flags = raw[i]
        (ln,) = struct.unpack(">I", raw[i + 1:i + 5])
        i += 5
        chunk = raw[i:i + ln]
        i += ln
        if flags & 0x02:
            try:
                err = json.loads(chunk)
                if "error" in err:
                    raise RuntimeError(json.dumps(err["error"])[:600])
            except json.JSONDecodeError:
                pass
            continue
        ev = _parse_event(chunk)
        if ev.get("model"):
            routed_model = ev["model"]
        if ev.get("tool_parts"):
            tool_parts.extend(ev["tool_parts"])
        t = ev.get("text", "")
        if t:
            t = _clean(t)
            if t:
                text_parts.append(t)
                if on_chunk:
                    on_chunk(t)
    result_text = "".join(text_parts)
    if routed_model:
        stream.last_model = routed_model
    return {"text": result_text, "tool_calls": _merge_tool_parts(tool_parts),
            "model": routed_model}


def stream(prompt: str, model: str = DEFAULT_MODEL, token: str | None = None,
           conversation_id: str | None = None, history=None,
           on_chunk=None, tools=None) -> str:
    return infer(prompt, model, token, conversation_id, history, tools, on_chunk)["text"]


stream.last_model = ""


