#!/usr/bin/env python3
# Copyright (c) 2026 Ilya Balobanov. MIT; see the skill LICENSE.
# Adapted from bablobanov/hermes-telegram-checklist at 519dcd21a0521298c6ec742ae7b0ff8a3cfe9f85.
"""Standalone native Telegram checklist CLI (MTProto user session).

Create/get/append/toggle checklists only; never general messages or login.
Creating requires Telegram Premium. Saved Messages is always allowed; all
other destinations require a negative group/chat ID in the owning profile's
skills.config.telegram_checklist.chats setting (whole chat or chat:topic).
Every command, reads included, enforces the same allowlist and topic gate.

Bind the owning HERMES_HOME at process launch. Non-secret settings live in
config.yaml; relative session paths resolve there. TELETHON_API_ID and
TELETHON_API_HASH come from the process environment or the profile's .env.
Legacy process TELETHON_CHECKLIST_CHATS and TELETHON_SESSION are fallbacks;
non-secret grants are deliberately NOT loaded from the secret .env file.

plan and create --dry-run are strictly offline: no Telethon import, client,
credential read, or session-directory writes. Append/toggle dry-runs read the
live list but send no writes. Online writes reread state in verified; a
failed reread or absent message ID is uncertain success, not safe to retry.

Output: one JSON object, ok:true or ok:false (exit 1); --help is plain text.
See SKILL.md and references/plan-contract.md for tool-framed invocations.
"""
import argparse
import asyncio
import json
import os
import random
import re
import sys
import unicodedata
from pathlib import Path

# JSON goes to stdout; force UTF-8 so emoji survive non-UTF-8 pipes (e.g. Windows)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
except (AttributeError, OSError, ValueError):
    pass

# Populated by the online dependency loader; offline commands use none of them.
TelegramClient = FloodWaitError = RPCError = None
SendMediaRequest = AppendTodoListRequest = ToggleTodoCompletedRequest = None
GetForumTopicsRequest = GetForumTopicsByIDRequest = None
InputMediaTodo = TodoList = TodoItem = TextWithEntities = None
InputReplyToMessage = MessageMediaToDo = None


def _load_telethon():
    """Online-only dependency: planning must work without Telethon installed."""
    import importlib

    modules = {
        "telethon": ["TelegramClient"],
        "telethon.errors": ["FloodWaitError", "RPCError"],
        "telethon.tl.functions.messages": [
            "SendMediaRequest", "AppendTodoListRequest", "ToggleTodoCompletedRequest",
            "GetForumTopicsRequest", "GetForumTopicsByIDRequest"],
        "telethon.tl.types": [
            "InputMediaTodo", "TodoList", "TodoItem", "TextWithEntities",
            "InputReplyToMessage", "MessageMediaToDo"],
    }
    try:
        for module_name, names in modules.items():
            module = importlib.import_module(module_name)
            for name in names:
                globals()[name] = getattr(module, name)
    except (ImportError, AttributeError) as exc:
        _die(f"telethon import failed: {exc} (need telethon>=1.44,<2)")

try:
    HERMES = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")  # health: allow HX001 -- standalone process; no core import; launch with owning HERMES_HOME
except (RuntimeError, OSError):  # no home: process env only, no .env fallback
    HERMES = None
if os.environ.get("TELETHON_SESSION"):
    SESSION = Path(os.environ["TELETHON_SESSION"])
elif HERMES is not None:
    SESSION = HERMES / "telethon" / "user.session"
else:  # neither a home nor an explicit session path - refuse cleanly, never traceback
    print(json.dumps({"ok": False, "error": "cannot resolve the home directory - "
                      "set HERMES_HOME or TELETHON_SESSION"}, ensure_ascii=False))
    sys.exit(1)

# Telegram server caps: appConfig todo_items_max / todo_title_length_max /
# todo_item_length_max (defaults as of 2026, see core.telegram.org/api/config).
# Lengths are measured in UTF-16 code units, like everything text in Telegram.
MAX_TASKS = 30
MAX_TITLE = 255
MAX_TASK = 200

PLAN_MEDIA = {"text", "photo", "document", "video", "audio", "voice", "webpage", "rich_media"}


class ChecklistError(Exception):
    """A checklist problem that maps to a clean {"ok": false} JSON."""


def _die(msg):
    print(json.dumps({"ok": False, "error": msg}, ensure_ascii=False))
    sys.exit(1)


def _load_env_file(path):
    vals = {}
    try:
        for line in Path(path).read_text(encoding="utf-8-sig", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            vals[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return vals


def _env(name, empty_wins=False):
    """Process environment first, ~/.hermes/.env as fallback.

    empty_wins: a set-but-empty variable counts as a value (lets an empty
    TELETHON_CHECKLIST_CHATS narrow the allowlist); credentials treat empty
    as unset and fall back to the file.
    """
    v = os.environ.get(name)
    if v is None or (v == "" and not empty_wins):
        if HERMES is None:
            return None
        return _load_env_file(HERMES / ".env").get(name)
    return v


def _creds():
    api_id = _env("TELETHON_API_ID")
    api_hash = _env("TELETHON_API_HASH")
    if not (api_id and api_hash):
        where = "the environment" if HERMES is None else "the owning profile .env or the environment"
        _die(f"missing TELETHON_API_ID / TELETHON_API_HASH (set them in {where})")
    try:
        api_id = int(str(api_id).strip())
    except ValueError:
        # fixed message on purpose: never echo the value (it may be a mispasted secret)
        _die("TELETHON_API_ID must be numeric")
    if api_id <= 0:
        _die("TELETHON_API_ID must be numeric")
    return api_id, api_hash


def _parse_allowlist(raw):
    """TELETHON_CHECKLIST_CHATS -> (allowlist, warnings).

    Comma-separated entries: '-100123' (whole chat) or '-100123:33' (topic 33
    only). Only negative chat ids (groups/channels) are accepted - user peers
    stay out of scope for a WRITE tool. Topic ids start at 2: the General
    topic (1) cannot be topic-restricted, allow the whole chat instead.
    Saved Messages ('me') is always allowed. Bad entries produce a warning
    that never echoes the raw value (it could be a mispasted secret).
    Representation: {chat: None} means any topic, {chat: {33, 41}} only those.
    """
    allowed = {"me": None}
    warnings = []
    for idx, part in enumerate((raw or "").replace(" ", "").split(","), 1):
        if not part:
            continue
        chat_s, sep, thread_s = part.partition(":")
        try:
            chat = int(chat_s)
        except ValueError:
            warnings.append(f"allowlist: ignored entry #{idx} (expected -100123 or -100123:33)")
            continue
        if chat >= 0:
            warnings.append(f"allowlist: ignored entry #{idx} "
                            "(chat id must be negative; user peers are not supported)")
            continue
        if not sep:
            allowed[chat] = None
            continue
        # len cap keeps int() safe from CPython's digit-count conversion limit
        if (not (thread_s.isascii() and thread_s.isdigit() and len(thread_s) <= 10)
                or int(thread_s) < 2):
            warnings.append(f"allowlist: ignored entry #{idx} (topic id must be >= 2; "
                            "the General topic cannot be topic-restricted)")
            continue
        if allowed.get(chat, set()) is None:
            continue  # the whole chat is already allowed
        allowed.setdefault(chat, set()).add(int(thread_s))
    return allowed, warnings


def _setting(key, default=None):
    """Read non-secret settings from the launch profile; never migrate files."""
    if HERMES is None or not (HERMES / "config.yaml").exists():
        return default
    unreadable = "cannot read telegram_checklist settings from profile config.yaml"
    try:
        try:
            import hermes_yaml as yaml
        except ImportError:
            import yaml
        settings = yaml.safe_load((HERMES / "config.yaml").read_text(encoding="utf-8-sig")) or {}
        for section in ("skills", "config", "telegram_checklist"):
            if not isinstance(settings, dict):
                _die(unreadable)
            # A bare "section:" header (the block commented out) parses to None, not {}:
            # read it as absent, like a missing key; only a present non-mapping fails closed.
            settings = settings.get(section) or {}
        if not isinstance(settings, dict):
            _die(unreadable)
        return settings.get(key, default)
    except Exception:  # health: allow BLE001 -- external YAML parser failures must fail closed as JSON
        _die(unreadable)


def _session_path():
    value = _setting("session")
    if value is None:
        return SESSION
    if not isinstance(value, str) or not value.strip():
        _die("telegram_checklist.session must be a non-empty path")
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if HERMES is None:
        _die("relative session paths require HERMES_HOME")
    return HERMES / path


def _allowed():
    """Profile config wins; legacy env grants are used only when absent."""
    missing = object()
    raw = _setting("chats", missing)
    if raw is missing:
        raw = os.environ.get("TELETHON_CHECKLIST_CHATS") or ""
    if not isinstance(raw, str):
        _die("telegram_checklist.chats must be a comma-separated string")
    return _parse_allowlist(raw)


def _fmt_allowlist(allowed):
    return {str(k): ("any" if v is None else sorted(v)) for k, v in allowed.items()}


def _norm_chat(chat):
    if isinstance(chat, (bool, float)):
        _die(f"chat must be 'me' or a numeric chat_id; got {chat!r}")
    s = str(chat).strip().lower()
    if s in ("me", "saved", "self"):
        return "me"
    try:
        return int(chat)
    except (ValueError, TypeError):
        _die(f"chat must be 'me' or a numeric chat_id; got {chat!r}")


def _check_chat(chat, allowed):
    target = _norm_chat(chat)
    if target not in allowed:
        _die(f"target {target} not allowed - only Saved ('me') and TELETHON_CHECKLIST_CHATS "
             f"entries; current allowlist: {_fmt_allowlist(allowed)}")
    return target


def _check_thread(target, thread, allowed):
    """Enforce topic-restricted allowlist entries ('-100123:33')."""
    threads = allowed.get(target)
    if threads is None:
        return
    if thread is None or int(thread) not in threads:
        _die(f"chat {target} is allowed only for topics {sorted(threads)}; got thread {thread}")


def _msg_topic_id(msg):
    """Forum topic id of a message (None outside a topic)."""
    r = getattr(msg, "reply_to", None)
    if r is None or not getattr(r, "forum_topic", False):
        return None
    return getattr(r, "reply_to_top_id", None) or getattr(r, "reply_to_msg_id", None)


def _check_msg_thread(target, msg, msg_id, allowed):
    """Topic gate for an already-fetched message (get / append / toggle).

    Unlike _check_thread (where the thread comes from the caller's own
    arguments), the topic here was learned from the server - so the refusal
    must not name it: under a per-topic allowlist a probe by message id would
    otherwise map the chat's messages to topics one refusal at a time."""
    threads = allowed.get(target)
    if threads is None:
        return
    topic = _msg_topic_id(msg)
    if topic is None or int(topic) not in threads:
        _die(f"message {int(msg_id)} is outside the allowed topics of chat {target}")


def build_link(chat_id, topic_id, message_id):
    """Link to a private supergroup message: t.me/c/<id without -100>/<topic>/<msg>.

    Not called internally - a documented utility for plan authors and tests.
    """
    internal = str(chat_id)
    internal = internal[4:] if internal.startswith("-100") else internal.lstrip("-")
    if topic_id:
        return f"https://t.me/c/{internal}/{topic_id}/{message_id}"
    return f"https://t.me/c/{internal}/{message_id}"


_URL_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                 "-._~:/?#@!$&*+,;=%()[]")  # [] belong to IPv6-literal hosts
_TRAILING_PUNCT = ".,;:!?)]}\"'"
# case-insensitive scheme anchor searched on the ORIGINAL text: a lowercased
# copy is a trap - str.lower() is not length-preserving in Unicode (Turkish
# U+0130 lowers to TWO chars), which desynchronizes indices
_SCHEME_RE = re.compile(r"https?://", re.IGNORECASE | re.ASCII)


def _iri_token_end(text, i):
    """End index of the full linkifiable URL/IRI token starting at i: ASCII URL
    characters, alphanumerics of ANY script, and apostrophes glued between them
    - everything a chat client folds into one clickable link. Used to consume a
    whole foreign IRI so a t.me link buried inside it is never seen as its own."""
    n = len(text)
    j = i
    while j < n:
        c = text[j]
        if c in _URL_CHARS or c.isalnum():
            j += 1
        elif c == "'":
            # consume apostrophe(s) only when the token continues past them
            # (internal to a longer URL); a trailing/closing quote ends it, so
            # 'LINK' quoting keeps matching while LINK'word / .../x''?u= do not
            k = j
            while k < n and text[k] == "'":
                k += 1
            if k < n and (text[k] in _URL_CHARS or text[k].isalnum()):
                j = k
            else:
                break
        else:
            break
    return j


def _link_in_text(link, text):
    """True when the link appears as a COMPLETE, STANDALONE URL token.

    The link must equal a whole token - never a prefix of a longer URL
    (.../50, .../5.evil), never a fragment of a foreign IRI (http://x/?u=<link>),
    never a run merged with adjacent letters (<link>foo). Only genuine trailing
    sentence punctuation is stripped before the equality check. Scheme anchors
    are found on the ORIGINAL text - a lowercased copy would desync indices,
    since str.lower() is not length-preserving in Unicode.
    """
    i, n = 0, len(text)
    while True:
        m = _SCHEME_RE.search(text, i)
        if not m:
            return False
        start = m.start()
        a = start                       # end of the strict ASCII run (a t.me URL is ASCII)
        while a < n and text[a] in _URL_CHARS:
            a += 1
        full = _iri_token_end(text, start)   # end of the whole linkified IRI
        if a == full and text[start:a].rstrip(_TRAILING_PUNCT) == link:
            return True
        # a prefix, spoof, or a link nested in a foreign IRI: resume PAST the
        # ENTIRE token. full > start always (text[start] is a scheme letter in
        # _URL_CHARS), so i strictly advances - no infinite loop.
        i = full


def _lock_session():
    """Best-effort chmod 0600 on the ACTUAL session file. Telethon decides the
    filename by STRING suffix (str.endswith('.session')), not pathlib .suffix -
    so a file literally named '.session' is used as-is, not doubled. chmod only
    a regular file (never a directory) and never let this raise."""
    try:
        s = str(SESSION)
        sfile = Path(s if s.endswith(".session") else s + ".session")
        if sfile.is_file():
            os.chmod(sfile, 0o600)
    except OSError:
        pass  # best effort; Windows ACLs are not chmod-representable


def _twe(text):
    return TextWithEntities(text=text, entities=[])


def _tg_len(s):
    """Telegram measures text limits in UTF-16 code units (emoji count as 2)."""
    try:
        return len(s.encode("utf-16-le")) // 2
    except UnicodeEncodeError:
        _die("text contains unpaired surrogates - not valid Telegram text")


def _check_task_texts(tasks):
    for t in tasks:
        if _tg_len(t) > MAX_TASK:
            _die(f"task too long ({_tg_len(t)} utf-16 units, Telegram limit {MAX_TASK})")


# ---------------------------------------------------------------- plan contract

def _plan_float(raw):
    """Reject non-finite numbers: 1e999 overflows to inf without ever hitting
    parse_constant, and json.dumps would then emit bare Infinity - not JSON."""
    v = float(raw)
    if v != v or v in (float("inf"), float("-inf")):
        _die("plan file is not valid JSON: non-finite numbers are not allowed")
    return v


def _load_plan(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"),
                          parse_float=_plan_float,
                          parse_constant=lambda c: _die(
                              f"plan file is not valid JSON: {c} is not allowed"))
    except (OSError, UnicodeDecodeError) as e:
        _die(f"cannot read plan file: {e}")
    except json.JSONDecodeError as e:
        _die(f"plan file is not valid JSON: {e}")
    if not isinstance(data, dict):
        _die("plan must be a JSON object")
    return data


def _validate_plan_task(t, i, collected, seen, source_map, warnings):
    if not isinstance(t, dict):
        _die(f"task #{i} must be an object with 'text'")
    text = t.get("text")
    if not isinstance(text, str) or not text.strip():
        _die(f"task #{i}: text must be a non-empty string")
    text = text.strip()
    if _tg_len(text) > MAX_TASK:
        _die(f"task #{i}: text too long ({_tg_len(text)} utf-16 units, Telegram limit {MAX_TASK})")
    # normalize FIRST, casefold after (then normalize again - casefold can
    # denormalize): zero-width and compatibility tricks must not sneak
    # visually identical duplicates past the gate
    norm = unicodedata.normalize("NFKC", text)
    norm = "".join(ch for ch in norm if unicodedata.category(ch) != "Cf")
    norm = unicodedata.normalize("NFKC", norm.casefold())
    norm = " ".join(norm.split())
    if norm in seen:
        _die(f"task #{i}: duplicate task text (merge semantic duplicates, keep all links)")
    seen.add(norm)
    sources = t.get("sources") or []
    if not isinstance(sources, list):
        _die(f"task #{i}: sources must be a list")
    # only a direct https://t.me/ link counts as evidence; a whitespace or
    # arbitrary substring must not slip through the presence check
    links = []
    for s in sources:
        if isinstance(s, dict) and isinstance(s.get("link"), str):
            link = s["link"].strip()
            if link.startswith("https://t.me/") and len(link) > len("https://t.me/"):
                links.append(link)
    if collected:
        if not sources:
            _die(f"task #{i}: collected_from_chat requires sources with links")
        if not links:
            _die(f"task #{i}: sources must contain at least one direct https://t.me/ link")
        if not any(_link_in_text(link, text) for link in links):
            _die(f"task #{i}: none of the source links is present in the task text")
    for s in sources:
        if not isinstance(s, dict):
            _die(f"task #{i}: each source must be an object")
        media = str(s.get("media") or "text")
        if media not in PLAN_MEDIA:
            warnings.append(f"task #{i}: unknown media type {media!r}")
        source_map.append({
            "task": i,
            "topic": s.get("topic"),
            "message_id": s.get("message_id"),
            "link": s.get("link"),
            "media": media,
            "says": s.get("says"),
        })
    return text


def _validate_plan(plan, allowed):
    """Validate plan.json; return (payload, source_map, warnings).

    The only road into create --from-plan: exactly what passed validation is
    what goes to Telegram. Any contract violation dies with ok:false.
    """
    warnings = []
    target = plan.get("target") or {}
    if not isinstance(target, dict) or target.get("chat") is None:
        _die("plan.target.chat is required")
    tchat = _check_chat(target.get("chat"), allowed)
    thread = target.get("thread")
    if thread is not None:
        if isinstance(thread, (bool, float)):
            _die(f"plan.target.thread must be an integer; got {target.get('thread')!r}")
        try:
            thread = int(thread)
        except (TypeError, ValueError):
            _die(f"plan.target.thread must be an integer; got {target.get('thread')!r}")
        if thread <= 0:
            _die("plan.target.thread must be a positive forum topic id")
        if thread == 1:
            _die("thread 1 is the General topic - omit thread to post to General")
    if tchat == "me" and thread:
        _die("plan.target.thread is not applicable for Saved Messages")
    _check_thread(tchat, thread, allowed)
    title = plan.get("title")
    if title is not None and not isinstance(title, str):
        _die("plan.title must be a string")
    title = (title or "").strip()
    if not title:
        _die("empty plan.title")
    if _tg_len(title) > MAX_TITLE:
        _die(f"title too long ({_tg_len(title)} utf-16 units, Telegram limit {MAX_TITLE})")
    raw_tasks = plan.get("tasks") or []
    if not isinstance(raw_tasks, list) or not raw_tasks:
        _die("plan.tasks must be a non-empty list")
    if len(raw_tasks) > MAX_TASKS:
        _die(f"too many tasks ({len(raw_tasks)}), Telegram limit {MAX_TASKS}")
    for key in ("shared", "collected_from_chat"):
        if key in plan and not isinstance(plan.get(key), bool):
            # a string like "false" is truthy - silently enabling shared would be worse
            _die(f"plan.{key} must be true or false (JSON boolean)")
    collected = bool(plan.get("collected_from_chat"))
    shared = bool(plan.get("shared"))
    texts, seen, source_map = [], set(), []
    for i, task in enumerate(raw_tasks, 1):
        texts.append(_validate_plan_task(task, i, collected, seen, source_map, warnings))
    payload = {
        "chat": tchat, "thread": thread, "title": title, "tasks": texts,
        "others_can_append": shared, "others_can_complete": shared,
    }
    source_map.sort(key=lambda s: (
        str(s.get("topic") or ""),
        not isinstance(s.get("message_id"), int),   # numeric ids first, in order
        s.get("message_id") if isinstance(s.get("message_id"), int) else 0,
        str(s.get("message_id") or ""),
        s["task"],
    ))
    return payload, source_map, warnings


def cmd_plan(args):
    """Offline: validate the plan and print the source map. No client is created."""
    allowed, warnings = _allowed()
    plan = _load_plan(args.file)
    payload, source_map, w2 = _validate_plan(plan, allowed)
    print(json.dumps({
        "ok": True, "kind": "plan", "dry_run": True,
        "would_send": payload, "source_map": source_map,
        "warnings": warnings + w2,
    }, ensure_ascii=False, default=str))


def _payload_from_args(args, allowed):
    """Payload from --title/--task (the user dictated the tasks, no plan file)."""
    target = _check_chat("me" if args.chat is None else args.chat, allowed)
    if args.thread is not None and args.thread <= 0:
        _die("--thread must be a positive forum topic id")
    if args.thread == 1:
        _die("thread 1 is the General topic - omit --thread to post to General")
    if target == "me" and args.thread:
        _die("--thread is not applicable for Saved Messages")
    _check_thread(target, args.thread, allowed)
    tasks = [t.strip() for t in (args.task or []) if t and t.strip()]
    if not tasks:
        _die("no tasks (repeat --task ...)")
    if len(tasks) > MAX_TASKS:
        _die(f"too many tasks ({len(tasks)}), Telegram limit {MAX_TASKS}")
    title = (args.title or "").strip()
    if not title:
        _die("empty --title")
    if _tg_len(title) > MAX_TITLE:
        _die(f"title too long ({_tg_len(title)} utf-16 units, Telegram limit {MAX_TITLE})")
    _check_task_texts(tasks)
    return {
        "chat": target, "thread": args.thread, "title": title, "tasks": tasks,
        "others_can_append": bool(args.others_append),
        "others_can_complete": bool(args.others_complete),
    }


def _create_payload(args):
    """Single offline assembly point for create (both input paths)."""
    allowed, warnings = _allowed()
    if args.from_plan:
        if (args.title is not None or args.task is not None or args.thread is not None
                or args.others_append or args.others_complete or args.chat is not None):
            _die("--from-plan is exclusive: do not combine with "
                 "--title/--task/--chat/--thread/--others-*")
        plan = _load_plan(args.from_plan)
        payload, source_map, w2 = _validate_plan(plan, allowed)
        return payload, source_map, warnings + w2
    if not args.title or not args.task:
        _die("either --from-plan or --title with --task is required")
    return _payload_from_args(args, allowed), [], warnings


def cmd_create_dry(args):
    """Offline dry-run: print what WOULD be sent, exit before any client exists."""
    payload, source_map, warnings = _create_payload(args)
    print(json.dumps({
        "ok": True, "kind": "checklist-create", "dry_run": True,
        "would_send": payload, "source_map": source_map, "warnings": warnings,
    }, ensure_ascii=False, default=str))


# ---------------------------------------------------------------- online helpers

async def _entity(client, target):
    return "me" if target == "me" else await client.get_entity(target)


async def _read_checklist(client, entity, msg_id):
    """Return (todo, tasks, msg); raise ChecklistError when missing/not a checklist."""
    msg = await client.get_messages(entity, ids=int(msg_id))
    if msg is None:
        raise ChecklistError(f"message {msg_id} not found in this chat")
    media = getattr(msg, "media", None)
    if not isinstance(media, MessageMediaToDo):
        raise ChecklistError(f"message {msg_id} is not a checklist")
    completed = {c.id for c in (media.completions or [])}
    tasks = [{"id": it.id, "text": it.title.text, "done": it.id in completed}
             for it in media.todo.list]
    return media.todo, tasks, msg


def _verified(todo, tasks, with_tasks=True):
    """JSON-ready snapshot of a re-read checklist."""
    out = {"title": todo.title.text, "total": len(tasks),
           "done": sum(1 for t in tasks if t["done"]),
           "others_can_append": bool(todo.others_can_append),
           "others_can_complete": bool(todo.others_can_complete)}
    if with_tasks:
        out["tasks"] = tasks
    return out


async def _verify_after_write(client, entity, msg_id, with_tasks=True):
    """Best-effort re-read after a successful write; never raises - a verify
    failure must not turn an already-applied write into ok:false. The snapshot
    build stays INSIDE the try. BaseException on purpose: an interrupt or task
    cancellation inside this narrow read-only window must not fake a failure
    of a write that already happened."""
    try:
        todo, tasks, _msg = await _read_checklist(client, entity, msg_id)
        return _verified(todo, tasks, with_tasks=with_tasks)
    except BaseException:  # health: allow BLE001 -- cancellation during verification must not report an applied write as failed
        return False


def _extract_msg_id(updates):
    best = None
    for u in getattr(updates, "updates", []) or []:
        mid = getattr(getattr(u, "message", None), "id", None)
        if mid is None:
            mid = getattr(u, "id", None)
        if isinstance(mid, int):
            best = mid
    if best is None:
        # e.g. UpdateShortSentMessage carries the id on the top-level object
        mid = getattr(updates, "id", None)
        if isinstance(mid, int):
            best = mid
    return best


# ---------------------------------------------------------------- commands

async def cmd_create(client, payload, warnings):
    entity = await _entity(client, payload["chat"])
    todo = TodoList(
        title=_twe(payload["title"]),
        list=[TodoItem(id=i + 1, title=_twe(t))
              for i, t in enumerate(payload["tasks"])],
        others_can_append=payload["others_can_append"],
        others_can_complete=payload["others_can_complete"],
    )
    reply_to = None
    if payload["thread"]:
        thread = int(payload["thread"])
        reply_to = InputReplyToMessage(reply_to_msg_id=thread, top_msg_id=thread)
    updates = await client(SendMediaRequest(
        peer=entity, media=InputMediaTodo(todo=todo), message="",
        random_id=random.randrange(-(2 ** 63), 2 ** 63), reply_to=reply_to,
    ))
    msg_id = _extract_msg_id(updates)
    verified = False
    if msg_id is None:
        warnings = warnings + ["server did not return a message_id; the checklist was "
                               "likely sent - check the chat before retrying"]
    else:
        verified = await _verify_after_write(client, entity, msg_id)
        if verified is False:
            warnings = warnings + ["created, but the post-write re-read failed - verify in the app"]
        elif (verified["total"] != len(payload["tasks"])
                or verified["title"] != payload["title"]
                or verified["others_can_append"] != payload["others_can_append"]
                or verified["others_can_complete"] != payload["others_can_complete"]):
            warnings = warnings + [
                f"post-check mismatch: sent {len(payload['tasks'])} tasks titled "
                f"{payload['title']!r} (others_can_append={payload['others_can_append']}, "
                f"others_can_complete={payload['others_can_complete']}), read back "
                f"{verified['total']} tasks titled {verified['title']!r} "
                f"(others_can_append={verified['others_can_append']}, "
                f"others_can_complete={verified['others_can_complete']})"]
    print(json.dumps({
        "ok": True, "kind": "checklist-create", "chat": payload["chat"],
        "thread": payload["thread"], "message_id": msg_id, "title": payload["title"],
        "tasks_created": len(payload["tasks"]), "verified": verified,
        "warnings": warnings,
    }, ensure_ascii=False, default=str))


async def cmd_get(client, args):
    allowed, warnings = _allowed()
    target = _check_chat(args.chat, allowed)
    entity = await _entity(client, target)
    try:
        todo, tasks, msg = await _read_checklist(client, entity, args.message_id)
    except ChecklistError as e:
        _die(str(e))
    # a per-topic allowlist entry gates READS too: the topic of a message is
    # known only after fetching it, so the check sits right after the read and
    # before anything from the checklist is printed (same order as append/toggle)
    _check_msg_thread(target, msg, args.message_id, allowed)
    print(json.dumps({
        "ok": True, "kind": "checklist-get", "chat": target,
        "message_id": int(args.message_id), "title": todo.title.text,
        "done": sum(1 for t in tasks if t["done"]), "total": len(tasks),
        "tasks": tasks,
        "others_can_append": bool(todo.others_can_append),
        "others_can_complete": bool(todo.others_can_complete),
        "warnings": warnings,
    }, ensure_ascii=False, default=str))


async def cmd_append(client, args):
    allowed, warnings = _allowed()
    target = _check_chat(args.chat, allowed)
    new = [t.strip() for t in (args.task or []) if t and t.strip()]
    if not new:
        _die("no tasks to append (repeat --task ...)")
    _check_task_texts(new)
    entity = await _entity(client, target)
    try:
        _todo, tasks, msg = await _read_checklist(client, entity, args.message_id)
    except ChecklistError as e:
        _die(str(e))
    _check_msg_thread(target, msg, args.message_id, allowed)
    if len(tasks) + len(new) > MAX_TASKS:
        _die(f"would exceed Telegram limit {MAX_TASKS} tasks")
    max_id = max([t["id"] for t in tasks], default=0)
    if max_id + len(new) > 2 ** 31 - 1:  # TL int is signed 32-bit
        _die("task id space exhausted for this checklist - rebuild it as a new list")
    items = [TodoItem(id=max_id + i + 1, title=_twe(t)) for i, t in enumerate(new)]
    if args.dry_run:
        print(json.dumps({
            "ok": True, "kind": "checklist-append", "dry_run": True, "chat": target,
            "message_id": int(args.message_id),
            "would_append": [{"id": it.id, "text": it.title.text} for it in items],
            "total_after": len(tasks) + len(items), "warnings": warnings,
        }, ensure_ascii=False, default=str))
        return
    peer = await client.get_input_entity(entity)
    await client(AppendTodoListRequest(peer=peer, msg_id=int(args.message_id), list=items))
    verified = await _verify_after_write(client, entity, args.message_id)
    if verified is False:
        warnings = warnings + ["appended, but the post-write re-read failed - verify in the app"]
    elif verified["total"] != len(tasks) + len(items):
        warnings = warnings + [f"post-check mismatch: expected {len(tasks) + len(items)} "
                               f"tasks after append, read back {verified['total']}"]
    print(json.dumps({
        "ok": True, "kind": "checklist-append", "chat": target,
        "message_id": int(args.message_id), "appended": len(items),
        "total": len(tasks) + len(items), "verified": verified,
        "warnings": warnings,
    }, ensure_ascii=False, default=str))


async def cmd_toggle(client, args):
    allowed, warnings = _allowed()
    target = _check_chat(args.chat, allowed)
    done = list(dict.fromkeys(int(x) for x in (args.done or [])))
    undone = list(dict.fromkeys(int(x) for x in (args.undone or [])))
    if not done and not undone:
        _die("nothing to toggle (use --done <id> and/or --undone <id>)")
    overlap = sorted(set(done) & set(undone))
    if overlap:
        _die(f"task ids in both --done and --undone: {overlap}")
    entity = await _entity(client, target)
    try:
        _todo, tasks, msg = await _read_checklist(client, entity, args.message_id)
    except ChecklistError as e:
        _die(str(e))
    _check_msg_thread(target, msg, args.message_id, allowed)
    ids = {t["id"] for t in tasks}
    bad = [x for x in done + undone if x not in ids]
    if bad:
        _die(f"unknown task ids {bad}; valid: {sorted(ids)}")
    if args.dry_run:
        print(json.dumps({
            "ok": True, "kind": "checklist-toggle", "dry_run": True, "chat": target,
            "message_id": int(args.message_id),
            "would_mark_done": done, "would_mark_undone": undone,
            "warnings": warnings,
        }, ensure_ascii=False, default=str))
        return
    peer = await client.get_input_entity(entity)
    await client(ToggleTodoCompletedRequest(peer=peer, msg_id=int(args.message_id),
                                            completed=done, incompleted=undone))
    verified = await _verify_after_write(client, entity, args.message_id)
    if verified is False:
        warnings = warnings + ["toggled, but the post-write re-read failed - verify in the app"]
    else:
        state = {t["id"]: t["done"] for t in verified.get("tasks", [])}
        # an id missing from the re-read is a mismatch for BOTH directions
        wrong = ([x for x in done if state.get(x) is not True]
                 + [x for x in undone if state.get(x) is not False])
        if wrong:
            warnings = warnings + [f"post-check mismatch: task ids {sorted(set(wrong))} "
                                   "did not change as requested"]
    print(json.dumps({
        "ok": True, "kind": "checklist-toggle", "chat": target,
        "message_id": int(args.message_id), "marked_done": done,
        "marked_undone": undone, "verified": verified, "warnings": warnings,
    }, ensure_ascii=False, default=str))


async def cmd_list_topics(client, args):
    """Forum topics of an allowlisted chat - to pick a source or a target topic."""
    allowed, warnings = _allowed()
    target = _check_chat(args.chat, allowed)
    if target == "me":
        _die("'me' has no forum topics - pass a forum chat id from TELETHON_CHECKLIST_CHATS")
    entity = await client.get_entity(target)
    threads = allowed.get(target)
    if threads is None:
        res = await client(GetForumTopicsRequest(
            peer=entity, offset_date=None, offset_id=0, offset_topic=0, limit=100))
    else:
        # per-topic allowlist: ask the server for exactly the allowed ids - the
        # chat's other topics are never fetched, not merely filtered out
        res = await client(GetForumTopicsByIDRequest(peer=entity, topics=sorted(threads)))
    topics = [{"id": t.id, "title": t.title, "closed": bool(getattr(t, "closed", False))}
              for t in getattr(res, "topics", [])
              if hasattr(t, "title") and (threads is None or t.id in threads)]  # fail-closed
    total = getattr(res, "count", None)
    if threads is not None:
        # a deleted / never-existing allowed id comes back as ForumTopicDeleted (no title)
        missing = sorted(threads - {t["id"] for t in topics})
        if missing:
            warnings = warnings + [f"allowed topics {missing} not found in chat {target}"]
    elif isinstance(total, int) and total > len(topics):
        warnings = warnings + [f"listed the first {len(topics)} of {total} topics - "
                               "the rest are not shown"]
    print(json.dumps({
        "ok": True, "kind": "list-topics", "chat": target,
        "total": total, "listed": len(topics),
        "topics": topics, "warnings": warnings,
    }, ensure_ascii=False, default=str))


async def run(args):
    global SESSION
    SESSION = _session_path()
    _load_telethon()
    client = None
    try:
        payload, warnings = None, []
        if args.cmd == "create":
            # full offline validation BEFORE any network: a refusal needs no client
            payload, _source_map, warnings = _create_payload(args)
        else:
            # allowlist refusal for every other command also needs no client
            pre_allowed, _pre_warnings = _allowed()
            target = _check_chat(args.chat, pre_allowed)
            if args.cmd == "list-topics" and target == "me":
                _die("'me' has no forum topics - pass a forum chat id from TELETHON_CHECKLIST_CHATS")
        api_id, api_hash = _creds()
        created_by_us = False
        try:
            SESSION.parent.mkdir(parents=True)  # exist_ok=False: learn atomically
            created_by_us = True                # whether WE created the directory
        except FileExistsError:
            # also raised when a FILE blocks the parent chain - not "already a dir"
            if not SESSION.parent.is_dir():
                _die("cannot prepare the session directory (NotADirectoryError) - "
                     "check TELETHON_SESSION / HERMES_HOME")
        except OSError as e:
            # type name only: the env-derived path must not be echoed back
            _die(f"cannot prepare the session directory ({type(e).__name__}) - "
                 "check TELETHON_SESSION / HERMES_HOME")
        try:  # the session file grants full account access - keep it private
            if created_by_us or (HERMES is not None
                                 and SESSION.parent == HERMES / "telethon"):
                # never tighten an unrelated pre-existing directory the user
                # merely pointed TELETHON_SESSION into
                os.chmod(SESSION.parent, 0o700)
        except OSError:
            pass  # best effort; Windows ACLs are not chmod-representable
        _lock_session()  # an already-existing session file
        try:
            client = TelegramClient(str(SESSION), api_id, api_hash)
        except Exception as e:  # health: allow BLE001 -- Telethon backends raise multiple exception types; redact session paths
            _die(f"cannot open the session file ({type(e).__name__}) - check TELETHON_SESSION")
        # telethon's sqlite session is created in the constructor - lock it down
        # now, and once more in the finally below (covers a failing connect)
        _lock_session()
        await client.connect()
        if not await client.is_user_authorized():
            _die("session not authorized - authorize it locally outside the agent (see SKILL.md)")
        if args.cmd == "create":
            await cmd_create(client, payload, warnings)
        else:
            handlers = {"get": cmd_get, "append": cmd_append,
                        "toggle": cmd_toggle, "list-topics": cmd_list_topics}
            handler = handlers.get(args.cmd)
            if handler is None:
                _die(f"unknown command: {args.cmd}")
            await handler(client, args)
    except FloodWaitError as e:
        _die(f"FloodWait {e.seconds}s - aborting, try later")
    except RPCError as e:
        _die(f"Telegram error: {type(e).__name__}: {e}")
    except Exception as e:  # health: allow BLE001 -- CLI boundary emits one JSON error rather than a traceback
        _die(f"{type(e).__name__}: {e}")
    finally:
        try:
            if client is not None and client.is_connected():
                await client.disconnect()
        except BaseException:  # health: allow BLE001 S110 -- disconnect errors must not overwrite an already printed write result
            # BaseException on purpose: the result is already printed - even an
            # interrupt here must not add a second JSON object or a traceback
            pass
        if client is not None:
            _lock_session()  # never raises; covers a session created before a failed connect


class _Parser(argparse.ArgumentParser):
    # keep the stdout JSON contract even for bad CLI arguments (plain argparse
    # prints usage to stderr and exits 2)
    def error(self, message):
        _die(f"bad arguments: {message}")


def _build_parser():
    p = _Parser(description="Native Telegram To-Do lists via Telethon (WRITE, allowlist-gated)")
    sub = p.add_subparsers(dest="cmd", required=True)

    lt = sub.add_parser("list-topics", help="list forum topics of an allowlisted forum chat")
    lt.add_argument("--chat", required=True, help="numeric chat_id of a forum supergroup from the allowlist")

    pl = sub.add_parser("plan", help="validate plan.json offline and print the source map (sends nothing)")
    pl.add_argument("--file", required=True, help="path to plan.json")

    c = sub.add_parser("create", help="create a new checklist")
    c.add_argument("--title", help="checklist title (direct path)")
    c.add_argument("--task", action="append", help="task text, repeat per task (direct path)")
    c.add_argument("--from-plan", dest="from_plan", help="path to plan.json (research contract; exclusive)")
    # default=None (not "me"): --from-plan must detect an explicitly passed --chat
    c.add_argument("--chat", default=None, help="'me' (default) or a numeric chat_id from the allowlist")
    c.add_argument("--thread", type=int, help="forum topic id to post into")
    c.add_argument("--others-append", action="store_true", help="let others add tasks")
    c.add_argument("--others-complete", action="store_true", help="let others tick tasks")
    c.add_argument("--dry-run", action="store_true",
                   help="validate and print the plan fully offline, send nothing")

    g = sub.add_parser("get", help="read a checklist's state")
    g.add_argument("--chat", default="me", help="'me' (default) or a numeric chat_id from the allowlist")
    g.add_argument("--message-id", dest="message_id", required=True, type=int)

    a = sub.add_parser("append", help="add tasks to an existing checklist")
    a.add_argument("--chat", default="me", help="'me' (default) or a numeric chat_id from the allowlist")
    a.add_argument("--message-id", dest="message_id", required=True, type=int)
    a.add_argument("--task", action="append", required=True, help="task text (repeat per task)")
    a.add_argument("--dry-run", action="store_true", help="validate against the live list, send nothing")

    tg = sub.add_parser("toggle", help="mark tasks done / not done")
    tg.add_argument("--chat", default="me", help="'me' (default) or a numeric chat_id from the allowlist")
    tg.add_argument("--message-id", dest="message_id", required=True, type=int)
    tg.add_argument("--done", action="append", type=int, help="task id to mark done (repeatable)")
    tg.add_argument("--undone", action="append", type=int, help="task id to mark not done (repeatable)")
    tg.add_argument("--dry-run", action="store_true", help="validate against the live list, send nothing")

    return p


def main():
    args = _build_parser().parse_args()
    try:
        if args.cmd == "plan":
            cmd_plan(args)          # strictly offline: no Telethon client, no network
        elif args.cmd == "create" and args.dry_run:
            cmd_create_dry(args)    # strictly offline: no Telethon client, no network
        else:
            asyncio.run(run(args))
    except KeyboardInterrupt:
        _die("interrupted")
    except Exception as e:  # health: allow BLE001 -- last-resort JSON CLI boundary; SystemExit passes through
        _die(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
