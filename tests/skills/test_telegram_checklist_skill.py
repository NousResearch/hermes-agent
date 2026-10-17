#!/usr/bin/env python3
# Copyright (c) 2026 Ilya Balobanov. MIT; see the skill LICENSE.
# Adapted from bablobanov/hermes-telegram-checklist at 519dcd21a0521298c6ec742ae7b0ff8a3cfe9f85.
"""Offline tests for telethon_checklist.py. Telethon is stubbed via sys.modules
BEFORE the module import: no real telethon, no network, no session needed.
Run through scripts/run_tests.sh tests/skills/test_telegram_checklist_skill.py.
"""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

HERE = (Path(__file__).resolve().parents[2] / "optional-skills" /
        "communication" / "telegram-checklist" / "scripts")
sys.path.insert(0, str(HERE))

# --------------------------------------------------------------- telethon stub

CALLS = []  # ("connect",) / ("get_entity", x) / ("get_messages", ids) / ("request", <req>)


class _Simple:
    """Fixture base: stores kwargs as attributes (like TL objects do)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        return f"{type(self).__name__}({self.__dict__})"


class TodoList(_Simple):
    pass


class TodoItem(_Simple):
    pass


class TextWithEntities(_Simple):
    pass


class InputMediaTodo(_Simple):
    pass


class InputReplyToMessage(_Simple):
    pass


class MessageMediaToDo(_Simple):
    pass


class SendMediaRequest(_Simple):
    pass


class AppendTodoListRequest(_Simple):
    pass


class ToggleTodoCompletedRequest(_Simple):
    pass


class GetForumTopicsRequest(_Simple):
    pass


class GetForumTopicsByIDRequest(_Simple):
    pass


class RPCError(Exception):
    pass


class FloodWaitError(RPCError):  # mirrors real telethon: FloodWait is an RPCError
    def __init__(self, seconds=60):
        super().__init__(f"flood wait {seconds}s")
        self.seconds = seconds


class FakeClient:
    """TelegramClient stub. Behavior is scripted through FakeClient.script:
    script["get_messages"] - object or callable(ids) -> object
    script["request"]      - object, Exception or callable(request) -> object
    """

    created = 0  # how many clients were constructed (offline paths must keep 0)
    script = {}

    def __init__(self, *a, **kw):
        type(self).created += 1

    async def connect(self):
        CALLS.append(("connect",))

    def is_connected(self):
        return True

    async def disconnect(self):
        CALLS.append(("disconnect",))

    async def is_user_authorized(self):
        return True

    async def get_entity(self, x):
        CALLS.append(("get_entity", x))
        return ("entity", x)

    async def get_input_entity(self, x):
        return ("input", x)

    async def get_messages(self, entity, ids=None):
        CALLS.append(("get_messages", ids))
        out = type(self).script.get("get_messages")
        return out(ids) if callable(out) else out

    async def __call__(self, request):
        CALLS.append(("request", request))
        out = type(self).script.get("request")
        if callable(out):
            out = out(request)
        if isinstance(out, Exception):
            raise out
        return out if out is not None else _Simple(updates=[])


_telethon = types.ModuleType("telethon")
_telethon.TelegramClient = FakeClient
_errors = types.ModuleType("telethon.errors")
_errors.FloodWaitError = FloodWaitError
_errors.RPCError = RPCError
_tl = types.ModuleType("telethon.tl")
_fx = types.ModuleType("telethon.tl.functions")
_fmsg = types.ModuleType("telethon.tl.functions.messages")
_fmsg.SendMediaRequest = SendMediaRequest
_fmsg.AppendTodoListRequest = AppendTodoListRequest
_fmsg.ToggleTodoCompletedRequest = ToggleTodoCompletedRequest
_fmsg.GetForumTopicsRequest = GetForumTopicsRequest
_fmsg.GetForumTopicsByIDRequest = GetForumTopicsByIDRequest
_ttypes = types.ModuleType("telethon.tl.types")
for _cls in (InputMediaTodo, TodoList, TodoItem, TextWithEntities,
             InputReplyToMessage, MessageMediaToDo):
    setattr(_ttypes, _cls.__name__, _cls)
_telethon.errors = _errors
_telethon.tl = _tl
_tl.functions = _fx
_fx.messages = _fmsg
_tl.types = _ttypes
for _name, _mod in {
    "telethon": _telethon,
    "telethon.errors": _errors,
    "telethon.tl": _tl,
    "telethon.tl.functions": _fx,
    "telethon.tl.functions.messages": _fmsg,
    "telethon.tl.types": _ttypes,
}.items():
    sys.modules[_name] = _mod

# point the session at a temp file BEFORE the import (module reads env at import);
# assign unconditionally so a developer's real TELETHON_SESSION is never touched
os.environ["TELETHON_SESSION"] = str(Path(tempfile.gettempdir()) / "tc_test.session")

import telethon_checklist as tc  # noqa: E402  import after the stub - intentional
tc._load_telethon()


# --------------------------------------------------------------- helpers

CHAT = -1001234567890            # fake supergroup id used across fixtures
TOPIC = 33
LINK = "https://t.me/c/1234567890/33/123"
EMOJI = "\U0001F600"             # counts as 2 UTF-16 code units


def run_cli(*argv):
    """Run tc.main() with argv; return (exit_code, stdout_text).

    Also enforces the "never output on stderr" half of the contract on every
    single invocation across the suite.
    """
    out, err = io.StringIO(), io.StringIO()
    code = 0
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        with mock.patch.object(sys, "argv", ["telethon_checklist.py", *argv]):
            try:
                tc.main()
            except SystemExit as e:
                if isinstance(e.code, int):
                    code = e.code
                elif e.code is not None:
                    code = 1
    if err.getvalue():
        raise AssertionError(f"stderr must stay empty, got: {err.getvalue()!r}")
    return code, out.getvalue()


def run_json(*argv):
    code, text = run_cli(*argv)
    return code, json.loads(text)


def dies_with(fn, *a, **kw):
    """Call fn expecting _die(); return (exit_code, parsed_json)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            fn(*a, **kw)
        except SystemExit as e:
            return e.code, json.loads(buf.getvalue())
    return None, None


def checklist_msg(items=("a", "b"), done=(), topic=None, title="Demo",
                  appendable=False, completable=False):
    """Fake checklist message built from the stub TL types."""
    todo = TodoList(
        title=TextWithEntities(text=title, entities=[]),
        list=[TodoItem(id=i + 1, title=TextWithEntities(text=t, entities=[]))
              for i, t in enumerate(items)],
        others_can_append=appendable,
        others_can_complete=completable,
    )
    reply = None
    if topic:
        reply = _Simple(forum_topic=True, reply_to_top_id=topic,
                        reply_to_msg_id=topic)
    return _Simple(
        media=MessageMediaToDo(todo=todo,
                               completions=[_Simple(id=i) for i in done]),
        reply_to=reply,
    )


def plan_dict(**over):
    """A valid plan.json; fields overridable via kwargs."""
    base = {
        "target": {"chat": CHAT, "thread": TOPIC},
        "title": "Weekly tasks",
        "shared": False,
        "collected_from_chat": True,
        "tasks": [{
            "text": f"Do X: {LINK}",
            "sources": [{
                "link": LINK,
                "topic": f"Ideas ({TOPIC})",
                "message_id": 123,
                "media": "text",
                "says": "the post proposes doing X",
            }],
        }],
    }
    base.update(over)
    return base


def write_plan(data):
    """Write a plan to a temp file; return its path."""
    d = tempfile.mkdtemp(prefix="tcplan")
    p = Path(d) / "plan.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(p)


def env_dir(text):
    """Create a temp dir holding a .env file with the given raw bytes/str."""
    d = Path(tempfile.mkdtemp(prefix="tcenv"))
    if isinstance(text, bytes):
        (d / ".env").write_bytes(text)
    else:
        (d / ".env").write_text(text, encoding="utf-8")
    return d


class Base(unittest.TestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ, {
            "TELETHON_API_ID": "1",
            "TELETHON_API_HASH": "h",
            "TELETHON_CHECKLIST_CHATS": str(CHAT),
        })
        self._env.start()
        self.addCleanup(self._env.stop)
        FakeClient.created = 0
        FakeClient.script = {}
        CALLS.clear()

    def requests(self, kind=None):
        reqs = [c[1] for c in CALLS if c[0] == "request"]
        return [r for r in reqs if kind is None or isinstance(r, kind)]


# --------------------------------------------------------------- CLI contract

class TestProfileSettings(Base):
    def test_offline_preview_never_reads_secret_file(self):
        with mock.patch.dict(os.environ):
            del os.environ["TELETHON_CHECKLIST_CHATS"]
            with mock.patch.object(tc, "_load_env_file", side_effect=AssertionError("secret read")):
                code, out = run_json("create", "--title", "Tasks", "--task", "A", "--dry-run")
        self.assertEqual(code, 0)
        self.assertTrue(out["dry_run"])
        self.assertEqual(FakeClient.created, 0)


    def test_session_setting_is_relative_to_selected_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            (home / "config.yaml").write_text(
                "skills:\n  config:\n    telegram_checklist:\n"
                "      session: telethon/selected.session\n", encoding="utf-8")
            with mock.patch.object(tc, "HERMES", home):
                self.assertEqual(tc._session_path(), home / "telethon/selected.session")

    def test_bare_config_sections_read_as_absent(self):
        """A bare ``skills:`` / ``config:`` / ``telegram_checklist:`` header (the usual
        way to comment a block out) parses to None, not {}; the offline commands must
        treat it exactly like a missing section instead of refusing the whole file."""
        for text in ("skills:\n",
                     "skills:\n  config:\n",
                     "skills:\n  config:\n    telegram_checklist:\n",
                     "skills:\n  # config:\n  #   telegram_checklist:\n  #     chats: '-100123'\n"):
            with self.subTest(config=text), tempfile.TemporaryDirectory() as directory:
                home = Path(directory)
                (home / "config.yaml").write_text(text, encoding="utf-8")
                with mock.patch.object(tc, "HERMES", home):
                    self.assertEqual(tc._setting("chats", "absent"), "absent")
                    self.assertEqual(tc._session_path(), tc.SESSION)
                    code, out = run_json("create", "--title", "Tasks", "--task", "A", "--dry-run")
                    self.assertEqual(code, 0, out)
                    self.assertTrue(out["dry_run"])
                    code, out = run_json("plan", "--file", write_plan(plan_dict()))
                    self.assertEqual(code, 0, out)
                    self.assertEqual(out["kind"], "plan")

    def test_non_mapping_config_sections_fail_closed(self):
        for text in ("- skills\n",
                     "skills: 1\n",
                     "skills:\n  config: [1]\n",
                     "skills:\n  config:\n    telegram_checklist: chats\n"):
            with self.subTest(config=text), tempfile.TemporaryDirectory() as directory:
                home = Path(directory)
                (home / "config.yaml").write_text(text, encoding="utf-8")
                with mock.patch.object(tc, "HERMES", home):
                    code, out = dies_with(tc._setting, "chats")
                self.assertEqual(code, 1)
                self.assertFalse(out["ok"])
                self.assertIn("config.yaml", out["error"])


class TestCLIContract(Base):
    def test_help_exits_zero_with_usage(self):
        code, text = run_cli("--help")
        self.assertEqual(code, 0)
        self.assertIn("create", text)

    def test_no_subcommand_is_json_exit_1(self):
        code, out = run_json()
        self.assertEqual(code, 1)
        self.assertFalse(out["ok"])
        self.assertIn("bad arguments", out["error"])

    def test_unknown_subcommand_is_json_exit_1(self):
        code, out = run_json("nonsense")
        self.assertEqual(code, 1)
        self.assertIn("bad arguments", out["error"])

    def test_missing_required_option_is_json(self):
        code, out = run_json("get")
        self.assertEqual(code, 1)
        self.assertIn("message-id", out["error"])

    def test_bad_int_value_is_json(self):
        code, out = run_json("toggle", "--message-id", "5", "--done", EMOJI)
        self.assertEqual(code, 1)
        self.assertFalse(out["ok"])
        self.assertIn(EMOJI, out["error"])


# --------------------------------------------------------------- allowlist

class TestAllowlistParse(Base):
    def test_default_is_me_only(self):
        allowed, warns = tc._parse_allowlist("")
        self.assertEqual(allowed, {"me": None})
        self.assertEqual(warns, [])

    def test_chat_and_topic_entries(self):
        allowed, warns = tc._parse_allowlist("-100123, -100555:33, -100555:41")
        self.assertIsNone(allowed[-100123])
        self.assertEqual(allowed[-100555], {33, 41})
        self.assertEqual(warns, [])

    def test_garbage_warned_not_silent(self):
        allowed, warns = tc._parse_allowlist("junk,-100777:xx,-100888")
        self.assertIsNone(allowed[-100888])
        self.assertNotIn(-100777, allowed)
        self.assertEqual(len(warns), 2)

    def test_multi_dash_garbage_warned_not_crash(self):
        allowed, warns = tc._parse_allowlist("--100,---5,-1005555555555")
        self.assertEqual(set(allowed), {"me", -1005555555555})
        self.assertEqual(len(warns), 2)

    def test_full_grant_wins_over_topic_entry(self):
        allowed, _ = tc._parse_allowlist("-100555:33,-100555")
        self.assertIsNone(allowed[-100555])
        allowed, _ = tc._parse_allowlist("-100555,-100555:33")
        self.assertIsNone(allowed[-100555])

    def test_positive_or_zero_chat_rejected_without_echo(self):
        allowed, warns = tc._parse_allowlist("123,0,-100555")
        self.assertEqual(set(allowed), {"me", -100555})
        self.assertEqual(len(warns), 2)
        self.assertFalse(any("123" in w for w in warns))

    def test_general_topic_entry_rejected(self):
        allowed, warns = tc._parse_allowlist("-100555:1,-100555:0")
        self.assertNotIn(-100555, allowed)
        self.assertEqual(len(warns), 2)

    def test_bad_entry_value_never_echoed(self):
        _allowed, warns = tc._parse_allowlist("-100555:12345:BOT_SECRET")
        self.assertEqual(len(warns), 1)
        self.assertNotIn("BOT_SECRET", warns[0])

    def test_unicode_digit_topic_is_warning_not_crash(self):
        # "²".isdigit() is True but int("²") raises - must stay a warning
        allowed, warns = tc._parse_allowlist("-100555:²")
        self.assertNotIn(-100555, allowed)
        self.assertEqual(len(warns), 1)
        self.assertNotIn("²", warns[0])

    def test_huge_digit_topic_is_warning_not_crash(self):
        # CPython 3.11+ refuses int() on very long digit strings - stay a warning
        allowed, warns = tc._parse_allowlist("-100555:" + "9" * 5000)
        self.assertNotIn(-100555, allowed)
        self.assertEqual(len(warns), 1)


class TestEnvAndCreds(Base):
    def test_env_file_quotes_and_bom(self):
        d = env_dir(b'\xef\xbb\xbfTELETHON_API_ID="777"\n# comment\nX=\'y\'\n')
        vals = tc._load_env_file(d / ".env")
        self.assertEqual(vals, {"TELETHON_API_ID": "777", "X": "y"})

    def test_env_file_non_utf8_survives(self):
        d = env_dir(b"# comment \xf2\xe5\xf1\xf2\nX=1\n")  # cp1251 bytes, invalid UTF-8
        self.assertEqual(tc._load_env_file(d / ".env")["X"], "1")

    def test_creds_from_file(self):
        d = env_dir("TELETHON_API_ID=12345\nTELETHON_API_HASH=abc\n")
        with mock.patch.object(tc, "HERMES", d), \
                mock.patch.dict(os.environ):
            del os.environ["TELETHON_API_ID"], os.environ["TELETHON_API_HASH"]
            self.assertEqual(tc._creds(), (12345, "abc"))

    def test_empty_exported_cred_falls_back_to_file(self):
        d = env_dir("TELETHON_API_ID=12345\nTELETHON_API_HASH=abc\n")
        with mock.patch.object(tc, "HERMES", d), \
                mock.patch.dict(os.environ, {"TELETHON_API_ID": "",
                                             "TELETHON_API_HASH": ""}):
            self.assertEqual(tc._creds(), (12345, "abc"))

    def test_empty_chats_env_narrows_allowlist(self):
        d = env_dir("TELETHON_CHECKLIST_CHATS=-100777\n")
        with mock.patch.object(tc, "HERMES", d), \
                mock.patch.dict(os.environ, {"TELETHON_CHECKLIST_CHATS": ""}):
            allowed, _ = tc._allowed()
            self.assertEqual(allowed, {"me": None})

    def test_chats_file_ignored_and_process_env_compatible(self):
        d = env_dir("TELETHON_CHECKLIST_CHATS=-100777\n")
        with mock.patch.object(tc, "HERMES", d), mock.patch.dict(os.environ):
            del os.environ["TELETHON_CHECKLIST_CHATS"]
            allowed, _ = tc._allowed()
            self.assertEqual(allowed, {"me": None})
            os.environ["TELETHON_CHECKLIST_CHATS"] = "-100888"
            allowed, _ = tc._allowed()
            self.assertIn(-100888, allowed)
            self.assertNotIn(-100777, allowed)

    def test_bad_api_id_not_echoed(self):
        with mock.patch.dict(os.environ, {"TELETHON_API_ID": "hash-pasted-by-mistake"}):
            code, out = dies_with(tc._creds)
        self.assertEqual(code, 1)
        self.assertNotIn("hash-pasted-by-mistake", out["error"])

    def test_missing_creds_message(self):
        d = env_dir("")
        with mock.patch.object(tc, "HERMES", d), mock.patch.dict(os.environ):
            del os.environ["TELETHON_API_ID"], os.environ["TELETHON_API_HASH"]
            code, out = dies_with(tc._creds)
        self.assertEqual(code, 1)
        self.assertIn("TELETHON_API_ID", out["error"])

    def test_unicode_digit_api_id_not_echoed(self):
        # "²" passes str.isdigit() but fails int(); the value must never be echoed
        with mock.patch.dict(os.environ, {"TELETHON_API_ID": "²"}):
            code, out = dies_with(tc._creds)
        self.assertEqual(code, 1)
        self.assertEqual(out["error"], "TELETHON_API_ID must be numeric")

    def test_nonpositive_api_id_rejected(self):
        for bad in ("-5", "0"):
            with mock.patch.dict(os.environ, {"TELETHON_API_ID": bad}):
                code, out = dies_with(tc._creds)
            self.assertEqual(code, 1)
            self.assertEqual(out["error"], "TELETHON_API_ID must be numeric")


class TestNormChatAndLimits(Base):
    def test_norm_aliases(self):
        for raw in ("me", " ME ", "saved", "self"):
            self.assertEqual(tc._norm_chat(raw), "me")
        self.assertEqual(tc._norm_chat("-100123"), -100123)

    def test_username_rejected_clean_json(self):
        code, out = dies_with(tc._norm_chat, "@user")
        self.assertEqual(code, 1)
        self.assertIn("numeric chat_id", out["error"])

    def test_tg_len_counts_utf16_units(self):
        self.assertEqual(tc._tg_len(EMOJI), 2)
        self.assertEqual(tc._tg_len("abc"), 3)

    def test_task_at_utf16_cap_passes(self):
        code, out = run_json("create", "--title", "T", "--task", EMOJI * 100,
                             "--dry-run")
        self.assertEqual(code, 0)
        self.assertEqual(FakeClient.created, 0)

    def test_task_over_utf16_cap_refused(self):
        code, out = run_json("create", "--title", "T", "--task", EMOJI * 101,
                             "--dry-run")
        self.assertEqual(code, 1)
        self.assertIn("202 utf-16 units", out["error"])

    def test_title_over_cap_refused(self):
        code, out = run_json("create", "--title", "x" * 256, "--task", "a",
                             "--dry-run")
        self.assertEqual(code, 1)
        self.assertIn("title too long", out["error"])

    def test_bool_and_float_chat_rejected(self):
        for bad in (True, 3.5):
            code, out = dies_with(tc._norm_chat, bad)
            self.assertEqual(code, 1)
            self.assertIn("numeric chat_id", out["error"])

    def test_lone_surrogate_dies_cleanly(self):
        code, out = dies_with(tc._tg_len, "x\ud800y")
        self.assertEqual(code, 1)
        self.assertIn("surrogates", out["error"])


class TestBuildLink(Base):
    def test_supergroup_with_topic(self):
        self.assertEqual(tc.build_link(CHAT, TOPIC, 123), LINK)

    def test_supergroup_without_topic(self):
        self.assertEqual(tc.build_link(CHAT, None, 9),
                         "https://t.me/c/1234567890/9")


# --------------------------------------------------------------- plan contract

class TestPlanCommand(Base):
    def test_valid_plan_prints_source_map_offline(self):
        code, out = run_json("plan", "--file", write_plan(plan_dict()))
        self.assertEqual(code, 0)
        self.assertTrue(out["ok"])
        self.assertTrue(out["dry_run"])
        self.assertEqual(out["would_send"]["chat"], CHAT)
        self.assertEqual(out["would_send"]["thread"], TOPIC)
        self.assertEqual(out["would_send"]["tasks"], [f"Do X: {LINK}"])
        self.assertEqual(len(out["source_map"]), 1)
        self.assertEqual(out["source_map"][0]["message_id"], 123)
        self.assertEqual(FakeClient.created, 0)  # the key guarantee: offline

    def test_target_chat_required(self):
        code, out = run_json("plan", "--file", write_plan(plan_dict(target={})))
        self.assertEqual(code, 1)
        self.assertIn("target.chat", out["error"])

    def test_chat_not_allowed(self):
        code, out = run_json(
            "plan", "--file",
            write_plan(plan_dict(target={"chat": -100999, "thread": 1})))
        self.assertEqual(code, 1)
        self.assertIn("not allowed", out["error"])

    def test_saved_forbids_thread(self):
        code, out = run_json(
            "plan", "--file",
            write_plan(plan_dict(target={"chat": "me", "thread": 3})))
        self.assertEqual(code, 1)
        self.assertIn("Saved", out["error"])

    def test_thread_must_be_integer(self):
        code, out = run_json(
            "plan", "--file",
            write_plan(plan_dict(target={"chat": CHAT, "thread": "x"})))
        self.assertEqual(code, 1)
        self.assertIn("integer", out["error"])

    def test_thread_must_be_positive(self):
        code, out = run_json(
            "plan", "--file",
            write_plan(plan_dict(target={"chat": CHAT, "thread": 0})))
        self.assertEqual(code, 1)
        self.assertIn("positive", out["error"])

    def test_topic_restricted_entry_wrong_thread(self):
        os.environ["TELETHON_CHECKLIST_CHATS"] = "-100555:33"
        d = plan_dict(target={"chat": -100555, "thread": 41})
        code, out = run_json("plan", "--file", write_plan(d))
        self.assertEqual(code, 1)
        self.assertIn("allowed only for topics", out["error"])

    def test_topic_restricted_entry_missing_thread(self):
        os.environ["TELETHON_CHECKLIST_CHATS"] = "-100555:33"
        d = plan_dict(target={"chat": -100555})
        code, out = run_json("plan", "--file", write_plan(d))
        self.assertEqual(code, 1)
        self.assertIn("allowed only for topics", out["error"])

    def test_collected_requires_sources(self):
        d = plan_dict()
        d["tasks"][0]["sources"] = []
        code, out = run_json("plan", "--file", write_plan(d))
        self.assertEqual(code, 1)
        self.assertIn("sources", out["error"])

    def test_link_must_be_in_text(self):
        d = plan_dict()
        d["tasks"][0]["text"] = "Do X without a link"
        code, out = run_json("plan", "--file", write_plan(d))
        self.assertEqual(code, 1)
        self.assertIn("link", out["error"])

    def test_duplicate_texts_rejected(self):
        d = plan_dict()
        d["tasks"] = [d["tasks"][0], json.loads(json.dumps(d["tasks"][0]))]
        d["tasks"][1]["text"] = f"  DO x: {LINK}  "
        code, out = run_json("plan", "--file", write_plan(d))
        self.assertEqual(code, 1)
        self.assertIn("duplicate", out["error"])

    def test_too_many_tasks(self):
        d = plan_dict(collected_from_chat=False)
        d["tasks"] = [{"text": f"t{i}"} for i in range(31)]
        code, out = run_json("plan", "--file", write_plan(d))
        self.assertEqual(code, 1)
        self.assertIn("too many", out["error"])

    def test_unknown_media_warns_not_dies(self):
        d = plan_dict()
        d["tasks"][0]["sources"][0]["media"] = "sticker"
        code, out = run_json("plan", "--file", write_plan(d))
        self.assertEqual(code, 0)
        self.assertTrue(any("media" in w for w in out["warnings"]))

    def test_bad_json_file(self):
        d = tempfile.mkdtemp(prefix="tcplan")
        p = Path(d) / "plan.json"
        p.write_text("{broken", encoding="utf-8")
        code, out = run_json("plan", "--file", str(p))
        self.assertEqual(code, 1)
        self.assertIn("JSON", out["error"])

    def test_plan_must_be_an_object(self):
        code, out = run_json("plan", "--file", write_plan([1, 2]))
        self.assertEqual(code, 1)
        self.assertIn("JSON object", out["error"])

    def test_empty_title_rejected(self):
        code, out = run_json("plan", "--file", write_plan(plan_dict(title="  ")))
        self.assertEqual(code, 1)
        self.assertIn("title", out["error"])

    def test_long_task_rejected_utf16(self):
        d = plan_dict(collected_from_chat=False)
        d["tasks"] = [{"text": "x" * 201}]
        code, out = run_json("plan", "--file", write_plan(d))
        self.assertEqual(code, 1)
        self.assertIn("201 utf-16 units", out["error"])

    def test_task_must_be_object_and_sources_a_list(self):
        d = plan_dict(collected_from_chat=False)
        d["tasks"] = ["just a string"]
        code, out = run_json("plan", "--file", write_plan(d))
        self.assertEqual(code, 1)
        self.assertIn("object", out["error"])
        d = plan_dict()
        d["tasks"][0]["sources"] = "nope"
        code, out = run_json("plan", "--file", write_plan(d))
        self.assertEqual(code, 1)
        self.assertIn("list", out["error"])

    def test_bom_plan_file_accepted(self):
        d = tempfile.mkdtemp(prefix="tcplan")
        p = Path(d) / "plan.json"
        p.write_bytes(b"\xef\xbb\xbf" +
                      json.dumps(plan_dict(), ensure_ascii=False).encode("utf-8"))
        code, out = run_json("plan", "--file", str(p))
        self.assertEqual(code, 0)
        self.assertTrue(out["ok"])

    def test_non_string_link_not_counted(self):
        d = plan_dict()
        d["tasks"][0]["sources"][0]["link"] = 123
        code, out = run_json("plan", "--file", write_plan(d))
        self.assertEqual(code, 1)
        self.assertIn("at least one", out["error"])

    def test_whitespace_or_non_tme_link_not_evidence(self):
        d = plan_dict()
        d["tasks"][0]["sources"][0]["link"] = " "
        code, out = run_json("plan", "--file", write_plan(d))
        self.assertEqual(code, 1)
        self.assertIn("at least one", out["error"])
        d = plan_dict()
        d["tasks"][0]["text"] = "Do X: gopher://evil"
        d["tasks"][0]["sources"][0]["link"] = "gopher://evil"
        code, out = run_json("plan", "--file", write_plan(d))
        self.assertEqual(code, 1)
        self.assertIn("at least one", out["error"])

    def test_shared_and_collected_must_be_json_booleans(self):
        code, out = run_json("plan", "--file",
                             write_plan(plan_dict(shared="false")))
        self.assertEqual(code, 1)
        self.assertIn("boolean", out["error"])
        code, out = run_json("plan", "--file",
                             write_plan(plan_dict(collected_from_chat="true")))
        self.assertEqual(code, 1)
        self.assertIn("boolean", out["error"])

    def test_nan_constant_rejected(self):
        d = tempfile.mkdtemp(prefix="tcplan")
        p = Path(d) / "plan.json"
        p.write_text('{"target":{"chat":"me"},"title":"T",'
                     '"collected_from_chat":false,'
                     '"tasks":[{"text":"a"}],"x":NaN}', encoding="utf-8")
        code, out = run_json("plan", "--file", str(p))
        self.assertEqual(code, 1)
        self.assertIn("NaN", out["error"])

    def test_numeric_overflow_rejected(self):
        # 1e999 overflows to inf WITHOUT hitting parse_constant
        d = tempfile.mkdtemp(prefix="tcplan")
        p = Path(d) / "plan.json"
        p.write_text('{"target":{"chat":"me"},"title":"T",'
                     '"collected_from_chat":false,'
                     '"tasks":[{"text":"a"}],"x":1e999}', encoding="utf-8")
        code, out = run_json("plan", "--file", str(p))
        self.assertEqual(code, 1)
        self.assertIn("non-finite", out["error"])

    def test_link_prefix_of_longer_url_is_not_evidence(self):
        d = plan_dict()
        d["tasks"][0]["text"] = f"Do X: {LINK}0"  # only .../1230 in the text
        code, out = run_json("plan", "--file", write_plan(d))
        self.assertEqual(code, 1)
        self.assertIn("present in the task text", out["error"])

    def test_link_followed_by_punctuation_is_evidence(self):
        d = plan_dict()
        d["tasks"][0]["text"] = f"Do X ({LINK})."
        code, out = run_json("plan", "--file", write_plan(d))
        self.assertEqual(code, 0)
        self.assertTrue(out["ok"])

    def test_link_extended_by_url_chars_is_not_evidence(self):
        for tail in (".evil", "%30", "?x=1"):
            d = plan_dict()
            d["tasks"][0]["text"] = f"Do X: {LINK}{tail}"
            code, out = run_json("plan", "--file", write_plan(d))
            self.assertEqual(code, 1, tail)
            self.assertIn("present in the task text", out["error"])

    def test_link_nested_in_foreign_url_is_not_evidence(self):
        # the source link appears only inside another site's query string
        d = plan_dict()
        d["tasks"][0]["text"] = f"Open https://evil.invalid/?u={LINK}"
        code, out = run_json("plan", "--file", write_plan(d))
        self.assertEqual(code, 1)
        self.assertIn("present in the task text", out["error"])

    def test_link_nested_in_http_or_mixed_case_url_is_not_evidence(self):
        # outer URLs with http:// or a cased scheme must also be consumed whole
        for outer in ("http://evil.invalid/?u=",
                      "HTTPS://evil.invalid/?u=",
                      "HtTp://evil.invalid/?u="):
            d = plan_dict()
            d["tasks"][0]["text"] = f"Open {outer}{LINK}"
            code, out = run_json("plan", "--file", write_plan(d))
            self.assertEqual(code, 1, outer)
            self.assertIn("present in the task text", out["error"])

    def test_link_after_length_changing_unicode_is_recognized(self):
        # Turkish U+0130 lowercases to TWO chars; anchor indices must stay
        # native to the original text or a verbatim link stops matching
        d = plan_dict()
        d["tasks"][0]["text"] = f"İstanbul chat says do X: {LINK}"
        code, out = run_json("plan", "--file", write_plan(d))
        self.assertEqual(code, 0)
        self.assertTrue(out["ok"])

    def test_pathological_unicode_prefix_terminates(self):
        # regression guard for the lower()-copy hang: must return, not spin
        self.assertFalse(tc._link_in_text(LINK, "İ" * 8 + "http://x"))
        self.assertFalse(tc._link_in_text(LINK, "İ" * 40 + "http://x"))
        self.assertTrue(tc._link_in_text(LINK, "İ" * 8 + " " + LINK))

    def test_iri_or_apostrophe_continuation_is_not_evidence(self):
        # a Cyrillic letter or 'word glued right after the link extends the
        # linkified IRI - the prefix must not count as a complete match
        for tail in ("\u044f", "'evil", "\u0451" + "123"):
            d = plan_dict()
            d["tasks"][0]["text"] = f"Do X: {LINK}{tail}"
            code, out = run_json("plan", "--file", write_plan(d))
            self.assertEqual(code, 1, repr(tail))
            self.assertIn("present in the task text", out["error"])

    def test_link_in_closing_apostrophe_quotes_is_evidence(self):
        d = plan_dict()
        d["tasks"][0]["text"] = f"see '{LINK}'"
        code, out = run_json("plan", "--file", write_plan(d))
        self.assertEqual(code, 0)
        self.assertTrue(out["ok"])

    def test_link_buried_in_foreign_iri_is_not_evidence(self):
        # a t.me link inside a foreign IRI's path/query is not standalone; the
        # whole outer IRI must be consumed, not re-scanned from inside it
        ya = "\u044f"  # Cyrillic small ya, an IRI-legal path char
        for host in (f"https://evil.invalid/{ya}?u=",
                     "https://evil.invalid/'evil?u="):
            self.assertFalse(tc._link_in_text(LINK, host + LINK), host)
            d = plan_dict()
            d["tasks"][0]["text"] = f"Do X: {host}{LINK}"
            code, out = run_json("plan", "--file", write_plan(d))
            self.assertEqual(code, 1, host)
            self.assertIn("present in the task text", out["error"])
        # a genuine standalone link after a real separator still counts
        self.assertTrue(
            tc._link_in_text(LINK, f"https://evil.invalid/{ya} {LINK}"))

    def test_link_nested_in_ipv6_or_apostrophe_iri_is_not_evidence(self):
        # IPv6-literal hosts use [ ], and a doubled apostrophe is IRI-internal;
        # both must be consumed as part of the outer token, not re-scanned
        ya = "\u044f"
        for host in ("https://[2001:db8::1]/?u=",
                     f"https://evil.invalid/{ya}''?u="):
            self.assertFalse(tc._link_in_text(LINK, host + LINK), host)
            d = plan_dict()
            d["tasks"][0]["text"] = f"Do X: {host}{LINK}"
            code, out = run_json("plan", "--file", write_plan(d))
            self.assertEqual(code, 1, host)
            self.assertIn("present in the task text", out["error"])

    def test_bracket_and_apostrophe_quoting_still_evidence(self):
        # [LINK] and 'LINK' quoting styles must keep matching (closing bracket
        # and closing apostrophe are trailing punctuation, not continuations)
        for wrapped in (f"[{LINK}]", f"'{LINK}'", f"{LINK}''"):
            self.assertTrue(tc._link_in_text(LINK, wrapped), wrapped)

    def test_unicode_equivalent_duplicate_rejected(self):
        # U+03D2 GREEK UPSILON WITH HOOK: NFKC gives capital upsilon, so only
        # normalize-then-casefold-then-normalize catches this pair
        d = plan_dict(collected_from_chat=False)
        d["tasks"] = [{"text": "buy ϒ cable"}, {"text": "buy υ cable"}]
        code, out = run_json("plan", "--file", write_plan(d))
        self.assertEqual(code, 1)
        self.assertIn("duplicate", out["error"])

    def test_general_topic_thread_rejected(self):
        code, out = run_json(
            "plan", "--file",
            write_plan(plan_dict(target={"chat": CHAT, "thread": 1})))
        self.assertEqual(code, 1)
        self.assertIn("General", out["error"])
        code, out = run_json("create", "--title", "T", "--task", "a",
                             "--chat", str(CHAT), "--thread", "1", "--dry-run")
        self.assertEqual(code, 1)
        self.assertIn("General", out["error"])

    def test_thread_bool_and_float_rejected(self):
        for bad in (True, 33.5):
            code, out = run_json(
                "plan", "--file",
                write_plan(plan_dict(target={"chat": CHAT, "thread": bad})))
            self.assertEqual(code, 1)
            self.assertIn("integer", out["error"])

    def test_non_string_title_and_text_rejected(self):
        code, out = run_json("plan", "--file",
                             write_plan(plan_dict(title={"a": 1})))
        self.assertEqual(code, 1)
        self.assertIn("string", out["error"])
        d = plan_dict(collected_from_chat=False)
        d["tasks"] = [{"text": 5}]
        code, out = run_json("plan", "--file", write_plan(d))
        self.assertEqual(code, 1)
        self.assertIn("string", out["error"])

    def test_invisible_duplicate_rejected(self):
        d = plan_dict()
        d["tasks"] = [d["tasks"][0],
                      {"text": "Do\u200b X: " + LINK,
                       "sources": d["tasks"][0]["sources"]}]
        code, out = run_json("plan", "--file", write_plan(d))
        self.assertEqual(code, 1)
        self.assertIn("duplicate", out["error"])

    def test_lone_surrogate_in_plan_rejected_cleanly(self):
        d = tempfile.mkdtemp(prefix="tcplan")
        p = Path(d) / "plan.json"
        p.write_text('{"target":{"chat":"me"},"title":"x\\ud800y",'
                     '"collected_from_chat":false,"tasks":[{"text":"a"}]}',
                     encoding="utf-8")
        code, out = run_json("plan", "--file", str(p))
        self.assertEqual(code, 1)
        self.assertIn("surrogates", out["error"])

    def test_mixed_source_message_ids_sort_safely(self):
        d = plan_dict()
        d["tasks"][0]["sources"].append(
            {"link": LINK, "topic": f"Ideas ({TOPIC})",
             "message_id": "abc", "media": "text"})
        d["tasks"][0]["sources"].append(
            {"link": LINK, "topic": f"Ideas ({TOPIC})",
             "message_id": 9, "media": "text"})
        code, out = run_json("plan", "--file", write_plan(d))
        self.assertEqual(code, 0)
        # numeric ids sort numerically and come before non-numeric ones
        self.assertEqual([s["message_id"] for s in out["source_map"]],
                         [9, 123, "abc"])


class TestCreateDryRun(Base):
    def test_dry_run_args_path_offline(self):
        code, out = run_json("create", "--title", "T", "--task", "a", "--dry-run")
        self.assertEqual(code, 0)
        self.assertTrue(out["dry_run"])
        self.assertEqual(out["would_send"]["chat"], "me")
        self.assertEqual(out["would_send"]["tasks"], ["a"])
        self.assertEqual(FakeClient.created, 0)  # offline guarantee

    def test_dry_run_from_plan_offline(self):
        code, out = run_json("create", "--from-plan", write_plan(plan_dict()),
                             "--dry-run")
        self.assertEqual(code, 0)
        self.assertEqual(len(out["source_map"]), 1)
        self.assertEqual(FakeClient.created, 0)

    def test_me_with_thread_refused_offline(self):
        code, out = run_json("create", "--title", "T", "--task", "a",
                             "--thread", "5", "--dry-run")
        self.assertEqual(code, 1)
        self.assertIn("Saved", out["error"])
        self.assertEqual(FakeClient.created, 0)

    def test_explicit_chat_me_direct_path_ok(self):
        code, out = run_json("create", "--title", "T", "--task", "a",
                             "--chat", "me", "--dry-run")
        self.assertEqual(code, 0)
        self.assertEqual(out["would_send"]["chat"], "me")

    def test_dry_run_preserves_shared_flags_without_network(self):
        code, out = run_json(
            "create", "--title", "Team", "--task", "a",
            "--chat", str(CHAT), "--thread", str(TOPIC),
            "--others-append", "--others-complete", "--dry-run")
        self.assertEqual(code, 0)
        self.assertTrue(out["ok"])
        self.assertTrue(out["dry_run"])
        self.assertEqual(out["would_send"]["chat"], CHAT)
        self.assertEqual(out["would_send"]["thread"], TOPIC)
        self.assertTrue(out["would_send"]["others_can_append"])
        self.assertTrue(out["would_send"]["others_can_complete"])
        self.assertEqual(FakeClient.created, 0)
        self.assertEqual(self.requests(), [])


class TestCreateFromPlan(Base):
    def test_sends_exactly_plan_and_verifies_clean(self):
        FakeClient.script["request"] = _Simple(
            updates=[_Simple(message=_Simple(id=777))])
        FakeClient.script["get_messages"] = checklist_msg(
            items=(f"Do X: {LINK}",), topic=TOPIC, title="Weekly tasks")
        code, out = run_json("create", "--from-plan", write_plan(plan_dict()))
        self.assertEqual(code, 0)
        self.assertTrue(out["ok"])
        self.assertEqual(out["message_id"], 777)
        self.assertEqual(out["verified"]["total"], 1)
        self.assertEqual(out["warnings"], [])  # exact match -> no warnings
        req = self.requests(SendMediaRequest)[0]
        self.assertEqual(req.message, "")
        self.assertEqual(req.reply_to.top_msg_id, TOPIC)
        self.assertEqual([it.title.text for it in req.media.todo.list],
                         [f"Do X: {LINK}"])

    def test_shared_plan_sets_flags(self):
        FakeClient.script["request"] = _Simple(
            updates=[_Simple(message=_Simple(id=5))])
        FakeClient.script["get_messages"] = checklist_msg(
            items=(f"Do X: {LINK}",), topic=TOPIC, title="Weekly tasks",
            appendable=True, completable=True)
        code, out = run_json("create", "--from-plan",
                             write_plan(plan_dict(shared=True)))
        self.assertEqual(code, 0)
        req = self.requests(SendMediaRequest)[0]
        self.assertTrue(req.media.todo.others_can_append)
        self.assertTrue(req.media.todo.others_can_complete)
        self.assertTrue(out["verified"]["others_can_append"])
        self.assertTrue(out["verified"]["others_can_complete"])

    def test_post_check_failure_is_warning_not_false_success(self):
        FakeClient.script["request"] = _Simple(
            updates=[_Simple(message=_Simple(id=5))])
        FakeClient.script["get_messages"] = None  # re-read fails
        code, out = run_json("create", "--from-plan", write_plan(plan_dict()))
        self.assertEqual(code, 0)
        self.assertTrue(out["ok"])
        self.assertIs(out["verified"], False)
        self.assertTrue(any("re-read failed" in w for w in out["warnings"]))

    def test_post_check_exception_is_swallowed(self):
        FakeClient.script["request"] = _Simple(
            updates=[_Simple(message=_Simple(id=5))])

        def boom(ids):
            raise RuntimeError("network blip")
        FakeClient.script["get_messages"] = boom
        code, out = run_json("create", "--from-plan", write_plan(plan_dict()))
        self.assertEqual(code, 0)
        self.assertTrue(out["ok"])
        self.assertIs(out["verified"], False)

    def test_post_check_mismatch_warns(self):
        FakeClient.script["request"] = _Simple(
            updates=[_Simple(message=_Simple(id=77))])
        # read back 3 tasks titled "Demo" while the plan sent 1 titled "Weekly tasks"
        FakeClient.script["get_messages"] = checklist_msg(items=("x", "y", "z"))
        code, out = run_json("create", "--from-plan", write_plan(plan_dict()))
        self.assertEqual(code, 0)
        self.assertTrue(out["ok"])
        self.assertTrue(any("post-check mismatch" in w for w in out["warnings"]))
        self.assertEqual(out["verified"]["total"], 3)

    def test_no_message_id_ok_true_with_warning(self):
        FakeClient.script["request"] = _Simple(updates=[])
        code, out = run_json("create", "--from-plan", write_plan(plan_dict()))
        self.assertEqual(code, 0)
        self.assertIsNone(out["message_id"])
        self.assertIs(out["verified"], False)
        self.assertTrue(any("message_id" in w for w in out["warnings"]))

    def test_top_level_update_id_fallback(self):
        FakeClient.script["request"] = _Simple(id=42)  # UpdateShortSentMessage shape
        FakeClient.script["get_messages"] = checklist_msg(
            items=(f"Do X: {LINK}",), topic=TOPIC, title="Weekly tasks")
        code, out = run_json("create", "--from-plan", write_plan(plan_dict()))
        self.assertEqual(code, 0)
        self.assertEqual(out["message_id"], 42)

    def test_floodwait_honest_failure(self):
        FakeClient.script["request"] = FloodWaitError(42)
        code, out = run_json("create", "--from-plan", write_plan(plan_dict()))
        self.assertEqual(code, 1)
        self.assertFalse(out["ok"])
        self.assertIn("FloodWait 42", out["error"])

    def test_rpc_error_honest_failure(self):
        FakeClient.script["request"] = RPCError("CHAT_WRITE_FORBIDDEN")
        code, out = run_json("create", "--from-plan", write_plan(plan_dict()))
        self.assertEqual(code, 1)
        self.assertIn("Telegram error", out["error"])
        self.assertIn("CHAT_WRITE_FORBIDDEN", out["error"])

    def test_from_plan_exclusive_with_flags(self):
        code, out = run_json("create", "--from-plan", write_plan(plan_dict()),
                             "--title", "T")
        self.assertEqual(code, 1)
        self.assertIn("exclusive", out["error"])

    def test_from_plan_exclusive_even_with_falsy_values(self):
        code, out = run_json("create", "--from-plan", write_plan(plan_dict()),
                             "--thread", "0")
        self.assertEqual(code, 1)
        self.assertIn("exclusive", out["error"])

    def test_from_plan_rejects_explicit_chat_me(self):
        code, out = run_json("create", "--from-plan", write_plan(plan_dict()),
                             "--chat", "me")
        self.assertEqual(code, 1)
        self.assertIn("exclusive", out["error"])

    def test_create_requires_input(self):
        code, out = run_json("create")
        self.assertEqual(code, 1)
        self.assertIn("--from-plan or --title", out["error"])


class TestCreateArgsPath(Base):
    def test_direct_path_clean_verified(self):
        FakeClient.script["request"] = _Simple(
            updates=[_Simple(message=_Simple(id=9))])
        FakeClient.script["get_messages"] = checklist_msg(items=("a", "b"))
        code, out = run_json("create", "--title", "Demo",
                             "--task", "a", "--task", "b")
        self.assertEqual(code, 0)
        self.assertEqual(out["message_id"], 9)
        self.assertEqual(out["tasks_created"], 2)
        self.assertEqual(out["verified"]["total"], 2)
        self.assertEqual(out["warnings"], [])

    def test_denied_chat_fails_before_network(self):
        code, out = run_json("create", "--title", "T", "--task", "a",
                             "--chat", "-100999")
        self.assertEqual(code, 1)
        self.assertIn("not allowed", out["error"])
        self.assertEqual(FakeClient.created, 0)

    def test_too_many_tasks_refused_before_network(self):
        args = ["create", "--title", "T"]
        for i in range(31):
            args += ["--task", f"t{i}"]
        code, out = run_json(*args)
        self.assertEqual(code, 1)
        self.assertIn("too many", out["error"])
        self.assertEqual(FakeClient.created, 0)

    def test_nonpositive_thread_refused_offline(self):
        code, out = run_json("create", "--title", "T", "--task", "a",
                             "--chat", str(CHAT), "--thread", "0", "--dry-run")
        self.assertEqual(code, 1)
        self.assertIn("positive", out["error"])
        self.assertEqual(FakeClient.created, 0)

    def test_unauthorized_session_honest_error(self):
        async def _no(self):
            return False
        with mock.patch.object(FakeClient, "is_user_authorized", _no):
            code, out = run_json("create", "--title", "T", "--task", "a")
        self.assertEqual(code, 1)
        self.assertIn("not authorized", out["error"])

    def test_connect_failure_keeps_json_contract(self):
        async def _boom(self):
            raise ConnectionError("boom")
        with mock.patch.object(FakeClient, "connect", _boom):
            code, out = run_json("create", "--title", "T", "--task", "a")
        self.assertEqual(code, 1)
        self.assertIn("ConnectionError", out["error"])


# --------------------------------------------------------------- read commands

class TestGet(Base):
    def test_denied_outside_allowlist(self):
        code, out = run_json("get", "--chat", "-100999", "--message-id", "5")
        self.assertEqual(code, 1)
        self.assertIn("not allowed", out["error"])
        self.assertNotIn(("get_messages", 5), CALLS)
        self.assertEqual(FakeClient.created, 0)  # refused before any client

    def test_reads_checklist_with_flags(self):
        FakeClient.script["get_messages"] = checklist_msg(
            items=("x", "y"), done=(1,), appendable=True)
        code, out = run_json("get", "--chat", "me", "--message-id", "7")
        self.assertEqual(code, 0)
        self.assertEqual(out["done"], 1)
        self.assertEqual(out["total"], 2)
        self.assertTrue(out["others_can_append"])
        self.assertFalse(out["others_can_complete"])

    def test_message_not_found_distinct_error(self):
        FakeClient.script["get_messages"] = None
        code, out = run_json("get", "--chat", "me", "--message-id", "7")
        self.assertEqual(code, 1)
        self.assertIn("not found", out["error"])

    def test_not_a_checklist_distinct_error(self):
        FakeClient.script["get_messages"] = _Simple(media=None)
        code, out = run_json("get", "--chat", "me", "--message-id", "7")
        self.assertEqual(code, 1)
        self.assertIn("not a checklist", out["error"])

    def test_topic_restriction_blocks_foreign_topic(self):
        # allowlist -100555:33 - a checklist living in topic 34 of that chat
        # must be refused on READ exactly as on write (gates-inventory class)
        os.environ["TELETHON_CHECKLIST_CHATS"] = "-100555:33"
        FakeClient.script["get_messages"] = checklist_msg(items=("secret",), topic=34)
        code, out = run_json("get", "--chat", "-100555", "--message-id", "7")
        self.assertEqual(code, 1)
        self.assertIn("outside the allowed topics", out["error"])
        self.assertNotIn("secret", json.dumps(out))
        self.assertNotIn("34", out["error"])  # the message's topic stays private

    def test_topic_restriction_blocks_general_topic(self):
        # a message outside any topic (General) is not topic 33 either
        os.environ["TELETHON_CHECKLIST_CHATS"] = "-100555:33"
        FakeClient.script["get_messages"] = checklist_msg(items=("secret",), topic=None)
        code, out = run_json("get", "--chat", "-100555", "--message-id", "7")
        self.assertEqual(code, 1)
        self.assertIn("outside the allowed topics", out["error"])
        self.assertNotIn("secret", json.dumps(out))

    def test_topic_restriction_allows_matching_topic(self):
        os.environ["TELETHON_CHECKLIST_CHATS"] = "-100555:33"
        FakeClient.script["get_messages"] = checklist_msg(items=("a", "b"), topic=33)
        code, out = run_json("get", "--chat", "-100555", "--message-id", "7")
        self.assertEqual(code, 0)
        self.assertEqual(out["total"], 2)


class TestAppend(Base):
    def test_new_ids_continue_after_max(self):
        msg = checklist_msg(items=("a", "b"))
        msg.media.todo.list[1].id = 3  # id gap: 1 and 3 exist
        FakeClient.script["get_messages"] = msg
        code, out = run_json("append", "--chat", "me", "--message-id", "7",
                             "--task", "d")
        self.assertEqual(code, 0)
        req = self.requests(AppendTodoListRequest)[-1]
        self.assertEqual([it.id for it in req.list], [4])
        self.assertEqual(out["verified"]["total"], 2)  # re-read of the fixture

    def test_over_cap_refused(self):
        FakeClient.script["get_messages"] = checklist_msg(
            items=tuple(f"t{i}" for i in range(30)))
        code, out = run_json("append", "--chat", "me", "--message-id", "7",
                             "--task", "x")
        self.assertEqual(code, 1)
        self.assertIn("exceed", out["error"])
        self.assertEqual(self.requests(), [])

    def test_too_long_task_refused_not_truncated(self):
        code, out = run_json("append", "--chat", "me", "--message-id", "7",
                             "--task", "x" * 201)
        self.assertEqual(code, 1)
        self.assertIn("201 utf-16 units", out["error"])
        self.assertEqual(self.requests(), [])  # refused before any request

    def test_topic_restriction_blocks_foreign_topic(self):
        os.environ["TELETHON_CHECKLIST_CHATS"] = "-100555:33"
        FakeClient.script["get_messages"] = checklist_msg(items=("a",), topic=41)
        code, out = run_json("append", "--chat", "-100555", "--message-id", "7",
                             "--task", "x")
        self.assertEqual(code, 1)
        self.assertIn("outside the allowed topics", out["error"])
        self.assertNotIn("41", out["error"])  # the message's topic stays private
        self.assertEqual(self.requests(AppendTodoListRequest), [])

    def test_topic_restriction_allows_matching_topic(self):
        os.environ["TELETHON_CHECKLIST_CHATS"] = "-100555:33"
        FakeClient.script["get_messages"] = checklist_msg(items=("a",), topic=33)
        code, out = run_json("append", "--chat", "-100555", "--message-id", "7",
                             "--task", "x")
        self.assertEqual(code, 0)
        self.assertTrue(out["ok"])

    def test_dry_run_reads_but_never_writes(self):
        FakeClient.script["get_messages"] = checklist_msg(items=("a", "b"))
        code, out = run_json("append", "--chat", "me", "--message-id", "7",
                             "--task", "c", "--dry-run")
        self.assertEqual(code, 0)
        self.assertTrue(out["dry_run"])
        self.assertEqual(out["would_append"], [{"id": 3, "text": "c"}])
        self.assertEqual(out["total_after"], 3)
        self.assertEqual(self.requests(), [])  # reads happened, no write request

    def test_verify_failure_after_write_is_warning(self):
        state = {"n": 0}

        def gm(ids):
            state["n"] += 1
            if state["n"] == 1:
                return checklist_msg(items=("a", "b"))
            raise RuntimeError("network blip")
        FakeClient.script["get_messages"] = gm
        code, out = run_json("append", "--chat", "me", "--message-id", "7",
                             "--task", "c")
        self.assertEqual(code, 0)
        self.assertTrue(out["ok"])
        self.assertIs(out["verified"], False)
        self.assertTrue(any("re-read failed" in w for w in out["warnings"]))

    def test_post_check_mismatch_warns(self):
        # the re-read still shows 2 tasks although 1 was appended
        FakeClient.script["get_messages"] = checklist_msg(items=("a", "b"))
        code, out = run_json("append", "--chat", "me", "--message-id", "7",
                             "--task", "c")
        self.assertEqual(code, 0)
        self.assertTrue(any("post-check mismatch" in w for w in out["warnings"]))

    def test_id_space_exhausted_refused(self):
        msg = checklist_msg(items=("a",))
        msg.media.todo.list[0].id = 2 ** 31 - 1
        FakeClient.script["get_messages"] = msg
        code, out = run_json("append", "--chat", "me", "--message-id", "7",
                             "--task", "x")
        self.assertEqual(code, 1)
        self.assertIn("exhausted", out["error"])
        self.assertEqual(self.requests(AppendTodoListRequest), [])


class TestToggle(Base):
    def test_unknown_id_rejected(self):
        FakeClient.script["get_messages"] = checklist_msg(items=("a", "b"))
        code, out = run_json("toggle", "--chat", "me", "--message-id", "7",
                             "--done", "9")
        self.assertEqual(code, 1)
        self.assertIn("unknown task ids", out["error"])

    def test_done_undone_overlap_rejected(self):
        code, out = run_json("toggle", "--chat", "me", "--message-id", "7",
                             "--done", "1", "--undone", "1")
        self.assertEqual(code, 1)
        self.assertIn("both --done and --undone", out["error"])
        self.assertEqual(self.requests(), [])

    def test_dry_run_never_writes(self):
        FakeClient.script["get_messages"] = checklist_msg(items=("a", "b"))
        code, out = run_json("toggle", "--chat", "me", "--message-id", "7",
                             "--done", "2", "--dry-run")
        self.assertEqual(code, 0)
        self.assertTrue(out["dry_run"])
        self.assertEqual(out["would_mark_done"], [2])
        self.assertEqual(self.requests(), [])

    def test_toggle_verified_targets_checked(self):
        FakeClient.script["get_messages"] = checklist_msg(items=("a", "b"),
                                                          done=(2,))
        code, out = run_json("toggle", "--chat", "me", "--message-id", "7",
                             "--done", "2")
        self.assertEqual(code, 0)
        req = self.requests(ToggleTodoCompletedRequest)[0]
        self.assertEqual(req.completed, [2])
        self.assertEqual(req.incompleted, [])
        self.assertEqual(out["verified"]["done"], 1)
        self.assertEqual(out["warnings"], [])  # requested id reads back as done

    def test_toggle_post_check_mismatch_warns(self):
        # the re-read shows the id still not done
        FakeClient.script["get_messages"] = checklist_msg(items=("a", "b"))
        code, out = run_json("toggle", "--chat", "me", "--message-id", "7",
                             "--done", "2")
        self.assertEqual(code, 0)
        self.assertTrue(any("did not change" in w for w in out["warnings"]))

    def test_missing_id_in_reread_counts_as_mismatch(self):
        state = {"n": 0}

        def gm(ids):
            state["n"] += 1
            if state["n"] == 1:
                return checklist_msg(items=("a", "b"), done=(2,))
            return checklist_msg(items=("a",))  # id 2 vanished from the re-read
        FakeClient.script["get_messages"] = gm
        code, out = run_json("toggle", "--chat", "me", "--message-id", "7",
                             "--undone", "2")
        self.assertEqual(code, 0)
        self.assertTrue(any("did not change" in w for w in out["warnings"]))

    def test_denied_chat_fails_before_network(self):
        code, out = run_json("toggle", "--chat", "-100999", "--message-id", "5",
                             "--done", "1")
        self.assertEqual(code, 1)
        self.assertIn("not allowed", out["error"])
        self.assertEqual(FakeClient.created, 0)

    def test_duplicate_done_ids_deduped(self):
        FakeClient.script["get_messages"] = checklist_msg(items=("a", "b"))
        code, out = run_json("toggle", "--chat", "me", "--message-id", "7",
                             "--done", "1", "--done", "1")
        self.assertEqual(code, 0)
        req = self.requests(ToggleTodoCompletedRequest)[0]
        self.assertEqual(req.completed, [1])

    def test_topic_restriction_applies_without_naming_the_topic(self):
        os.environ["TELETHON_CHECKLIST_CHATS"] = "-100555:33"
        FakeClient.script["get_messages"] = checklist_msg(items=("a",), topic=41)
        code, out = run_json("toggle", "--chat", "-100555", "--message-id", "7",
                             "--done", "1")
        self.assertEqual(code, 1)
        self.assertIn("outside the allowed topics", out["error"])
        self.assertNotIn("41", out["error"])
        self.assertEqual(self.requests(ToggleTodoCompletedRequest), [])


class TestListTopics(Base):
    def test_lists_topics_of_allowed_chat(self):
        FakeClient.script["request"] = _Simple(count=7, topics=[
            _Simple(id=41, title="Ideas", closed=False),
            _Simple(id=43, title="Monitoring", closed=True),
            _Simple(id=1),  # service entry without a title - filtered out
        ])
        code, out = run_json("list-topics", "--chat", str(CHAT))
        self.assertEqual(code, 0)
        self.assertEqual([t["id"] for t in out["topics"]], [41, 43])
        self.assertEqual(out["total"], 7)
        self.assertEqual(out["listed"], 2)
        req = self.requests(GetForumTopicsRequest)[0]
        self.assertEqual(req.limit, 100)
        self.assertTrue(hasattr(req, "peer"))

    def test_denied_chat_no_request(self):
        code, out = run_json("list-topics", "--chat", "-100999")
        self.assertEqual(code, 1)
        self.assertIn("not allowed", out["error"])
        self.assertEqual(self.requests(), [])
        self.assertEqual(FakeClient.created, 0)

    def test_saved_has_no_topics(self):
        code, out = run_json("list-topics", "--chat", "me")
        self.assertEqual(code, 1)
        self.assertIn("forum topics", out["error"])
        self.assertEqual(FakeClient.created, 0)

    def test_chat_argument_required(self):
        code, out = run_json("list-topics")
        self.assertEqual(code, 1)
        self.assertIn("chat", out["error"])

    def test_partial_listing_warns(self):
        FakeClient.script["request"] = _Simple(count=135, topics=[
            _Simple(id=i + 2, title=f"t{i}", closed=False) for i in range(100)])
        code, out = run_json("list-topics", "--chat", str(CHAT))
        self.assertEqual(code, 0)
        self.assertTrue(any("first 100 of 135" in w for w in out["warnings"]))

    def test_per_topic_allowlist_lists_only_allowed_topics_by_id(self):
        os.environ["TELETHON_CHECKLIST_CHATS"] = "-100555:41,-100555:33"

        def answer(req):
            self.assertIsInstance(req, GetForumTopicsByIDRequest)
            self.assertEqual(req.topics, [33, 41])
            return _Simple(count=2, topics=[
                _Simple(id=33, title="Ops", closed=False),
                _Simple(id=41, title="Ideas", closed=True),
            ])
        FakeClient.script["request"] = answer
        code, out = run_json("list-topics", "--chat", "-100555")
        self.assertEqual(code, 0)
        self.assertEqual([t["id"] for t in out["topics"]], [33, 41])
        self.assertEqual(out["listed"], 2)
        self.assertEqual(self.requests(GetForumTopicsRequest), [])  # never the full listing

    def test_per_topic_allowlist_missing_topic_is_a_warning(self):
        os.environ["TELETHON_CHECKLIST_CHATS"] = "-100555:33,-100555:41"
        FakeClient.script["request"] = _Simple(count=2, topics=[
            _Simple(id=33, title="Ops", closed=False),
            _Simple(id=41),  # ForumTopicDeleted: id only, no title
        ])
        code, out = run_json("list-topics", "--chat", "-100555")
        self.assertEqual(code, 0)
        self.assertEqual([t["id"] for t in out["topics"]], [33])
        self.assertTrue(any("41" in w and "not found" in w for w in out["warnings"]), out)
        self.assertFalse(any("first" in w for w in out["warnings"]), out)  # no bogus partial-listing warning

    def test_per_topic_allowlist_drops_unrequested_topics_from_the_answer(self):
        # fail-closed: whatever the server answers, only allowlisted ids are printed
        os.environ["TELETHON_CHECKLIST_CHATS"] = "-100555:33"
        FakeClient.script["request"] = _Simple(count=2, topics=[
            _Simple(id=33, title="Ops", closed=False),
            _Simple(id=99, title="Leak", closed=False),
        ])
        code, out = run_json("list-topics", "--chat", "-100555")
        self.assertEqual(code, 0)
        self.assertEqual([t["id"] for t in out["topics"]], [33])
        self.assertNotIn("Leak", json.dumps(out))


class TestTransportSeams(Base):
    def test_disconnect_failure_does_not_corrupt_output(self):
        async def _boom(self):
            raise RuntimeError("transport died during disconnect")
        FakeClient.script["get_messages"] = checklist_msg()
        with mock.patch.object(FakeClient, "disconnect", _boom):
            code, out = run_json("get", "--chat", "me", "--message-id", "7")
        self.assertEqual(code, 0)          # run_json would fail on a second JSON object
        self.assertTrue(out["ok"])

    def test_disconnect_failure_after_refusal_keeps_single_json(self):
        async def _boom(self):
            raise RuntimeError("transport died during disconnect")
        FakeClient.script["get_messages"] = _Simple(media=None)
        with mock.patch.object(FakeClient, "disconnect", _boom):
            code, out = run_json("get", "--chat", "me", "--message-id", "1")
        self.assertEqual(code, 1)
        self.assertIn("not a checklist", out["error"])

    def test_is_connected_failure_does_not_corrupt_output(self):
        def _boom(self):
            raise RuntimeError("state probe died")
        FakeClient.script["get_messages"] = checklist_msg()
        with mock.patch.object(FakeClient, "is_connected", _boom):
            code, out = run_json("get", "--chat", "me", "--message-id", "7")
        self.assertEqual(code, 0)
        self.assertTrue(out["ok"])

    def test_verify_snapshot_build_failure_is_swallowed(self):
        # the re-read succeeds but the snapshot build hits an unexpected shape
        state = {"n": 0}

        def gm(ids):
            state["n"] += 1
            msg = checklist_msg(items=("a",))
            if state["n"] > 1:
                msg.media.todo.title = None
            return msg
        FakeClient.script["get_messages"] = gm
        code, out = run_json("append", "--chat", "me", "--message-id", "7",
                             "--task", "b")
        self.assertEqual(code, 0)
        self.assertTrue(out["ok"])
        self.assertIs(out["verified"], False)

    def test_keyboard_interrupt_stays_json(self):
        async def _ki(self):
            raise KeyboardInterrupt()
        with mock.patch.object(FakeClient, "connect", _ki):
            code, out = run_json("create", "--title", "T", "--task", "a")
        self.assertEqual(code, 1)
        self.assertIn("interrupted", out["error"])

    def test_interrupt_during_verify_keeps_write_success(self):
        state = {"n": 0}

        def gm(ids):
            state["n"] += 1
            if state["n"] == 1:
                return checklist_msg(items=("a",))
            raise KeyboardInterrupt()  # interrupt inside the post-write re-read
        FakeClient.script["get_messages"] = gm
        code, out = run_json("append", "--chat", "me", "--message-id", "7",
                             "--task", "b")
        self.assertEqual(code, 0)
        self.assertTrue(out["ok"])
        self.assertIs(out["verified"], False)

    def test_interrupt_during_disconnect_stays_single_json(self):
        async def _ki(self):
            raise KeyboardInterrupt()
        FakeClient.script["get_messages"] = checklist_msg()
        with mock.patch.object(FakeClient, "disconnect", _ki):
            code, out = run_json("get", "--chat", "me", "--message-id", "7")
        self.assertEqual(code, 0)
        self.assertTrue(out["ok"])

    def test_lock_session_targets_suffixed_file(self):
        # telethon appends .session when the configured path lacks it
        d = Path(tempfile.mkdtemp())
        (d / "sess.session").write_text("", encoding="utf-8")
        with mock.patch.object(tc, "SESSION", d / "sess"):
            tc._lock_session()  # must neither raise nor create d/"sess"
        self.assertFalse((d / "sess").exists())
        self.assertTrue((d / "sess.session").exists())

    def test_lock_session_dotsession_file_used_as_is(self):
        # a file literally named ".session": str endswith wins over pathlib .suffix
        d = Path(tempfile.mkdtemp())
        (d / ".session").write_text("", encoding="utf-8")
        with mock.patch.object(tc, "SESSION", d / ".session"):
            tc._lock_session()  # must target d/".session", not ".session.session"
        self.assertFalse((d / ".session.session").exists())

    def test_lock_session_ignores_a_directory(self):
        # a directory named like a session must never be chmod'd to 0600
        d = Path(tempfile.mkdtemp())
        (d / "archive.session").mkdir()
        before = (d / "archive.session").stat().st_mode
        with mock.patch.object(tc, "SESSION", d / "archive.session"):
            tc._lock_session()
        self.assertEqual((d / "archive.session").stat().st_mode, before)

    def test_lock_session_root_path_never_raises(self):
        # empty-name path (drive/anchor root) must not raise or echo anything
        with mock.patch.object(tc, "SESSION", Path(tc.SESSION.anchor or "/")):
            tc._lock_session()  # just must not raise

    def test_session_dir_failure_does_not_echo_path(self):
        blocker = Path(tempfile.mkdtemp()) / "blocker"
        blocker.write_text("x", encoding="utf-8")
        bad = blocker / "LEAKED_DIR" / "s.session"  # parent chain crosses a file
        with mock.patch.object(tc, "SESSION", bad):
            code, out = run_json("get", "--chat", "me", "--message-id", "7")
        self.assertEqual(code, 1)
        self.assertIn("session directory", out["error"])
        self.assertNotIn("LEAKED_DIR", out["error"])


class TestModuleBootstrap(unittest.TestCase):
    def test_unresolvable_home_stays_clean_json(self):
        """No HOME/USERPROFILE and no HERMES_HOME: still one JSON, never a
        traceback (fresh subprocess with forced home-resolution failure)."""
        env = {k: v for k, v in os.environ.items()
               if k not in ("HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH",
                            "HERMES_HOME", "TELETHON_SESSION")}
        probe = (
            "import runpy; from pathlib import Path; from unittest.mock import patch; "
            "patch.object(Path, 'home', side_effect=RuntimeError('no home')).start(); "
            "runpy.run_path(__import__('sys').argv[1], run_name='__main__')")
        r = subprocess.run(
            [sys.executable, "-c", probe, str(HERE / "telethon_checklist.py"), "--help"],
            capture_output=True, text=True, env=env)
        data = json.loads(r.stdout)
        self.assertFalse(data["ok"])
        self.assertEqual(r.returncode, 1)
        self.assertEqual(r.stderr, "")

    def test_session_env_alone_rescues_bootstrap(self):
        """TELETHON_SESSION without any resolvable home must be enough for the
        offline paths - the error hint promises exactly that."""
        # A standalone offline path no longer requires Telethon. Force home
        # resolution failure instead of letting the OS expose the real home.
        probe = (
            "import runpy; from pathlib import Path; from unittest.mock import patch; "
            "patch.object(Path, 'home', side_effect=RuntimeError('no home')).start(); "
            "sys = __import__('sys'); sys.argv = sys.argv[1:]; "
            "runpy.run_path(sys.argv[0], run_name='__main__')")
        env = {k: v for k, v in os.environ.items()
               if k not in ("HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH",
                            "HERMES_HOME")}
        env["TELETHON_SESSION"] = str(Path(tempfile.gettempdir()) / "tc_boot.session")
        env["TELETHON_CHECKLIST_CHATS"] = ""
        r = subprocess.run(
            [sys.executable, "-c", probe, str(HERE / "telethon_checklist.py"),
                         "create", "--title", "T", "--task", "a", "--dry-run"],
            capture_output=True, text=True, env=env)
        data = json.loads(r.stdout)
        self.assertTrue(data["ok"], data)
        self.assertEqual(r.returncode, 0)
        self.assertEqual(r.stderr, "")


class TestRealTelethonIntegration(unittest.TestCase):
    """Construct every request with the script's exact kwargs against the REAL
    installed telethon (fresh subprocess, so the in-process stub does not
    interfere). Skipped when telethon is not installed."""

    def test_constructor_kwargs_match_installed_telethon(self):
        probe = (
            "import sys\n"
            "try:\n"
            "    import telethon\n"
            "except Exception:\n"
            "    print('SKIP'); sys.exit(0)\n"
            "from telethon.tl.functions.messages import (SendMediaRequest,\n"
            "    AppendTodoListRequest, ToggleTodoCompletedRequest, GetForumTopicsRequest,\n"
            "    GetForumTopicsByIDRequest)\n"
            "from telethon.tl.types import (InputMediaTodo, TodoList, TodoItem,\n"
            "    TextWithEntities, InputReplyToMessage, MessageMediaToDo)\n"
            "from telethon.errors import FloodWaitError, RPCError\n"
            "twe = TextWithEntities(text='t', entities=[])\n"
            "todo = TodoList(title=twe, list=[TodoItem(id=1, title=twe)],\n"
            "                others_can_append=False, others_can_complete=False)\n"
            "SendMediaRequest(peer='me', media=InputMediaTodo(todo=todo), message='',\n"
            "                 random_id=1, reply_to=InputReplyToMessage(\n"
            "                     reply_to_msg_id=3, top_msg_id=3))\n"
            "AppendTodoListRequest(peer='me', msg_id=1, list=[TodoItem(id=2, title=twe)])\n"
            "ToggleTodoCompletedRequest(peer='me', msg_id=1, completed=[1], incompleted=[])\n"
            "GetForumTopicsRequest(peer='me', offset_date=None, offset_id=0,\n"
            "                      offset_topic=0, limit=100)\n"
            "GetForumTopicsByIDRequest(peer='me', topics=[33])\n"
            "assert issubclass(FloodWaitError, RPCError)\n"
            "print('OK')\n"
        )
        r = subprocess.run([sys.executable, "-c", probe],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        if "SKIP" in r.stdout:
            self.skipTest("real telethon not installed")
        self.assertIn("OK", r.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
