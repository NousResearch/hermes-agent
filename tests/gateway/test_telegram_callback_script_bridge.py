"""Tests for profile-local Telegram callback scripts."""

import asyncio
import importlib
import json
import os
import stat
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import psutil

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


def _ensure_telegram_mock():
    try:
        importlib.import_module("telegram")
        importlib.import_module("telegram.ext")
        importlib.import_module("telegram.constants")
        importlib.import_module("telegram.request")
        importlib.import_module("telegram.error")
        return
    except ImportError:
        pass

    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return

    telegram_mod = ModuleType("telegram")
    ext_mod = ModuleType("telegram.ext")
    constants_mod = ModuleType("telegram.constants")
    request_mod = ModuleType("telegram.request")
    error_mod = ModuleType("telegram.error")

    setattr(telegram_mod, "Update", object)
    setattr(telegram_mod, "Bot", object)
    setattr(telegram_mod, "Message", object)
    setattr(telegram_mod, "InlineKeyboardButton", lambda *args, **kwargs: SimpleNamespace(args=args, kwargs=kwargs))
    setattr(telegram_mod, "InlineKeyboardMarkup", lambda inline_keyboard: SimpleNamespace(inline_keyboard=inline_keyboard))
    setattr(constants_mod, "ParseMode", SimpleNamespace(MARKDOWN="Markdown", MARKDOWN_V2="MarkdownV2", HTML="HTML"))
    setattr(constants_mod, "ChatType", SimpleNamespace(PRIVATE="private", GROUP="group", SUPERGROUP="supergroup", CHANNEL="channel"))
    setattr(ext_mod, "Application", object)
    setattr(ext_mod, "CommandHandler", object)
    setattr(ext_mod, "CallbackQueryHandler", object)
    setattr(ext_mod, "MessageHandler", object)
    setattr(ext_mod, "ContextTypes", SimpleNamespace(DEFAULT_TYPE=type(None)))
    setattr(ext_mod, "filters", object)
    setattr(request_mod, "HTTPXRequest", object)
    setattr(error_mod, "NetworkError", type("NetworkError", (OSError,), {}))
    setattr(error_mod, "TimedOut", type("TimedOut", (OSError,), {}))
    setattr(error_mod, "BadRequest", type("BadRequest", (Exception,), {}))
    setattr(telegram_mod, "constants", constants_mod)
    setattr(telegram_mod, "ext", ext_mod)
    setattr(telegram_mod, "error", error_mod)
    setattr(telegram_mod, "request", request_mod)

    sys.modules.setdefault("telegram", telegram_mod)
    sys.modules.setdefault("telegram.ext", ext_mod)
    sys.modules.setdefault("telegram.constants", constants_mod)
    sys.modules.setdefault("telegram.request", request_mod)
    sys.modules.setdefault("telegram.error", error_mod)


_ensure_telegram_mock()

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import _CALLBACK_SCRIPT_MAX_OUTPUT_BYTES, TelegramAdapter


def _make_adapter():
    # Short synthetic placeholder; tests never contact Telegram with this value.
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake", extra={}))
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


def _make_query(data: str, *, user_id=111, chat_id=12345, thread_id=7, chat_type="supergroup"):
    reply_markup = SimpleNamespace(name="existing-inline-keyboard")
    message = SimpleNamespace(
        chat_id=chat_id,
        message_id=42,
        message_thread_id=thread_id,
        text="Original text",
        reply_markup=reply_markup,
        chat=SimpleNamespace(type=chat_type),
    )
    query = SimpleNamespace(
        data=data,
        message=message,
        from_user=SimpleNamespace(id=user_id, first_name="Ada"),
        answer=AsyncMock(),
        edit_message_text=AsyncMock(),
        edit_message_reply_markup=AsyncMock(),
    )
    return query


def _write_script(callback_dir: Path, name: str, source: str) -> Path:
    callback_dir.mkdir(parents=True, exist_ok=True)
    path = callback_dir / name
    path.write_text(source, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


async def _dispatch(adapter, query):
    await adapter._handle_callback_query(
        SimpleNamespace(callback_query=query),
        SimpleNamespace(),
    )


@pytest.fixture
def callback_dir(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    directory = hermes_home / "scripts" / "telegram-callbacks"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "111")
    return directory


@pytest.mark.asyncio
async def test_success_invokes_script_with_context_env_and_edits(callback_dir, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    record_path = callback_dir.parent / "record.json"
    _write_script(
        callback_dir,
        "task.py",
        f"""#!/usr/bin/env python3
import json, os, sys
ctx = json.load(sys.stdin)
record = {{
    "argv": sys.argv,
    "stdin": ctx,
    "env": {{
        "HERMES_HOME": os.environ.get("HERMES_HOME"),
        "HERMES_TELEGRAM_CALLBACK_DATA": os.environ.get("HERMES_TELEGRAM_CALLBACK_DATA"),
        "HERMES_TELEGRAM_CALLBACK_PREFIX": os.environ.get("HERMES_TELEGRAM_CALLBACK_PREFIX"),
        "HERMES_TELEGRAM_USER_ID": os.environ.get("HERMES_TELEGRAM_USER_ID"),
        "HERMES_TELEGRAM_CHAT_ID": os.environ.get("HERMES_TELEGRAM_CHAT_ID"),
        "HERMES_TELEGRAM_CHAT_TYPE": os.environ.get("HERMES_TELEGRAM_CHAT_TYPE"),
        "HERMES_TELEGRAM_MESSAGE_ID": os.environ.get("HERMES_TELEGRAM_MESSAGE_ID"),
        "HERMES_TELEGRAM_THREAD_ID": os.environ.get("HERMES_TELEGRAM_THREAD_ID"),
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
    }},
}}
open({str(record_path)!r}, "w", encoding="utf-8").write(json.dumps(record))
print(json.dumps({{"ok": True, "answer_text": "Done", "edit_message_text": "Updated", "remove_keyboard": True}}))
""",
    )
    adapter = _make_adapter()
    query = _make_query("task:123")

    await _dispatch(adapter, query)

    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["argv"][1] == "task:123"
    assert record["stdin"]["callback_data"] == "task:123"
    assert record["stdin"]["prefix"] == "task"
    assert record["stdin"]["telegram"]["user_id"] == "111"
    assert record["stdin"]["telegram"]["chat_id"] == "12345"
    assert record["stdin"]["telegram"]["chat_type"] == "supergroup"
    assert record["stdin"]["telegram"]["message_id"] == "42"
    assert record["stdin"]["telegram"]["message_thread_id"] == "7"
    assert record["stdin"]["hermes"]["home"] == str(callback_dir.parents[1])
    assert record["env"]["HERMES_TELEGRAM_CALLBACK_DATA"] == "task:123"
    assert record["env"]["HERMES_TELEGRAM_CALLBACK_PREFIX"] == "task"
    assert record["env"]["HERMES_TELEGRAM_USER_ID"] == "111"
    assert record["env"]["HERMES_TELEGRAM_CHAT_ID"] == "12345"
    assert record["env"]["HERMES_TELEGRAM_CHAT_TYPE"] == "supergroup"
    assert record["env"]["HERMES_TELEGRAM_MESSAGE_ID"] == "42"
    assert record["env"]["HERMES_TELEGRAM_THREAD_ID"] == "7"
    assert record["env"]["OPENAI_API_KEY"] is None
    query.answer.assert_awaited_once_with(text="Done", show_alert=False)
    query.edit_message_text.assert_awaited_once_with(text="Updated", reply_markup=None)
    query.edit_message_reply_markup.assert_not_awaited()


@pytest.mark.asyncio
async def test_message_edit_falls_back_to_bot_api_when_callback_edit_fails(callback_dir):
    _write_script(
        callback_dir,
        "edit.py",
        """#!/usr/bin/env python3
import json
print(json.dumps({"ok": True, "answer_text": "Done", "edit_message_text": "Updated", "remove_keyboard": True}))
""",
    )
    adapter = _make_adapter()
    query = _make_query("edit:1")
    query.edit_message_text.side_effect = RuntimeError("callback edit failed")

    await _dispatch(adapter, query)

    query.answer.assert_awaited_once_with(text="Done", show_alert=False)
    query.edit_message_text.assert_awaited_once_with(text="Updated", reply_markup=None)
    bot = adapter._bot
    assert bot is not None
    bot.edit_message_text.assert_awaited_once_with(
        chat_id=12345,
        message_id=42,
        text="Updated",
        reply_markup=None,
    )


@pytest.mark.asyncio
async def test_message_edit_preserves_inline_keyboard_by_default(callback_dir):
    _write_script(
        callback_dir,
        "keep.py",
        """#!/usr/bin/env python3
import json
print(json.dumps({"ok": True, "answer_text": "Done", "edit_message_text": "Updated"}))
""",
    )
    adapter = _make_adapter()
    query = _make_query("keep:1")

    await _dispatch(adapter, query)

    query.answer.assert_awaited_once_with(text="Done", show_alert=False)
    query.edit_message_text.assert_awaited_once_with(
        text="Updated",
        reply_markup=query.message.reply_markup,
    )


@pytest.mark.asyncio
async def test_message_edit_fallback_preserves_inline_keyboard_by_default(callback_dir):
    _write_script(
        callback_dir,
        "keep.py",
        """#!/usr/bin/env python3
import json
print(json.dumps({"ok": True, "answer_text": "Done", "edit_message_text": "Updated"}))
""",
    )
    adapter = _make_adapter()
    query = _make_query("keep:1")
    query.edit_message_text.side_effect = RuntimeError("callback edit failed")

    await _dispatch(adapter, query)

    query.answer.assert_awaited_once_with(text="Done", show_alert=False)
    query.edit_message_text.assert_awaited_once_with(
        text="Updated",
        reply_markup=query.message.reply_markup,
    )
    bot = adapter._bot
    assert bot is not None
    bot.edit_message_text.assert_awaited_once_with(
        chat_id=12345,
        message_id=42,
        text="Updated",
        reply_markup=query.message.reply_markup,
    )


@pytest.mark.asyncio
async def test_show_alert_true_success(callback_dir):
    _write_script(
        callback_dir,
        "alert",
        """#!/usr/bin/env python3
import json
print(json.dumps({"ok": True, "answer_text": "Look", "show_alert": True}))
""",
    )
    query = _make_query("alert:1")

    await _dispatch(_make_adapter(), query)

    query.answer.assert_awaited_once_with(text="Look", show_alert=True)


@pytest.mark.asyncio
async def test_remove_keyboard_only_uses_reply_markup_edit(callback_dir):
    _write_script(
        callback_dir,
        "clean.sh",
        """#!/bin/sh
printf '%s\n' '{"ok": true, "answer_text": "Cleared", "remove_keyboard": true}'
""",
    )
    query = _make_query("clean:anything")

    await _dispatch(_make_adapter(), query)

    query.answer.assert_awaited_once_with(text="Cleared", show_alert=False)
    query.edit_message_reply_markup.assert_awaited_once_with(reply_markup=None)
    query.edit_message_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_unauthorized_user_does_not_execute(callback_dir, monkeypatch):
    marker = callback_dir.parent / "ran"
    _write_script(
        callback_dir,
        "task",
        f"""#!/usr/bin/env python3
from pathlib import Path
Path({str(marker)!r}).write_text("ran")
print('{{"ok": true}}')
""",
    )
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "999")
    query = _make_query("task:123", user_id=111)

    await _dispatch(_make_adapter(), query)

    assert not marker.exists()
    query.answer.assert_awaited_once()
    assert "not authorized" in query.answer.call_args.kwargs["text"].lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data",
    [
        "missing:1",
        "bad.prefix:1",
        "../escape:1",
        "~home:1",
        "slash/prefix:1",
        r"back\\slash:1",
        "toolongprefixname_over_32_chars_here:1",
        "nocolon",
    ],
)
async def test_missing_or_invalid_prefix_is_ignored(callback_dir, data):
    query = _make_query(data)

    await _dispatch(_make_adapter(), query)

    query.answer.assert_not_awaited()
    query.edit_message_text.assert_not_awaited()
    query.edit_message_reply_markup.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="Symlinks require elevated privileges on Windows")
async def test_symlink_escape_rejected(callback_dir, tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text("#!/usr/bin/env python3\nprint('{\"ok\": true, \"answer_text\": \"bad\"}')\n", encoding="utf-8")
    outside.chmod(outside.stat().st_mode | stat.S_IXUSR)
    callback_dir.mkdir(parents=True, exist_ok=True)
    (callback_dir / "escape.py").symlink_to(outside)
    query = _make_query("escape:1")

    await _dispatch(_make_adapter(), query)

    query.answer.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("script_body", "expected"),
    [
        ("import sys; sys.exit(3)", "Callback action failed."),
        ("print('not json')", "Callback action failed."),
        ("import json; print(json.dumps({'ok': False, 'answer_text': 'Could not finish'}))", "Could not finish"),
    ],
)
async def test_script_failures_answer_once_with_safe_text(callback_dir, script_body, expected):
    _write_script(callback_dir, "fail.py", f"#!/usr/bin/env python3\n{script_body}\n")
    query = _make_query("fail:secret-payload")

    await _dispatch(_make_adapter(), query)

    query.answer.assert_awaited_once()
    text = query.answer.call_args.kwargs["text"]
    assert text == expected
    assert "secret-payload" not in text
    assert query.answer.await_count == 1


@pytest.mark.asyncio
async def test_nonzero_exit_with_valid_json_does_not_apply_response(callback_dir):
    _write_script(
        callback_dir,
        "nonzero.py",
        """#!/usr/bin/env python3
import json
import sys
print(json.dumps({"ok": True, "answer_text": "Unsafe", "edit_message_text": "Do not apply"}))
sys.exit(7)
""",
    )
    query = _make_query("nonzero:1")

    await _dispatch(_make_adapter(), query)

    query.answer.assert_awaited_once_with(text="Callback action failed.")
    query.edit_message_text.assert_not_awaited()
    query.edit_message_reply_markup.assert_not_awaited()


@pytest.mark.asyncio
async def test_callback_chat_type_accepts_enum_like_value(callback_dir):
    record_path = callback_dir.parent / "record.json"
    _write_script(
        callback_dir,
        "enumtype.py",
        f"""#!/usr/bin/env python3
import json, os, sys
ctx = json.load(sys.stdin)
open({str(record_path)!r}, "w", encoding="utf-8").write(json.dumps({{
    "stdin_chat_type": ctx["telegram"]["chat_type"],
    "env_chat_type": os.environ.get("HERMES_TELEGRAM_CHAT_TYPE"),
}}))
print(json.dumps({{"ok": True, "answer_text": "Done"}}))
""",
    )
    query = _make_query("enumtype:1", chat_type=SimpleNamespace(value="SUPERGROUP"))

    await _dispatch(_make_adapter(), query)

    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record == {"stdin_chat_type": "supergroup", "env_chat_type": "supergroup"}
    query.answer.assert_awaited_once_with(text="Done", show_alert=False)


@pytest.mark.asyncio
async def test_split_script_stdout_json_is_read_until_eof(callback_dir):
    _write_script(
        callback_dir,
        "split.py",
        """#!/usr/bin/env python3
import sys
import time
sys.stdout.write('{"ok": true, "answer_text": "Hel')
sys.stdout.flush()
time.sleep(0.05)
sys.stdout.write('lo"}')
sys.stdout.flush()
""",
    )
    query = _make_query("split:1")

    await _dispatch(_make_adapter(), query)

    query.answer.assert_awaited_once_with(text="Hello", show_alert=False)


@pytest.mark.asyncio
async def test_oversized_script_stdout_is_rejected_with_safe_text(callback_dir):
    _write_script(
        callback_dir,
        "huge.py",
        f"""#!/usr/bin/env python3
import sys
sys.stdout.write("x" * {_CALLBACK_SCRIPT_MAX_OUTPUT_BYTES + 1})
sys.stdout.flush()
""",
    )
    query = _make_query("huge:secret-payload")

    await _dispatch(_make_adapter(), query)

    query.answer.assert_awaited_once_with(text="Callback action failed.")
    assert "secret-payload" not in query.answer.call_args.kwargs["text"]


@pytest.mark.asyncio
async def test_oversized_script_stdout_after_initial_json_is_rejected(callback_dir):
    _write_script(
        callback_dir,
        "overflow.py",
        f"""#!/usr/bin/env python3
import sys
import time
sys.stdout.write('{{"ok": true, "answer_text": "Unsafe"}}')
sys.stdout.flush()
time.sleep(0.05)
sys.stdout.write("x" * {_CALLBACK_SCRIPT_MAX_OUTPUT_BYTES + 1})
sys.stdout.flush()
""",
    )
    query = _make_query("overflow:secret-payload")

    await _dispatch(_make_adapter(), query)

    query.answer.assert_awaited_once_with(text="Callback action failed.")
    assert "secret-payload" not in query.answer.call_args.kwargs["text"]


@pytest.mark.asyncio
async def test_completed_stdout_is_not_applied_when_stderr_overflows(callback_dir):
    _write_script(
        callback_dir,
        "stderr_overflow.py",
        f"""#!/usr/bin/env python3
import sys
sys.stdout.write('{{"ok": true, "answer_text": "Unsafe", "edit_message_text": "Do not apply"}}')
sys.stdout.flush()
sys.stdout.close()
sys.stderr.write("e" * {_CALLBACK_SCRIPT_MAX_OUTPUT_BYTES + 1})
sys.stderr.flush()
""",
    )
    query = _make_query("stderr_overflow:secret-payload")

    await _dispatch(_make_adapter(), query)

    query.answer.assert_awaited_once_with(text="Callback action failed.")
    query.edit_message_text.assert_not_awaited()
    query.edit_message_reply_markup.assert_not_awaited()
    assert "secret-payload" not in query.answer.call_args.kwargs["text"]


async def _process_is_gone(pid: int) -> bool:
    for _ in range(50):
        if not psutil.pid_exists(pid):
            return True
        try:
            process = psutil.Process(pid)
            if process.status() == psutil.STATUS_ZOMBIE:
                return True
        except psutil.NoSuchProcess:
            return True
        await asyncio.sleep(0.05)
    return False


@pytest.mark.asyncio
async def test_timeout_kills_script_process_group(callback_dir, tmp_path, monkeypatch):
    from plugins.platforms.telegram import adapter as telegram_mod

    monkeypatch.setattr(telegram_mod, "_CALLBACK_SCRIPT_TIMEOUT_S", 0.2)
    marker = tmp_path / "child.pid"
    _write_script(
        callback_dir,
        "spawn.py",
        f"""#!/usr/bin/env python3
import subprocess
import sys
import time
from pathlib import Path

child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
Path({str(marker)!r}).write_text(str(child.pid), encoding="utf-8")
time.sleep(30)
""",
    )
    query = _make_query("spawn:child")

    await _dispatch(_make_adapter(), query)

    query.answer.assert_awaited_once_with(text="Callback action failed.")
    child_pid = int(marker.read_text(encoding="utf-8"))
    assert await _process_is_gone(child_pid)


@pytest.mark.asyncio
async def test_script_timeout_is_killed_and_answers_once(callback_dir, monkeypatch):
    marker = callback_dir.parent / "finished"
    _write_script(
        callback_dir,
        "slow.py",
        f"""#!/usr/bin/env python3
import json, time
time.sleep(5)
open({str(marker)!r}, "w", encoding="utf-8").write("finished")
print(json.dumps({{"ok": True, "answer_text": "too late"}}))
""",
    )
    monkeypatch.setattr("plugins.platforms.telegram.adapter._CALLBACK_SCRIPT_TIMEOUT_S", 0.1)
    query = _make_query("slow:1")

    await _dispatch(_make_adapter(), query)

    assert not marker.exists()
    query.answer.assert_awaited_once_with(text="Callback action failed.")


@pytest.mark.asyncio
async def test_shell_metacharacters_are_passed_as_argv(callback_dir, tmp_path):
    marker = tmp_path / "shell-ran"
    record = tmp_path / "argv.json"
    _write_script(
        callback_dir,
        "safe",
        f"""#!/usr/bin/env python3
import json, sys
open({str(record)!r}, "w", encoding="utf-8").write(json.dumps(sys.argv))
print(json.dumps({{"ok": True, "answer_text": "Safe"}}))
""",
    )
    data = f"safe:$(touch {marker});`touch {marker}`"
    query = _make_query(data)

    await _dispatch(_make_adapter(), query)

    assert not marker.exists()
    assert json.loads(record.read_text(encoding="utf-8"))[1] == data
    query.answer.assert_awaited_once_with(text="Safe", show_alert=False)


@pytest.mark.asyncio
async def test_answer_text_is_truncated_to_200_chars(callback_dir):
    _write_script(
        callback_dir,
        "long",
        """#!/usr/bin/env python3
import json
print(json.dumps({"ok": True, "answer_text": "x" * 250}))
""",
    )
    query = _make_query("long:1")

    await _dispatch(_make_adapter(), query)

    assert len(query.answer.call_args.kwargs["text"]) == 200


@pytest.mark.asyncio
async def test_builtin_prefix_still_wins(callback_dir):
    marker = callback_dir.parent / "ran"
    _write_script(
        callback_dir,
        "gt",
        f"""#!/usr/bin/env python3
from pathlib import Path
Path({str(marker)!r}).write_text("ran")
print('{{"ok": true, "answer_text": "GENERIC SCRIPT USED", "edit_message_text": "GENERIC EDIT"}}')
""",
    )
    adapter = _make_adapter()
    adapter._handle_gmail_triage_callback = AsyncMock()
    query = _make_query("gt:send:abc")

    await _dispatch(adapter, query)

    adapter._handle_gmail_triage_callback.assert_awaited_once()
    assert not marker.exists()
    query.answer.assert_not_awaited()
    query.edit_message_text.assert_not_awaited()
