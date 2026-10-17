"""A leading ``/`` is not enough to make text a slash command.

``get_command`` has always refused a command name containing ``/`` — "``/tmp/x is broken`` is
text, not a command" (see the note in ``session_envelope``) — but ``MessageEvent.is_command`` and
every platform adapter's inbound classification only tested ``startswith("/")``. The two disagreed:

* the adapter tagged ``/root/notes.md done`` ``MessageType.COMMAND``;
* ``session_envelope._event_clause`` then saw ``message_type == COMMAND`` and refused the message
  as ``invalid_params (command)``;
* the user got only "Hermes refused this message" and the turn was lost.

So the path never reached a handler, and the refusal is indistinguishable from a malformed
command. ``parse_slash_command`` is now the single classifier: the event methods and every
adapter's classification route through it, so they cannot disagree again.

These tests pin:

* the classifier itself (what is and is not a command);
* that ``is_command`` and ``get_command`` never disagree;
* that a real adapter classifies a path as TEXT;
* that admission passes the path through instead of refusing it.
"""

import pytest

from gateway.platforms.event import MessageEvent, MessageType, parse_slash_command


# ---------------------------------------------------------------------------
# parse_slash_command — the single classifier
# ---------------------------------------------------------------------------

class TestParseSlashCommand:
    @pytest.mark.parametrize("text, expected", [
        ("/new", "new"),
        ("/NEW", "new"),                       # command names are lower-cased
        ("/new prompt", "new"),                # arguments are not part of the name
        ("/model gpt", "model"),
        ("/new@hermesbot", "new"),             # a "@bot" suffix on the command name is stripped
        ("/restart@bot now", "restart"),
        ("/restart", "restart"),
    ])
    def test_a_real_command_is_recognised(self, text, expected):
        assert parse_slash_command(text) == expected

    @pytest.mark.parametrize("text", [
        "/root/.hermes/secrets/token.txt 写在这里了",     # the reported case
        "/mnt/tf/backup/dump.sql 放好了",
        "/tmp/x is broken",                             # the case named in session_envelope
        "/home/user/notes.md",
        "/usr/bin/env python3",
        "/root/",                                       # trailing slash, no name after it
        "/新建/文件.txt",
        "///",                                          # first token is "//"
        # A mention with no command name: the original code returned "" here — is_command() True,
        # get_command() empty — the same disagreement as the file path, just unnoticed.
        "/@hermesbot new",
    ])
    def test_a_file_path_is_not_a_command(self, text):
        assert parse_slash_command(text) is None

    @pytest.mark.parametrize("text", [
        None,
        "",
        "   ",
        "new",                  # no leading slash
        "path /root/x.txt",     # slash not first
        "把 /root/Project 删了",
        "/",                    # a lone slash carries no name
        "/@bot",                # mention only, no command name
        "：/new",               # full-width colon, not a leading slash
    ])
    def test_text_without_a_command_name(self, text):
        assert parse_slash_command(text) is None

    def test_desktop_attachment_refs_do_not_hide_the_command(self):
        # buildContextText prepends "@file:<path>"; the command after it must still be found.
        assert parse_slash_command("@file:/tmp/a.png\n\n/new") == "new"
        assert parse_slash_command("@image:/tmp/a.png\n@url:https://x/y\n/status") == "status"

    def test_desktop_attachment_ref_over_a_path_stays_text(self):
        assert parse_slash_command("@file:/tmp/a.png\n\n/root/notes.md done") is None

    def test_none_and_empty_are_not_commands(self):
        # str(None) is "None" — it must not be treated as a slash token.
        assert parse_slash_command(None) is None


# ---------------------------------------------------------------------------
# MessageEvent: is_command and get_command must agree
# ---------------------------------------------------------------------------

def _event(text, *, allow_gateway_control=True, message_type=MessageType.TEXT):
    return MessageEvent(text=text, message_type=message_type, message_id='m1',
                        allow_gateway_control=allow_gateway_control)


class TestEventMethodsAgree:
    @pytest.mark.parametrize("text", [
        "/root/.hermes/secrets/token.txt 写在这里了",
        "/mnt/tf/backup/dump.sql 放好了",
        "/tmp/x is broken",
        "/new",
        "/",
        "普通消息",
        "把 /root/Project 删了",
        "",
    ])
    def test_is_command_and_get_command_never_disagree(self, text):
        # The original bug: is_command() said "command" while get_command() said None, and the
        # gateway refused the message for having a command with no name.
        ev = _event(text)
        assert ev.is_command() is (ev.get_command() is not None)

    def test_a_file_path_is_text(self):
        path = '/root/.hermes/secrets/github_pr_token.txt 写在这里了'
        ev = _event(path)
        assert ev.is_command() is False
        assert ev.get_command() is None

    def test_the_arguments_of_text_are_the_whole_text(self):
        # get_command_args() falls back to text for non-commands, so nothing is silently dropped.
        path = '/root/.hermes/secrets/token.txt 写在这里了'
        assert _event(path).get_command_args() == path

    def test_a_command_keeps_its_arguments(self):
        ev = _event('/model gpt-4o')
        assert ev.is_command() is True
        assert ev.get_command() == 'model'
        assert ev.get_command_args() == 'gpt-4o'

    def test_a_platform_that_opted_out_has_no_commands(self):
        ev = _event('/new', allow_gateway_control=False)
        assert ev.is_command() is False
        assert ev.get_command() is None


# ---------------------------------------------------------------------------
# A real adapter's classification
# ---------------------------------------------------------------------------

class TestAdapterClassification:
    def test_weixin_classifies_a_path_as_text(self):
        from plugins.platforms.weixin.adapter import _message_type_from_media
        assert _message_type_from_media([], '/root/notes.md done') == MessageType.TEXT
        assert _message_type_from_media([], '/new') == MessageType.COMMAND
        # Media still wins over text, exactly as before.
        assert _message_type_from_media(['image/png'], '/new') == MessageType.PHOTO
        assert _message_type_from_media(['application/pdf'], '/new') == MessageType.DOCUMENT

    def test_yuanbao_classifies_a_path_as_text(self):
        from plugins.platforms.yuanbao.adapter import ClassifyMessageTypeMiddleware
        classify = ClassifyMessageTypeMiddleware._classify
        assert classify('/mnt/tf/dump.sql 放好了', []) == MessageType.TEXT
        assert classify('/new', []) == MessageType.COMMAND


# ---------------------------------------------------------------------------
# Admission: the path must pass through, not be refused
# ---------------------------------------------------------------------------

class TestAdmission:
    def test_a_path_event_carries_no_command_clause(self):
        """The clause ``_event_clause`` would name — a path must not produce one."""
        from gateway.platforms.event import MessageType as MT
        path = '/root/.hermes/secrets/token.txt 写在这里了'
        # How every adapter now classifies inbound text:
        classified = MT.COMMAND if parse_slash_command(path) is not None else MT.TEXT
        assert classified == MT.TEXT
        ev = _event(path, message_type=classified)
        # Both halves of session_envelope's check must be false.
        assert not (ev.message_type == MT.COMMAND or ev.get_command())

    def test_a_real_command_still_takes_the_command_clause(self):
        # Regression guard: a genuine command is still refused on the native text lane, because
        # commands are dispatched elsewhere — the fix must not open that lane up.
        from gateway.platforms.event import MessageType as MT
        ev = _event('/new', message_type=MT.COMMAND)
        assert ev.message_type == MT.COMMAND or ev.get_command()
