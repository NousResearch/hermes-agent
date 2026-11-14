"""Fail-closed redaction tests for sessions export --redact (#90361).

``redact_session_data`` used to walk only the ``messages``/``segments``
keys, so session-level string fields — most importantly the auto-generated
title, which quotes the first user message — escaped redaction and rendered
into the export header in plaintext despite an explicit ``--redact``. The
walk now covers every string in the export dict.
"""


def _fake_botfather_token() -> str:
    # BotFather shape: <10-digit id>:<35-char base64url-ish>, assembled to
    # avoid a credential-shaped literal in source.
    digits = "8" * 10
    tail = ("A" * 34) + "h"
    return digits + ":" + tail


def _fake_api_key() -> str:
    return "-".join(("sk", "probe" * 3, "key9")) + "XYZabcd"


def _session_fixture() -> dict:
    token = _fake_botfather_token()
    return {
        "id": "s1",
        "title": f"telegram setup with {token}",
        "cwd": "/tmp/x",
        "message_count": 3,
        "started_at": 1710000000,
        "messages": [
            {"role": "user", "content": f"use {token} please"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "function": {
                            "name": "send_message",
                            "arguments": f'{{"token": "{token}"}}',
                        },
                    }
                ],
            },
            {"role": "tool", "content": f"sent via {token}"},
        ],
    }


class TestRedactSessionData:
    def test_title_token_is_redacted(self):
        """The title quotes the first user message — a credential there must
        not ride into the export header in plaintext (#90361)."""
        from hermes_cli.session_export_md import redact_session_data

        token = _fake_botfather_token()
        out = redact_session_data(_session_fixture())
        assert token not in out["title"]

    def test_message_content_and_tool_args_still_redacted(self):
        from hermes_cli.session_export_md import redact_session_data

        token = _fake_botfather_token()
        out = redact_session_data(_session_fixture())
        assert token not in out["messages"][0]["content"]
        assert token not in out["messages"][1]["tool_calls"][0]["function"]["arguments"]
        assert token not in out["messages"][2]["content"]

    def test_non_string_scalars_preserved(self):
        from hermes_cli.session_export_md import redact_session_data

        out = redact_session_data(_session_fixture())
        assert out["id"] == "s1"
        assert out["message_count"] == 3
        assert out["started_at"] == 1710000000

    def test_api_key_in_top_level_field_redacted(self):
        from hermes_cli.session_export_md import redact_session_data

        session = _session_fixture()
        session["summary"] = f"rotated to {_fake_api_key()} today"
        out = redact_session_data(session)
        assert _fake_api_key() not in out["summary"]

    def test_credential_shaped_dict_key_redacted(self):
        """Dict KEYS are strings too — a misbehaving emitter writing the
        credential as a key must not pass through verbatim (review #90361)."""
        from hermes_cli.session_export_md import redact_session_data

        session = _session_fixture()
        key = _fake_api_key()
        session["tool_result"] = {key: "payload"}
        out = redact_session_data(session)
        assert key not in str(out["tool_result"]), out["tool_result"]
        assert "payload" in str(out["tool_result"])

    def test_original_session_not_mutated(self):
        from hermes_cli.session_export_md import redact_session_data

        session = _session_fixture()
        token = _fake_botfather_token()
        redact_session_data(session)
        assert token in session["title"], "redaction must return a copy"


class TestRenderedExportIsClean:
    def test_markdown_render_contains_no_token(self):
        """End-to-end: --redact output file must not contain the token
        anywhere — header included."""
        from hermes_cli.session_export_md import (
            redact_session_data,
            render_session_markdown,
        )

        token = _fake_botfather_token()
        md = render_session_markdown(redact_session_data(_session_fixture()), fmt="md")
        assert token not in md
