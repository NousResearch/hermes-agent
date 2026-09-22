"""Slack authored-text deduplication tests.

``TestSlackAuthoredTextDeduplication`` with its module-level permalink
constants, moved byte-verbatim out of ``tests/gateway/test_slack.py``.

The shared ``adapter`` fixture, the two module-level ``autouse`` fixtures and
the rich-text block helpers stay in ``test_slack`` and are imported here, so
pytest fixture discovery is unchanged.

Part of #79914, #78647
"""

import pytest

from tests.gateway.test_slack import (
    _pin_legacy_assistant_threads_api,
    _redirect_cache,
    _rich_text_blocks,
    _rich_text_section,
    _slack_mod,
    adapter,
)


# ---------------------------------------------------------------------------
# TestSlackAuthoredTextDeduplication
# ---------------------------------------------------------------------------


# A "Copy link" URL for a Slack thread always carries query parameters, so
# Slack HTML-escapes the ``&`` in ``event.text`` while leaving the same URL
# raw inside ``blocks[].link.url``.
_THREAD_PERMALINK = (
    "https://example.slack.com/archives/C0BCDG3H66P/p1786102118226679"
    "?thread_ts=1786102118.226679&cid=C0BCDG3H66P"
)
_THREAD_PERMALINK_ESCAPED = _THREAD_PERMALINK.replace("&", "&amp;")

# A permalink as the Slack client pastes it — no query parameters, delivered as
# a ``message_mention`` element rather than a plain ``link``.
_PERMALINK = "https://example.slack.com/archives/C0BCDG3H66P/p1786102118226679"


class TestSlackAuthoredTextDeduplication:
    """One authored Slack message must never be appended to itself.

    Slack delivers the same authored text twice — flat in ``event.text`` and
    structurally in ``event.blocks`` — and HTML-escapes ``&``/``<``/``>`` in
    the flat copy only. Whenever the two representations fail to compare
    equal, the block rendering is mistaken for additional content and the
    user sees their own message twice. Both merge sites are covered:
    ``_handle_slack_message`` (live inbound) and ``_render_message_text``
    (thread/parent hydration).
    """

    @staticmethod
    def _thread_link_blocks(*trailing):
        return _rich_text_blocks(
            _rich_text_section(
                {"type": "user", "user_id": "U_BOT"},
                {"type": "text", "text": " do you see "},
                {"type": "link", "url": _THREAD_PERMALINK},
                {"type": "text", "text": " ?"},
            ),
            *trailing,
        )

    @staticmethod
    def _thread_link_text():
        return f"<@U_BOT> do you see <{_THREAD_PERMALINK_ESCAPED}> ?"

    # -- helper-level equivalence -----------------------------------------

    @pytest.mark.parametrize(
        "flat_text,elements",
        [
            # Thread permalink: query params make Slack escape ``&`` in text
            # while ``blocks[].link.url`` stays raw. The reported bug.
            (
                f"look <{_THREAD_PERMALINK_ESCAPED}> here",
                [
                    {"type": "text", "text": "look "},
                    {"type": "link", "url": _THREAD_PERMALINK},
                    {"type": "text", "text": " here"},
                ],
            ),
            # Bare ampersand in prose.
            ("AT&amp;T outage", [{"type": "text", "text": "AT&T outage"}]),
            # Literal angle brackets the user typed.
            ("use &lt;div&gt; here", [{"type": "text", "text": "use <div> here"}]),
            # Labelled link whose label carries an ampersand.
            (
                "see <https://x.example|AT&amp;T>",
                [
                    {"type": "text", "text": "see "},
                    {"type": "link", "url": "https://x.example", "text": "AT&T"},
                ],
            ),
        ],
    )
    def test_escaped_entities_compare_equal(self, flat_text, elements):
        assert (
            _slack_mod._extract_additional_text_from_slack_blocks(
                _rich_text_blocks(_rich_text_section(*elements)), flat_text
            )
            == ""
        )

    def test_genuine_quote_still_appended_next_to_escaped_link(self):
        """Negative case: the fix must not swallow real structured content."""
        blocks = self._thread_link_blocks(
            {
                "type": "rich_text_quote",
                "elements": [
                    _rich_text_section({"type": "text", "text": "quoted context"})
                ],
            }
        )

        assert (
            _slack_mod._extract_additional_text_from_slack_blocks(
                blocks, self._thread_link_text(), bot_uid="U_BOT"
            )
            == "> quoted context"
        )

    # -- live inbound path -------------------------------------------------

    @pytest.mark.asyncio
    async def test_live_inbound_thread_permalink_not_duplicated(self, adapter):
        await adapter._handle_slack_message(
            {
                "text": self._thread_link_text(),
                "blocks": self._thread_link_blocks(),
                "user": "U_USER",
                "client_msg_id": "cm-1",
                "channel": "D_DM",
                "channel_type": "im",
                "ts": "123.456",
                "team": "T_TEAM",
            }
        )

        adapter.handle_message.assert_awaited_once()
        text = adapter.handle_message.await_args.args[0].text
        assert text.count("p1786102118226679") == 1
        assert text.count("do you see") == 1

    # -- thread/parent hydration path --------------------------------------

    def test_hydration_thread_permalink_not_duplicated(self, adapter):
        rendered = adapter._render_message_text(
            {"text": self._thread_link_text(), "blocks": self._thread_link_blocks()},
            bot_uid="U_BOT",
        )

        assert rendered.count("p1786102118226679") == 1
        assert rendered.count("do you see") == 1

    def test_hydration_skips_message_unfurl_attachment(self, adapter):
        """A permalink unfurl echoes the *linked* message — the live path
        already skips it, so hydration must not re-append it either."""
        rendered = adapter._render_message_text(
            {
                "text": f"<{_THREAD_PERMALINK_ESCAPED}>",
                "attachments": [
                    {
                        "is_msg_unfurl": True,
                        "text": "the linked message body",
                        "fallback": "linked message fallback",
                    }
                ],
            }
        )

        assert "the linked message body" not in rendered
        assert "linked message fallback" not in rendered

    def test_hydration_still_surfaces_regular_attachments(self, adapter):
        """Alert-bot content lives only in attachments — keep surfacing it."""
        rendered = adapter._render_message_text(
            {
                "text": "",
                "attachments": [
                    {"is_msg_unfurl": True, "text": "echoed message body"},
                    {"title": "FiringAlert", "text": "disk usage 95%"},
                ],
            }
        )

        assert "echoed message body" not in rendered
        assert "FiringAlert" in rendered
        assert "disk usage 95%" in rendered

    # -- Block Kit payload dump --------------------------------------------

    @pytest.mark.asyncio
    async def test_block_kit_dump_leaves_out_the_authored_rich_text(self, adapter):
        """A single non-rich_text block must not drag the message in with it.

        The dump exists for the interactive blocks bots post, and its
        allowlist deliberately drops ``url``. Serializing the authored
        ``rich_text`` alongside them therefore repeats the user's own
        sentence with its links deleted — the "second copy without the
        link" a reporter sees.
        """
        await adapter._handle_slack_message(
            {
                "text": self._thread_link_text(),
                "blocks": self._thread_link_blocks()
                + [{"type": "section", "text": {"type": "mrkdwn", "text": "extra"}}],
                "user": "U_USER",
                "client_msg_id": "cm-2",
                "channel": "D_DM",
                "channel_type": "im",
                "ts": "123.457",
                "team": "T_TEAM",
            }
        )

        text = adapter.handle_message.await_args.args[0].text
        assert text.count("do you see") == 1
        assert text.count("p1786102118226679") == 1
        # The block the agent cannot otherwise read is still surfaced.
        assert "extra" in text

    @pytest.mark.asyncio
    async def test_no_block_kit_dump_for_a_plain_authored_message(self, adapter):
        await adapter._handle_slack_message(
            {
                "text": self._thread_link_text(),
                "blocks": self._thread_link_blocks(),
                "user": "U_USER",
                "client_msg_id": "cm-3",
                "channel": "D_DM",
                "channel_type": "im",
                "ts": "123.458",
                "team": "T_TEAM",
            }
        )

        text = adapter.handle_message.await_args.args[0].text
        assert "[Slack Block Kit payload for this message]" not in text

    # -- inline elements the renderer does not know ------------------------

    @staticmethod
    def _mention_blocks(element, *trailing):
        """The blocks Slack sends for ``@bot do you see <permalink> ?``."""
        return _rich_text_blocks(
            _rich_text_section(
                {"type": "user", "user_id": "U_BOT"},
                {"type": "text", "text": " do you see "},
                element,
                {"type": "text", "text": " ?"},
            ),
            *trailing,
        )

    @staticmethod
    def _mention_text():
        """``event.text`` for a pasted permalink: label equals the URL."""
        return f"<@U_BOT> do you see <{_PERMALINK}|{_PERMALINK}> ?"

    @pytest.mark.parametrize(
        "element",
        [
            # Slack's own element for a pasted message permalink, as the
            # client sends it: required ids plus an optional url/label.
            {
                "type": "message_mention",
                "channel_id": "C0BCDG3H66P",
                "message_ts": "1786102118.226679",
                "url": _PERMALINK,
                "text": _PERMALINK,
            },
            # Same element with the optional label omitted.
            {
                "type": "message_mention",
                "channel_id": "C0BCDG3H66P",
                "message_ts": "1786102118.226679",
                "url": _PERMALINK,
            },
            # Slack adds inline element types without notice; one that carries
            # a url must render rather than vanish.
            {"type": "an_element_slack_adds_later", "url": _PERMALINK},
            # ... and one that carries only a label.
            {"type": "an_element_slack_adds_later", "text": _PERMALINK},
        ],
    )
    def test_url_bearing_inline_elements_render_instead_of_vanishing(self, element):
        assert (
            _slack_mod._extract_additional_text_from_slack_blocks(
                self._mention_blocks(element), self._mention_text(), bot_uid="U_BOT"
            )
            == ""
        )
        assert _PERMALINK in _slack_mod._extract_text_from_slack_blocks(
            self._mention_blocks(element)
        )

    @pytest.mark.parametrize(
        "element,rendered",
        [
            # Block Kit carries text as an object in many places, so an unknown
            # element may hold one where a string belongs.
            (
                {
                    "type": "an_element_slack_adds_later",
                    "text": {"type": "plain_text", "text": "oops"},
                },
                "",
            ),
            # A string field next to it is still read.
            (
                {
                    "type": "an_element_slack_adds_later",
                    "text": {"type": "plain_text", "text": "oops"},
                    "fallback": _PERMALINK,
                },
                _PERMALINK,
            ),
            # A known type reading a field of its own is no different.
            ({"type": "color", "value": {"type": "plain_text", "text": "#fff"}}, ""),
            (
                {
                    "type": "date",
                    "timestamp": 1786102118,
                    "fallback": {"type": "plain_text", "text": "Aug 7th"},
                },
                "",
            ),
            ({"type": "text", "text": {"type": "plain_text", "text": "oops"}}, ""),
        ],
    )
    def test_inline_element_with_an_object_field_keeps_the_message(
        self, element, rendered
    ):
        """A non-string field must not reach the caller's ``str.join``."""
        blocks = self._mention_blocks(element)
        flat_text = f"<@U_BOT> do you see {rendered} ?"

        assert _slack_mod._extract_text_from_slack_blocks(blocks) == flat_text
        assert (
            _slack_mod._extract_additional_text_from_slack_blocks(
                blocks, flat_text, bot_uid="U_BOT"
            )
            == ""
        )

    @pytest.mark.parametrize(
        "flat_text",
        [
            # The permalink as pasted...
            f"<@U_BOT> do you see <{_PERMALINK}|{_PERMALINK}> ?",
            # ...and its "Copy link" form, whose query parameters the element
            # cannot rebuild.
            f"<@U_BOT> do you see <{_THREAD_PERMALINK_ESCAPED}> ?",
        ],
    )
    def test_url_less_message_mention_is_not_duplicated(self, flat_text):
        """``url`` is optional on this element; ``channel_id`` and
        ``message_ts`` are not, and they rebuild the permalink's tail."""
        blocks = self._mention_blocks(
            {
                "type": "message_mention",
                "channel_id": "C0BCDG3H66P",
                "message_ts": "1786102118.226679",
            }
        )

        assert (
            _slack_mod._extract_additional_text_from_slack_blocks(
                blocks, flat_text, bot_uid="U_BOT"
            )
            == ""
        )
        assert (
            "archives/C0BCDG3H66P/p1786102118226679"
            in _slack_mod._extract_text_from_slack_blocks(blocks)
        )

    @pytest.mark.parametrize(
        "element",
        [
            # The element's own ``url`` never carries the query parameters the
            # flat text has...
            {
                "type": "message_mention",
                "channel_id": "C0BCDG3H66P",
                "message_ts": "1786102118.226679",
                "url": _PERMALINK,
                "text": "Custom label",
            },
            # ...and it may not carry a ``url`` at all.
            {
                "type": "message_mention",
                "channel_id": "C0BCDG3H66P",
                "message_ts": "1786102118.226679",
                "text": "Custom label",
            },
        ],
    )
    def test_labelled_permalink_with_query_params_is_not_duplicated(self, element):
        """A labelled link is canonicalized to ``label (url)``, so reducing the
        permalink must stop at the query and leave the closing parenthesis."""
        blocks = self._mention_blocks(element)

        assert (
            _slack_mod._extract_additional_text_from_slack_blocks(
                blocks,
                f"<@U_BOT> do you see <{_THREAD_PERMALINK_ESCAPED}|Custom label> ?",
                bot_uid="U_BOT",
            )
            == ""
        )

    def test_quote_beside_a_url_less_message_mention_appended_alone(self):
        """The quote is the only addition: the sentence around the permalink
        must not come back as a second copy with the link blanked."""
        blocks = self._mention_blocks(
            {
                "type": "message_mention",
                "channel_id": "C0BCDG3H66P",
                "message_ts": "1786102118.226679",
            },
            {
                "type": "rich_text_quote",
                "elements": [
                    _rich_text_section({"type": "text", "text": "quoted context"})
                ],
            },
        )

        assert (
            _slack_mod._extract_additional_text_from_slack_blocks(
                blocks, self._mention_text(), bot_uid="U_BOT"
            )
            == "> quoted context"
        )

    def test_quote_containing_an_unrenderable_element_is_still_appended(self):
        """Negative case: a quote is absent from ``event.text`` by construction,
        so it is never a duplicate of it."""
        blocks = _rich_text_blocks(
            _rich_text_section({"type": "text", "text": "look at this"}),
            {
                "type": "rich_text_quote",
                "elements": [
                    {"type": "text", "text": "see "},
                    {"type": "an_element_slack_adds_later"},
                    {"type": "text", "text": " please"},
                ],
            },
        )

        additional = _slack_mod._extract_additional_text_from_slack_blocks(
            blocks, "look at this", bot_uid="U_BOT"
        )

        assert "see" in additional
        assert "please" in additional

    @pytest.mark.parametrize(
        ("element", "flat"),
        [
            # ``fallback`` and ``url`` are both optional on the rich-text date
            # element, so an element with neither renders as nothing.
            ({}, "<!date^1786102118^{date_short}>"),
            ({"fallback": "Aug 7"}, "<!date^1786102118^{date_short}^|Aug 7>"),
            (
                {"url": "https://cal/x", "fallback": "Aug 7"},
                "<!date^1786102118^{date_short}^https://cal/x|Aug 7>",
            ),
            (
                {"url": "https://cal/x"},
                "<!date^1786102118^{date_short}^https://cal/x>",
            ),
        ],
    )
    def test_date_element_is_not_read_as_new_content(self, element, flat):
        """The flat field carries ``<!date^…>`` while the rich text renders the
        fallback or the url, so both sides need reading down to one value."""
        blocks = _rich_text_blocks(
            _rich_text_section(
                {"type": "user", "user_id": "U_BOT"},
                {"type": "text", "text": " meet at "},
                {
                    "type": "date",
                    "timestamp": 1786102118,
                    "format": "{date_short}",
                    **element,
                },
                {"type": "text", "text": " ok?"},
            )
        )

        assert (
            _slack_mod._extract_additional_text_from_slack_blocks(
                blocks, f"<@U_BOT> meet at {flat} ok?", bot_uid="U_BOT"
            )
            == ""
        )

    @pytest.mark.parametrize("flat_text", ["", "New alert"])
    def test_app_message_keeps_its_body(self, flat_text):
        """Negative case: an app posts its body in the blocks, with a flat
        ``text`` field that is empty or a short notification of its own."""
        blocks = _rich_text_blocks(
            _rich_text_section(
                {"type": "text", "text": "Build failed on "},
                # ``team`` carries neither a url nor a label.
                {"type": "team", "team_id": "T123"},
                {"type": "text", "text": " see logs"},
            )
        )

        additional = _slack_mod._extract_additional_text_from_slack_blocks(
            blocks, flat_text, bot_uid="U_BOT"
        )

        assert "Build failed on" in additional
        assert "see logs" in additional

    def test_hydrated_app_message_without_flat_text_keeps_its_body(self, adapter):
        rendered = adapter._render_message_text(
            {
                "text": "",
                "blocks": _rich_text_blocks(
                    _rich_text_section(
                        {"type": "text", "text": "Build failed on "},
                        {"type": "team", "team_id": "T123"},
                        {"type": "text", "text": " see logs"},
                    )
                ),
            },
            bot_uid="U_BOT",
        )

        assert "Build failed on" in rendered
        assert "see logs" in rendered

    def test_workspace_mention_is_not_read_as_new_content(self):
        """A workspace mention renders into the flat form Slack sends."""
        blocks = _rich_text_blocks(
            _rich_text_section(
                {"type": "user", "user_id": "U_BOT"},
                {"type": "text", "text": " ping "},
                {"type": "team", "team_id": "T123"},
                {"type": "text", "text": " now"},
            )
        )

        assert (
            _slack_mod._extract_additional_text_from_slack_blocks(
                blocks, "<@U_BOT> ping <!team^T123> now", bot_uid="U_BOT"
            )
            == ""
        )

    def test_color_element_is_not_read_as_new_content(self):
        """The composer keeps the typed hex code in the flat text."""
        blocks = _rich_text_blocks(
            _rich_text_section(
                {"type": "user", "user_id": "U_BOT"},
                {"type": "text", "text": " brand is "},
                {"type": "color", "value": "#FF0000"},
                {"type": "text", "text": " ok?"},
            )
        )

        assert (
            _slack_mod._extract_additional_text_from_slack_blocks(
                blocks, "<@U_BOT> brand is #FF0000 ok?", bot_uid="U_BOT"
            )
            == ""
        )

    @pytest.mark.parametrize(
        ("element", "flat"),
        [
            (
                {"type": "channel", "channel_id": "C024BE7LR"},
                "<@U_BOT> see <#C024BE7LR|general> please",
            ),
            (
                {"type": "usergroup", "usergroup_id": "SAZ94GDB8"},
                "<@U_BOT> see <!subteam^SAZ94GDB8|@marketing> please",
            ),
            (
                {"type": "user", "user_id": "U024BE7LH"},
                "<@U_BOT> see <@U024BE7LH|nikita> please",
            ),
            (
                {"type": "broadcast", "range": "here"},
                "<@U_BOT> see <!here|@here> please",
            ),
        ],
    )
    def test_labelled_mention_is_not_read_as_new_content(self, element, flat):
        """Slack may label any mention in the flat text while the blocks carry
        the bare id."""
        blocks = _rich_text_blocks(
            _rich_text_section(
                {"type": "user", "user_id": "U_BOT"},
                {"type": "text", "text": " see "},
                element,
                {"type": "text", "text": " please"},
            )
        )

        assert (
            _slack_mod._extract_additional_text_from_slack_blocks(
                blocks, flat, bot_uid="U_BOT"
            )
            == ""
        )

    def test_section_of_a_single_untrusted_element_is_still_delivered(self):
        """Negative case: a mismatch is never a reason to drop content."""
        blocks = _rich_text_blocks(
            _rich_text_section({"type": "team", "team_id": "T123"})
        )

        assert _slack_mod._extract_additional_text_from_slack_blocks(
            blocks, "New alert", bot_uid="U_BOT"
        )

    @pytest.mark.parametrize(
        "flat",
        [
            "hey <@U_BOT|hermes> please look",
            "hey <@U_BOT> please look",
            "hey &lt;@U_BOT&gt; please look",
        ],
    )
    def test_labelled_bot_mention_is_not_read_as_new_content(self, flat):
        """The render drops the bot mention, so every flat form of it must be
        dropped from the flat text too."""
        blocks = _rich_text_blocks(
            _rich_text_section(
                {"type": "text", "text": "hey "},
                {"type": "user", "user_id": "U_BOT"},
                {"type": "text", "text": " please look"},
            )
        )

        assert (
            _slack_mod._extract_additional_text_from_slack_blocks(
                blocks, flat, bot_uid="U_BOT"
            )
            == ""
        )

    def test_non_http_scheme_link_is_not_read_as_new_content(self):
        """Autolinks are not limited to the schemes we happened to list."""
        blocks = _rich_text_blocks(
            _rich_text_section(
                {"type": "user", "user_id": "U_BOT"},
                {"type": "text", "text": " call "},
                {"type": "link", "url": "tel:+15551234567"},
                {"type": "text", "text": " now"},
            )
        )

        assert (
            _slack_mod._extract_additional_text_from_slack_blocks(
                blocks, "<@U_BOT> call <tel:+15551234567> now", bot_uid="U_BOT"
            )
            == ""
        )

    @pytest.mark.asyncio
    async def test_live_inbound_pasted_permalink_not_duplicated(self, adapter):
        await adapter._handle_slack_message(
            {
                "text": self._mention_text(),
                "blocks": self._mention_blocks(
                    {
                        "type": "message_mention",
                        "channel_id": "C0BCDG3H66P",
                        "message_ts": "1786102118.226679",
                        "url": _PERMALINK,
                        "text": _PERMALINK,
                    }
                ),
                "user": "U_USER",
                "client_msg_id": "cm-4",
                "channel": "D_DM",
                "channel_type": "im",
                "ts": "123.459",
                "team": "T_TEAM",
            }
        )

        text = adapter.handle_message.await_args.args[0].text
        # One line, and no second copy with the permalink blanked out.
        assert text.count("do you see") == 1
        assert "\n" not in text
        assert _PERMALINK in text

    def test_hydration_pasted_permalink_not_duplicated(self, adapter):
        rendered = adapter._render_message_text(
            {
                "text": self._mention_text(),
                "blocks": self._mention_blocks(
                    {
                        "type": "message_mention",
                        "channel_id": "C0BCDG3H66P",
                        "message_ts": "1786102118.226679",
                        "url": _PERMALINK,
                    }
                ),
            },
            bot_uid="U_BOT",
        )

        assert rendered.count("do you see") == 1
        assert "\n" not in rendered
        assert _PERMALINK in rendered

    def test_block_kit_dump_still_describes_bot_ui_blocks(self):
        """Negative case: UI-heavy bot blocks are why the dump exists."""
        payload = _slack_mod._serialize_slack_blocks_for_agent(
            [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "Deploy failed"},
                },
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "action_id": "rollback",
                            "text": {"type": "plain_text", "text": "Roll back"},
                        }
                    ],
                },
            ]
        )

        assert "Deploy failed" in payload
        assert "rollback" in payload
        assert "Roll back" in payload
