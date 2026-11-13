"""Tests for the Buzz adapter's @mention resolution (PR #83414).

Covers ``_mention_pubkeys_for`` / ``_channel_member_pubkeys`` and the
``send()`` recovery paths: membership-accurate candidate sourcing, token
boundaries on both sides of the @name, duplicate-name ambiguity, and the
non-member / unresolvable-token publish retries.
"""

import asyncio
import json

import pytest

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_buzz_mod = load_plugin_adapter("buzz")

BuzzAdapter = _buzz_mod.BuzzAdapter
_MENTION_LOOKUP_CAP = _buzz_mod._MENTION_LOOKUP_CAP

SELF_PUBKEY = "9fd5c7ba6d3ef224da78f541e0fcb9c50f72cc63edb19aae76ac6a0474dfa860"
FIZZ_PUBKEY = "b" * 64
BUZZ_PUBKEY = "c" * 64
DUPE_PUBKEY = "d" * 64
CHANNEL = "ccc2bc1a-7a82-5a8f-8c4e-57a070cbe7cd"

_ENV_VARS = (
    "BUZZ_RELAY_URL",
    "BUZZ_PRIVATE_KEY",
    "BUZZ_CHANNELS",
    "BUZZ_HOME_CHANNEL",
    "BUZZ_ALLOWED_USERS",
    "BUZZ_ALLOW_ALL_USERS",
    "BUZZ_POLL_INTERVAL",
    "BUZZ_CLI_PATH",
    "BUZZ_CREDENTIALS_FILE",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(_buzz_mod, "_DEFAULT_CREDENTIALS_DIR", tmp_path / "no-creds")
    yield


def _make_adapter(extra=None):
    from gateway.config import PlatformConfig

    cfg = PlatformConfig(enabled=True, extra={"relay_url": "https://test.relay", **(extra or {})})
    adapter = BuzzAdapter(cfg)
    adapter._self_pubkey = SELF_PUBKEY
    adapter._private_key = "nsec1test"
    return adapter


class _ScriptedCli:
    """Fake ``_run_cli`` that routes on the buzz subcommand and records calls."""

    def __init__(self):
        self.responses = {}
        self.calls = []

    def script(self, group, cmd, payload, code=0, stderr=""):
        stdout = payload if isinstance(payload, str) else json.dumps(payload)
        self.responses.setdefault((group, cmd), []).append((code, stdout, stderr))

    async def __call__(self, args, *, input_text=None):
        self.calls.append((list(args), input_text))
        queue = self.responses.get((args[0], args[1]), [])
        if len(queue) > 1:
            return queue.pop(0)
        if queue:
            return queue[0]
        return 0, "[]", ""


def _wire(adapter, cli):
    adapter._run_cli = cli
    return cli


def _members(*pubkeys):
    return [{"pubkey": pk} for pk in pubkeys]


def _profile(pubkey, name):
    return [{"pubkey": pubkey, "display_name": name}]


# ── candidate sourcing ────────────────────────────────────────────────────


class TestChannelMemberPubkeys:

    @pytest.mark.asyncio
    async def test_members_subcommand_is_primary_source(self):
        adapter = _make_adapter()
        cli = _wire(adapter, _ScriptedCli())
        cli.script("channels", "members", _members(FIZZ_PUBKEY, BUZZ_PUBKEY))

        pks = await adapter._channel_member_pubkeys(CHANNEL)

        assert pks == [FIZZ_PUBKEY, BUZZ_PUBKEY]
        assert ("messages", "get") not in {(c[0][0], c[0][1]) for c in cli.calls}

    @pytest.mark.asyncio
    async def test_falls_back_to_recent_traffic_when_members_unavailable(self):
        adapter = _make_adapter()
        cli = _wire(adapter, _ScriptedCli())
        cli.script("channels", "members", "", code=1, stderr="unknown subcommand")
        cli.script(
            "messages",
            "get",
            [
                {"pubkey": FIZZ_PUBKEY, "tags": [["p", BUZZ_PUBKEY]]},
            ],
        )

        pks = await adapter._channel_member_pubkeys(CHANNEL)

        assert FIZZ_PUBKEY in pks and BUZZ_PUBKEY in pks


# ── token matching ────────────────────────────────────────────────────────


class TestMentionTokenMatching:

    async def _resolve(self, content, members=None, profiles=None):
        adapter = _make_adapter()
        cli = _wire(adapter, _ScriptedCli())
        cli.script("channels", "members", _members(*(members or [FIZZ_PUBKEY])))
        for pk, name in (profiles or {FIZZ_PUBKEY: "Fizz"}).items():
            cli.script("users", "get", _profile(pk, name))
        return await adapter._mention_pubkeys_for(CHANNEL, content)

    @pytest.mark.asyncio
    async def test_clean_mention_resolves(self):
        assert await self._resolve("hey @Fizz, ping") == [FIZZ_PUBKEY]

    @pytest.mark.asyncio
    async def test_trailing_punctuation_resolves(self):
        assert await self._resolve("@Fizz!! wake up") == [FIZZ_PUBKEY]

    @pytest.mark.asyncio
    async def test_right_boundary_rejects_longer_token(self):
        assert await self._resolve("@FizzBuzz is a game") == []

    @pytest.mark.asyncio
    async def test_left_boundary_rejects_email_like_text(self):
        assert await self._resolve("mail me at email@Fizz today") == []

    @pytest.mark.asyncio
    async def test_left_boundary_rejects_adjacent_word_char(self):
        assert await self._resolve("x@Fizz") == []

    @pytest.mark.asyncio
    async def test_left_boundary_rejects_double_at(self):
        assert await self._resolve("@@Fizz") == []

    @pytest.mark.asyncio
    async def test_left_boundary_is_unicode_aware(self):
        # 山 is a word character: "山田@Fizz" is email-shaped text in a
        # non-ASCII script, not a mention.
        assert await self._resolve("山田@Fizz") == []


    @pytest.mark.asyncio
    async def test_longest_name_wins_and_consumes_span(self):
        result = await self._resolve(
            "@Hermes Matt please review",
            members=[FIZZ_PUBKEY, BUZZ_PUBKEY],
            profiles={FIZZ_PUBKEY: "Hermes Matt", BUZZ_PUBKEY: "Hermes"},
        )
        assert result == [FIZZ_PUBKEY]

    @pytest.mark.asyncio
    async def test_duplicate_display_names_tag_nobody(self):
        result = await self._resolve(
            "@Fizz which one of you is real",
            members=[FIZZ_PUBKEY, DUPE_PUBKEY],
            profiles={FIZZ_PUBKEY: "Fizz", DUPE_PUBKEY: "Fizz"},
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_self_is_never_a_candidate(self):
        result = await self._resolve(
            "@Chip and @Fizz",
            members=[SELF_PUBKEY, FIZZ_PUBKEY],
            profiles={SELF_PUBKEY: "Chip", FIZZ_PUBKEY: "Fizz"},
        )
        assert result == [FIZZ_PUBKEY]


# ── caching ───────────────────────────────────────────────────────────────


class TestResolutionCaching:

    @pytest.mark.asyncio
    async def test_member_list_cached_within_ttl(self):
        adapter = _make_adapter()
        cli = _wire(adapter, _ScriptedCli())
        cli.script("channels", "members", _members(FIZZ_PUBKEY))
        cli.script("users", "get", _profile(FIZZ_PUBKEY, "Fizz"))

        assert await adapter._mention_pubkeys_for(CHANNEL, "@Fizz one") == [FIZZ_PUBKEY]
        assert await adapter._mention_pubkeys_for(CHANNEL, "@Fizz two") == [FIZZ_PUBKEY]

        member_calls = [c for c in cli.calls if (c[0][0], c[0][1]) == ("channels", "members")]
        assert len(member_calls) == 1, "second resolve must hit the member cache"

    @pytest.mark.asyncio
    async def test_profile_name_expires_after_ttl(self, monkeypatch):
        adapter = _make_adapter()
        cli = _wire(adapter, _ScriptedCli())
        cli.script("channels", "members", _members(FIZZ_PUBKEY))
        cli.script("users", "get", _profile(FIZZ_PUBKEY, "Fizz"))
        cli.script("users", "get", _profile(FIZZ_PUBKEY, "FizzRenamed"))

        clock = [1000.0]
        monkeypatch.setattr(_buzz_mod.time, "monotonic", lambda: clock[0])

        assert await adapter._mention_pubkeys_for(CHANNEL, "@Fizz hi") == [FIZZ_PUBKEY]
        # Inside both TTLs: rename not visible yet, no new lookups needed.
        assert await adapter._mention_pubkeys_for(CHANNEL, "@FizzRenamed hi") == []
        # Past the name TTL (and member TTL): the rename resolves.
        clock[0] += _buzz_mod._PROFILE_NAME_TTL + 1
        assert await adapter._mention_pubkeys_for(CHANNEL, "@FizzRenamed hi") == [FIZZ_PUBKEY]


# ── send() recovery paths ─────────────────────────────────────────────────


class TestSendRecovery:

    def _sending_adapter(self):
        adapter = _make_adapter()
        cli = _wire(adapter, _ScriptedCli())
        cli.script("channels", "members", _members(FIZZ_PUBKEY))
        cli.script("users", "get", _profile(FIZZ_PUBKEY, "Fizz"))
        return adapter, cli

    def _send_calls(self, cli):
        return [c for c in cli.calls if (c[0][0], c[0][1]) == ("messages", "send")]

    @pytest.mark.asyncio
    async def test_resolved_mention_attached_to_publish(self):
        adapter, cli = self._sending_adapter()
        cli.script("messages", "send", {"accepted": True, "event_id": "e1"})

        result = await adapter.send(CHANNEL, "@Fizz hello")

        assert result.success
        sends = self._send_calls(cli)
        assert len(sends) == 1
        assert ["--mention", FIZZ_PUBKEY] == sends[0][0][-2:]

    @pytest.mark.asyncio
    async def test_non_member_mention_retries_without_mentions(self):
        adapter, cli = self._sending_adapter()
        cli.script(
            "messages", "send", "",
            code=1,
            stderr=json.dumps({"error": "user_error", "message": "mentioned pubkeys are not channel members"}),
        )
        cli.script("messages", "send", {"accepted": True, "event_id": "e2"})

        result = await adapter.send(CHANNEL, "@Fizz hello")

        assert result.success
        sends = self._send_calls(cli)
        assert len(sends) == 2
        assert "--mention" in sends[0][0]
        assert "--mention" not in sends[1][0]

    @pytest.mark.asyncio
    async def test_unresolvable_token_retries_with_self_mention(self):
        adapter = _make_adapter()
        cli = _wire(adapter, _ScriptedCli())
        cli.script("channels", "members", _members())  # nobody to resolve
        unresolved = json.dumps({
            "error": "user_error",
            "message": "mention '@-mention' does not match a current channel member",
        })
        # Composed ladder (#82646 + #83414): the presentation-escape retry
        # fires first; when the CLI still rejects, the self-mention downgrade
        # is the last rung.
        cli.script("messages", "send", "", code=1, stderr=unresolved)
        cli.script("messages", "send", "", code=1, stderr=unresolved)
        cli.script("messages", "send", {"accepted": True, "event_id": "e3"})

        result = await adapter.send(CHANNEL, "just @-mention me")

        assert result.success
        sends = self._send_calls(cli)
        assert len(sends) == 3
        # Rung 3: escaped presentation token, no mention flags.
        assert "--mention" not in sends[1][0]
        assert "\u200b" in (sends[1][1] or "")
        # Rung 4: self-mention downgrade with the original content.
        assert ["--mention", SELF_PUBKEY] == sends[2][0][-2:]
        assert sends[2][1] == "just @-mention me"

    @pytest.mark.asyncio
    async def test_unresolvable_token_escape_retry_delivers(self):
        adapter = _make_adapter()
        cli = _wire(adapter, _ScriptedCli())
        cli.script("channels", "members", _members())
        cli.script(
            "messages", "send", "",
            code=1,
            stderr=json.dumps({
                "error": "user_error",
                "message": "mention '@-mention' does not match a current channel member",
            }),
        )
        cli.script("messages", "send", {"accepted": True, "event_id": "e4"})

        result = await adapter.send(CHANNEL, "just @-mention me")

        assert result.success
        sends = self._send_calls(cli)
        assert len(sends) == 2
        assert "--mention" not in sends[1][0]
        assert sends[1][1] == "just @\u200b-mention me"


# ── inbound: @Name that starts a longer member name ───────────────────────

CHIP_SALES_PUBKEY = "e" * 64
SENDER_PUBKEY = "a" * 64
CHIP_ING_PUBKEY = "f" * 64


class TestLongerMemberNameMention:

    @pytest.fixture
    def adapter(self):
        a = _make_adapter()
        a._display_name = "Chip"
        a.set_authorization_check(lambda *_args: True)
        a._dispatched = []

        async def capture(**kwargs):
            a._dispatched.append(kwargs)

        a._dispatch_message = capture
        a._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}
        cli = _wire(a, _ScriptedCli())
        cli.script("channels", "members", _members(SELF_PUBKEY, CHIP_SALES_PUBKEY))
        cli.script("users", "get", _profile(CHIP_SALES_PUBKEY, "Chip Sales"))
        a._cli = cli
        return a

    async def _poll(self, adapter, content, p_tags=(), then_empty=False):
        event = {
            "id": "e1", "pubkey": SENDER_PUBKEY, "content": content, "created_at": 10, "kind": 9,
            "tags": [["h", CHANNEL], *(["p", pk] for pk in p_tags)],
        }
        adapter._cli.script("messages", "get", [event])
        if then_empty:  # roster fallback reads recent traffic after the poll consumed the event
            adapter._cli.script("messages", "get", [])
        await adapter._poll_channel(CHANNEL)

    @pytest.mark.asyncio
    async def test_longer_name_mention_does_not_wake_agent(self, adapter):
        await self._poll(adapter, "@Chip Sales hola")
        assert adapter._dispatched == []

    @pytest.mark.asyncio
    async def test_plain_mention_still_dispatches_and_strips(self, adapter):
        await self._poll(adapter, "@Chip hola")
        assert [d["text"] for d in adapter._dispatched] == ["hola"]

    @pytest.mark.asyncio
    async def test_second_plain_mention_survives_longer_name(self, adapter):
        await self._poll(adapter, "@Chip Sales y @Chip, hola")
        assert len(adapter._dispatched) == 1

    @pytest.mark.asyncio
    async def test_longer_name_from_p_tags_when_roster_empty(self, adapter):
        cli = adapter._cli
        cli.responses[("channels", "members")] = [(0, "[]", "")]
        await self._poll(adapter, "@Chip Sales hola", p_tags=[CHIP_SALES_PUBKEY], then_empty=True)
        assert adapter._dispatched == []

    @pytest.mark.asyncio
    async def test_no_lookups_without_self_name_in_text(self, adapter):
        await self._poll(adapter, "hello everyone")
        assert adapter._dispatched == []
        assert {(c[0][0], c[0][1]) for c in adapter._cli.calls} <= {("messages", "get")}

    def test_strip_mention_leaves_longer_name_alone(self, adapter):
        assert adapter._strip_mention("@Chip Sales /whoami", ["Chip Sales"]) == "@Chip Sales /whoami"
        assert adapter._strip_mention("@Chip /whoami", ["Chip Sales"]) == "/whoami"

    @pytest.mark.asyncio
    async def test_name_lookups_are_bounded_per_event(self, adapter):
        cli = adapter._cli
        cli.responses[("channels", "members")] = [(0, "[]", "")]
        valid = [f"{i:02x}" * 32 for i in range(40)]
        junk = [f"junk-{i}" for i in range(10)]
        # No roster: the fallback re-reads recent traffic, which includes this very event.
        await self._poll(adapter, "@Chip hola", p_tags=junk[:5] + valid + junk[5:])
        looked_up = [c[0] for c in cli.calls if c[0][:2] == ["users", "get"] and c[0][3] != SENDER_PUBKEY]  # sender is resolved after the gate
        assert 0 < len(looked_up) <= _MENTION_LOOKUP_CAP
        assert not any(j in a for c in cli.calls for a in c[0] for j in junk)
        assert [d["text"] for d in adapter._dispatched] == ["hola"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "content",
        ["[]", {"display_name": "Chip Sales"}, "[" * 200_000],
        ids=["empty-list", "object-profile", "deeply-nested"],
    )
    async def test_malformed_profile_content_keeps_mention(self, adapter, content):
        adapter._cli.responses[("users", "get")] = [
            (0, json.dumps([{"pubkey": CHIP_SALES_PUBKEY, "display_name": "", "name": "", "content": content}]), "")
        ]
        await self._poll(adapter, "@Chip hola")
        assert [d["text"] for d in adapter._dispatched] == ["hola"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stdout", ['{"error": "not found"}', "42"], ids=["json-object", "json-number"])
    async def test_non_list_users_get_output_keeps_mention(self, adapter, stdout):
        adapter._cli.responses[("users", "get")] = [(0, stdout, "")]
        await self._poll(adapter, "@Chip hola")
        assert [d["text"] for d in adapter._dispatched] == ["hola"]

    @pytest.mark.asyncio
    async def test_p_tags_are_read_before_the_roster(self, adapter):
        cli = adapter._cli
        fillers = [f"{i:02x}" * 32 for i in range(_MENTION_LOOKUP_CAP)]
        cli.responses[("channels", "members")] = [(0, json.dumps(_members(*fillers, CHIP_SALES_PUBKEY)), "")]
        inner = cli.__call__

        async def by_pubkey(args, *, input_text=None):  # only the p-tagged member has a longer name
            code, out, err = await inner(args, input_text=input_text)
            if args[:2] == ["users", "get"]:
                hit = CHIP_SALES_PUBKEY in args
                return 0, (json.dumps(_profile(CHIP_SALES_PUBKEY, "Chip Sales")) if hit else "[]"), ""
            return code, out, err

        adapter._run_cli = by_pubkey
        await self._poll(adapter, "@Chip Sales hola", p_tags=[CHIP_SALES_PUBKEY])
        assert adapter._dispatched == []

    @pytest.mark.asyncio
    async def test_lookup_cli_failure_fails_open(self, adapter):
        cli = adapter._cli
        cli.responses[("channels", "members")] = [(1, "", "boom")]
        cli.responses[("messages", "get")] = [(1, "", "boom")]
        event = {
            "id": "e1", "pubkey": SENDER_PUBKEY, "content": "@Chip hola", "created_at": 10, "kind": 9,
            "tags": [["h", CHANNEL]],
        }
        await adapter._handle_events(CHANNEL, adapter._channel_state[CHANNEL], [event])
        assert [d["text"] for d in adapter._dispatched] == ["hola"]

    @pytest.mark.asyncio
    async def test_lookup_exception_keeps_mention(self, adapter):
        inner = adapter._run_cli

        async def raising(args, *, input_text=None):
            if args[:2] == ["users", "get"] and CHIP_SALES_PUBKEY in args:
                raise OSError("cannot spawn cli")
            return await inner(args, input_text=input_text)

        adapter._run_cli = raising
        await self._poll(adapter, "@Chip hola")
        assert [d["text"] for d in adapter._dispatched] == ["hola"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("failure", ["raises", "null"])
    async def test_roster_lookup_failure_keeps_mention(self, adapter, failure):
        inner = adapter._run_cli

        async def broken(args, *, input_text=None):
            if args[:2] == ["channels", "members"]:
                if failure == "raises":
                    raise OSError("cannot spawn cli")
                return 0, "null", ""
            return await inner(args, input_text=input_text)

        adapter._run_cli = broken
        await self._poll(adapter, "@Chip hola")
        assert [d["text"] for d in adapter._dispatched] == ["hola"]

    @pytest.mark.asyncio
    async def test_unauthorized_sender_triggers_no_name_lookups(self, adapter):
        adapter._allowed_pubkeys = {FIZZ_PUBKEY}
        adapter._reaction_only_pubkeys = set()
        await self._poll(adapter, "@Chip Sales hola", p_tags=[CHIP_SALES_PUBKEY])
        assert adapter._dispatched == []
        assert not [c for c in adapter._cli.calls if c[0][:2] in (["channels", "members"], ["users", "get"])]

    def _two_longer_members(self, adapter):
        cli = adapter._cli
        cli.responses[("channels", "members")] = [(0, json.dumps(_members(SELF_PUBKEY, CHIP_SALES_PUBKEY, CHIP_ING_PUBKEY)), "")]
        cli.script("users", "get", _profile(CHIP_ING_PUBKEY, "Chip Ing"))

    @pytest.mark.asyncio
    async def test_two_longer_names_do_not_wake_agent(self, adapter):
        self._two_longer_members(adapter)
        await self._poll(adapter, "@Chip Sales y @Chip Ing hola")
        assert adapter._dispatched == []

    @pytest.mark.asyncio
    async def test_bare_mention_after_two_longer_names_dispatches_intact(self, adapter):
        self._two_longer_members(adapter)
        await self._poll(adapter, "@Chip Sales, @Chip Ing y @Chip")
        assert [d["text"] for d in adapter._dispatched] == ["@Chip Sales, @Chip Ing y @Chip"]

    @pytest.mark.asyncio
    async def test_p_tag_to_self_still_wakes_on_longer_name(self, adapter):
        await self._poll(adapter, "@Chip Sales hola", p_tags=[SELF_PUBKEY])
        assert [d["text"] for d in adapter._dispatched] == ["@Chip Sales hola"]

    @pytest.mark.asyncio
    async def test_punctuation_only_suffix_is_not_a_longer_name(self, adapter):
        adapter._cli.responses[("users", "get")] = [(0, json.dumps(_profile(CHIP_SALES_PUBKEY, "Chip,")), "")]
        await self._poll(adapter, "@Chip, hola")
        assert [d["text"] for d in adapter._dispatched] == ["hola"]

    @pytest.mark.asyncio
    async def test_p_tags_outside_roster_are_not_looked_up(self, adapter):
        await self._poll(adapter, "@Chip hola", p_tags=[CHIP_ING_PUBKEY])
        assert [d["text"] for d in adapter._dispatched] == ["hola"]
        assert not any(c[0][:2] == ["users", "get"] and CHIP_ING_PUBKEY in c[0] for c in adapter._cli.calls)

    @pytest.mark.asyncio
    async def test_longer_name_check_uses_the_lowercased_name(self, adapter):
        # "İ".lower() is two characters, so slicing the original-case name by len(own) misses the suffix.
        adapter._display_name = "Aİ"
        other = "b" * 64
        adapter._cli.responses[("channels", "members")] = [(0, json.dumps(_members(SELF_PUBKEY, other)), "")]
        adapter._cli.responses[("users", "get")] = [(0, json.dumps(_profile(other, "Aİ7")), "")]
        event = {"id": "e1", "pubkey": SENDER_PUBKEY, "content": "@Aİ7 hola", "created_at": 10, "kind": 9, "tags": [["h", CHANNEL]]}
        assert await adapter._longer_member_names(CHANNEL, event) == ["aİ7".lower()]

    @pytest.mark.asyncio
    async def test_cancelled_roster_read_propagates(self, adapter):
        async def cancelled(channel_id):
            raise asyncio.CancelledError

        adapter._channel_member_pubkeys = cancelled
        event = {"id": "e1", "pubkey": SENDER_PUBKEY, "content": "@Chip hola", "created_at": 10, "kind": 9, "tags": [["h", CHANNEL]]}
        with pytest.raises(asyncio.CancelledError):
            await adapter._longer_member_names(CHANNEL, event)
