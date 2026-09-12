"""Tests for the Buzz adapter's @mention resolution (PR #83414).

Covers ``_mention_pubkeys_for`` / ``_channel_member_pubkeys`` and the
``send()`` recovery paths: membership-accurate candidate sourcing, token
boundaries on both sides of the @name, duplicate-name ambiguity, and the
non-member / unresolvable-token publish retries.
"""

import json

import pytest

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_buzz_mod = load_plugin_adapter("buzz")

BuzzAdapter = _buzz_mod.BuzzAdapter

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
    async def test_no_at_sign_short_circuits_without_cli_calls(self):
        adapter = _make_adapter()
        cli = _wire(adapter, _ScriptedCli())
        assert await adapter._mention_pubkeys_for(CHANNEL, "no mentions here") == []
        assert cli.calls == []

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
