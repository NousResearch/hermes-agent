"""Discord ``$<profile>`` mention dispatch, END TO END: every case through the real gateway.

Each test sends a Discord message through ``discord_gateway_rig`` (real ``DiscordAdapter`` -> real
multiplexed ``GatewayRunner`` -> the session authority's real admission and provenance binding ->
the real turn setup and reply delivery; only the model call and the Discord wire are stand-ins)
and asserts on what actually ran: which profile's home and session got a turn, with what prompt,
what the channel was sent, and that nothing was refused.

Why this file exists: every earlier test of this feature checked the adapter's hand-over or called
``resolve_identity`` directly, and all of them passed while the gateway refused every dispatched
prompt (``AuthorityRefusal('profile_mismatch')`` from ``session_ingress_context._binding``; the
user got "Something went wrong"). Test a change to this feature here first.

The cases are issue #159776's acceptance table (cases 1-6), the review's findings and each later
round's fix (duplicate dispatch with two bots, prompt formatting, slash commands, bare handles with
backfill, attachment text, handle-prefixed commands, skill commands), plus the routing shapes the
binding has to accept (a secondary profile's bot, a custom handle, a busy session, a
``profile_routes`` channel, missed-message recovery).
"""
import pytest
import pytest_asyncio

from tests.plugins.platforms.discord_gateway_rig import discord_gateway_rig, ensure_discord_mock, text_attachment

ensure_discord_mock()

import plugins.platforms.discord.adapter_profile_mentions as profile_mentions

#: ``agent-one``/``agent-two`` opted in through their own config.yaml; ``quiet`` is served but did not.
MENTIONABLE = {"agent-one": {"mentionable": True}, "agent-two": {"mentionable": True}, "quiet": None}
_DOC = "# Runbook\n\nPage $agent-two if prod is down.\n  Indented: costs $agent-one-ish, $5\n"


@pytest_asyncio.fixture
async def rig(tmp_path, monkeypatch, authority_cwd_in_tmp):
    """A free-response channel (every message reaches the gateway), one bot owned by ``default``."""
    async with discord_gateway_rig(tmp_path, monkeypatch, profiles=MENTIONABLE) as rig:
        yield rig


def _ran_in(rig, out, profile, prompt):
    """*out* is exactly one turn, in *profile*'s own home and session, with *prompt*, and nothing refused."""
    assert out.refusals == [] and out.errors == [], out.summary()
    assert [(t.profile, t.message) for t in out.turns] == [(profile, prompt)], out.summary()
    (turn,) = out.turns
    assert turn.home == rig.home(profile), out.summary()
    expected_namespace = "main" if profile in (None, "default") else profile
    assert turn.session_key.startswith(f"agent:{expected_namespace}:discord:"), turn.session_key
    return turn


def _labelled(out, profile):
    """The reply the channel got carries the profile's label (and no failure notice)."""
    texts = out.reply_texts()
    assert texts and all("Something went wrong" not in t for t in texts), out.summary()
    assert any(f"— {profile}" in t for t in texts), out.summary()


# --- issue #159776 acceptance cases ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_case1_a_leading_handle_runs_one_labelled_turn_in_that_profile(rig):
    out = await rig.send("$agent-one what's the status?")
    _ran_in(rig, out, "agent-one", "what's the status?")
    _labelled(out, "agent-one")


@pytest.mark.asyncio
async def test_case2_a_handle_mid_message_routes_too(rig):
    out = await rig.send("hey $agent-one, any update?")
    _ran_in(rig, out, "agent-one", "hey, any update?")


@pytest.mark.asyncio
async def test_case3_several_handles_each_run_in_their_own_session(rig):
    out = await rig.send("$agent-one $agent-two compare your dependency staleness")
    assert out.refusals == [] and out.errors == [], out.summary()
    assert out.turns_by_profile() == {"agent-one": ["compare your dependency staleness"],
                                      "agent-two": ["compare your dependency staleness"]}, out.summary()
    homes = {t.profile: t.home for t in out.turns}
    assert homes == {"agent-one": rig.home("agent-one"), "agent-two": rig.home("agent-two")}
    assert len({t.session_id for t in out.turns}) == 2
    for profile in ("agent-one", "agent-two"):
        _labelled(out, profile)


@pytest.mark.asyncio
async def test_case4_a_message_without_a_handle_stays_in_the_channels_session(rig):
    out = await rig.send("anyone know if CI is green?")
    _ran_in(rig, out, None, "anyone know if CI is green?")
    assert all("— " not in t for t in out.reply_texts()), out.summary()


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["$not-a-real-profile ping", "$quiet hello"], ids=["unknown", "not-opted-in"])
async def test_case5_an_unknown_or_unmentionable_handle_is_ordinary_text(rig, content):
    out = await rig.send(content)
    _ran_in(rig, out, None, content)


@pytest.mark.asyncio
async def test_case6_a_bare_handle_without_backfill_runs_nothing(rig):
    out = await rig.send("$agent-one")
    assert out.handed == [] and out.turns == [] and out.refusals == [], out.summary()


@pytest.mark.asyncio
@pytest.mark.parametrize("busy", [True, False], ids=["channel-activity", "no-activity"])
async def test_case6_a_bare_handle_with_default_backfill_is_a_catch_me_up_turn(tmp_path, monkeypatch, authority_cwd_in_tmp, busy):
    """History backfill at its default (on): like a bare ``@bot``, a catch-me-up turn when the channel
    has activity, nothing otherwise; in agent-one's session."""
    async with discord_gateway_rig(tmp_path, monkeypatch, profiles=MENTIONABLE,
                                   env={"DISCORD_HISTORY_BACKFILL": None, "DISCORD_REQUIRE_MENTION": "true"}) as rig:
        history = ["CI went red on main"] if busy else []
        out = await rig.send("$agent-one", channel_history=history)
        if not busy:
            assert out.handed == [] and out.turns == [], out.summary()
            return
        assert out.refusals == [] and out.errors == [], out.summary()
        assert [t.profile for t in out.turns] == ["agent-one"], out.summary()
        (turn,) = out.turns
        assert turn.home == rig.home("agent-one") and "CI went red on main" in (turn.message or "")


# --- the review's findings and each later round's fix ---------------------------------------------


@pytest.mark.asyncio
async def test_two_bots_in_one_gateway_run_the_turn_once(tmp_path, monkeypatch, authority_cwd_in_tmp):
    """Round 1: both bots receive every message; each named profile gets ONE turn, in its own session.
    And always through the same bot: the gateway binds a session lane to the bot that carried its
    first turn and PAUSES the lane ("could not be re-authorized (not_found)") when a later turn
    arrives through another bot, so the bot that wins a later message's race must not matter."""
    profiles = {**MENTIONABLE, "agent-two": {"mentionable": True}}
    async with discord_gateway_rig(tmp_path, monkeypatch, profiles=profiles, bots=("default", "agent-two")) as rig:
        out = await rig.send("$agent-one status?")
        _ran_in(rig, out, "agent-one", "status?")
        assert [bot for bot, _ in out.replies] == ["default"], out.summary()
        for order in (["agent-two", "default"], ["default", "agent-two"], ["agent-two", "default"]):
            out = await rig.send("$agent-one $agent-two again", to=order)
            assert out.refusals == [] and out.errors == [], (order, out.summary())
            assert out.turns_by_profile() == {"agent-one": ["again"], "agent-two": ["again"]}, (order, out.summary())
            assert {t.profile: t.home for t in out.turns} == {p: rig.home(p) for p in ("agent-one", "agent-two")}
            assert all("paused" not in text for text in out.reply_texts()), (order, out.summary())
            # agent-one's lane stays on the bot that carried its first turn
            assert sorted(bot for bot, _ in out.replies) == ["agent-two", "default"], (order, out.summary())

        profile_mentions._CLAIMS.clear()  # a restart: the in-memory claims are gone, the session store is not
        out = await rig.send("$agent-one after a restart", to=["agent-two", "default"])
        _ran_in(rig, out, "agent-one", "after a restart")
        assert [bot for bot, _ in out.replies] == ["default"], out.summary()


@pytest.mark.asyncio
async def test_two_bots_racing_on_a_new_lane_keep_it_on_one_bot(tmp_path, monkeypatch, authority_cwd_in_tmp):
    """Two quick messages to a profile with no session in the channel yet, the bots seeing them in
    opposite orders: the second arrives before the first one's turn has created the lane, so only
    the in-memory lane claim keeps both on the bot that took the first."""
    async with discord_gateway_rig(tmp_path, monkeypatch, profiles=MENTIONABLE, bots=("default", "agent-two")) as rig:
        out = await rig.send("$agent-one first", to=["default", "agent-two"], wait=False)
        await rig.send("$agent-one second", to=["agent-two", "default"], wait=False, into=out)
        await rig.settle()
        assert out.refusals == [] and out.errors == [], out.summary()
        assert sorted(t.message for t in out.turns if t.profile == "agent-one") == ["first", "second"], out.summary()
        assert len(out.turns) == 2, out.summary()
        assert [bot for bot, _ in out.replies] == ["default", "default"], out.summary()


_FENCED = "$agent-one\n```py\ndef f():\n    return  1\n```\n\n  keep   this   spacing\n"


@pytest.mark.asyncio
async def test_code_fences_and_whitespace_reach_the_profile_intact(rig):
    """Round 2: only the handle goes; the code block and inner spacing arrive byte for byte."""
    out = await rig.send(_FENCED)
    _ran_in(rig, out, "agent-one", "```py\ndef f():\n    return  1\n```\n\n  keep   this   spacing")


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["/reset $agent-one", "<@999> /reset $agent-one"])
async def test_a_slash_command_with_a_handle_runs_in_the_channels_session(rig, content):
    """Round 3: the command runs where it would without the handle; agent-one's session is untouched."""
    out = await rig.send(content)
    assert out.refusals == [] and out.errors == [], out.summary()
    assert [(c, p, h) for c, p, h in out.commands] == [("reset", None, rig.home(None))], out.summary()
    assert out.turns == [], out.summary()


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["$agent-one /new", "$agent-one: /reset", "$agent-one\n/compact",
                                     "$agent-one $agent-two /new", "$Agent-One   /model x"])
async def test_a_handle_in_front_of_a_slash_command_never_runs_it_in_that_profile(rig, content):
    """Round 8: a plain-text turn in the channel's own session; no command in agent-one's home."""
    out = await rig.send(content)
    assert out.commands == [], out.summary()
    _ran_in(rig, out, None, content)


@pytest.mark.asyncio
async def test_the_rig_can_run_a_command_in_agent_ones_session(rig):
    """Why the round-8 rule matters, proven in this rig: the copy dispatch used to make of
    ``$agent-one /new`` (a TEXT event whose text is ``/new``, re-homed to agent-one) runs the real
    ``/new`` in agent-one's own home. Commands skip the FIFO admission, so no binding stops it."""
    import dataclasses

    from gateway.platforms.event import MessageEvent
    source = rig.bots["default"].build_source(chat_id="555", chat_type="group", user_id="42", guild_id="777")
    copy = MessageEvent(text="/new", message_id="900",
                        source=dataclasses.replace(source, profile="agent-one", profile_mention=True))
    out = await rig.hand_over(copy)
    assert out.commands == [("reset", "agent-one", rig.home("agent-one"))], out.summary()


@pytest.mark.asyncio
async def test_a_slash_later_in_the_prompt_still_dispatches(rig):
    out = await rig.send("$agent-one please explain what /new does")
    _ran_in(rig, out, "agent-one", "please explain what /new does")
    assert out.commands == []


@pytest.mark.asyncio
async def test_a_handle_inside_an_attached_document_never_routes(rig):
    """Round 5: the document's ``$agent-two`` addresses nobody; the turn is the channel's, document intact."""
    attachment = text_attachment(rig, _DOC)
    out = await rig.send("please summarize", attachments=[attachment])
    turn = _ran_in(rig, out, None, out.turns[0].message if out.turns else None)
    assert _DOC.strip() in turn.message and turn.message.endswith("please summarize"), out.summary()


@pytest.mark.asyncio
async def test_a_typed_handle_routes_with_the_attached_document_intact(rig):
    attachment = text_attachment(rig, _DOC)
    out = await rig.send("$agent-one summarize this", attachments=[attachment])
    turn = _ran_in(rig, out, "agent-one", out.turns[0].message if out.turns else None)
    assert _DOC.strip() in turn.message and turn.message.endswith("summarize this"), out.summary()
    assert "$agent-one summarize" not in turn.message

    out = await rig.send("$agent-one", attachments=[text_attachment(rig, _DOC)])  # the document is the prompt
    turn = _ran_in(rig, out, "agent-one", out.turns[0].message if out.turns else None)
    assert _DOC.strip() in turn.message and "summarize" not in turn.message, out.summary()


@pytest.mark.asyncio
async def test_pin_known_gap_a_skill_slash_command_keeps_its_handle_as_text(rig, monkeypatch):
    """Post-review check (round 7), a documented limitation: ``/demo $agent-one X`` runs the skill in
    the channel's own session with the handle left as text. A tripwire: update, don't revert."""
    import agent.skill_bundles as bundles
    import agent.skill_commands as skills
    from agent import skill_utils
    monkeypatch.setattr(skills, "get_skill_commands", lambda: {"/demo": {"name": "demo"}})
    monkeypatch.setattr(skills, "resolve_skill_command_key", lambda command, **_: "/demo" if command == "demo" else None)
    monkeypatch.setattr(skills, "split_stacked_skill_commands", lambda rest, **_: ([], rest))
    monkeypatch.setattr(skills, "build_skill_invocation_message", lambda key, text="", **_: f"[demo skill] {text}")
    monkeypatch.setattr(bundles, "resolve_bundle_command_key", lambda command: None)
    monkeypatch.setattr(skill_utils, "get_disabled_skill_names", lambda **_: set())
    out = await rig.send("/demo $agent-one summarize X")
    _ran_in(rig, out, None, "[demo skill] $agent-one summarize X")


# --- routing shapes the gateway's binding has to accept --------------------------------------------


@pytest.mark.asyncio
async def test_a_require_mention_channel_lets_a_handle_through_to_the_profile(tmp_path, monkeypatch, authority_cwd_in_tmp):
    async with discord_gateway_rig(tmp_path, monkeypatch, profiles=MENTIONABLE,
                                   env={"DISCORD_REQUIRE_MENTION": "true"}) as rig:
        out = await rig.send("$agent-one status?")
        _ran_in(rig, out, "agent-one", "status?")
        for content in ("anyone know if CI is green?", "$not-a-real-profile ping", "/reset $agent-one"):
            out = await rig.send(content)
            assert out.handed == [] and out.turns == [] and out.commands == [], (content, out.summary())


@pytest.mark.asyncio
async def test_a_custom_handle_and_the_bots_own_profile(tmp_path, monkeypatch, authority_cwd_in_tmp):
    """``mention_handle: a1`` reaches agent-one; the bot's own (launch) profile can be named too."""
    async with discord_gateway_rig(tmp_path, monkeypatch, profiles={"agent-one": {"mentionable": True, "mention_handle": "a1"}},
                                   default_discord={"mentionable": True}) as rig:
        _ran_in(rig, await rig.send("$a1 hi"), "agent-one", "hi")
        _ran_in(rig, await rig.send("$agent-one hi"), None, "$agent-one hi")  # the name is not its handle
        _ran_in(rig, await rig.send("$default hi"), "default", "hi")


@pytest.mark.asyncio
async def test_a_secondary_profiles_bot_hands_a_message_to_another_profile(tmp_path, monkeypatch, authority_cwd_in_tmp):
    """Only agent-two's bot is in the channel: ``$agent-one`` runs in agent-one, received (and
    authorized) by agent-two's bot; ``$agent-two`` runs in agent-two's own session."""
    async with discord_gateway_rig(tmp_path, monkeypatch, profiles=MENTIONABLE, bots=("default", "agent-two")) as rig:
        out = await rig.send("$agent-one status?", to=["agent-two"])
        _ran_in(rig, out, "agent-one", "status?")
        assert {bot for bot, _ in out.replies} == {"agent-two"}, out.summary()
        _ran_in(rig, await rig.send("$agent-two status?", to=["agent-two"]), "agent-two", "status?")


@pytest.mark.asyncio
async def test_a_follow_up_while_the_profile_is_busy_is_not_refused(rig):
    """A second ``$agent-one`` message while agent-one's first turn is still running."""
    from tests.waiting import async_wait_until
    gate = rig.hold_turns()
    out = await rig.send("$agent-one first", wait=False)
    await async_wait_until(lambda: out.turns or out.refusals or out.errors, timeout=15, what="the first turn")
    await rig.send("$agent-one second", wait=False, into=out)
    gate.set()
    await rig.settle()
    assert out.refusals == [] and out.errors == [], out.summary()
    assert [t.profile for t in out.turns] == ["agent-one", "agent-one"], out.summary()
    assert {t.home for t in out.turns} == {rig.home("agent-one")}
    assert "second" in out.turns[1].message, out.summary()


@pytest.mark.asyncio
async def test_a_handle_in_a_profile_routes_channel_wins_over_the_route(rig):
    """The channel routes to agent-two (``profile_routes``); naming agent-one addresses agent-one,
    while an unaddressed message keeps following the route."""
    from gateway.profile_routing import ProfileRoute
    rig.runner.config.profile_routes = [ProfileRoute(name="r", platform="discord", profile="agent-two", chat_id="555")]
    _ran_in(rig, await rig.send("plain"), "agent-two", "plain")
    _ran_in(rig, await rig.send("$agent-one status?"), "agent-one", "status?")


@pytest.mark.asyncio
async def test_a_recovered_missed_message_runs_in_the_named_profile(rig, monkeypatch):
    """Missed-message recovery re-applies the gates itself, then dispatches like a live message."""
    bot = rig.bots["default"]
    monkeypatch.setattr(bot, "_discord_message_admission", lambda message, claim: (True, False))
    monkeypatch.setattr(bot, "_in_bot_thread", lambda message: False)
    _ran_in(rig, await rig.recover("$agent-two rerun the audit"), "agent-two", "rerun the audit")


# --- what the session binding trusts ----------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(("profile", "marked", "runs"), [
    ("agent-one", True, True),     # what the bot dispatches: admitted
    ("agent-one", False, False),   # a stamped profile alone proves nothing
    ("quiet", True, False),        # served, but never opted in
], ids=["marked-opted-in", "unmarked", "marked-not-opted-in"])
async def test_the_binding_admits_only_a_marked_copy_for_an_opted_in_profile(rig, profile, marked, runs):
    """The gateway re-homes a copy only when the receiving bot marked it in process
    (``profile_mention``, never serialized) AND the bot still vouches for the profile (served and
    opted in: ``profile_mention_targets``). Anything else is ``profile_mismatch``, as before."""
    import dataclasses

    from gateway.platforms.event import MessageEvent
    bot = rig.bots["default"]
    source = dataclasses.replace(bot.build_source(chat_id="555", chat_type="group", user_id="42", guild_id="777"),
                                 profile=profile, profile_mention=marked)
    out = await rig.hand_over(MessageEvent(text="hi", source=source, message_id="900"))
    if runs:
        _ran_in(rig, out, profile, "hi")
        return
    assert out.turns == [], out.summary()
    assert [r.reason for r in out.refusals] == ["profile_mismatch"], out.summary()
    assert out.refusals[0].where == "_binding", out.summary()
