"""Mirror eligibility for origin-fallback and explicit cron delivery targets.

Field report (enterprise, 2026-08-17): `cron.mirror_delivery: true` with
`deliver: origin` delivered the brief to Slack but never appended it to the
reply-facing gateway session, so a user reply hit a session with no context.

Root cause: the job was created by a provisioning script, so it carries no
captured origin. `deliver: origin` falls back to the home channel — the
user's actual conversation — but `_target_matches_origin` returns False for
an empty origin, so the mirror and the in_channel seed never fire. The June
origin-scoping refactor (c06ceb3232) correctly excluded broadcasts, but the
origin-FALLBACK target is not a broadcast: it is the best available stand-in
for the user's primary conversation.

Design under test:
- Delivery targets carry a `mirror_eligibility` tag set at resolution time:
  * origin match            -> eligible (unchanged)
  * origin-fallback (deliver=origin, no origin) -> eligible (NEW)
  * explicit platform:chat  -> eligible ONLY with per-job attach_to_session
    (NEW, opt-in; the global flag never activates explicit targets)
  * `all` / bare-platform expansion -> never eligible (unchanged invariant)
- Dedup across tokens (e.g. "origin,all" hitting the same chat) OR-merges
  eligibility so token order cannot strip it.
- The in_channel flat-session seed requires a DM-shaped target or a known
  user_id: group-channel session keys are user-isolated, and a seed without
  user_id would create an orphan session no reply ever resolves to.
"""

import pytest

from cron.scheduler import (
    _deliver_result,
    _resolve_delivery_targets,
    _target_mirror_eligible,
)


@pytest.fixture(autouse=True)
def _home_channel(monkeypatch):
    monkeypatch.setenv("SLACK_HOME_CHANNEL", "D0HOME")
    monkeypatch.delenv("TELEGRAM_HOME_CHANNEL", raising=False)
    monkeypatch.delenv("DISCORD_HOME_CHANNEL", raising=False)


class TestMirrorEligibilityResolution:
    def test_origin_target_is_eligible(self):
        job = {
            "deliver": "origin",
            "origin": {"platform": "slack", "chat_id": "D0AAA", "chat_type": "dm"},
        }
        targets = _resolve_delivery_targets(job)
        assert len(targets) == 1
        assert _target_mirror_eligible(job, targets[0], global_mirror=True)

    def test_origin_fallback_target_is_eligible(self):
        """deliver=origin with no captured origin: the home-channel fallback
        is the user's conversation, not a broadcast — mirror it."""
        job = {"deliver": "origin", "origin": None}
        targets = _resolve_delivery_targets(job)
        assert len(targets) == 1
        assert targets[0]["chat_id"] == "D0HOME"
        assert _target_mirror_eligible(job, targets[0], global_mirror=True)

    def test_all_expansion_is_never_eligible(self):
        """Broadcast targets stay unmirrored even with the global flag on."""
        job = {"deliver": "all", "origin": None}
        targets = _resolve_delivery_targets(job)
        assert targets, "home channel should expand from 'all'"
        for t in targets:
            assert not _target_mirror_eligible(job, t, global_mirror=True)

    def test_bare_platform_target_is_not_eligible(self):
        job = {"deliver": "slack", "origin": None}
        targets = _resolve_delivery_targets(job)
        assert len(targets) == 1
        assert not _target_mirror_eligible(job, targets[0], global_mirror=True)

    def test_explicit_target_not_eligible_under_global_flag(self):
        """Global mirror_delivery must not write sessions into arbitrary
        explicitly-addressed chats."""
        job = {"deliver": "slack:D0EXPL", "origin": None}
        targets = _resolve_delivery_targets(job)
        assert len(targets) == 1
        assert not _target_mirror_eligible(job, targets[0], global_mirror=True)

    def test_explicit_target_eligible_with_per_job_attach(self):
        """attach_to_session=true on the job is the author declaring the
        explicit target a conversation — managed per-user DM crons."""
        job = {
            "deliver": "slack:D0EXPL",
            "origin": None,
            "attach_to_session": True,
        }
        targets = _resolve_delivery_targets(job)
        assert len(targets) == 1
        assert _target_mirror_eligible(job, targets[0], global_mirror=False)

    def test_origin_and_all_dedup_keeps_eligibility(self):
        """'origin,all' resolving to the same home chat must not lose the
        fallback's eligibility to dedup order."""
        job = {"deliver": "origin,all", "origin": None}
        targets = _resolve_delivery_targets(job)
        # Home channel deduped to one target.
        slack_targets = [t for t in targets if t["platform"].lower() == "slack"]
        assert len(slack_targets) == 1
        assert _target_mirror_eligible(job, slack_targets[0], global_mirror=True)

    def test_all_and_origin_reversed_order_keeps_eligibility(self):
        job = {"deliver": "all,origin", "origin": None}
        targets = _resolve_delivery_targets(job)
        slack_targets = [t for t in targets if t["platform"].lower() == "slack"]
        assert len(slack_targets) == 1
        assert _target_mirror_eligible(job, slack_targets[0], global_mirror=True)

    def test_explicit_other_chat_with_origin_not_eligible(self):
        """An explicit target that is NOT the origin stays unmirrored under
        the global flag even when the job has an origin elsewhere."""
        job = {
            "deliver": "slack:D0OTHER",
            "origin": {"platform": "slack", "chat_id": "D0AAA", "chat_type": "dm"},
        }
        targets = _resolve_delivery_targets(job)
        assert len(targets) == 1
        assert not _target_mirror_eligible(job, targets[0], global_mirror=True)


class TestFallbackMirrorEndToEnd:
    """Drive _deliver_result with a stubbed sender + mirror recorder."""

    @pytest.fixture()
    def slack_env(self, monkeypatch, tmp_path):
        home = tmp_path / "hermes-home"
        home.mkdir()
        (home / "config.yaml").write_text(
            "cron:\n  mirror_delivery: true\n"
            "platforms:\n  slack:\n    enabled: true\n    token: xoxb-test\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(home))

        send_calls = []

        async def fake_sender(pconfig, chat_id, message, *, thread_id=None,
                              media_files=None, force_document=False, caption=None):
            send_calls.append({"chat_id": chat_id, "thread_id": thread_id})
            return {"success": True, "chat_id": chat_id, "message_id": "1.2"}

        import gateway.platform_registry as reg
        import hermes_cli.plugins as hp

        entry = reg.platform_registry.get("slack")
        if entry is None:
            hp.discover_plugins()
            entry = reg.platform_registry.get("slack")
        if entry is None:
            pytest.skip("slack platform entry not registered")
        monkeypatch.setattr(entry, "standalone_sender_fn", fake_sender)
        monkeypatch.setattr(hp, "discover_plugins", lambda *a, **k: None)

        mirror_calls = []

        import cron.scheduler as sched

        def fake_mirror(platform, chat_id, text, source_label="cli",
                        thread_id=None, user_id=None, role="assistant"):
            mirror_calls.append({
                "platform": platform, "chat_id": chat_id,
                "thread_id": thread_id, "user_id": user_id, "role": role,
            })
            return True

        import gateway.mirror as mirror_mod

        monkeypatch.setattr(mirror_mod, "mirror_to_session", fake_mirror)
        return {"send": send_calls, "mirror": mirror_calls}

    def test_origin_fallback_job_mirrors_brief(self, slack_env):
        """The field repro: managed cron, deliver=origin, no origin captured.
        The brief must be mirrored into the home-channel session."""
        job = {"id": "j1", "name": "brief", "deliver": "origin", "origin": None}
        err = _deliver_result(job, "Risk-off close brief", adapters=None, loop=None)
        assert err is None
        assert len(slack_env["send"]) == 1
        assert len(slack_env["mirror"]) == 1, (
            "origin-fallback delivery must mirror the brief into the "
            "home-channel session (the reply-continuity bug)"
        )
        assert slack_env["mirror"][0]["chat_id"] == "D0HOME"
        assert slack_env["mirror"][0]["role"] == "user"

    def test_all_broadcast_does_not_mirror(self, slack_env):
        job = {"id": "j2", "name": "cast", "deliver": "all", "origin": None}
        err = _deliver_result(job, "broadcast text", adapters=None, loop=None)
        assert err is None
        assert len(slack_env["send"]) == 1
        assert len(slack_env["mirror"]) == 0

    def test_explicit_target_with_attach_mirrors(self, slack_env):
        job = {
            "id": "j3", "name": "managed-dm", "deliver": "slack:D0USER7",
            "origin": None, "attach_to_session": True,
        }
        err = _deliver_result(job, "managed brief", adapters=None, loop=None)
        assert err is None
        assert len(slack_env["mirror"]) == 1
        assert slack_env["mirror"][0]["chat_id"] == "D0USER7"

    def test_explicit_target_without_attach_does_not_mirror(self, slack_env):
        job = {
            "id": "j4", "name": "plain-explicit", "deliver": "slack:D0USER8",
            "origin": None,
        }
        err = _deliver_result(job, "plain text", adapters=None, loop=None)
        assert err is None
        assert len(slack_env["mirror"]) == 0

    def test_origin_job_still_mirrors_unchanged(self, slack_env):
        """Regression control: the June origin-scoped behavior is untouched."""
        job = {
            "id": "j5", "name": "origin-job", "deliver": "origin",
            "origin": {"platform": "slack", "chat_id": "D0AAA", "chat_type": "dm"},
        }
        err = _deliver_result(job, "origin brief", adapters=None, loop=None)
        assert err is None
        assert len(slack_env["mirror"]) == 1
        assert slack_env["mirror"][0]["chat_id"] == "D0AAA"


class TestInChannelSeedUserIdGuard:
    """Group-channel seeds are user-keyed; a seed with no user_id would create
    an orphan session. DM targets are safe (key has no user_id)."""

    def test_seed_requires_dm_or_user_id(self):
        from cron.scheduler import _inchannel_seed_allowed

        # DM-shaped chat, no user_id: allowed (DM keys don't embed user).
        assert _inchannel_seed_allowed(is_dm=True, user_id=None)
        # Group chat with known user: allowed.
        assert _inchannel_seed_allowed(is_dm=False, user_id="U123")
        # Group chat, no user: refused — would orphan the session.
        assert not _inchannel_seed_allowed(is_dm=False, user_id=None)
