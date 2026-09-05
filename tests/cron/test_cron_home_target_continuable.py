"""Continuable-cron eligibility for user-written bare-platform home targets.

Field report (enterprise, 2026-09-02): managed crons provisioned with
``deliver: "slack"`` (no captured origin) deliver their brief to the Slack
home channel, but the transcript mirror and the in_channel session seed are
skipped — the agent never sees its own delivery, and a reply lands in a
context-less session. The same job with an ``origin`` present works.

Root cause: bare-platform targets carry no ``_resolved_from`` provenance,
so ``_target_mirror_eligible`` treats them like an ``all`` broadcast
expansion. But a user-WRITTEN bare-platform token is a deliberate address
to the operator-configured home channel — the same destination and the
same conversation semantics as the ``origin_fallback`` lane (580daa7b96).
The broadcast exclusion from the June origin-scoping refactor (c06ceb3232)
is kept: only tokens expanded from ``all`` remain untagged and ineligible.

Design under test:
- ``deliver: "slack"`` written by the user resolves with
  ``_resolved_from: "home"``; the identical token produced by expanding
  ``all`` stays untagged.
- ``home`` targets are mirror-eligible under the SAME flags as
  ``origin_fallback``: per-job ``attach_to_session`` wins, else the global
  ``cron.mirror_delivery`` flag. Explicit ``platform:chat`` rules are
  unchanged (per-job opt-in only).
- Dedup provenance OR-merge covers the new tag.
"""

import pytest

from cron.scheduler import (
    _deliver_result,
    _resolve_delivery_targets,
)
from cron.scheduler_delivery import _target_mirror_eligible


@pytest.fixture(autouse=True)
def _home_channel(monkeypatch):
    monkeypatch.setenv("SLACK_HOME_CHANNEL", "D0HOME")
    monkeypatch.delenv("TELEGRAM_HOME_CHANNEL", raising=False)
    monkeypatch.delenv("DISCORD_HOME_CHANNEL", raising=False)


class TestHomeProvenanceResolution:
    def test_user_written_bare_platform_is_tagged_home(self):
        job = {"deliver": "slack", "origin": None}
        targets = _resolve_delivery_targets(job)
        assert len(targets) == 1
        assert targets[0]["chat_id"] == "D0HOME"
        assert targets[0].get("_resolved_from") == "home"

    def test_all_expansion_stays_untagged(self):
        """The same bare token produced by expanding ``all`` is a broadcast
        and must NOT gain the home tag."""
        job = {"deliver": "all", "origin": None}
        targets = _resolve_delivery_targets(job)
        assert targets
        for t in targets:
            assert t.get("_resolved_from") is None

    def test_bare_platform_with_origin_elsewhere_is_tagged_home(self):
        """A job with a Telegram origin and ``deliver: "slack"`` addresses
        the Slack home channel deliberately."""
        job = {
            "deliver": "slack",
            "origin": {"platform": "telegram", "chat_id": "123", "chat_type": "dm"},
        }
        targets = _resolve_delivery_targets(job)
        assert len(targets) == 1
        assert targets[0].get("_resolved_from") == "home"


class TestHomeMirrorEligibility:
    def test_home_eligible_under_global_flag(self):
        """The Coatue shape: managed cron, deliver=slack, no origin,
        global mirror flag on."""
        job = {"deliver": "slack", "origin": None}
        targets = _resolve_delivery_targets(job)
        assert _target_mirror_eligible(job, targets[0], global_mirror=True)

    def test_home_eligible_with_per_job_attach(self):
        job = {"deliver": "slack", "origin": None, "attach_to_session": True}
        targets = _resolve_delivery_targets(job)
        assert _target_mirror_eligible(job, targets[0], global_mirror=False)

    def test_home_per_job_false_beats_global_true(self):
        """Same precedence as origin_fallback: an explicit per-job False
        must win over the global flag."""
        job = {"deliver": "slack", "origin": None, "attach_to_session": False}
        targets = _resolve_delivery_targets(job)
        assert not _target_mirror_eligible(job, targets[0], global_mirror=True)

    def test_home_not_eligible_without_any_flag(self):
        job = {"deliver": "slack", "origin": None}
        targets = _resolve_delivery_targets(job)
        assert not _target_mirror_eligible(job, targets[0], global_mirror=False)

    def test_all_expansion_still_never_eligible(self):
        job = {"deliver": "all", "origin": None, "attach_to_session": True}
        targets = _resolve_delivery_targets(job)
        assert targets
        for t in targets:
            assert not _target_mirror_eligible(job, t, global_mirror=True)

    def test_dedup_bare_and_all_keeps_home_tag(self):
        """'slack,all' resolves the home chat twice; the user-written
        token's provenance must survive dedup in either order."""
        for deliver in ("slack,all", "all,slack"):
            job = {"deliver": deliver, "origin": None}
            targets = _resolve_delivery_targets(job)
            slack_targets = [t for t in targets if t["platform"].lower() == "slack"]
            assert len(slack_targets) == 1
            assert slack_targets[0].get("_resolved_from") == "home", deliver

    def test_dedup_origin_fallback_outranks_or_ties_home(self):
        """'origin,slack' (origin-less) resolves the same home chat via two
        eligible lanes; eligibility must hold regardless of order."""
        for deliver in ("origin,slack", "slack,origin"):
            job = {"deliver": deliver, "origin": None}
            targets = _resolve_delivery_targets(job)
            slack_targets = [t for t in targets if t["platform"].lower() == "slack"]
            assert len(slack_targets) == 1
            assert _target_mirror_eligible(
                job, slack_targets[0], global_mirror=True
            ), deliver


class TestHomeMirrorEndToEnd:
    """Drive _deliver_result with a stubbed sender + mirror recorder,
    mirroring the origin_fallback E2E harness."""

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

    def test_managed_bare_platform_job_mirrors_brief(self, slack_env):
        """The field repro: managed cron, deliver=slack, no origin. The
        brief must be mirrored into the home-channel session so a reply
        continues in context."""
        job = {"id": "h1", "name": "managed", "deliver": "slack", "origin": None}
        err = _deliver_result(job, "morning brief", adapters=None, loop=None)
        assert err is None
        assert len(slack_env["send"]) == 1
        assert len(slack_env["mirror"]) == 1, (
            "user-written bare-platform delivery must mirror the brief into "
            "the home-channel session (the reply-continuity bug)"
        )
        assert slack_env["mirror"][0]["chat_id"] == "D0HOME"
        assert slack_env["mirror"][0]["role"] == "user"

    def test_all_broadcast_still_does_not_mirror(self, slack_env):
        job = {"id": "h2", "name": "cast", "deliver": "all", "origin": None}
        err = _deliver_result(job, "broadcast text", adapters=None, loop=None)
        assert err is None
        assert len(slack_env["send"]) == 1
        assert len(slack_env["mirror"]) == 0
