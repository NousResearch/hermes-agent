"""Cron delivery honours the profile's Slack posting policy (education design
5b): a delivery is free text, so it never reaches a gated channel, and a
profile with slack.post_allowed_channels delivers only there."""
import cron.scheduler as sch

GUARD = {"notion": {"write_guard": {"enabled": True}}}


def _t(chat, platform="slack"):
    return {"platform": platform, "chat_id": chat}


def test_no_policy_is_inert():
    targets = [_t("C1"), _t("123", "telegram")]
    assert sch._apply_post_policy(targets, {}) == (targets, [])


def test_allowlist_and_gated_filtering():
    cfg = {"slack": {"post_allowed_channels": ["CHOME001", "CREL0001"], "post_gated_channels": ["CREL0001"]}}
    kept, refused = sch._apply_post_policy(
        [_t("chome001"), _t("CREL0001"), _t("COTHER01"), _t("#release-updates"), _t("9", "telegram")], cfg)
    assert [t["chat_id"] for t in kept] == ["chome001", "9"]
    assert len(refused) == 3


def test_guarded_profile_without_allowlist_delivers_nowhere_on_slack():
    kept, refused = sch._apply_post_policy([_t("C1")], GUARD)
    assert kept == [] and "post_allowed_channels" in refused[0]


def test_deliver_result_refuses_before_sending(monkeypatch):
    import agent.boundary_config as bc
    monkeypatch.setattr(bc, "read_boundary_config", lambda home=None: {"slack": {"post_gated_channels": ["CREL0001"]}})
    sent = []
    import tools.send_message_tool as smt
    monkeypatch.setattr(smt, "_send_to_platform", lambda *a, **k: sent.append(a))
    err = sch._deliver_result({"id": "j1", "name": "feed", "deliver": "slack:CREL0001"}, "hello")
    assert err and "posting policy" in err
    assert sent == []


def test_owner_alerted_when_policy_refused_every_target(monkeypatch):
    import agent.boundary_config as bc
    monkeypatch.setattr(bc, "read_boundary_config", lambda home=None: {"slack": {"post_gated_channels": ["CREL0001"]}})
    job = {"id": "j1", "name": "feed", "deliver": "slack:CREL0001"}
    assert sch._resolve_delivery_targets(job)
    assert sch._apply_post_policy(sch._resolve_delivery_targets(job))[0] == []
