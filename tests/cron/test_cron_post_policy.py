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
    cfg = {"slack": {"post_allowed_channels": ["CHOME", "CREL"], "post_gated_channels": ["CREL"]}}
    kept, refused = sch._apply_post_policy([_t("chome"), _t("CREL"), _t("COTHER"), _t("9", "telegram")], cfg)
    assert [t["chat_id"] for t in kept] == ["chome", "9"]
    assert len(refused) == 2


def test_guarded_profile_without_allowlist_delivers_nowhere_on_slack():
    kept, refused = sch._apply_post_policy([_t("C1")], GUARD)
    assert kept == [] and "post_allowed_channels" in refused[0]


def test_deliver_result_refuses_before_sending(monkeypatch):
    import hermes_cli.config as hc
    monkeypatch.setattr(hc, "read_raw_config", lambda: {"slack": {"post_gated_channels": ["CREL"]}})
    sent = []
    import tools.send_message_tool as smt
    monkeypatch.setattr(smt, "_send_to_platform", lambda *a, **k: sent.append(a))
    err = sch._deliver_result({"id": "j1", "name": "feed", "deliver": "slack:CREL"}, "hello")
    assert err and "posting policy" in err
    assert sent == []
