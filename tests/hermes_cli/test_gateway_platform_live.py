"""``hermes config set`` reports what the running gateway did, from the gateway's own verdict.

A platform the gateway tried to start and could not bring up (Feishu with its SDK not importable) was
listed as "applied it now: started feishu" and, on the next line, as "could not start": both cannot be true.
"""
from hermes_cli import gateway_platform_live


def _note(monkeypatch, tmp_path, answer):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(gateway_platform_live, "reconcile_running_gateway_platforms", lambda name, home: answer)
    return gateway_platform_live.config_set_live_note("FEISHU_APP_SECRET")


def test_a_platform_that_did_not_come_up_is_never_reported_as_applied(monkeypatch, tmp_path):
    note = _note(monkeypatch, tmp_path, {"applied": True, "changed": True, "started": ["feishu"], "restarted": [],
                                         "stopped": [], "platforms": {"feishu": "unavailable"}, "reasons": {}})

    assert "applied it now" not in note
    assert note.count("feishu") == 1 and "could not start feishu" in note


def test_each_platform_is_reported_once_with_its_own_outcome(monkeypatch, tmp_path):
    note = _note(monkeypatch, tmp_path, {
        "applied": True, "changed": True, "started": ["qqbot", "feishu"], "restarted": [], "stopped": ["slack"],
        "platforms": {"qqbot": "retrying", "feishu": "unavailable"},
        "reasons": {"feishu": "the running gateway cannot import this platform's Python dependencies"}})

    applied, failed = note.splitlines()
    assert applied.strip() == "Running gateway applied it now: started qqbot; stopped slack"
    assert "could not start feishu: the running gateway cannot import" in failed
