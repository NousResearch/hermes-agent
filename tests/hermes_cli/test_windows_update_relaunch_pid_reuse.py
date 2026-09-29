"""Regression tests for Windows post-update gateway relaunch semantics."""


def test_post_update_relaunch_does_not_wait_on_stale_pids(monkeypatch):
    from hermes_cli import gateway
    from hermes_cli import update_cmd_windows as update_windows

    calls = []

    monkeypatch.setattr(
        gateway,
        "launch_detached_profile_gateway_restart",
        lambda *args, **kwargs: calls.append(("profile", args, kwargs)) or True,
    )
    monkeypatch.setattr(
        gateway,
        "launch_detached_gateway_restart_by_cmdline",
        lambda *args, **kwargs: calls.append(("unmapped", args, kwargs)) or True,
    )

    relaunched, unmapped = update_windows._relaunch_paused_gateways(
        {}, {"default": 1234}, [{"pid": 5678, "argv": ["python", "gateway"]}]
    )

    assert relaunched == ["default"]
    assert unmapped == 1
    assert [call[2]["wait_for_exit"] for call in calls] == [False, False]
