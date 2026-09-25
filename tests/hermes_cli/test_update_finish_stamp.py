"""Historical takeover publishes source identity before restarting runtimes."""

from hermes_cli.update_finish import finish_update


def test_successful_historical_completion_stamps_before_restart(tmp_path, monkeypatch):
    import hermes_cli.source_stamp as source_stamp
    import hermes_cli.update_cmd as update_cmd

    events = []
    monkeypatch.setattr(update_cmd, "_run_post_update_maintenance", lambda **kwargs: True)
    monkeypatch.setattr(
        source_stamp, "write_source_stamp", lambda root: events.append(("stamp", root)),
    )
    monkeypatch.setattr(
        update_cmd, "_restart_gateway_fleet_after_update",
        lambda plan, gateway_mode: events.append(("restart", plan)) or object(),
    )
    monkeypatch.setattr(
        update_cmd, "_resume_windows_gateways_and_merge_outcome", lambda *args: None,
    )
    monkeypatch.setattr(update_cmd, "_verify_fleet_after_update", lambda *args, **kwargs: None)

    finish_update(
        root=tmp_path,
        assume_yes=True,
        gateway_mode=False,
        pre_update_snapshot_id=None,
        had_desktop_app_before_update=False,
        pre_update_version=None,
        plan=None,
        windows_resume=None,
    )

    assert events == [("stamp", tmp_path), ("restart", None)]
