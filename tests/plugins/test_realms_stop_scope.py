"""Native fail-closed StopUnit regression; no private desktop input."""
from pathlib import Path

import pytest


@pytest.mark.linux_only
@pytest.mark.integration
@pytest.mark.parametrize("post_query_unavailable", [False, True], ids=["active", "unknown"])
def test_failed_stop_requires_fresh_terminal_observation(
    tmp_path, monkeypatch, post_query_unavailable
):
    from runpy import run_path

    root = Path(__file__).resolve().parents[2] / "plugins/hermes-realms"
    load = run_path(str(root / "realms/_binding.py"))["load_runtime"]
    lifecycle = load("lifecycle")
    supervisor = load("supervisor")
    Manager = load("manager").Manager

    # The transport is broken only after the real ownership observation. No
    # unit states or subprocess results are fabricated, and the scope stays live.
    manager = Manager(tmp_path)
    record = manager.start("failed-stop-observation")
    original_run = lifecycle.subprocess.run
    stop_errors = []
    post_queries = []
    unavailable = {
        "XDG_RUNTIME_DIR": str(tmp_path / "absent-runtime"),
        "DBUS_SESSION_BUS_ADDRESS": "unix:path=" + str(tmp_path / "absent-bus"),
    }

    def fail_stop_transport(command, **kwargs):
        if command == ["systemctl", "--user", "stop", record["scope"]]:
            kwargs["env"] = dict(kwargs["env"], **unavailable)
            try:
                return original_run(command, **kwargs)
            except lifecycle.subprocess.CalledProcessError as exc:
                stop_errors.append(exc)
                raise
        if stop_errors and command[:4] == ["systemctl", "--user", "show", record["scope"]]:
            if post_query_unavailable:
                kwargs["env"] = dict(kwargs["env"], **unavailable)
            try:
                result = original_run(command, **kwargs)
            except lifecycle.subprocess.CalledProcessError as exc:
                post_queries.append(exc)
                raise
            post_queries.append(result)
            return result
        return original_run(command, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(lifecycle.subprocess, "run", fail_stop_transport)
            if post_query_unavailable:
                with pytest.raises(lifecycle.RealmError, match="inspect"):
                    supervisor.cleanup(tmp_path, record["id"])
            else:
                with pytest.raises(lifecycle.subprocess.CalledProcessError) as failure:
                    supervisor.cleanup(tmp_path, record["id"])
                assert failure.value is stop_errors[0]
        assert len(stop_errors) == len(post_queries) == 1
        assert stop_errors[0].returncode != 0 and stop_errors[0].stderr
        if post_query_unavailable:
            assert isinstance(post_queries[0], lifecycle.subprocess.CalledProcessError)
            assert post_queries[0].returncode != 0 and post_queries[0].stderr
        else:
            assert post_queries[0].returncode == 0
            observed = dict(
                line.split("=", 1)
                for line in post_queries[0].stdout.splitlines() if "=" in line
            )
            assert observed["ActiveState"] == "active"
            assert observed["InvocationID"] == record["invocation_id"]
        assert manager.registry.get(record["id"]) == record
        assert Path(record["runtime_dir"]).exists()
        lifecycle.validate_live(record)
    finally:
        # Restore actual transports even on RED, then retire only this owner.
        lifecycle.stop_scope(record)
        manager.registry.put(record)
        manager.stop(record["id"])
        assert not manager.registry.path(record["id"]).exists()
        assert not Path(record["runtime_dir"]).exists()
