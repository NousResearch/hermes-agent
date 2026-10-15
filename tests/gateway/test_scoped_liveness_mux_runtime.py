"""A served profile's live owner outranks its obsolete standalone runtime record."""

import pytest

from gateway import status


@pytest.mark.parametrize("source", ["pid", "runtime_status"])
@pytest.mark.parametrize("same_owner", [True, False], ids=["same-owner", "different-owner"])
def test_live_scoped_owner_only_adopts_its_own_multiplexer_runtime(
    tmp_path, monkeypatch, source, same_owner
):
    profile_home = tmp_path / ".hermes" / "profiles" / "worker"
    profile_home.mkdir(parents=True)
    owner_pid = 4242
    stale_runtime = {
        "pid": 1111,
        "gateway_state": "stopped",
        "platforms": {"a2a": {"state": "disconnected"}},
    }
    canonical_runtime = {
        "pid": owner_pid if same_owner else 9999,
        "gateway_state": "running",
        "served_profiles": ["default", "worker"],
        "platforms": {"worker:a2a": {"state": "connected"}},
    }

    def multiplexer_for_home(home):
        assert home == profile_home
        return canonical_runtime["pid"], canonical_runtime

    monkeypatch.setattr(status, "multiplexer_liveness_for_profile", multiplexer_for_home)

    result = status.resolve_gateway_liveness(
        profile_dir=profile_home,
        runtime=stale_runtime,
        pid_probe=lambda *args, **kwargs: owner_pid if source == "pid" else None,
        runtime_pid_probe=lambda *args, **kwargs: owner_pid,
    )

    assert result.running is True
    assert result.pid == owner_pid
    assert result.source == source
    if same_owner:
        assert result.runtime is canonical_runtime
        platforms = status.profile_platforms_from_multiplexer(result.runtime, "worker")
        assert platforms["a2a"]["state"] == "connected"
    else:
        assert result.runtime is None
