"""A profile directory copied from another HERMES_HOME must never read as running.

Copying a profile directory wholesale (sandbox injection, restore-from-backup,
cloning a profile) carries that home's ``gateway.pid`` and ``gateway_state.json``
along. The liveness ladder trusted those records' PID, and its last rung matched the
default multiplexer by profile *name* -- which the copied directory keeps. So the
dashboard and ``hermes -p X status`` reported a phantom gateway as running (with the
production gateway's PID) while ``hermes cron`` / ``gateway list`` said the opposite on
the same machine.

These pin the ladder's contract: a record naming another home is not this home's
identity, the pooled-layout premise is required for the name-matching rung, and a
record naming *this* home is still believed.
"""

import json

from gateway import status
import hermes_constants

_LIVE_PID = 4242


def _pooled_profile_dir(name="eagle"):
    """A real ``<default root>/profiles/<name>`` -- the pooled layout rung 4 assumes."""
    return hermes_constants.get_default_hermes_root() / "profiles" / name


def _write_pid_record(profile_dir, hermes_home, pid=_LIVE_PID):
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "gateway.pid").write_text(
        json.dumps(
            {
                "pid": pid,
                "kind": "hermes-gateway",
                "argv": ["python", "-m", "hermes_cli.main", "gateway"],
                "hermes_home": str(hermes_home),
            }
        ),
        encoding="utf-8",
    )


def _liveness(profile_dir, **overrides):
    kwargs = dict(
        profile_dir=profile_dir,
        runtime=None,
        health_probe=None,
        pid_probe=lambda *a, **k: _LIVE_PID,
        runtime_reader=lambda **k: {},
        runtime_pid_probe=lambda *a, **k: None,
        use_cache=False,
    )
    kwargs.update(overrides)
    return status.resolve_gateway_liveness(**kwargs)


def test_identity_files_recorded_for_another_home_never_report_running(tmp_path, monkeypatch):
    """Rung 1 (gateway.pid) and rung 3 (gateway_state.json) both carry a foreign identity."""
    # Isolate the identity rungs: no live multiplexer serves this profile name.
    monkeypatch.setattr(status, "multiplexer_liveness_for_profile", lambda *a, **k: None)
    profile_dir = _pooled_profile_dir()
    foreign_home = tmp_path / "production-home"

    _write_pid_record(profile_dir, hermes_home=foreign_home)
    pid_rung = _liveness(profile_dir)
    assert pid_rung.running is False
    assert pid_rung.source == "none"

    runtime = {"gateway_state": "running", "pid": _LIVE_PID, "hermes_home": str(foreign_home)}
    runtime_rung = _liveness(
        profile_dir,
        pid_probe=lambda *a, **k: None,
        runtime=runtime,
        runtime_pid_probe=lambda *a, **k: _LIVE_PID,
    )
    assert runtime_rung.running is False
    assert runtime_rung.source == "none"


def test_identity_files_recorded_for_this_home_still_report_running():
    """The guard must not disable the PID rung for a gateway that really owns this home."""
    profile_dir = _pooled_profile_dir()
    _write_pid_record(profile_dir, hermes_home=profile_dir)

    result = _liveness(profile_dir)

    assert result.running is True
    assert result.source == "pid"
    assert result.pid == _LIVE_PID


def test_a_profile_dir_outside_the_default_root_never_takes_the_multiplexer_rung(tmp_path, monkeypatch):
    """Rung 4 matches a profile *name*; that premise only holds under ``<root>/profiles/``."""
    profile_dir = tmp_path / "sandbox-home" / "profiles" / "eagle"
    profile_dir.mkdir(parents=True)
    assert profile_dir != _pooled_profile_dir()
    # The live multiplexer claims to serve this profile name -- true only for the pooled layout.
    monkeypatch.setattr(
        status, "multiplexer_liveness_for_profile", lambda *a, **k: (_LIVE_PID, {"gateway_state": "running"})
    )

    result = _liveness(
        profile_dir,
        pid_probe=lambda *a, **k: None,
        runtime_pid_probe=lambda *a, **k: None,
    )

    assert result.running is False
    assert result.source == "none"


def test_a_scoped_read_that_found_nothing_does_not_borrow_the_process_home_record(tmp_path, monkeypatch):
    """A missing per-profile ``gateway_state.json`` must not fall back to the PROCESS home's file.

    ``get_runtime_status_running_pid(None, ...)`` re-reads the DEFAULT home's ``gateway_state.json``
    -- and a copied profile directory keeps the profile *name*, so the live serving gateway's record
    would lend it that gateway's PID. Rung 3 must be handed "no record", never ``None``.
    """
    monkeypatch.setattr(status, "multiplexer_liveness_for_profile", lambda *a, **k: None)
    profile_dir = tmp_path / "sandbox-home" / "profiles" / "eagle"
    profile_dir.mkdir(parents=True)
    handed_to_rung3 = []

    def _runtime_pid_probe(record, **kw):
        handed_to_rung3.append(record)
        # Mirrors the production probe: ``None`` means "re-read the default home's record".
        return _LIVE_PID if record is None else None

    result = _liveness(
        profile_dir,
        pid_probe=lambda *a, **k: None,
        runtime_reader=lambda **k: None,  # the profile's own gateway_state.json is absent
        runtime_pid_probe=_runtime_pid_probe,
    )

    assert handed_to_rung3 == [{}]
    assert result.running is False
