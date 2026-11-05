"""A profile the gateway stops serving and serves again logs no ``hermes_state`` reopen warning.

A CLI rename or delete pauses the profile (unserve, then serve again), and a delete followed by a same-name recreate
goes through the reconcile. Each unserve closes the profile's store; the authority built by the next serve opens it
again, which is a fresh open and not a reuse under a live handle, so the gateway logs nothing about it.
"""
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from persistence_assembly import close_stores_under


def _reopen_warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "hermes_state" and "reopening" in r.getMessage()]


@pytest.fixture
def served(tmp_path, monkeypatch):
    """A multiplexed runner for the launch profile and 'alpha', set up as startup leaves it."""
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.run_profile_reconcile import profile_serve_signature
    from gateway.runtime_ownership import process_ownership
    root = tmp_path / ".hermes"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    alpha = root / "profiles" / "alpha"
    alpha.mkdir(parents=True, mode=0o700)
    (alpha / "config.yaml").write_text("{}")
    homes = [("default", root), ("alpha", alpha)]
    process_ownership.reserve([home for _, home in homes])
    config = GatewayConfig()
    config.multiplex_profiles = True
    config._runtime_profile_homes = tuple(homes)
    runner = GatewayRunner(config)
    runner._running = True
    runner._primary_profile_name = "default"
    runner._served_profile_homes = {"alpha": alpha}
    runner._served_profile_signatures = {"alpha": profile_serve_signature(alpha)}
    try:
        with patch("hermes_cli.profiles.get_active_profile_name", return_value="default"):
            yield runner, alpha
    finally:
        for authority in list(getattr(runner, "session_authorities", None) or []):
            close_stores_under(authority.persistence.scope.profile_dir)
        for _, home in homes:
            process_ownership.release(home)


@pytest.mark.asyncio
async def test_unserve_then_serve_again_logs_no_reopen(served, caplog):
    from gateway.run_runtime import initialize_gateway_runtime
    runner, alpha = served
    await initialize_gateway_runtime(runner)
    first_epoch = runner.session_authorities.for_home(alpha.resolve()).epoch
    await runner._unserve_profile("alpha", alpha)
    assert runner.session_authorities.for_home(alpha.resolve()) is None

    with caplog.at_level(logging.WARNING, logger="hermes_state"):
        assert await runner._serve_profile_runtime("alpha", alpha) is None
        authority = runner.session_authorities.for_home(alpha.resolve())
        assert authority.epoch > first_epoch
        assert authority.persistence.sessions.get("nope") is None
    assert _reopen_warnings(caplog) == []


@pytest.mark.asyncio
async def test_delete_then_same_name_recreate_logs_no_reopen(served, caplog):
    import shutil
    from gateway.run_runtime import initialize_gateway_runtime
    from hermes_constants import clear_named_profile_deleted, mark_named_profile_deleted
    runner, alpha = served
    await initialize_gateway_runtime(runner)
    mark_named_profile_deleted(alpha)
    assert (await runner.reconcile_served_profiles())["removed"] == ["alpha"]
    shutil.rmtree(alpha)
    clear_named_profile_deleted(alpha)
    alpha.mkdir(mode=0o700)
    (alpha / "config.yaml").write_text("{}")

    with caplog.at_level(logging.WARNING, logger="hermes_state"):
        assert (await runner.reconcile_served_profiles())["added"] == ["alpha"]
        authority = runner.session_authorities.for_home(alpha.resolve())
        assert authority.persistence.sessions.get("nope") is None
    assert _reopen_warnings(caplog) == []
