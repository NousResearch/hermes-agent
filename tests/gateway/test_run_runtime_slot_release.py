"""A failed profile-authority build releases its reserved slot without replacing the real error.

``_build_profile_authority`` reserves a slot, builds the authority and, on failure, drops the slot
and re-raises. The registry refuses to remove the launch slot, so on the launch profile the drop
raised ``RuntimeError('the launch session authority cannot be removed')`` and that replaced the
error being handled: a ``database is locked`` at boot surfaced as a code-bug message.
"""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from gateway.session_authorities import SessionAuthorities


@pytest.mark.asyncio
async def test_a_failed_launch_build_raises_the_original_error(monkeypatch, tmp_path):
    from gateway import run_runtime
    import gateway.run as gateway_run
    import gateway.session_authority as session_authority
    import persistence_assembly

    home = tmp_path / "launch"
    registry = SessionAuthorities(home)
    persistence = SimpleNamespace(scope=SimpleNamespace(profile_dir=home))

    class _Reference:
        def close(self):
            pass

    monkeypatch.setattr(run_runtime, "open_profile_store", lambda runner, h: persistence)
    monkeypatch.setattr(gateway_run, "_profile_runtime_scope", lambda *a, **k: nullcontext())
    monkeypatch.setattr(persistence_assembly, "open_persistence", lambda *a, **k: _Reference())
    monkeypatch.setattr(persistence_assembly, "store_location", lambda scope: None)

    async def locked(*args, **kwargs):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(session_authority, "initialize_session_authority", locked)

    runner = SimpleNamespace(
        session_authorities=registry,
        session_runtime_descriptor={"instance_id": "boot"},
    )

    with pytest.raises(RuntimeError) as raised:
        await run_runtime._build_profile_authority(runner, "default", home, register=True)

    assert "database is locked" in str(raised.value), (
        "the build's own error must reach the caller, not the launch-slot refusal that replaced it"
    )
    assert registry.starting_homes() == [str(home)], "the launch slot must survive the failed build"
