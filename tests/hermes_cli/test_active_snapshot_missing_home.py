"""Registry discovery must not create absent homes or cross profile boundaries."""
import pytest
from hermes_cli import active_sessions as registry


@pytest.mark.parametrize('rename_during_lock_entry', [False, True])
def test_snapshot_does_not_recreate_a_missing_home(tmp_path, monkeypatch, rename_during_lock_entry):
    home = tmp_path / 'profiles' / 'target'
    if rename_during_lock_entry:
        (home / 'runtime').mkdir(parents=True)
        enter = registry._FileLock.__enter__

        def rename_then_enter(lock):
            home.rename(home.with_name('renamed'))
            return enter(lock)

        monkeypatch.setattr(registry._FileLock, '__enter__', rename_then_enter)
    assert registry.active_session_registry_snapshot(registry_home=home) == []
    assert not home.exists()


def test_snapshot_keeps_existing_profile_leases_separate(tmp_path):
    homes = [tmp_path / 'profiles' / name for name in ('a', 'b')]
    leases = []
    try:
        for home in homes:
            home.mkdir(parents=True)
            lease, refusal = registry.try_acquire_active_session(
                session_id=home.name, surface='cli', config={}, registry_home=home)
            assert refusal is None
            leases.append(lease)
        for home in (homes[0], homes[1], homes[0]):
            rows = registry.active_session_registry_snapshot(registry_home=home)
            assert [row['session_id'] for row in rows] == [home.name]
    finally:
        for lease in leases:
            lease.release()
