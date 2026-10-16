"""``CredentialPool._pick_and_rotate`` is the one strategy seam ``select()`` goes through."""

from agent.credential_pool import CredentialPool, PooledCredential


def _entry(idx):
    return PooledCredential(provider="openrouter", id=f"k{idx}", label=f"key-{idx}", auth_type="api_key",
                            priority=idx, source="manual", access_token=f"sk-{idx}")


def test_select_goes_through_the_pick_and_rotate_override(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    calls = []

    class LastFirstPool(CredentialPool):
        def _pick_and_rotate(self, available, *, count):
            calls.append([e.id for e in available])
            return available[-1]

    assert CredentialPool("openrouter", [_entry(0), _entry(1)]).select().id == "k0"  # positive control
    pool = LastFirstPool("openrouter", [_entry(0), _entry(1)])
    assert pool.select().id == "k1"
    assert calls == [["k0", "k1"]]
    assert pool.current().id == "k1"
