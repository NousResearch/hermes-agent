"""Security regressions for authenticated WhatsApp bridge IPC (#83395 / #80096)."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def test_concurrent_first_use_callers_agree_on_one_persisted_token(tmp_path: Path):
    """Gateway start and a cron delivery can both touch the session first; both must end up with THE token, 0600."""
    import os
    import stat
    from plugins.platforms.whatsapp.adapter import _load_or_create_bridge_token, _read_bridge_token

    with ThreadPoolExecutor(max_workers=8) as executor:
        tokens = set(executor.map(_load_or_create_bridge_token, [tmp_path] * 8))

    assert len(tokens) == 1 and len(next(iter(tokens))) >= 32
    assert _read_bridge_token(tmp_path) == next(iter(tokens))
    if os.name != "nt":
        assert stat.S_IMODE((tmp_path / ".bridge-token").stat().st_mode) == 0o600


def test_proof_is_bound_to_both_token_and_challenge(tmp_path: Path):
    """A replayed proof (other challenge) or a short/garbage persisted token never passes."""
    from plugins.platforms.whatsapp.adapter import _bridge_auth_proof, _load_or_create_bridge_token

    assert _bridge_auth_proof("tok", "a") != _bridge_auth_proof("tok", "b")
    assert _bridge_auth_proof("tok", "a") != _bridge_auth_proof("other", "a")
    (tmp_path / ".bridge-token").write_text("truncated", encoding="utf-8")
    assert len(_load_or_create_bridge_token(tmp_path)) >= 32  # garbage replaced, never trusted
