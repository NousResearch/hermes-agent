"""Exact discard authority, bounded per-file fixture workload."""
import pytest
from tests.gateway.test_canonical_peer_target_setup import target  # noqa: F401
from tests.gateway.test_canonical_peer_files_target import files_target  # noqa: F401
from tests.gateway.peer_output_discard_checks import assert_discard_denied

@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['owner', 'request_digest', 'unknown', 'generation', 'result', 'horizon', 'revoked'])
async def test_discard_requires_current_exact_right_run_and_result(files_target, monkeypatch, change):
    await assert_discard_denied(files_target, monkeypatch, change)
