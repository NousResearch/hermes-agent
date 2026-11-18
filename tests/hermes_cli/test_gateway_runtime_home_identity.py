"""Endpoint discovery matches served homes by canonical identity on BOTH sides.

``_canonical_home`` resolves symlinks and case-folds (Windows ``normcase``). A caller
passing an un-canonicalized spelling of the same directory (the managed worker's
``WorkerRPC``, a symlinked HERMES_HOME) must still find its owner; otherwise the worker
sees ``profile_mismatch`` -> ``owner_unavailable`` although the daemon is ready.
"""
import os
from pathlib import Path

import pytest


def _ready_payload(home: str) -> dict:
    return {'runtime_protocol': 1, 'state': 'ready', 'instance_id': 'inst', 'authority_epoch': 3,
            'capabilities': ['session-authority-v1'], 'api_origin': 'http://127.0.0.1:4321',
            'supervisor': 'none', 'served_profiles': [{'profile_id': home, 'home': home}]}


def test_endpoint_matches_served_home_under_any_canonical_spelling(tmp_path):
    from hermes_cli.gateway_runtime import _canonical_home, _endpoint
    real = tmp_path / 'real-home'
    real.mkdir()
    served = str(real.resolve())
    spellings = [real, Path(served), Path(_canonical_home(served))]
    if os.name == 'nt':
        spellings.append(Path(served.swapcase()))
    else:
        link = tmp_path / 'link-home'
        link.symlink_to(real, target_is_directory=True)
        spellings.append(link)
    for home in spellings:
        discovery = _endpoint(_ready_payload(served), home)
        assert discovery.state == 'ready', (home, discovery)
        assert discovery.endpoint.profile_id == served
    # A different directory is still refused: canonical matching is not prefix or fuzzy matching.
    other = tmp_path / 'other-home'
    other.mkdir()
    assert _endpoint(_ready_payload(served), other).reason_code == 'profile_mismatch'
    assert _endpoint(_ready_payload(served), Path(served + '-suffix')).reason_code == 'profile_mismatch'


def test_ready_endpoint_carries_the_owner_boot_commit(tmp_path):
    """`gateway ensure --json` publishes the code the owner BOOTED from, so Desktop can restart a
    gateway that outlived `hermes update` instead of re-attaching it forever (503 Restart required)."""
    from hermes_cli.gateway_runtime import _endpoint
    home = tmp_path / 'home'
    home.mkdir()
    served = str(home.resolve())
    sha = '0123456789abcdef0123456789abcdef01234567'
    assert _endpoint({**_ready_payload(served), 'code_sha': sha}, home).endpoint.code_sha == sha
    # An owner that cannot name its code (non-git, unreadable stamp) or answers garbage is unknown, not a mismatch.
    assert _endpoint(_ready_payload(served), home).endpoint.code_sha is None
    assert _endpoint({**_ready_payload(served), 'code_sha': ['x']}, home).endpoint.code_sha is None
