from pathlib import Path

import pytest

from hermes_cli.db_ownership import (
    DbOpenDecision,
    decide_direct_db_open,
)


@pytest.mark.parametrize("role", ["cli", "web", "tui"])
def test_active_gateway_blocks_direct_writes(role, tmp_path: Path):
    decision = decide_direct_db_open(
        role=role,
        operation="write",
        db_path=tmp_path / "state.db",
        gateway_pid=1234,
    )

    assert decision is DbOpenDecision.ROUTE_VIA_GATEWAY


@pytest.mark.parametrize("role", ["cli", "web", "tui"])
def test_active_gateway_allows_only_safe_reads(role, tmp_path: Path):
    decision = decide_direct_db_open(
        role=role,
        operation="read",
        db_path=tmp_path / "state.db",
        gateway_pid=1234,
    )

    assert decision is DbOpenDecision.SAFE_READ


def test_maintenance_requires_gateway_to_be_stopped(tmp_path: Path):
    assert decide_direct_db_open(
        role="maintenance",
        operation="write",
        db_path=tmp_path / "state.db",
        gateway_pid=None,
    ) is DbOpenDecision.ALLOW

    assert decide_direct_db_open(
        role="maintenance",
        operation="write",
        db_path=tmp_path / "state.db",
        gateway_pid=1234,
    ) is DbOpenDecision.REFUSE


def test_test_role_must_not_open_canonical_home(tmp_path: Path):
    canonical = tmp_path / "hermes"
    test_db = canonical / "state.db"

    with pytest.raises(RuntimeError, match="isolated"):
        decide_direct_db_open(
            role="test",
            operation="write",
            db_path=test_db,
            gateway_pid=None,
            hermes_home=canonical,
        )
