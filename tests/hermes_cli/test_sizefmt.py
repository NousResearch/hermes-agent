"""Tests for the shared byte formatter (hermes_cli.sizefmt).

Consolidates five near-identical private formatters (backup, checkpoints,
doctor, context_references, curator_backup). The contract below locks the
shared behavior, including the two deliberate changes vs the old copies:
a real TB tier (doctor/context_references/curator_backup previously
rendered 1 TiB as '1024.0 GB') and non-raising fallback for None/garbage.
"""

import pytest

from hermes_cli.sizefmt import format_bytes


@pytest.mark.parametrize(
    "n,expected",
    [
        (0, "0 B"),
        (1, "1 B"),
        (512, "512 B"),
        (1023, "1023 B"),
        (1024, "1.0 KB"),
        (1536, "1.5 KB"),
        (1024**2, "1.0 MB"),
        (1234567, "1.2 MB"),
        (1024**3, "1.0 GB"),
        (1024**4, "1.0 TB"),
        (2 * 1024**4, "2.0 TB"),
    ],
)
def test_format_bytes_tiers(n, expected):
    assert format_bytes(n) == expected


def test_format_bytes_tb_tier_not_gb_overflow():
    """The old doctor/context_references/curator_backup copies topped out at
    GB and rendered 1 TiB as '1024.0 GB' — the shared helper must not."""
    assert "TB" in format_bytes(1024**4)
    assert "1024" not in format_bytes(1024**4)


def test_format_bytes_never_raises():
    assert format_bytes(None) == "?"
    assert format_bytes("garbage") == "?"
    assert format_bytes("2048") == "2.0 KB"  # numeric strings accepted
    assert format_bytes(None, fallback="unknown") == "unknown"


def test_delegating_aliases_share_the_implementation():
    """The five migrated call sites must all resolve to the shared helper
    (checkpoints wraps it to preserve its None -> '0 B' display)."""
    from agent.context_references import _human_bytes as ctx
    from agent.curator_backup import format_size as curator
    from hermes_cli.backup import _format_size as backup
    from hermes_cli.checkpoints import _fmt_bytes as checkpoints
    from hermes_cli.doctor import _human_bytes as doctor

    assert backup is format_bytes
    assert ctx is format_bytes
    assert curator is format_bytes
    assert doctor is format_bytes
    assert checkpoints(None) == "0 B"
    assert checkpoints(2048) == format_bytes(2048)
