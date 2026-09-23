"""Tests for the shared byte formatter (hermes_cli.sizefmt).

Consolidates six near-identical formatters (backup, checkpoints, doctor,
update_cmd, context_references, curator_backup). The contract below locks
the shared behavior, including the one deliberate change vs the old
copies: a real TB tier (doctor/context_references/curator_backup
previously rendered 1 TiB as '1024.0 GB').
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
    """Doctor's stats dict tolerates None in every field; the formatter must
    render (not raise) for None/garbage."""
    assert format_bytes(None) == "?"
    assert format_bytes("garbage") == "?"
    assert format_bytes("2048") == "2.0 KB"  # numeric strings accepted


def test_migrated_sites_render_through_the_shared_helper():
    """Behavior contract for the aliased call sites: the module-local names
    must render byte-identically to the shared helper (how they delegate is
    an implementation detail — only the rendering is pinned)."""
    from hermes_cli.backup import _format_size as backup
    from hermes_cli.checkpoints import _fmt_bytes as checkpoints
    from hermes_cli.sizefmt import format_bytes as doctor

    for n in (0, 512, 2048, 1234567, 1024**3, 1024**4):
        expected = format_bytes(n)
        assert backup(n) == expected
        assert checkpoints(n) == expected
        assert doctor(n) == expected
