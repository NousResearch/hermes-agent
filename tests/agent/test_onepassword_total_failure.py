"""A 1Password pull that resolves nothing must report failure, not an empty success.

``FetchResult.ok`` is ``error is None`` and a per-ref ``op read`` error is only a
warning, so a total failure looked like a successful pull of zero secrets.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.secret_sources import onepassword as op


# A genuinely rejected credential.
PERMANENT_ERR = (
    "error initializing client: Validation: (failed to session.DecodeSACredentials), "
    "Server: (failed to DecodeSACredentials), failed to parseToken, format is invalid"
)


@pytest.fixture(autouse=True)
def _reset_caches():
    op._reset_cache_for_tests()
    yield
    op._reset_cache_for_tests()


def _ok(value: str):
    return mock.Mock(returncode=0, stdout=value, stderr="")


def _err(code: int, stderr: str):
    return mock.Mock(returncode=code, stdout="", stderr=stderr)


# ---------------------------------------------------------------------------
# The false-success bug: a total failure must report failure
# ---------------------------------------------------------------------------
def test_total_failure_reports_not_ok(monkeypatch, tmp_path):
    """A pull that resolved NOTHING must not report ok=True.

    ``FetchResult.ok`` is ``error is None``; a per-ref read error is only a warning,
    so a total outage used to look like a successful empty pull.
    """
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    fake_op.chmod(0o755)
    monkeypatch.setattr(op.subprocess, "run", lambda *a, **k: _err(1, PERMANENT_ERR))

    source = op.OnePasswordSource()
    result = source.fetch({"env": {"K": "op://V/I/F"}, "binary_path": str(fake_op)}, tmp_path)
    assert result.ok is False
    assert result.error is not None


def test_partial_failure_still_reports_ok(monkeypatch, tmp_path):
    """Some refs resolved: the source stays usable, so it must stay ok=True."""
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    fake_op.chmod(0o755)

    def fake_run(argv, **k):
        # Refs are read in sorted order, so match on the reference, not the call count.
        if "op://V/I/G" in argv:
            return _err(1, PERMANENT_ERR)
        return _ok("GOOD_TOKEN\n")

    monkeypatch.setattr(op.subprocess, "run", fake_run)

    source = op.OnePasswordSource()
    result = source.fetch(
        {"env": {"K": "op://V/I/F", "J": "op://V/I/G"}, "binary_path": str(fake_op)}, tmp_path
    )
    assert result.ok is True
    assert result.secrets.get("K") == "GOOD_TOKEN"


def test_successful_pull_reports_ok(monkeypatch, tmp_path):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    fake_op.chmod(0o755)
    monkeypatch.setattr(op.subprocess, "run", lambda *a, **k: _ok("TOKEN\n"))

    source = op.OnePasswordSource()
    result = source.fetch({"env": {"K": "op://V/I/F"}, "binary_path": str(fake_op)}, tmp_path)
    assert result.ok is True
    assert result.secrets == {"K": "TOKEN"}
