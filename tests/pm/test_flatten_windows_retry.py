"""Windows extraction can briefly hold a directory open after unpacking."""
import pytest
from pathlib import Path
from pm.store import flatten_single_dir

pytestmark = pytest.mark.platforms("windows")


def test_flatten_recovers_from_transient_hold(tmp_path, monkeypatch):
    tmp_path = tmp_path / "extract"
    inner = tmp_path / "archive"
    (inner / "bin").mkdir(parents=True)
    (inner / "bin" / "tool.exe").write_bytes(b"verified payload")
    original = Path.rename
    attempts = []

    def rename(path, target):
        attempts.append(path)
        if len(attempts) < 3:
            error = PermissionError("temporary hold")
            error.winerror = 5
            raise error
        return original(path, target)

    monkeypatch.setattr(Path, "rename", rename)
    flatten_single_dir(tmp_path)
    assert (tmp_path / "bin" / "tool.exe").read_bytes() == b"verified payload"
    assert not inner.exists()
    assert len(attempts) == 3


@pytest.mark.parametrize("winerror,expected_attempts", [(5, 21), (87, 1)])
def test_flatten_preserves_payload_when_move_fails(tmp_path, monkeypatch, winerror, expected_attempts):
    tmp_path = tmp_path / "extract"
    inner = tmp_path / "archive"
    inner.mkdir(parents=True)
    payload = inner / "payload"
    payload.write_bytes(b"verified payload")
    attempts = []

    def rename(path, target):
        attempts.append(path)
        error = OSError("move failed")
        error.winerror = winerror
        raise error

    monkeypatch.setattr(Path, "rename", rename)
    with pytest.raises(OSError):
        flatten_single_dir(tmp_path)
    assert payload.read_bytes() == b"verified payload"
    assert len(attempts) == expected_attempts
