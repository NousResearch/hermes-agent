import os
from pathlib import Path

from hermes_cli import config
import utils


def test_env_rewrite_publishes_secure_mode_and_preserves_owner(
    tmp_path: Path, monkeypatch,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("OLD=1\n", encoding="utf-8")
    os.chmod(env_path, 0o644)
    original = env_path.stat()
    chowns = []
    real_fchown = os.fchown
    monkeypatch.setattr(
        utils.os, "fchown",
        lambda fd, uid, gid: (chowns.append((uid, gid)), real_fchown(fd, uid, gid))[1],
    )

    config._write_env_lines(env_path, ["NEW=2\n"], preserve_mode=False)

    rewritten = env_path.stat()
    assert env_path.read_text(encoding="utf-8") == "NEW=2\n"
    assert rewritten.st_mode & 0o777 == 0o600
    assert (rewritten.st_uid, rewritten.st_gid) == (original.st_uid, original.st_gid)
    assert chowns == [(original.st_uid, original.st_gid)]

    os.chmod(env_path, 0o640)
    config._write_env_lines(env_path, ["NEW=3\n"], preserve_mode=True)
    assert env_path.stat().st_mode & 0o777 == 0o640

    new_path = tmp_path / "new.env"
    config._write_env_lines(new_path, ["TOKEN=x\n"], preserve_mode=True)
    assert new_path.stat().st_mode & 0o777 == 0o600