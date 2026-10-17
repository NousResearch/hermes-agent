"""Execute the documented backup script against disposable worlds (#134910)."""

import os
from pathlib import Path
import subprocess
import tarfile

import pytest


SKILL = (
    Path(__file__).resolve().parents[2]
    / "optional-skills/gaming/minecraft-modpack-server/SKILL.md"
)


@pytest.fixture
def backup_env(tmp_path):
    home = tmp_path / "home"
    root = home / "minecraft-server"
    world = root / "server/world"
    world.mkdir(parents=True)
    (world / "level.dat").write_bytes(b"disposable world data")
    backups = root / "backups"
    backups.mkdir()
    for number in range(24):
        archive = backups / f"world_2000-01-01_00-00-{number:02}.tar.gz"
        archive.write_bytes(f"previous backup {number}".encode())
        os.utime(archive, (number + 1, number + 1))
    # Run the actual installation heredoc, not a second implementation of it.
    document = SKILL.read_text()
    installer = document.split("```bash\ncat > ~/minecraft-server/backup.sh", 1)[1]
    installer = "cat > ~/minecraft-server/backup.sh" + installer.split("```", 1)[0]
    env = {**os.environ, "HOME": str(home)}
    subprocess.run(["bash", "-c", installer], env=env, check=True, timeout=10)
    return root, backups, env


@pytest.mark.platforms("linux", "macos")
@pytest.mark.parametrize("failure", ["tar", "mv"])
def test_failed_backup_keeps_all_previous_archives(backup_env, tmp_path, failure):
    root, backups, env = backup_env
    before = {path.name: path.read_bytes() for path in backups.iterdir()}
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / failure
    stub.write_text(
        "#!/usr/bin/env bash\n"
        + ('printf partial > "$2"\n' if failure == "tar" else "")
        + "echo 'simulated backup failure' >&2\nexit 2\n"
    )
    stub.chmod(0o755)
    env["PATH"] = str(bin_dir) + os.pathsep + env["PATH"]

    result = subprocess.run(
        ["bash", str(root / "backup.sh")], env=env, capture_output=True,
        text=True, timeout=10,
    )

    assert result.returncode != 0
    assert "Saved:" not in result.stdout
    assert "Pruned" not in result.stdout
    assert "Done" not in result.stdout
    assert {path.name: path.read_bytes() for path in backups.iterdir()} == before


@pytest.mark.platforms("linux", "macos")
def test_successful_backup_publishes_world_before_retention(backup_env):
    root, backups, env = backup_env
    before = {path.name for path in backups.iterdir()}
    result = subprocess.run(
        ["bash", str(root / "backup.sh")], env=env, capture_output=True,
        text=True, timeout=10,
    )

    assert result.returncode == 0, result.stderr
    after = {path.name for path in backups.iterdir()}
    assert len(after) == 24
    assert before - after == {"world_2000-01-01_00-00-00.tar.gz"}
    new_archive, = after - before
    with tarfile.open(backups / new_archive, "r:gz") as archive:
        assert archive.extractfile("world/level.dat").read() == b"disposable world data"
    assert "Saved:" in result.stdout
    assert "Pruned 1 old backup(s)" in result.stdout
    assert "Done" in result.stdout
