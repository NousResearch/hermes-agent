"""Regression tests for the minecraft-modpack-server optional skill.

The backup script template embedded in SKILL.md must not report success or
run the retention prune when archive creation fails: a partial archive that
keeps the final ``world_*.tar.gz`` name enters the retention set and selects
a good backup for deletion (#134910).
"""

import os
import subprocess
from pathlib import Path

SKILL_PATH = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "gaming"
    / "minecraft-modpack-server"
    / "SKILL.md"
)

OLDEST = "world_2000-01-01_00-00-00.tar.gz"
MAX_BACKUPS = 24

# A tar double that writes a few bytes to its -f target and then fails, like a
# full disk would: the archive exists but is garbage.
TAR_STUB = """\
#!/usr/bin/env bash
target=""
prev=""
for a in "$@"; do
    if [ "$prev" = "-f" ]; then target="$a"; fi
    prev="$a"
done
if [ -n "$target" ]; then printf 'partial-garbage' > "$target"; fi
echo "tar: simulated write failure" >&2
exit 2
"""

# An rm double that records its arguments and deletes nothing, so the test can
# prove the retention prune never ran.
RM_STUB = """\
#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$RM_LOG"
exit 0
"""


def _extract_backup_script() -> str:
    lines = SKILL_PATH.read_text(encoding="utf-8").splitlines()
    start = lines.index("cat > ~/minecraft-server/backup.sh << 'SCRIPT'") + 1
    end = lines.index("SCRIPT", start)
    return "\n".join(lines[start:end]) + "\n"


def _make_home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    world = home / "minecraft-server/server/world"
    world.mkdir(parents=True)
    (world / "level.dat").write_bytes(b"level")
    backups = home / "minecraft-server/backups"
    backups.mkdir()
    for i in range(MAX_BACKUPS):
        name = OLDEST if i == 0 else f"world_2000-01-{i + 1:02d}_00-00-00.tar.gz"
        target = backups / name
        target.write_bytes(b"good")
        stamp = 946684800 + i * 86400  # 2000-01-01 UTC + i days, oldest first
        os.utime(target, (stamp, stamp))
    return home


def _run_backup(
    home: Path, extra_path: Path | None = None
) -> subprocess.CompletedProcess:
    script = home.parent / "backup.sh"
    script.write_text(_extract_backup_script())
    env = dict(os.environ, HOME=str(home))
    if extra_path is not None:
        env["PATH"] = f"{extra_path}{os.pathsep}{env['PATH']}"
        env["RM_LOG"] = str(home.parent / "rm-calls.log")
    return subprocess.run(
        ["/bin/bash", str(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_failed_tar_reports_failure_and_never_prunes(tmp_path):
    home = _make_home(tmp_path)
    stub_bin = tmp_path / "stub-bin"
    stub_bin.mkdir()
    for name, body in (("tar", TAR_STUB), ("rm", RM_STUB)):
        stub = stub_bin / name
        stub.write_text(body)
        stub.chmod(0o755)

    result = _run_backup(home, extra_path=stub_bin)

    assert result.returncode == 1, result.stdout
    assert "FAILED: tar exited non-zero" in result.stderr
    assert "Saved:" not in result.stdout
    assert "Done" not in result.stdout
    backups = sorted(p.name for p in (home / "minecraft-server/backups").iterdir())
    assert len(backups) == MAX_BACKUPS  # no partial archive entered the set
    assert OLDEST in backups  # good backups were kept
    assert not any(name.endswith(".partial") for name in backups)
    rm_calls = (tmp_path / "rm-calls.log").read_text().splitlines()
    assert rm_calls, "expected cleanup of the staged partial file"
    assert all(call.endswith(".partial") for call in rm_calls), rm_calls


def test_successful_backup_publishes_and_prunes_oldest(tmp_path):
    home = _make_home(tmp_path)

    result = _run_backup(home)

    assert result.returncode == 0, result.stderr
    assert "Saved:" in result.stdout
    backups = {p.name for p in (home / "minecraft-server/backups").iterdir()}
    assert OLDEST not in backups  # pruned to stay at MAX_BACKUPS
    assert len(backups) == MAX_BACKUPS  # 24 synthetic + 1 new - 1 pruned
    assert not any(name.endswith(".partial") for name in backups)
    newest = max(
        home.glob("minecraft-server/backups/world_*.tar.gz"),
        key=lambda p: p.stat().st_mtime,
    )
    listing = subprocess.run(
        ["tar", "-tzf", str(newest)], capture_output=True, text=True, timeout=60
    )
    assert "world/level.dat" in listing.stdout, listing.stderr
