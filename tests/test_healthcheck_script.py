import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HEALTHCHECK = REPO_ROOT / "scripts" / "healthcheck-host.sh"


def run_healthcheck(tmp_path, df_output, meminfo, **env_overrides):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_df = bin_dir / "df"
    fake_df.write_text(f"#!/usr/bin/env bash\ncat <<'EOF'\n{df_output}\nEOF\n")
    fake_df.chmod(0o755)

    meminfo_path = tmp_path / "meminfo"
    meminfo_path.write_text(meminfo)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "HERMES_HEALTHCHECK_MEMINFO": str(meminfo_path),
        }
    )
    env.update(env_overrides)

    return subprocess.run(
        ["bash", str(HEALTHCHECK)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_healthcheck_passes_under_thresholds(tmp_path):
    result = run_healthcheck(
        tmp_path,
        """Filesystem 1024-blocks Used Available Capacity Mounted on
/dev/root 1000 500 500 50% /
/dev/data 1000 700 300 70% /var/lib/hermes data""",
        "MemTotal:       1000 kB\nMemAvailable:    400 kB\n",
        DISK_USAGE_THRESHOLD="90",
        MEMORY_USAGE_THRESHOLD="90",
    )

    assert result.returncode == 0
    assert "OK   /var/lib/hermes data: 70% used (/dev/data)" in result.stdout
    assert "Healthcheck passed" in result.stdout


def test_healthcheck_fails_when_disk_reaches_threshold(tmp_path):
    result = run_healthcheck(
        tmp_path,
        """Filesystem 1024-blocks Used Available Capacity Mounted on
/dev/root 1000 900 100 90% /""",
        "MemTotal:       1000 kB\nMemAvailable:    500 kB\n",
        DISK_USAGE_THRESHOLD="90",
        MEMORY_USAGE_THRESHOLD="90",
    )

    assert result.returncode == 1
    assert "FAIL /: 90% used (/dev/root)" in result.stdout
    assert "Healthcheck failed: 1 issue(s) found" in result.stdout


def test_healthcheck_fails_when_memory_reaches_threshold(tmp_path):
    result = run_healthcheck(
        tmp_path,
        """Filesystem 1024-blocks Used Available Capacity Mounted on
/dev/root 1000 100 900 10% /""",
        "MemTotal:       1000 kB\nMemAvailable:    100 kB\n",
        DISK_USAGE_THRESHOLD="90",
        MEMORY_USAGE_THRESHOLD="90",
    )

    assert result.returncode == 1
    assert "FAIL 90% used (threshold: 90%)" in result.stdout


def test_healthcheck_rejects_invalid_threshold(tmp_path):
    result = run_healthcheck(
        tmp_path,
        """Filesystem 1024-blocks Used Available Capacity Mounted on
/dev/root 1000 100 900 10% /""",
        "MemTotal:       1000 kB\nMemAvailable:    900 kB\n",
        DISK_USAGE_THRESHOLD="abc",
    )

    assert result.returncode == 2
    assert "DISK_USAGE_THRESHOLD must be an integer between 1 and 100" in result.stderr


def test_healthcheck_reports_unreadable_meminfo_as_failure(tmp_path):
    result = run_healthcheck(
        tmp_path,
        """Filesystem 1024-blocks Used Available Capacity Mounted on
/dev/root 1000 100 900 10% /""",
        "MemTotal:       1000 kB\n",
        DISK_USAGE_THRESHOLD="90",
        MEMORY_USAGE_THRESHOLD="90",
    )

    assert result.returncode == 1
    assert "FAIL unable to read memory information" in result.stdout


def test_healthcheck_reports_impossible_meminfo_as_failure(tmp_path):
    result = run_healthcheck(
        tmp_path,
        """Filesystem 1024-blocks Used Available Capacity Mounted on
/dev/root 1000 100 900 10% /""",
        "MemTotal:       1000 kB\nMemAvailable:    1001 kB\n",
        DISK_USAGE_THRESHOLD="90",
        MEMORY_USAGE_THRESHOLD="90",
    )

    assert result.returncode == 1
    assert "FAIL unable to read memory information" in result.stdout


def test_healthcheck_reports_malformed_df_line_as_failure(tmp_path):
    result = run_healthcheck(
        tmp_path,
        """Filesystem 1024-blocks Used Available Capacity Mounted on
malformed df line""",
        "MemTotal:       1000 kB\nMemAvailable:    900 kB\n",
        DISK_USAGE_THRESHOLD="90",
        MEMORY_USAGE_THRESHOLD="90",
    )

    assert result.returncode == 1
    assert "FAIL unable to parse filesystem usage line: malformed df line" in result.stdout
