"""The macOS upgrade arm waits for the relaunched app's backend wherever it runs.

In rehearsal rc.5-v0.23.2 Squirrel.Mac installed 0.23.2 and relaunched the app, and the arm
failed 6 s later with "could not discover the relaunched backend's listening port". It looked
once, only among the app's children for a `serve` process. The relaunched app was still
starting, and 0.23.2 runs its gateway as a launchd job (`osascript` -> `python3 -m hermes_cli.main
gateway run`, parent pid 1), outside the app's process tree.
"""

import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "tests" / "install" / "e2e-assets" / "mac-backend-health.sh"
PAYLOAD = "/Users/runner/apps/Hermes Agent.app/Contents/Resources/agent-payload"
GATEWAY = f"{PAYLOAD}/tools/python-3.14/bin/python3 -m hermes_cli.main --profile default gateway run --quiet"

# ps, lsof and curl read a process table: one "pid|etime|args|port|first poll it listens on|healthy" per line.
FAKE_PS = r"""#!/usr/bin/env bash
table="$FAKE_TABLE"
if [ "$1" = "-o" ]; then
  awk -F'|' -v p="$4" '$1 == p { print $2 }' "$table"
else
  awk -F'|' '{ printf "%6s %11s %s\n", $1, $2, $3 }' "$table"
fi
"""
FAKE_LSOF = r"""#!/usr/bin/env bash
pid=""
while [ $# -gt 0 ]; do [ "$1" = "-p" ] && pid="$2"; shift; done
echo x >> "$FAKE_POLLS.$pid"
polls=$(wc -l < "$FAKE_POLLS.$pid")
awk -F'|' -v p="$pid" -v n="$polls" '$1 == p && $4 != "" && n >= $5 { print "p" p; print "f9"; print "n127.0.0.1:" $4 }' "$FAKE_TABLE"
"""
FAKE_CURL = r"""#!/usr/bin/env bash
out="" url=""
while [ $# -gt 0 ]; do
  case "$1" in -o) out="$2"; shift ;; http*) url="$1" ;; esac
  shift
done
port="${url#http://127.0.0.1:}"; port="${port%%/*}"
grep -q "^[^|]*|[^|]*|[^|]*|$port|[^|]*|yes$" "$FAKE_TABLE" || exit 7
printf '{"status":"ok","port":%s}\n' "$port" > "$out"
"""


def _run(tmp_path: Path, rows: list[tuple], *, ref: int, timeout: int = 20):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("ps", FAKE_PS), ("lsof", FAKE_LSOF), ("curl", FAKE_CURL)):
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    table = tmp_path / "table"
    table.write_text("".join("|".join(str(field) for field in row) + "\n" for row in rows))
    out = tmp_path / "backend-health.json"
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "FAKE_TABLE": str(table),
           "FAKE_POLLS": str(tmp_path / "polls")}
    result = subprocess.run(
        ["bash", str(HELPER), "--payload", PAYLOAD, "--newer-than", str(ref), "--timeout", str(timeout),
         "--out", str(out)],
        capture_output=True, text=True, timeout=timeout + 30, env=env, check=False,
    )
    return result, out


@pytest.mark.platforms("posix")
def test_waits_for_a_backend_outside_the_app_process_tree(tmp_path):
    rows = [
        (3884, "00:06", "/Users/runner/apps/Hermes Agent.app/Contents/MacOS/Hermes Agent", "", "", "no"),
        (3968, "00:04", f"/usr/bin/osascript -l JavaScript -e exec {GATEWAY}", "", "", "no"),
        (3971, "00:04", GATEWAY, 49512, 3, "yes"),
        (1186, "05:12", "/opt/homebrew/bin/node feed-server.mjs", 49385, 1, "yes"),
    ]
    result, out = _run(tmp_path, rows, ref=3884)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "49512"
    assert '"port":49512' in out.read_text()


@pytest.mark.platforms("posix")
def test_a_payload_process_older_than_the_relaunch_does_not_count(tmp_path):
    rows = [
        (3884, "00:06", "/Users/runner/apps/Hermes Agent.app/Contents/MacOS/Hermes Agent", "", "", "no"),
        (1750, "01:40", f"{PAYLOAD}/tools/python-3.14/bin/python3 -m hermes_cli.main serve", 49394, 1, "yes"),
    ]
    result, out = _run(tmp_path, rows, ref=3884, timeout=3)
    assert result.returncode == 1
    assert "newer than pid 3884 answered /api/health within 3s" in result.stderr
    assert "hermes_cli.main serve" in result.stderr
    assert not out.exists()
