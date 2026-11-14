"""marker.sh runs EVERY case of the shared update-marker corpus (A7 rule 7).

The judge cases go through marker_judge, the release cases through the real
marker_release_locked under the real A7 sidecar lock against a real file. The
corpus pins the process table, our incarnation and the clock, so they are
injected by redefining marker.sh's three probes (pid_alive, proc_ct,
marker_now) -- the parsing, identity and release logic are the shipped code.
"""

from __future__ import annotations

import json
from pathlib import Path
import shlex
import subprocess

import pytest

pytestmark = pytest.mark.platforms("posix")

ROOT = Path(__file__).resolve().parents[3]
MARKER_SH = ROOT / "scripts" / "desktop-update" / "marker.sh"
CORPUS = json.loads((ROOT / "tests" / "fixtures" / "update_marker_corpus.json").read_text(encoding="utf-8-sig"))

PRELUDE = f"""
set -u
log() {{ :; }}
. {shlex.quote(str(MARKER_SH))}
pid_alive() {{ case " $LIVE " in *" $1="*) return 0 ;; esac; return 1; }}
proc_ct() {{ case " $LIVE " in *" $1="*) local r=" $LIVE "; r="${{r#* $1=}}"; printf '%s' "${{r%% *}}" ;; esac; }}
marker_now() {{ echo {CORPUS["now"]}; }}
"""


def _live(table: dict) -> str:
    return " ".join(f"{pid}={'' if ct is None else repr(ct)}" for pid, ct in table.items())


def _run(script: str) -> list[str]:
    proc = subprocess.run(["bash", "-c", PRELUDE + script], capture_output=True, text=True, timeout=120, check=False)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.splitlines()


def test_every_judge_case(tmp_path):
    lines = []
    for i, case in enumerate(CORPUS["judge"]):
        (tmp_path / f"{i}.txt").write_bytes(case["text"].encode("utf-8"))
        our_ct = case.get("our_ct", CORPUS["our_ct"])
        lines.append(
            f"LIVE={shlex.quote(_live(case['live']))} MY_PID={case.get('our_pid', CORPUS['our_pid'])} "
            f"MY_CT={'' if our_ct is None else repr(our_ct)}; "
            f'marker_judge "$(cat {shlex.quote(str(tmp_path / f"{i}.txt"))})"; '
            f'printf "%s|%s|%s\\n" "$J_VERDICT" "$J_OWNER" "$M_RUN"'
        )
    out = _run("\n".join(lines) + "\n")
    assert len(out) == len(CORPUS["judge"])
    wrong = []
    for case, line in zip(CORPUS["judge"], out):
        verdict, owner, run = line.split("|")
        got = {"verdict": verdict, "owner": int(owner) if owner else None, "run": run or None}
        if got != case["expect"]:
            wrong.append((case["name"], got, case["expect"]))
    assert not wrong


def test_every_release_case(tmp_path):
    lines = []
    for i, case in enumerate(CORPUS["release"]):
        marker = tmp_path / f"{i}" / ".hermes-update-in-progress"
        marker.parent.mkdir()
        marker.write_bytes(case["text"].encode("utf-8"))
        lines.append(
            f"LIVE={shlex.quote(_live(case['live']))} MY_PID={case['releaser_pid']} MY_CT={case['releaser_ct']!r} "
            f"MARKER={shlex.quote(str(marker))}; marker_locked marker_release_locked; echo rc=$?"
        )
    out = _run("\n".join(lines) + "\n")
    assert out == ["rc=0"] * len(CORPUS["release"])
    wrong = []
    for i, case in enumerate(CORPUS["release"]):
        marker = tmp_path / f"{i}" / ".hermes-update-in-progress"
        if not marker.exists():
            got = {"action": "delete"}
        elif marker.read_bytes() == case["text"].encode("utf-8"):
            got = {"action": "keep"}
        else:
            got = {"action": "rewrite", "text": marker.read_text(encoding="utf-8-sig")}
        if got != case["expect"]:
            wrong.append((case["name"], got, case["expect"]))
        assert (marker.parent / ".hermes-update-in-progress.lock").exists()  # the sidecar is never deleted
    assert not wrong
