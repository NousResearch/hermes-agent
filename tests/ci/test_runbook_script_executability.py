"""deploy/RUNBOOK.md must be executable as written (card t_0a660daa).

The RUNBOOK tells operators to paste its commands verbatim; `deploy/deploy-test.sh`
was committed mode 100644, so the documented bare invocation died with
"Permission denied" (exit 126) on a fresh checkout while the doc still promised
it works. This test parses the RUNBOOK's shell fences and fails on any
command-position path under deploy/ or scripts/ whose git mode lacks the exec
bit — unless the same position carries an explicit interpreter (`bash`,
`python`, `run-in-hermes-env python …`), which makes the mode moot by design.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOC = "deploy/RUNBOOK.md"

FENCE_OPEN = re.compile(r"^\s*```(\w*)\s*$")
# a repo-relative script path sitting at the head of a command segment
SCRIPT_HEAD = re.compile(r"^(\./)?((?:deploy|scripts)/[\w./+-]+)(?=$|[\s;&|)]|\s#)")
# the head token itself is an interpreter: the mode bit does not matter then
INTERPRETERS = {"bash", "sh", "zsh", "python", "python3", "env", "node", "source", "."}


def _git_index_modes() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    proc = subprocess.run(["git", "ls-files", "-s"], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=60, check=True)
    modes = {}
    for line in proc.stdout.splitlines():
        meta, path = line.split("\t", 1)
        modes[path] = meta.split()[0]
    return modes


def _shell_lines(text: str) -> list[tuple[int, str]]:
    """(file_lineno, line) for every line inside a shell-ish fenced block."""
    out: list[tuple[int, str]] = []
    lang: str | None = None
    inside = False
    for lineno, line in enumerate(text.splitlines(), 1):
        m = FENCE_OPEN.match(line)
        if m:
            inside = not inside
            lang = m.group(1).lower() if inside else None
            continue
        if inside and lang in ("bash", "sh", "shell", "console", ""):
            out.append((lineno, line))
    return out


def _bare_script_refs(text: str) -> list[tuple[int, str]]:
    """(lineno, path) for tracked-file scripts invoked by bare path in the doc."""
    modes = _git_index_modes()
    refs: list[tuple[int, str]] = []
    for lineno, line in _shell_lines(text):
        line = line.strip()
        if line[:2] in ("$ ", "# "):  # console prompt / comment line
            line = line[2:].strip()
        if not line or line.startswith("#"):
            continue
        for segment in re.split(r"&&|;|\|", line):
            segment = re.split(r"\s#", segment.strip())[0].strip()
            m = SCRIPT_HEAD.match(segment)
            if not m:
                continue
            head = m.group(2)
            if head.rsplit("/", 1)[-1] in INTERPRETERS:
                continue
            if modes.get(head) is not None and modes[head] != "100755":
                refs.append((lineno, head))
    return refs


def test_runbook_actually_invokes_deploy_scripts_bare() -> None:
    """Guard the guard: the scanner must see the deployer, or the check is vacuous."""
    text = (REPO / DOC).read_text(encoding="utf-8")
    heads = []
    for _, line in _shell_lines(text):
        for segment in re.split(r"&&|;|\|", re.sub(r"^\$ ", "", line.strip())):
            m = SCRIPT_HEAD.match(re.split(r"\s#", segment.strip())[0].strip())
            if m:
                heads.append(re.sub(r"^\./", "", m.group(2)))
    assert "deploy/deploy-test.sh" in heads, (
        f"the scanner no longer finds deploy/deploy-test.sh invoked in {DOC}; "
        "the RUNBOOK changed shape — update this test deliberately, not by silence"
    )
    assert "scripts/run-in-hermes-env" in heads
    # qa_gateway.py is only ever run through an explicit interpreter; that is
    # why its 100644 mode is fine and must not become a bare head unnoticed.
    assert "deploy/qa_gateway.py" not in heads


def test_runbook_bare_invocations_are_executable_in_git() -> None:
    text = (REPO / DOC).read_text(encoding="utf-8")
    bad = _bare_script_refs(text)
    assert not bad, (
        f"{DOC} documents bare-path invocations of git-mode-100644 files "
        "(they fail with exit 126 on a fresh checkout): "
        + ", ".join(f"line {lineno}: {path}" for lineno, path in bad)
        + " — fix by `git update-index --chmod=+x <path>` or by giving the "
        "documented command an explicit interpreter (e.g. `bash deploy/…`)"
    )
