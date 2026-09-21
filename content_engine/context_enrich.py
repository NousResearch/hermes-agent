"""Turn a thin signal into a rich, size-capped context blob for the educational
generator: commit diffs, paper abstracts, repo READMEs, config diffs. Read-only,
best-effort, degrades to the signal summary."""
from __future__ import annotations
import base64
import os, subprocess
from pathlib import Path

MAX_BLOB = 2400  # chars fed to the LLM per signal


def _repo_path(name: str) -> Path:
    return (Path(os.path.expanduser("~/.hermes/profiles")) if name == "profiles"
            else Path(os.path.expanduser(f"~/repos/{name}")))


def _git_show(repo: str, sha: str) -> str:
    try:
        r = subprocess.run(["git","-C",str(_repo_path(repo)),"show","--stat","-p","--no-color",
                            f"{sha}", "-1"], capture_output=True, text=True, timeout=15)
        return r.stdout if r.returncode == 0 else ""
    except Exception:
        return ""


def _gh_readme(full_name: str) -> str:
    try:
        r = subprocess.run(["gh","api",f"repos/{full_name}/readme","--jq",".content"],
                           capture_output=True, text=True, timeout=15)
        if r.returncode == 0 and r.stdout.strip():
            return base64.b64decode(r.stdout).decode("utf-8", "ignore")
    except Exception:
        pass
    return ""


def enrich(signal: dict) -> str:
    t = signal.get("signal_type", "")
    summary = (signal.get("summary") or signal.get("title") or "").strip()
    detail = ""
    if t in ("harness_change", "github_push", "hermes_pr") and signal.get("sha"):
        detail = _git_show(signal.get("repo", "KenseiAgent"), signal["sha"])
    elif t == "gitradar_repo" and signal.get("full_name"):
        detail = _gh_readme(signal["full_name"])
    elif t in ("research_tool", "research_signal"):
        detail = signal.get("summary", "") + "\n" + signal.get("body", "")
    blob = (summary + "\n\n" + detail).strip()
    return blob[:MAX_BLOB]
