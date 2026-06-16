"""Hermes bridge for https://github.com/virgiliojr94/book-to-skill."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from hermes_constants import display_hermes_home, get_hermes_home

UPSTREAM_REPO = "https://github.com/virgiliojr94/book-to-skill.git"
DEFAULT_REF = "main"
SKILL_NAME = "book-to-skill"


def plugin_dir() -> Path:
    return Path(__file__).resolve().parent


def vendor_root() -> Path:
    return plugin_dir() / "vendor" / "book-to-skill"


def user_skill_path() -> Path:
    return get_hermes_home() / "skills" / SKILL_NAME


def generated_skills_dir() -> Path:
    return get_hermes_home() / "skills"


def extract_script_path() -> Path:
    return vendor_root() / "scripts" / "extract.py"


def _json_ok(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _run_git(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def clone_upstream(*, force: bool = False, ref: str | None = None) -> dict[str, Any]:
    """Shallow-clone upstream into ``plugins/book-to-skill/vendor/book-to-skill``."""
    dest = vendor_root()
    git_dir = dest / ".git"
    if git_dir.exists():
        if not force:
            return {
                "ok": True,
                "skipped": True,
                "action": "already_cloned",
                "path": str(dest),
            }
        shutil.rmtree(dest)

    dest.parent.mkdir(parents=True, exist_ok=True)
    branch = (ref or DEFAULT_REF).strip() or DEFAULT_REF
    attempts = [
        ["git", "clone", "--depth", "1", "--branch", branch, UPSTREAM_REPO, str(dest)],
        ["git", "clone", "--depth", "1", UPSTREAM_REPO, str(dest)],
    ]
    last: subprocess.CompletedProcess[str] | None = None
    for cmd in attempts:
        last = _run_git(cmd)
        if last.returncode == 0 and (dest / "SKILL.md").is_file():
            return {
                "ok": True,
                "action": "cloned",
                "path": str(dest),
                "ref": branch,
            }

    return {
        "ok": False,
        "action": "clone_failed",
        "path": str(dest),
        "ref": branch,
        "stderr": (last.stderr if last else "").strip(),
        "stdout": (last.stdout if last else "").strip(),
    }


def _remove_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink():
        path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink()


def _materialize_skill_source(src: Path, dst: Path) -> dict[str, Any]:
    """Symlink upstream skill into ``~/.hermes/skills/book-to-skill`` (copy fallback on Windows)."""
    if not (src / "SKILL.md").is_file():
        return {
            "ok": False,
            "error": f"Missing SKILL.md in upstream checkout: {src}",
        }

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        _remove_path(dst)

    try:
        os.symlink(str(src.resolve()), str(dst))
        return {
            "ok": True,
            "action": "symlink",
            "source": str(src.resolve()),
            "destination": str(dst),
        }
    except (OSError, NotImplementedError) as sym_err:
        if sys.platform != "win32":
            return {"ok": False, "error": str(sym_err), "action": "symlink_failed"}

        try:
            shutil.copytree(
                str(src.resolve()),
                str(dst),
                symlinks=True,
                ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"),
                dirs_exist_ok=False,
            )
            return {
                "ok": True,
                "action": "copytree",
                "source": str(src.resolve()),
                "destination": str(dst),
                "note": f"Symlink failed ({sym_err}); used copy fallback.",
            }
        except Exception as copy_err:
            return {
                "ok": False,
                "action": "copy_failed",
                "error": str(copy_err),
                "symlink_error": str(sym_err),
            }


def sync_skill_link(*, force: bool = False) -> dict[str, Any]:
    """Expose upstream ``SKILL.md`` to Hermes slash-command discovery."""
    src = vendor_root()
    dst = user_skill_path()
    if not (src / "SKILL.md").is_file():
        return {
            "ok": False,
            "error": "Upstream not installed. Run: hermes book-to-skill install",
            "vendor_path": str(src),
        }
    if dst.exists() and not force:
        return {
            "ok": True,
            "skipped": True,
            "action": "already_linked",
            "destination": str(dst),
            "is_symlink": dst.is_symlink(),
        }
    return _materialize_skill_source(src, dst)


def install(*, force: bool = False, ref: str | None = None) -> dict[str, Any]:
    clone = clone_upstream(force=force, ref=ref)
    if not clone.get("ok"):
        return {"ok": False, "clone": clone}
    link = sync_skill_link(force=True)
    return {
        "ok": bool(link.get("ok")),
        "clone": clone,
        "skill_link": link,
        "slash_command": f"/{SKILL_NAME}",
        "generated_skills_dir": str(generated_skills_dir()),
        "display_generated_skills_dir": f"{display_hermes_home()}/skills",
        "next_steps": [
            "Start a new Hermes session (or run /skills reload in the CLI).",
            f"Use /{SKILL_NAME} <document-path> [slug] to convert a book.",
            f"Write generated book skills under {display_hermes_home()}/skills/<slug>/",
        ],
    }


def status() -> dict[str, Any]:
    vendor = vendor_root()
    skill = user_skill_path()
    extract_py = extract_script_path()
    return {
        "ok": True,
        "plugin": SKILL_NAME,
        "upstream_repo": UPSTREAM_REPO,
        "vendor": {
            "path": str(vendor),
            "present": vendor.is_dir(),
            "has_skill_md": (vendor / "SKILL.md").is_file(),
            "has_extract_py": extract_py.is_file(),
        },
        "hermes_skill": {
            "path": str(skill),
            "present": skill.exists() or skill.is_symlink(),
            "is_symlink": skill.is_symlink() if skill.exists() else False,
        },
        "generated_skills_dir": str(generated_skills_dir()),
        "display_generated_skills_dir": f"{display_hermes_home()}/skills",
        "ready": (vendor / "SKILL.md").is_file() and skill.exists(),
    }


def check_extractors() -> dict[str, Any]:
    script = extract_script_path()
    if not script.is_file():
        return {
            "ok": False,
            "error": "extract.py not found. Run: hermes book-to-skill install",
            "script": str(script),
        }
    py = sys.executable
    proc = subprocess.run(
        [py, str(script), "--check"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        cwd=str(script.parent),
    )
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "script": str(script),
    }


def handle_status_json() -> str:
    return _json_ok(status())
