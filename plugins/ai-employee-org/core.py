"""Core bootstrap for the AI employee org plugin."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from hermes_constants import display_hermes_home, get_hermes_home

PLUGIN_NAME = "ai-employee-org"
BOARD_SLUG = "ai-company"

PROFILES: tuple[dict[str, str], ...] = (
    {
        "name": "secretary",
        "description": "Orchestrator: triage, decompose, schedule, human handoff.",
        "soul": "SOUL-secretary.md",
    },
    {
        "name": "job-recruiter",
        "description": "Creates and publishes job postings; tracks applicants.",
        "soul": "SOUL-job-recruiter.md",
    },
    {
        "name": "job-seeker",
        "description": "Finds roles, drafts applications, tracks pipeline.",
        "soul": "SOUL-job-seeker.md",
    },
    {
        "name": "self-improver",
        "description": "Reviews skills, memory, failures; proposes improvements.",
        "soul": "SOUL-self-improver.md",
    },
    {
        "name": "delivery-worker",
        "description": "Executes contracted deliverables end-to-end.",
        "soul": "SOUL-delivery-worker.md",
    },
)

OPS_DIRS: tuple[Path, ...] = (
    Path(r"C:/Users/downl/Documents/ops/job-seeker"),
    Path(r"C:/Users/downl/Documents/ops/job-recruiter"),
    Path(r"C:/Users/downl/Documents/ops/delivery"),
    Path(r"C:/Users/downl/Documents/ops/cursor-learning-inbox"),
)


def plugin_dir() -> Path:
    return Path(__file__).resolve().parent


def skill_source_dir() -> Path:
    return plugin_dir() / "skill"


def scripts_dir() -> Path:
    return plugin_dir() / "scripts"


def stack_file() -> Path:
    return plugin_dir() / "config" / "ai-employee-stack.yaml"


def _profiles_root() -> Path:
    return Path.home() / ".hermes" / "profiles"


def _run_hermes(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "hermes_cli.main", *args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=check,
    )


def _profile_exists(name: str) -> bool:
    proc = _run_hermes(["profile", "show", name], check=False)
    return proc.returncode == 0


def _materialize_skill(dest: Path, *, force: bool = False) -> dict[str, Any]:
    src = skill_source_dir()
    if not (src / "SKILL.md").is_file():
        return {"ok": False, "error": f"Missing bundled skill at {src}"}
    if dest.exists() or dest.is_symlink():
        if not force:
            return {"ok": True, "action": "exists", "destination": str(dest)}
        if dest.is_symlink():
            dest.unlink()
        elif dest.is_dir():
            shutil.rmtree(dest)
        else:
            dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(str(src.resolve()), str(dest))
        return {"ok": True, "action": "symlink", "destination": str(dest)}
    except (OSError, NotImplementedError):
        shutil.copytree(
            src,
            dest,
            symlinks=True,
            ignore=shutil.ignore_patterns(".git", "__pycache__"),
            dirs_exist_ok=True,
        )
        return {"ok": True, "action": "copytree", "destination": str(dest)}


def install_skill(*, profiles: bool = True, force: bool = False) -> dict[str, Any]:
    """Link bundled skill into default and worker profile homes."""
    results: dict[str, Any] = {"default": None, "profiles": {}}
    default_dest = get_hermes_home() / "skills" / "ai-employee-org"
    results["default"] = _materialize_skill(default_dest, force=force)
    if profiles:
        for spec in PROFILES:
            dest = _profiles_root() / spec["name"] / "skills" / "ai-employee-org"
            results["profiles"][spec["name"]] = _materialize_skill(dest, force=force)
    return results


def setup_profiles(*, clone: bool = False) -> dict[str, Any]:
    """Create profiles and copy SOUL templates."""
    template_dir = skill_source_dir() / "templates"
    out: dict[str, Any] = {"profiles": {}, "souls": {}}
    for spec in PROFILES:
        name = spec["name"]
        if _profile_exists(name):
            out["profiles"][name] = "exists"
        else:
            args = ["profile", "create", name, "--description", spec["description"]]
            if clone:
                args.append("--clone")
            proc = _run_hermes(args, check=False)
            out["profiles"][name] = "created" if proc.returncode == 0 else proc.stderr.strip()[:200]
        soul_src = template_dir / spec["soul"]
        soul_dest = _profiles_root() / name / "SOUL.md"
        if soul_src.is_file():
            soul_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(soul_src, soul_dest)
            out["souls"][name] = str(soul_dest)
    return out


def setup_kanban(*, board_slug: str = BOARD_SLUG) -> dict[str, Any]:
    proc = _run_hermes(
        ["kanban", "boards", "create", board_slug, "--name", "AI Company", "--switch"],
        check=False,
    )
    board_result = "created" if proc.returncode == 0 else "exists_or_failed"
    init_proc = _run_hermes(["kanban", "init"], check=False)
    return {
        "board": board_result,
        "init": "ok" if init_proc.returncode == 0 else init_proc.stderr.strip()[:200],
        "board_slug": board_slug,
    }


def ensure_ops_dirs() -> dict[str, str]:
    created: dict[str, str] = {}
    subdirs = {
        OPS_DIRS[0]: ("reports", "applications"),
        OPS_DIRS[1]: ("postings", "drafts", "templates", "reports"),
        OPS_DIRS[2]: ("projects", "reports"),
        OPS_DIRS[3]: ("processed",),
    }
    for base in OPS_DIRS:
        base.mkdir(parents=True, exist_ok=True)
        for sub in subdirs.get(base, ()):
            (base / sub).mkdir(exist_ok=True)
        seen = OPS_DIRS[0] / "seen.json"
        if base == OPS_DIRS[0] and not seen.exists():
            seen.write_text("[]\n", encoding="utf-8")
        created[str(base)] = "ready"
    return created


def status() -> dict[str, Any]:
    from . import cron_install
    from . import stack as stack_mod

    skill_dest = get_hermes_home() / "skills" / "ai-employee-org"
    return {
        "plugin": PLUGIN_NAME,
        "plugin_dir": str(plugin_dir()),
        "skill_linked": skill_dest.exists() or skill_dest.is_symlink(),
        "skill_path": str(skill_dest),
        "profiles": {p["name"]: _profile_exists(p["name"]) for p in PROFILES},
        "ops_dirs": {str(p): p.is_dir() for p in OPS_DIRS},
        "stack_file": str(stack_file()),
        "stack_exists": stack_file().is_file(),
        "cron_scripts": cron_install.list_installers(),
        "hermes_home": display_hermes_home(),
    }


def install_all(
    *,
    clone: bool = False,
    board_slug: str = BOARD_SLUG,
    apply_stack: bool = True,
    install_crons: bool = True,
    telegram_chat_id: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    from . import cron_install
    from . import stack as stack_mod

    result: dict[str, Any] = {
        "success": True,
        "dry_run": dry_run,
        "steps": {},
        "next_steps": [
            "hermes plugins enable ai-employee-org",
            "hermes ai-employees status",
            "hermes gateway run",
            "hermes kanban list --status ready,running,blocked",
        ],
    }
    if dry_run:
        result["steps"] = {
            "plugin": "would_enable",
            "skill": "would_install",
            "profiles": "would_setup",
            "kanban": "would_init",
            "ops": "would_create",
            "stack": "would_apply" if apply_stack else "skipped",
            "cron": "would_install" if install_crons else "skipped",
        }
        return result

    result["steps"]["plugin"] = stack_mod.enable_plugin(dry_run=False)
    if not result["steps"]["plugin"].get("ok", True):
        result["success"] = False

    result["steps"]["skill"] = install_skill(profiles=True, force=True)
    result["steps"]["profiles"] = setup_profiles(clone=clone)
    result["steps"]["kanban"] = setup_kanban(board_slug=board_slug)
    result["steps"]["ops"] = ensure_ops_dirs()

    if apply_stack:
        stack_out = stack_mod.apply_ai_employee_stack(dry_run=False)
        result["steps"]["stack"] = stack_out
        if not stack_out.get("ok", True):
            result["success"] = False

    if install_crons:
        cron_out = cron_install.install_all_crons(telegram_chat_id=telegram_chat_id)
        result["steps"]["cron"] = cron_out
        if not cron_out.get("ok", True):
            result["success"] = False

    return result


def json_dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)
