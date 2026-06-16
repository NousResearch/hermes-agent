"""Hermes bridge for in-tree ``vendor/openclaw-mirror`` (OpenClaw extensions + packages)."""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from hermes_constants import display_hermes_home, get_hermes_home

SKILL_NAME_RE = re.compile(r"^name:\s*(.+?)\s*$", re.MULTILINE)

# OpenClaw extensions under vendor/openclaw-mirror/extensions/
KNOWN_EXTENSIONS: list[dict[str, Any]] = [
    {
        "id": "hypura-harness",
        "description": "Hypura harness daemon, evolution skills, channel readiness",
        "skills_subdir": "skills",
        "hermes_cli": ["harness", "hypura"],
        "toolsets": ["openclaw", "vrchat"],
    },
    {
        "id": "hypura-provider",
        "description": "Hypura local LLM provider profile (OpenClaw)",
        "skills_subdir": None,
        "hermes_cli": [],
        "toolsets": [],
    },
    {
        "id": "vrchat-relay",
        "description": "VRChat relay OSC integration (TypeScript in harness)",
        "skills_subdir": None,
        "hermes_cli": ["harness"],
        "toolsets": ["vrchat", "openclaw"],
    },
]

# Top-level vendor packages (sibling of extensions/)
VENDOR_PACKAGES: list[dict[str, Any]] = [
    {
        "id": "AI-Scientist",
        "marker": "launch_scientist.py",
        "hermes_tool": "ai_scientist_research",
    },
    {
        "id": "ShinkaEvolve",
        "marker": "shinka/__init__.py",
        "hermes_tool": "shinka_run",
    },
    {"id": "ATLAS", "marker": None, "hermes_tool": None},
    {"id": "a2ui", "marker": None, "hermes_tool": None},
    {"id": "elan", "marker": None, "hermes_tool": None},
    {"id": "nc-kart-proof", "marker": None, "hermes_tool": None},
]

# Other in-repo plugins that complement vendor (not under vendor/)
SIBLING_PLUGINS: list[dict[str, str]] = [
    {
        "id": "questframe-fh6vr",
        "path": "plugins/questframe_fh6vr",
        "note": "QuestFrame 0.36 DIBR/6DoF lab — enable in plugins.enabled separately",
    },
    {
        "id": "book-to-skill",
        "path": "plugins/book-to-skill",
        "note": "External book-to-skill upstream clone plugin",
    },
]


def plugin_dir() -> Path:
    return Path(__file__).resolve().parent


def repo_root() -> Path:
    return plugin_dir().parents[1]


def vendor_mirror_root() -> Path:
    return repo_root() / "vendor" / "openclaw-mirror"


def extensions_root() -> Path:
    return vendor_mirror_root() / "extensions"


def skills_home() -> Path:
    return get_hermes_home() / "skills"


def _json_ok(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


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
    if not (src / "SKILL.md").is_file():
        return {
            "ok": False,
            "error": f"Missing SKILL.md: {src}",
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


def read_skill_name(skill_dir: Path) -> str:
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        return skill_dir.name
    text = skill_md.read_text(encoding="utf-8", errors="replace")
    match = SKILL_NAME_RE.search(text)
    if match:
        return match.group(1).strip()
    return skill_dir.name


def _is_vendor_extension_dir(ext_dir: Path) -> bool:
    if not ext_dir.is_dir():
        return False
    name = ext_dir.name
    if name.startswith("_") or name.startswith("."):
        return False
    return (ext_dir / "openclaw.plugin.json").is_file() or (ext_dir / "skills").is_dir()


def discover_skill_sources(
    *,
    extension_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return skill source dirs under extensions/*/skills/*."""
    root = extensions_root()
    if not root.is_dir():
        return []

    sources: list[dict[str, Any]] = []
    for ext_dir in sorted(root.iterdir()):
        if not _is_vendor_extension_dir(ext_dir):
            continue
        ext_id = ext_dir.name
        if extension_id and ext_id != extension_id:
            continue
        skills_root = ext_dir / "skills"
        if not skills_root.is_dir():
            continue
        for skill_dir in sorted(skills_root.iterdir()):
            if not skill_dir.is_dir():
                continue
            if not (skill_dir / "SKILL.md").is_file():
                continue
            skill_name = read_skill_name(skill_dir)
            sources.append(
                {
                    "extension": ext_id,
                    "skill_name": skill_name,
                    "source": skill_dir,
                    "destination": skills_home() / skill_name,
                }
            )
    return sources


def sync_skill_link(
    skill_name: str,
    src: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    dst = skills_home() / skill_name
    if dst.exists() and not force:
        return {
            "ok": True,
            "skipped": True,
            "skill_name": skill_name,
            "destination": str(dst),
            "is_symlink": dst.is_symlink() if dst.exists() else False,
        }
    result = _materialize_skill_source(src, dst)
    result["skill_name"] = skill_name
    return result


def sync_all_skills(
    *,
    force: bool = False,
    extension_id: str | None = None,
) -> dict[str, Any]:
    mirror = vendor_mirror_root()
    if not mirror.is_dir():
        return {
            "ok": False,
            "error": f"Vendor mirror missing: {mirror}",
            "hint": "Run scripts/sync_openclaw_vendor.py or ensure vendor/openclaw-mirror is checked out",
        }

    sources = discover_skill_sources(extension_id=extension_id)
    if extension_id and not sources:
        return {
            "ok": False,
            "error": f"No skills found for extension '{extension_id}'",
            "extensions_root": str(extensions_root()),
        }

    links: list[dict[str, Any]] = []
    ok = True
    for item in sources:
        link = sync_skill_link(
            item["skill_name"],
            item["source"],
            force=force,
        )
        link["extension"] = item["extension"]
        links.append(link)
        ok = ok and bool(link.get("ok"))

    return {
        "ok": ok,
        "action": "sync_skills",
        "vendor_mirror": str(mirror),
        "skills_home": str(skills_home()),
        "display_skills_home": f"{display_hermes_home()}/skills",
        "count": len(links),
        "skills": links,
    }


def _package_row(pkg: dict[str, Any]) -> dict[str, Any]:
    pkg_id = pkg["id"]
    pkg_path = vendor_mirror_root() / pkg_id
    marker = pkg.get("marker")
    present = pkg_path.is_dir()
    marker_ok = True
    if marker:
        marker_ok = (pkg_path / marker).exists()
    return {
        "id": pkg_id,
        "path": str(pkg_path),
        "present": present,
        "marker": marker,
        "marker_ok": marker_ok if marker else present,
        "hermes_tool": pkg.get("hermes_tool"),
        "tool_ready": bool(marker_ok) if pkg.get("hermes_tool") else present,
    }


def _extension_row(ext_id: str) -> dict[str, Any]:
    meta = next((e for e in KNOWN_EXTENSIONS if e["id"] == ext_id), None)
    ext_path = extensions_root() / ext_id
    plugin_json = ext_path / "openclaw.plugin.json"
    skills = discover_skill_sources(extension_id=ext_id)
    return {
        "id": ext_id,
        "path": str(ext_path),
        "present": ext_path.is_dir(),
        "has_openclaw_plugin_json": plugin_json.is_file(),
        "description": (meta or {}).get("description"),
        "hermes_cli": (meta or {}).get("hermes_cli", []),
        "toolsets": (meta or {}).get("toolsets", []),
        "skill_count": len(skills),
        "skills": [s["skill_name"] for s in skills],
    }


def list_units() -> dict[str, Any]:
    mirror = vendor_mirror_root()
    ext_ids = sorted(
        p.name
        for p in extensions_root().iterdir()
        if _is_vendor_extension_dir(p)
    ) if extensions_root().is_dir() else []

    return {
        "ok": True,
        "vendor_mirror": str(mirror),
        "mirror_present": mirror.is_dir(),
        "extensions": [_extension_row(ext_id) for ext_id in ext_ids],
        "packages": [_package_row(pkg) for pkg in VENDOR_PACKAGES],
        "sibling_plugins": [
            {
                **row,
                "abs_path": str(repo_root() / row["path"]),
                "present": (repo_root() / row["path"] / "plugin.yaml").is_file(),
            }
            for row in SIBLING_PLUGINS
        ],
    }


def status() -> dict[str, Any]:
    listing = list_units()
    sources = discover_skill_sources()
    linked: list[dict[str, Any]] = []
    for item in sources:
        dst = item["destination"]
        linked.append(
            {
                "skill_name": item["skill_name"],
                "extension": item["extension"],
                "linked": dst.exists() or dst.is_symlink(),
                "is_symlink": dst.is_symlink() if dst.exists() else False,
                "path": str(dst),
            }
        )

    packages = listing["packages"]
    tools_ready = all(
        p.get("tool_ready", True) for p in packages if p.get("hermes_tool")
    )
    skills_ready = all(entry["linked"] for entry in linked) if linked else False

    return {
        "ok": listing["mirror_present"],
        "plugin": "openclaw-vendor",
        "vendor_mirror": listing["vendor_mirror"],
        "mirror_present": listing["mirror_present"],
        "extensions": listing["extensions"],
        "packages": packages,
        "sibling_plugins": listing["sibling_plugins"],
        "skills": linked,
        "skills_home": str(skills_home()),
        "display_skills_home": f"{display_hermes_home()}/skills",
        "ready": listing["mirror_present"] and skills_ready and tools_ready,
    }


def install(*, force: bool = False, extension_id: str | None = None) -> dict[str, Any]:
    sync = sync_all_skills(force=force, extension_id=extension_id)
    st = status()
    slash_commands = sorted(
        {f"/{s['skill_name']}" for s in sync.get("skills", []) if s.get("ok")}
    )
    return {
        "ok": bool(sync.get("ok")) and st.get("mirror_present", False),
        "sync": sync,
        "status": {
            "ready": st.get("ready"),
            "skills_home": st.get("skills_home"),
        },
        "slash_commands": slash_commands,
        "next_steps": [
            "Add openclaw-vendor to plugins.enabled in ~/.hermes/config.yaml if not already.",
            "Start a new session or run /skills reload so slash commands refresh.",
            "Enable toolsets openclaw and/or vrchat via hermes tools for harness/VRChat tools.",
            "Start harness: hermes harness start",
            "QuestFrame 0.36 lab: enable questframe-fh6vr plugin separately.",
        ],
    }


def handle_status_json() -> str:
    return _json_ok(status())
