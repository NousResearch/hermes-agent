#!/usr/bin/env python3
"""Build a publishable skill-set catalog from local skill directories.

Publisher-side companion to ``hermes skills install-set``. Given one or more
skill directories (each containing a ``SKILL.md``), this produces a static
site tree implementing the agentskills #254 discovery index plus an AI
Catalog (https://ai-catalog.io) wrapper entry:

    out/
    ├── .well-known/
    │   ├── ai-catalog.json                  # AI Catalog with one skill-set entry
    │   └── agent-skills/
    │       ├── index.json                   # #254 discovery index
    │       ├── <single-file-skill>/SKILL.md # type: skill-md
    │       └── <multi-file-skill>.tar.gz    # type: archive
    └── (serve this directory over HTTPS)

Skills with only a SKILL.md are published as ``type: "skill-md"``; skills
with supporting files (scripts/, references/, ...) are packed into a
``.tar.gz`` with SKILL.md at the archive root, per the spec. Every artifact
gets a required ``sha256:`` digest.

Usage:
    python scripts/publish_skill_set.py \
        --name "Backend Dev" \
        --command backend-dev \
        --description "Everything for backend feature work." \
        --instruction "Prefer TDD. Run the linter before opening a PR." \
        --host-name "Example Corp" \
        --out ./public \
        path/to/skill-a path/to/skill-b ...

Then serve ``./public`` at your origin and install with:
    hermes skills install-set https://your-origin.example
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

INDEX_SCHEMA = "https://schemas.agentskills.io/discovery/0.2.0/schema.json"
SKILL_SET_ENTRY_TYPE = "application/agent-skills+json"
HERMES_SET_EXTENSION = "io.hermes.skill-set"


def _digest(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _frontmatter_field(skill_md: str, field: str) -> str:
    """Cheap YAML frontmatter single-field read (name/description)."""
    lines = skill_md.splitlines()
    if not lines or lines[0].strip() != "---":
        return ""
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if line.startswith(f"{field}:"):
            return line.split(":", 1)[1].strip().strip("\"'")
    return ""


def _pack_tar_gz(skill_dir: Path) -> bytes:
    """Deterministic tar.gz of a skill directory, SKILL.md at archive root."""
    buf = io.BytesIO()
    # mtime=0 + sorted names -> byte-stable archives, so digests only change
    # when content changes (plays nice with #254's digest-based caching).
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tf:
            for path in sorted(skill_dir.rglob("*")):
                if not path.is_file() or path.is_symlink():
                    continue
                rel = path.relative_to(skill_dir).as_posix()
                info = tarfile.TarInfo(name=rel)
                data = path.read_bytes()
                info.size = len(data)
                info.mtime = 0
                info.mode = 0o644
                tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    ap.add_argument("skill_dirs", nargs="+", type=Path,
                    help="Skill directories (each must contain SKILL.md)")
    ap.add_argument("--name", required=True, help="Skill set display name")
    ap.add_argument("--description", default="", help="Skill set description")
    ap.add_argument("--command", default="",
                    help="Suggested load-alias (io.hermes.skill-set extension)")
    ap.add_argument("--instruction", default="",
                    help="Shared instruction preamble (io.hermes.skill-set extension)")
    ap.add_argument("--host-name", default="", help="AI Catalog host displayName")
    ap.add_argument("--host-id", default="", help="AI Catalog host identifier (e.g. did:web:...)")
    ap.add_argument("--out", type=Path, default=Path("public"), help="Output directory")
    args = ap.parse_args()

    well_known = args.out / ".well-known"
    skills_dir = well_known / "agent-skills"
    skills_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for skill_dir in args.skill_dirs:
        skill_md_path = skill_dir / "SKILL.md"
        if not skill_md_path.is_file():
            print(f"error: {skill_dir} has no SKILL.md", file=sys.stderr)
            return 1
        skill_md = skill_md_path.read_text(encoding="utf-8")
        name = _frontmatter_field(skill_md, "name") or skill_dir.name
        description = _frontmatter_field(skill_md, "description")

        extra_files = [p for p in skill_dir.rglob("*")
                       if p.is_file() and p != skill_md_path]
        if extra_files:
            artifact = _pack_tar_gz(skill_dir)
            (skills_dir / f"{name}.tar.gz").write_bytes(artifact)
            entries.append({
                "name": name,
                "type": "archive",
                "description": description,
                "url": f"/.well-known/agent-skills/{name}.tar.gz",
                "digest": _digest(artifact),
            })
            print(f"  archive   {name}  ({len(extra_files) + 1} files, "
                  f"{len(artifact):,} bytes)")
        else:
            content = skill_md.encode("utf-8")
            dest = skills_dir / name
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "SKILL.md").write_bytes(content)
            entries.append({
                "name": name,
                "type": "skill-md",
                "description": description,
                "url": f"/.well-known/agent-skills/{name}/SKILL.md",
                "digest": _digest(content),
            })
            print(f"  skill-md  {name}")

    index = {"$schema": INDEX_SCHEMA, "skills": entries}
    (skills_dir / "index.json").write_text(
        json.dumps(index, indent=2) + "\n", encoding="utf-8")

    set_entry = {
        "identifier": f"urn:air:{args.host_id or 'example'}:skill-set:"
                      f"{args.command or args.name.lower().replace(' ', '-')}",
        "displayName": args.name,
        "description": args.description,
        "type": SKILL_SET_ENTRY_TYPE,
        "url": "/.well-known/agent-skills/index.json",
    }
    ext: dict = {}
    if args.command:
        ext["command"] = args.command
    if args.instruction:
        ext["instruction"] = args.instruction
    if ext:
        set_entry["extensions"] = {HERMES_SET_EXTENSION: ext}

    catalog = {
        "specVersion": "1.0",
        "host": {
            "displayName": args.host_name or args.name,
            **({"identifier": args.host_id} if args.host_id else {}),
        },
        "entries": [set_entry],
    }
    (well_known / "ai-catalog.json").write_text(
        json.dumps(catalog, indent=2) + "\n", encoding="utf-8")

    print(f"\nWrote {args.out}/.well-known/ai-catalog.json "
          f"and agent-skills/index.json ({len(entries)} skills).")
    print("Serve the output directory at your origin, then:")
    print("  hermes skills install-set https://<your-origin>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
