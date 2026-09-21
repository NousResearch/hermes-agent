"""Generate small, provenance-bound Obsidian role notes from the federation registry."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile

MARKER = "<!-- hermes-federation-navigation:v1 -->"


def render_notes(payload: bytes) -> dict[str, str]:
    registry = json.loads(payload)
    departments = registry["departments"]
    roles = [role for department in departments for role in department["roles"]]
    ids = [role["id"] for role in roles]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[a-z0-9-]+", name) for name in ids):
        raise ValueError("role ids must be unique safe note names")
    if any(target not in ids for role in roles for target in role.get("handoffs", [])):
        raise ValueError("role handoff has no registry entry")
    digest = sha256(payload).hexdigest()
    provenance = f"\nNarrative navigation only; no runtime or approval authority.\nSource registry SHA-256: `{digest}`.\n"
    notes = {}
    index = [MARKER, "# Federation navigation", provenance,
             f"{len(departments)} departments; {len(roles)} roles. Open only the role needed."]
    for department in departments:
        index += ["", "## " + department["display_name"], "", department["description"], ""]
        for role in department["roles"]:
            name = role["id"]
            index.append(f"- [{role['display_name']}](roles/{name}.md)")
            handoffs = ", ".join(f"[{target}]({target}.md)" for target in role.get("handoffs", [])) or "None declared"
            notes[f"roles/{name}.md"] = "\n".join([
                MARKER, "# " + role["display_name"], "", role["description"], "",
                f"Department: {department['display_name']}. Declared authority: {role['authority']}. Schedule: {role['schedule']}.",
                "", "Handoffs: " + handoffs, "", "[Department index](../Index.md)", provenance,
            ])
    notes["Index.md"] = "\n".join(index) + "\n"
    return notes


def write_notes(root: Path, notes: dict[str, str]) -> int:
    targets = []
    for relative, content in notes.items():
        target = root / relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("note path escapes navigation root")
        if any(parent.is_symlink() for parent in (target, *target.parents)):
            raise ValueError("navigation paths must not be symlinks")
        if target.exists() and not target.read_text(encoding="utf-8").startswith(MARKER):
            raise ValueError(f"refusing to overwrite authored note: {target}")
        targets.append((target, content))
    updated = 0
    for target, content in targets:
        if target.exists() and target.read_text(encoding="utf-8") == content:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".navigation-", dir=target.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
        updated += 1
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = args.registry.read_bytes()
    notes = render_notes(payload)
    print(json.dumps({"notes": len(notes), "updated": write_notes(args.output, notes),
                      "source_sha256": sha256(payload).hexdigest()}))


if __name__ == "__main__":
    main()
