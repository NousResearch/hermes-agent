"""Name the picks pnpm's minimumReleaseAge refuses during `hermes pm update --pnpm`.

`pnpm update --no-save` reports a too-young pick only as ERR_PNPM_STRICT_MIN_RELEASE_AGE_REQUIRES_SAVE,
which names no package (pnpm 12 returns before it formats the violations). The same update with
saving allowed and no terminal to prompt on lists them as ERR_PNPM_NO_MATURE_MATCHING_VERSION; it
may write the manifests, so it runs on a scratch copy of the workspace's resolution inputs.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from pm.filesystem import native

REQUIRES_SAVE = "ERR_PNPM_STRICT_MIN_RELEASE_AGE_REQUIRES_SAVE"
NO_MATURE = "ERR_PNPM_NO_MATURE_MATCHING_VERSION"
_VIOLATION = re.compile(r"^\s+(\S+@\S+) was published at ", re.MULTILINE)
# What a lockfile-only resolution reads at the workspace root besides each project's package.json.
_ROOT_INPUTS = ("pnpm-workspace.yaml", "pnpm-lock.yaml", ".npmrc", ".pnpmfile.cjs", ".pnpmfile.mjs")


def is_release_age_failure(output: str) -> bool:
    return REQUIRES_SAVE in output or NO_MATURE in output


def _run(command: list[str], *, cwd: Path, env: dict[str, str], timeout: int) -> subprocess.CompletedProcess:
    # No terminal on stdin: pnpm fails with the list instead of prompting to write it.
    return subprocess.run(command, check=False, cwd=native(cwd), env=env, stdin=subprocess.DEVNULL, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=timeout)


def _stage_inputs(pnpm: str, root: Path, scratch: Path, env: dict[str, str]) -> None:
    listed = _run([pnpm, "list", "--recursive", "--depth", "-1", "--json"], cwd=root, env=env, timeout=120)
    if listed.returncode != 0:
        raise RuntimeError(listed.stdout + listed.stderr)
    for project in json.loads(listed.stdout):
        relative = Path(project["path"]).resolve().relative_to(root.resolve())
        (scratch / relative).mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative / "package.json", scratch / relative / "package.json")
    for name in _ROOT_INPUTS:
        if (root / name).is_file():
            shutil.copy2(root / name, scratch / name)


def immature_picks(pnpm: str, root: Path, env: dict[str, str], *, timeout: int = 3600) -> list[str]:
    """Each `name@version` the recursive lockfile-only update would need excluded; never writes `root`.

    Empty when the scratch copy cannot be made or its update fails for another reason.
    """
    with tempfile.TemporaryDirectory(prefix="pm-pnpm-release-age-") as temporary:
        scratch = Path(temporary)
        try:
            _stage_inputs(pnpm, root, scratch, env)
        except (RuntimeError, OSError, ValueError, KeyError) as exc:
            print(f"  could not copy the workspace to list the too-young versions: {exc}")
            return []
        result = _run([pnpm, "update", "--recursive", "--lockfile-only", "--reporter=ndjson"],
                      cwd=scratch, env=env, timeout=timeout)
    picks = set()
    for line in (result.stdout + "\n" + result.stderr).splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and record.get("code") == NO_MATURE:
            picks.update(_VIOLATION.findall(str(record.get("message", ""))))
    return sorted(picks)


def _locked_versions(lock_text: str) -> dict[str, set[str]]:
    import hermes_yaml

    locked: dict[str, set[str]] = {}
    for key in (hermes_yaml.safe_load(lock_text) or {}).get("packages") or {}:
        name, _, version = str(key).partition("(")[0].rpartition("@")
        if name:
            locked.setdefault(name, set()).add(version)
    return locked


def _kept_versions(entry: str, locked: dict[str, set[str]]) -> list[str] | None:
    """The versions of an exact `name@v1 || v2` entry the lock still holds; None for any other shape."""
    name, _, versions = entry.rpartition("@")
    if not name or "*" in name or not versions:
        return None
    return [v for v in (part.strip() for part in versions.split("||")) if v in locked.get(name, set())]


def _yaml_scalar(value: str) -> str:
    """`value` as a YAML list item pnpm reads back unchanged: plain when that round-trips, else quoted.

    A plain scalar can't start with `@` (every scoped name), `!`, `&`, `*`, `?`, `%`, a quote and the
    like, and `: ` or ` #` inside one changes its meaning; pnpm then refuses to load the workspace.
    """
    import hermes_yaml

    try:
        plain = hermes_yaml.safe_load(f"- {value}") == [value]
    except hermes_yaml.YAMLError:
        plain = False
    return value if plain else "'" + value.replace("'", "''") + "'"


def prune_excludes(workspace_yaml: Path, lock_yaml: Path) -> list[str]:
    """Drop the minimumReleaseAgeExclude versions `lock_yaml` no longer holds; returns the dropped entries.

    pnpm's own minimumReleaseAgeExcludePrune would run on every install, user builds included, and
    rewrite a tracked file there; this runs only where the lock is refreshed. Each entry is one
    line, `  - name@version  # why`, so a dropped line takes its reason with it.
    """
    import hermes_yaml

    text = workspace_yaml.read_text(encoding="utf-8")
    entries = (hermes_yaml.safe_load(text) or {}).get("minimumReleaseAgeExclude") or []
    locked = _locked_versions(lock_yaml.read_text(encoding="utf-8"))
    dropped, lines, in_block = [], [], False
    for line in text.splitlines(keepends=True):
        if not line.startswith((" ", "\t", "#")) and line.strip():
            in_block = line.startswith("minimumReleaseAgeExclude:")
        item = re.match(r"(\s+- )(['\"]?)([^'\"#\s][^'\"#]*?)\2(\s*(?:#.*)?)$", line.rstrip("\r\n")) if in_block else None
        if item and item.group(3).strip() in entries:
            entry = item.group(3).strip()
            kept = _kept_versions(entry, locked)
            if kept is not None and len(kept) < len(entry.rpartition("@")[2].split("||")):
                dropped.append(entry)
                if not kept:
                    continue
                newline = line[len(line.rstrip("\r\n")):]
                narrowed = f"{entry.rpartition('@')[0]}@{' || '.join(kept)}"
                quote = item.group(2)
                value = f"{quote}{narrowed}{quote}" if quote else _yaml_scalar(narrowed)
                line = f"{item.group(1)}{value}{item.group(4)}{newline}"
        lines.append(line)
    if dropped:
        workspace_yaml.write_text("".join(lines), encoding="utf-8")
    return dropped


def exclude_hint(picks: list[str]) -> str:
    if not picks:
        return ("  a range's only matches are newer than minimumReleaseAge (pnpm-workspace.yaml): wait for them\n"
                "  to age, or add each exact name@version to minimumReleaseAgeExclude")
    entries = "\n".join(f"    - {_yaml_scalar(pick)}  # <why it cannot wait>" for pick in picks)
    return ("  only versions newer than minimumReleaseAge (pnpm-workspace.yaml) match: wait for them to age,\n"
            f"  or add them to minimumReleaseAgeExclude:\n{entries}")
