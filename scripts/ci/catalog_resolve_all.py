#!/usr/bin/env python3
"""Lock every catalog plugin together with Hermes core, the way an install resolves them.

Each entry is fetched at its pinned sha; PM's own workspace generator builds the root
(core's pyproject, the 14-day release quarantine, the plugins' honoured exemptions) and
uv locks it. Locking is the conflict check: it resolves for every platform at once and
never downloads a wheel it does not need for metadata.

Two passes:
  solo      — each plugin alone with core; failure = it cannot install at all.
  together  — plugins added one at a time onto the set that already locks; a plugin
              that breaks the set is bisected against it to name the first partner.

Exit status: every fetch, declaration, solo or combined resolution failure blocks.
The accepted subset is diagnostic only; excluding a broken plugin cannot make the
catalog pass.

``--source`` is the Hermes checkout whose ``pm`` code and ``uv.lock`` are used.
Run under the Hermes environment
(``scripts/run-in-hermes-env python3 ...``).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def fetch(entry, cache: Path) -> Path:
    """The entry's plugin directory at its pinned sha (depth-1 fetch, full clone fallback)."""
    clone = cache / entry.name
    if not (clone / ".git").is_dir() or _head(clone) != entry.sha:
        shutil.rmtree(clone, ignore_errors=True)
        clone.mkdir(parents=True)
        git = ["git", "-C", str(clone)]
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
        subprocess.run([*git, "init", "-q"], check=True, env=env, stdin=subprocess.DEVNULL, timeout=60)
        shallow = subprocess.run([*git, "fetch", "-q", "--depth", "1", entry.repo, entry.sha],
                                 env=env, stdin=subprocess.DEVNULL, capture_output=True, timeout=600)
        if shallow.returncode:
            subprocess.run([*git, "fetch", "-q", entry.repo], check=True, env=env,
                           stdin=subprocess.DEVNULL, capture_output=True, timeout=900)
        subprocess.run([*git, "-c", "advice.detachedHead=false", "checkout", "-q", entry.sha],
                       check=True, env=env, stdin=subprocess.DEVNULL, capture_output=True, timeout=600)
    plugin = (clone / entry.subdir).resolve() if entry.subdir else clone.resolve()
    if plugin != clone.resolve() and clone.resolve() not in plugin.parents:
        raise ValueError(f"subdir {entry.subdir!r} resolves outside the pinned clone")
    return plugin


def _head(clone: Path) -> str:
    result = subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"], capture_output=True,
                            text=True, stdin=subprocess.DEVNULL, timeout=60)
    return result.stdout.strip()


def lock(plugins: list[Path], *, source: Path, scratch: Path, seed: Path | None = None) -> tuple[Path | None, str]:
    """``(uv.lock, "")`` when core + *plugins* resolve, else ``(None, uv's reason)``.

    Seeded like an install (core's own lock by default) so uv keeps core's versions and only
    resolves what the plugins add; an unseeded lock re-resolves all of core from scratch.
    """
    from pm.operations import lock_project
    from pm.package import InstallError
    from pm.workspace import _generate_pyproject

    root = Path(tempfile.mkdtemp(prefix="ws-", dir=scratch)) / "workspace"
    try:
        _generate_pyproject(plugins, root, source=source)
        shutil.copyfile(seed or source / "uv.lock", root / "uv.lock")
        lock_project(root, explicit=True)
    except (InstallError, ValueError) as exc:
        return None, str(exc).strip()
    return root / "uv.lock", ""


def _headline(reason: str) -> str:
    """uv's explanation (``╰─▶`` / ``Because`` / ``error:``) rather than its interpreter banner."""
    lines = [line.strip() for line in reason.splitlines()]
    for marker in ("╰─▶", "Because ", "error: ", "hint: ", "×"):
        for line in lines:
            if marker in line:
                return line[line.index(marker):].lstrip("╰─▶× ")
    return lines[0].split(": ", 1)[-1] if lines else ""


def first_partner(name: str, accepted: list[str], dirs: dict[str, Path], *, source: Path, scratch: Path) -> str:
    """The earliest accepted plugin whose prefix stops *name* resolving (adding members only
    adds constraints, so failure is monotone in the prefix length)."""
    low, high = 0, len(accepted)
    while low < high:
        mid = (low + high) // 2
        locked, _ = lock([dirs[n] for n in accepted[:mid + 1]] + [dirs[name]], source=source, scratch=scratch)
        if locked is None:
            high = mid
        else:
            low = mid + 1
    return accepted[low] if low < len(accepted) else "?"


def main() -> int:
    parser = argparse.ArgumentParser(description="Lock every catalog plugin together with Hermes core.")
    parser.add_argument("--catalog", type=Path, default=Path("plugin-catalog"))
    parser.add_argument("--source", type=Path, default=Path("."), help="Hermes core checkout to lock against")
    parser.add_argument("--cache", type=Path, default=Path(tempfile.gettempdir()) / "catalog-clones")
    parser.add_argument("--report", type=Path, help="write the JSON report here")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    source = args.source.resolve()
    sys.path.insert(0, str(source))
    from hermes_cli.plugin_catalog import load_catalog
    from pm.plugin_declarations import read_python_declaration

    entries = {entry.name: entry for entry in load_catalog(args.catalog)}
    failures: dict[str, str] = {}
    dirs: dict[str, Path] = {}
    with concurrent.futures.ThreadPoolExecutor(args.jobs * 2) as pool:
        futures = {pool.submit(fetch, entry, args.cache): name for name, entry in entries.items()}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                plugin = future.result()
                if read_python_declaration(plugin).is_member:
                    dirs[name] = plugin
            except (subprocess.SubprocessError, OSError, ValueError) as exc:  # one bad repo must not hide the rest
                failures[name] = f"fetch/declaration: {str(exc).strip()[-400:]}"
    print(f"{len(entries)} entries, {len(dirs)} with Python dependencies", flush=True)

    with tempfile.TemporaryDirectory(prefix="catalog-resolve-") as tmp:
        scratch = Path(tmp)
        with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
            solo = dict(zip(dirs, pool.map(lambda n: lock([dirs[n]], source=source, scratch=scratch), dirs)))
        for name, (locked, reason) in solo.items():
            if locked is None:
                failures[name] = f"solo: {reason[-1200:]}"
        order = sorted(n for n in dirs if n not in failures)
        accepted: list[str] = []
        seed: Path | None = None
        for name in order:
            locked, reason = lock([dirs[n] for n in accepted + [name]], source=source, scratch=scratch, seed=seed)
            if locked is None:
                partner = first_partner(name, accepted, dirs, source=source, scratch=scratch)
                failures[name] = f"conflicts with {partner}: {reason[-1200:]}"
                continue
            accepted.append(name)
            seed = scratch / f"seed-{len(accepted)}.lock"
            shutil.copyfile(locked, seed)
            print(f"  + {name} ({len(accepted)}/{len(order)})", flush=True)

    for name, reason in sorted(failures.items()):
        prefix = reason.split(":", 1)[0]
        print(f"::error file={args.catalog / (name + '.yaml')}::{name}: {prefix}: {_headline(reason)}")
    if args.report:
        args.report.write_text(json.dumps({"locked_together": accepted, "failures": failures}, indent=2) + "\n", encoding="utf-8")
    print(f"{len(accepted)} plugins lock together; {len(failures)} failing")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
