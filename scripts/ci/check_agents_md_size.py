#!/usr/bin/env python3
"""Fail when a tracked AGENTS.md would be truncated when an agent loads it.

An agent started in a directory gets every AGENTS.md from the git root down to that directory as
ONE project-context block: ``agent/prompt_builder.py`` joins the chain before truncating it, at 6%
of the context window (30,720 chars on a 128k-token model, a 20k floor on small ones). So the
budget is per CHAIN: each AGENTS.md plus its ancestors and provenance labels must fit the
runtime floor (read from prompt_builder.py), and the root, which is in every
chain, stays within ROOT_MAX_CHARS so area files have room. An area file reached as a
subdirectory hint is also capped by ``_MAX_HINT_CHARS`` in ``agent/subdirectory_hints.py``, read
from the source here so the two can never drift. Shrink a file (move long form to the developer
guide or a sibling doc) instead of raising any number.

Exit codes: 0 = all within their caps, 1 = a file or chain is over (paths and sizes printed).
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path, PurePosixPath

ROOT_MAX_CHARS = 12_000



def _int_constant(source: Path, name: str) -> int:
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8-sig"))):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return int(ast.literal_eval(node.value))
    raise SystemExit(f"{source}: no {name} assignment; update scripts/ci/check_agents_md_size.py")


def _chain(rel: str, sizes: dict[str, int]) -> list[str]:
    """Every tracked AGENTS.md from the root down to ``rel``, root first."""
    ancestors = [str(p / "AGENTS.md") if str(p) != "." else "AGENTS.md" for p in PurePosixPath(rel).parents]
    return [p for p in reversed(ancestors) if p in sizes and p != rel] + [rel]


def main(repo: Path) -> int:
    hint_cap = _int_constant(repo / "agent" / "subdirectory_hints.py", "_MAX_HINT_CHARS")
    chain_cap = _int_constant(repo / "agent" / "prompt_builder.py", "CONTEXT_FILE_MAX_CHARS")
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "--", "AGENTS.md", "*/AGENTS.md"], cwd=repo, capture_output=True,
        check=True, stdin=subprocess.DEVNULL, timeout=60,
    ).stdout.decode("utf-8").split("\0")
    sizes = {rel: len((repo / rel).read_text(encoding="utf-8-sig")) for rel in filter(None, tracked)}
    over = []
    for rel, size in sorted(sizes.items()):
        cap = ROOT_MAX_CHARS if rel == "AGENTS.md" else hint_cap
        if size > cap:
            over.append(f"{rel}: {size} chars > {cap} cap")
        chain = _chain(rel, sizes)
        # Runtime wraps each file with "## <relative label>\n\n" and joins sections
        # with two newlines. Reserve extra room for context-scanner annotations.
        total = sum(sizes[p] + 4 + len(p) + 2 for p in chain) + 2 * (len(chain) - 1) + 64
        if len(chain) > 1 and total > chain_cap:
            over.append(f"{' + '.join(chain)}: {total} rendered chars > {chain_cap} floor cap")
    for line in over:
        print(line)
    if over:
        print("Move long form into website/docs/developer-guide/ or a sibling doc instead of raising a cap.")
    return 1 if over else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]))
