"""Mem0 OSS provider SDKs must ship in the managed mem0 closure.

Regression for #122242: ``_oss_providers.py`` tags each OSS backend with its
``pip_dep`` (``ollama`` for the Ollama LLM/embedder, ``psycopg2-binary`` for
pgvector), but after the #102765 bundles change nothing consumed those tags —
the ``mem0`` extra resolved to mem0ai alone. Selecting such a backend then
blew up inside ``Memory.from_config`` (mem0ai's lazy-install ``input()``
prompt reads EOF in the gateway), surfacing as the misleading
``Mem0 backend not initialized: EOF when reading a line``.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.utils import canonicalize_name

from plugins.memory.mem0._oss_providers import SECTION_REGISTRIES


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _oss_pip_deps() -> set[str]:
    deps = set()
    for _section, registry in SECTION_REGISTRIES:
        for definition in registry.values():
            if dep := definition.get("pip_dep"):
                deps.add(canonicalize_name(dep))
    return deps


def _mem0_lock_closure() -> set[str]:
    """Every distribution the locked ``mem0`` extra can install.

    Roots are the workspace root's resolved ``mem0`` optional-dependencies;
    the fixpoint over ``[[package]]`` dependencies adds what mem0ai already
    pulls transitively (e.g. qdrant-client), so only genuinely absent SDKs
    fail the assertion.
    """
    lock = tomllib.loads((_repo_root() / "uv.lock").read_text(encoding="utf-8"))
    by_name = {p["name"]: p for p in lock["package"]}
    hermes = by_name["hermes-agent"]
    seeds = [d["name"] for d in hermes["optional-dependencies"]["mem0"]]
    seen: set[str] = set()
    stack = list(seeds)
    while stack:
        name = stack.pop()
        key = canonicalize_name(name)
        if key in seen:
            continue
        seen.add(key)
        if (pkg := by_name.get(name)) is not None:
            stack.extend(d["name"] for d in pkg.get("dependencies", []))
    return seen


def test_oss_provider_pip_deps_ship_in_mem0_closure():
    missing = _oss_pip_deps() - _mem0_lock_closure()
    assert not missing, (
        f"Mem0 OSS backends need {sorted(missing)} but the locked mem0 extra "
        "does not provide them; declare them in the mem0 extra in pyproject.toml "
        "and relock (see #122242)"
    )
