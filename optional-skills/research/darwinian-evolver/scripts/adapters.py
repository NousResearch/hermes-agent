"""Tier 2 (external evolvers) and Tier 3 (DSPy / GEPA bridge) adapters.

The core library (Tier 1) is MIT-licensed and in-process. Two adapters
extend it:

Tier 2 — Heavy code evolution
    Uses external CLIs, never Python imports, so license-viral binaries
    (Imbue's ``darwinian-evolver``, AGPL v3) can be invoked as mere
    aggregation. Supported backends:

      * ``openevolve`` (Apache 2.0) — default for Tier 2.
      * ``darwinian-evolver`` (AGPL v3) — opt-in.

    Each backend is detected via ``shutil.which``; if the binary is
    absent the adapter raises :class:`AdapterUnavailable` with an
    install hint instead of crashing.

Tier 3 — DSPy / GEPA bridge
    Exports an experiment's lineage as DSPy-compatible JSONL so the
    ``hermes-agent-self-evolution`` pipeline (a separate Nous repo that
    uses DSPy + GEPA for skill/tool/prompt evolution) can ingest this
    skill's offline runs. This is a pure data contract — no DSPy
    import in this file.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import storage


class AdapterUnavailable(RuntimeError):
    """Raised when an external backend binary is not on PATH."""


# ---------------------------------------------------------------------------
# Tier 2 — external CLI adapters
# ---------------------------------------------------------------------------


@dataclass
class ExternalEvolverAdapter:
    """Thin subprocess wrapper around an external evolver CLI.

    The skill never imports Tier 2 code. It only invokes the binary via
    ``subprocess.run`` (mere aggregation), so the AGPL-v3 licensing of
    Imbue's binary does not propagate into the MIT-licensed Hermes repo.
    The adapter detects the binary lazily and fails closed.
    """

    binary: str
    install_hint: str

    def ensure_available(self) -> str:
        """Return the resolved binary path or raise :class:`AdapterUnavailable`."""
        path = shutil.which(self.binary)
        if not path:
            raise AdapterUnavailable(
                f"{self.binary!r} not found on PATH — {self.install_hint}"
            )
        return path

    def run(self, argv: list[str], *, cwd: Optional[Path] = None, timeout_s: float = 3600) -> subprocess.CompletedProcess:
        """Run ``<binary> <argv>`` and return the completed process.

        The caller owns stdout/stderr parsing — we deliberately don't
        translate the external schema into our :class:`Individual` type
        because the mapping is backend-specific. The ``SKILL.md``
        documents the conventional invocation pattern for each backend.
        """
        path = self.ensure_available()
        return subprocess.run(
            [path, *argv],
            cwd=str(cwd) if cwd else None,
            timeout=timeout_s,
            capture_output=True,
            text=True,
            check=False,
        )


def openevolve_adapter() -> ExternalEvolverAdapter:
    """Apache 2.0, default Tier 2 backend.

    OpenEvolve is the open-source reimplementation of DeepMind's
    AlphaEvolve. Installed via ``pip install openevolve``; it exposes
    an ``openevolve`` CLI after installation.
    """
    return ExternalEvolverAdapter(
        binary="openevolve",
        install_hint="install with `pip install openevolve` (Apache 2.0)",
    )


def darwinian_evolver_adapter() -> ExternalEvolverAdapter:
    """Imbue's ``darwinian-evolver``, AGPL v3.

    Hermes does NOT import this tool. The adapter runs it as an opaque
    subprocess so Imbue's code never enters the Hermes process. Users
    install it themselves via ``pip install darwinian-evolver`` after
    accepting the AGPL-v3 license; this skill only decides whether to
    shell out to it.
    """
    return ExternalEvolverAdapter(
        binary="darwinian-evolver",
        install_hint=(
            "install with `pip install darwinian-evolver` (AGPL v3 — review license before use)."
        ),
    )


# ---------------------------------------------------------------------------
# Tier 3 — DSPy / GEPA JSONL bridge
# ---------------------------------------------------------------------------


def export_dspy_jsonl(
    conn: sqlite3.Connection,
    out_path: Path,
    *,
    include_all_generations: bool = False,
) -> int:
    """Write one JSONL record per candidate, DSPy-compatible.

    Each record is the minimal shape DSPy's ``ProgramOfThought`` and
    ``ChainOfThought`` trainers expect for offline data:

    ``{"text": <genome>, "metric": {<objective>: <value>}, ...}``

    Additional fields (``candidate_id``, ``generation``, ``lineage``,
    ``operator``, ``source``) are preserved for GEPA reflective
    selection; DSPy ignores unknown keys.

    Returns the number of records written.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Best training-split score per (candidate, objective).
    rows = conn.execute(
        """
        SELECT c.id, c.genome, c.generation, f.objective, f.value, f.held_out
          FROM candidates c
          JOIN fitness   f ON f.candidate_id = c.id
         WHERE f.held_out = 0
         ORDER BY c.generation, c.id, f.objective
        """
    ).fetchall()

    grouped: dict[str, dict] = {}
    for r in rows:
        rec = grouped.setdefault(r["id"], {
            "candidate_id": r["id"],
            "text":         r["genome"],
            "generation":   r["generation"],
            "metric":       {},
            "source":       "darwinian-evolver",
            "schema":       "dspy-offline/v1",
        })
        rec["metric"][r["objective"]] = float(r["value"])

    # Attach parent edges for GEPA's reflective trace.
    for cid, rec in grouped.items():
        edges = conn.execute(
            "SELECT parent_id, operator FROM lineage WHERE child_id = ?",
            (cid,),
        ).fetchall()
        rec["lineage"] = [{"parent": e["parent_id"], "operator": e["operator"]} for e in edges]

    # Filter to the best per generation unless the caller asks for all.
    if not include_all_generations:
        best_by_gen: dict[int, dict] = {}
        for rec in grouped.values():
            g = rec["generation"]
            if g not in best_by_gen or _best_score(rec) > _best_score(best_by_gen[g]):
                best_by_gen[g] = rec
        records = list(best_by_gen.values())
    else:
        records = list(grouped.values())

    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


def export_gepa_trace(
    conn: sqlite3.Connection,
    out_path: Path,
) -> int:
    """Write a GEPA-style reflective trace JSONL.

    GEPA (Agrawal et al. 2025) expects triples of ``(candidate, feedback,
    revision)`` — our ``critique_then_edit`` operator already produces
    this shape natively. This exporter filters the lineage down to those
    edges and emits them in order.

    Other operator edges are omitted; GEPA's trainer only uses the
    reflective ones.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = conn.execute(
        """
        SELECT l.parent_id, l.child_id, l.operator, l.prompt_hash,
               pc.genome AS parent_genome, cc.genome AS child_genome,
               pc.generation AS parent_gen, cc.generation AS child_gen
          FROM lineage    l
          JOIN candidates pc ON pc.id = l.parent_id
          JOIN candidates cc ON cc.id = l.child_id
         WHERE l.operator IN ('critique_then_edit', 'meta_mutator')
         ORDER BY cc.generation, l.child_id
        """
    ).fetchall()
    with out_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({
                "schema":      "gepa-trace/v1",
                "operator":    r["operator"],
                "parent":      {"id": r["parent_id"], "genome": r["parent_genome"], "generation": r["parent_gen"]},
                "child":       {"id": r["child_id"],  "genome": r["child_genome"],  "generation": r["child_gen"]},
                "prompt_hash": r["prompt_hash"],
            }, ensure_ascii=False) + "\n")
    return len(rows)


def _best_score(rec: dict) -> float:
    """Return the representative score from a DSPy record for ranking."""
    metric = rec.get("metric") or {}
    if not metric:
        return float("-inf")
    return max(float(v) for v in metric.values())
