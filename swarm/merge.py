from __future__ import annotations

import difflib
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Conflict:
    file_path: str
    line_start: int
    line_end: int
    agents: list[str]
    base_lines: list[str]
    variant_lines: dict[str, list[str]]
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass
class Resolution:
    conflict_id: str
    chosen_agent: str  # agent id or "merged"
    merged_lines: list[str]
    reason: str = ""


class ConflictResolver:
    def __init__(self, strategy: str = "auto") -> None:
        if strategy not in ("auto", "ours", "theirs", "manual"):
            raise ValueError(f"Unknown strategy: {strategy!r}")
        self.strategy = strategy

    # ------------------------------------------------------------------
    # Diff helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _changed_ranges(base_lines: list[str], mod_lines: list[str]) -> list[tuple[int, int, list[str]]]:
        """Return list of (start, end, replacement_lines) for every changed region."""
        sm = difflib.SequenceMatcher(None, base_lines, mod_lines, autojunk=False)
        ranges: list[tuple[int, int, list[str]]] = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            ranges.append((i1, i2, mod_lines[j1:j2]))
        return ranges

    @staticmethod
    def _ranges_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
        return a[0] < b[1] and b[0] < a[1]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_conflicts(
        self,
        base_content: str,
        changes: list[tuple[str, str]],
        file_path: str = "",
    ) -> list[Conflict]:
        base_lines = base_content.splitlines(keepends=True)
        # Collect per-agent changed ranges
        agent_ranges: list[tuple[str, int, int, list[str]]] = []
        for agent_id, modified in changes:
            mod_lines = modified.splitlines(keepends=True)
            for start, end, repl in self._changed_ranges(base_lines, mod_lines):
                agent_ranges.append((agent_id, start, end, repl))

        # Group overlapping ranges into conflicts
        conflicts: list[Conflict] = []
        used: set[int] = set()
        for i, (aid_i, s_i, e_i, repl_i) in enumerate(agent_ranges):
            if i in used:
                continue
            cluster = [i]
            lo, hi = s_i, e_i
            # Iteratively absorb all overlapping ranges
            changed = True
            while changed:
                changed = False
                for j, (aid_j, s_j, e_j, _repl_j) in enumerate(agent_ranges):
                    if j in used or j in {c for c in cluster}:
                        continue
                    if self._ranges_overlap((lo, hi), (s_j, e_j)):
                        cluster.append(j)
                        lo = min(lo, s_j)
                        hi = max(hi, e_j)
                        changed = True

            # Only a conflict if more than one agent touches the region
            agents_in_cluster = list(dict.fromkeys(agent_ranges[c][0] for c in cluster))
            if len(agents_in_cluster) < 2:
                continue

            used.update(cluster)
            variant_lines: dict[str, list[str]] = {}
            for c in cluster:
                a, _s, _e, r = agent_ranges[c]
                variant_lines.setdefault(a, []).extend(r)

            conflicts.append(Conflict(
                file_path=file_path,
                line_start=lo,
                line_end=hi,
                agents=agents_in_cluster,
                base_lines=base_lines[lo:hi],
                variant_lines=variant_lines,
            ))

        return conflicts

    def resolve_trivial(self, conflicts: list[Conflict]) -> list[Resolution]:
        resolutions: list[Resolution] = []
        for c in conflicts:
            variants = list(c.variant_lines.values())
            # All agents produced the same replacement — trivially resolved
            if all(v == variants[0] for v in variants):
                resolutions.append(Resolution(
                    conflict_id=c.id,
                    chosen_agent="merged",
                    merged_lines=variants[0],
                    reason="all agents agree",
                ))
            elif self.strategy == "ours":
                first = c.agents[0]
                resolutions.append(Resolution(
                    conflict_id=c.id,
                    chosen_agent=first,
                    merged_lines=c.variant_lines[first],
                    reason="strategy=ours",
                ))
            elif self.strategy == "theirs":
                last = c.agents[-1]
                resolutions.append(Resolution(
                    conflict_id=c.id,
                    chosen_agent=last,
                    merged_lines=c.variant_lines[last],
                    reason="strategy=theirs",
                ))
            # auto and manual leave unresolved conflicts alone
        return resolutions

    def merge_changes(
        self,
        base: str,
        changes: list[tuple[str, str]],
        conflicts: list[Conflict] | None = None,
        resolutions: list[Resolution] | None = None,
    ) -> str:
        base_lines = base.splitlines(keepends=True)

        if conflicts is None:
            conflicts = self.detect_conflicts(base, changes)
        if resolutions is None:
            resolutions = self.resolve_trivial(conflicts)

        resolution_map: dict[str, Resolution] = {r.conflict_id: r for r in resolutions}
        conflict_regions: dict[int, Conflict] = {}
        for c in conflicts:
            for ln in range(c.line_start, c.line_end):
                conflict_regions[ln] = c

        # Collect non-overlapping edits from each agent
        all_edits: dict[tuple[int, int], list[str]] = {}
        for agent_id, modified in changes:
            mod_lines = modified.splitlines(keepends=True)
            for start, end, repl in self._changed_ranges(base_lines, mod_lines):
                in_conflict = any(ln in conflict_regions for ln in range(start, max(end, start + 1)))
                if not in_conflict:
                    all_edits[(start, end)] = repl

        # Apply conflict resolutions
        for c in conflicts:
            res = resolution_map.get(c.id)
            if res is not None:
                all_edits[(c.line_start, c.line_end)] = res.merged_lines
            else:
                # Unresolved — emit conflict markers
                markers = [f"<<<<<<< CONFLICT ({', '.join(c.agents)})\n"]
                for agent, lines in c.variant_lines.items():
                    markers.append(f"======= {agent}\n")
                    markers.extend(lines)
                markers.append(">>>>>>>\n")
                all_edits[(c.line_start, c.line_end)] = markers

        # Build output by replaying edits in reverse order
        result = list(base_lines)
        for (start, end), repl in sorted(all_edits.items(), reverse=True):
            result[start:end] = repl

        return "".join(result)

    def three_way_merge(
        self,
        base: str,
        ours: str,
        theirs: str,
    ) -> tuple[str, list[Conflict]]:
        changes = [("ours", ours), ("theirs", theirs)]
        conflicts = self.detect_conflicts(base, changes)
        resolutions = self.resolve_trivial(conflicts)
        merged = self.merge_changes(base, changes, conflicts, resolutions)
        unresolved = [
            c for c in conflicts
            if c.id not in {r.conflict_id for r in resolutions}
        ]
        return merged, unresolved

    @staticmethod
    def format_conflict_report(conflicts: list[Conflict]) -> str:
        if not conflicts:
            return "No conflicts detected."
        parts: list[str] = [f"Conflict report: {len(conflicts)} conflict(s)\n"]
        for i, c in enumerate(conflicts, 1):
            parts.append(f"\n--- Conflict {i} [{c.id}] ---")
            if c.file_path:
                parts.append(f"  File: {c.file_path}")
            parts.append(f"  Lines: {c.line_start + 1}-{c.line_end}")
            parts.append(f"  Agents: {', '.join(c.agents)}")
            parts.append(f"  Base ({len(c.base_lines)} lines):")
            for ln in c.base_lines:
                parts.append(f"    | {ln.rstrip()}")
            for agent, lines in c.variant_lines.items():
                parts.append(f"  {agent} ({len(lines)} lines):")
                for ln in lines:
                    parts.append(f"    | {ln.rstrip()}")
        return "\n".join(parts) + "\n"
