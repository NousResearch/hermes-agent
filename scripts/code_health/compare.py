"""Base vs head: the ratchet verdict.

Every unit has its own cap: a function or file already over target may not grow past the value
it has on the base revision; anything new must meet the target. Pattern rules compare multisets
of fingerprints, so fixing one violation and adding another still fails.

Code that moves keeps its cap and its existing violations. Head units are matched to base units
ONE-TO-ONE (a base unit is consumed by at most one head unit), in this order:

1. same file, same name, same body (unchanged code; reserved first, so a copy of it is new);
2. same file, same body (a rename, or an anonymous callback whose ordinal shifted);
3. any file, same body, when the origin's name is gone from its own file (a real move);
4. same file, same name (edited in place).

Every base hit is then owned by exactly one head scope: the head unit its unit matched, else the
same scope in the file's head path. Each old occurrence pays for one new occurrence, never two.

An occurrence is old only if it still sits where it was: inside a unit whose body is unchanged
(matched by passes 1-3, so moves and renames keep their debt), or on a line that survives from
the base file (compared without comments or whitespace, so dropping an allow comment or
re-indenting is not new debt). An identical violation re-added on a new line is new, even in
the same function as one that was removed.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher

from scripts.code_health.config import RULES_BY_ID, TARGETS
from scripts.code_health.gitio import Change
from scripts.code_health.model import MODULE_SCOPE, FileMeasure, Finding, Hit, Unit

Key = tuple[str, str]  # (path, qualname)


class _Matcher:
    def __init__(self, base: dict[str, FileMeasure], head: dict[str, FileMeasure],
                 base_of: dict[str, str | None], head_of: dict[str, str | None]) -> None:
        self.base, self.head, self.base_of, self.head_of = base, head, base_of, head_of
        self.taken: set[Key] = set()
        self.match: dict[Key, tuple[str, Unit]] = {}
        self.same_body: set[Key] = set()  # head units matched to an identical base body
        self.by_hash: dict[str, list[tuple[str, Unit]]] = defaultdict(list)
        for bpath, bf in sorted(base.items()):
            for unit in bf.units.values():
                self.by_hash[unit.body_hash].append((bpath, unit))

    def _base_file(self, hpath: str) -> tuple[str | None, FileMeasure | None]:
        bpath = self.base_of.get(hpath, hpath)
        return bpath, (self.base.get(bpath) if bpath else None)

    def _name_gone(self, bpath: str, qual: str) -> bool:
        if "<anon>" in qual:  # ordinals are positional, never evidence that the unit survived
            return True
        hpath = self.head_of.get(bpath, bpath)
        hf = self.head.get(hpath) if hpath else None
        return hf is None or qual not in hf.units

    def _claim(self, key: Key, origin: tuple[str, Unit]) -> None:
        self.match[key] = origin
        self.taken.add((origin[0], origin[1].qualname))

    def _free(self, bpath: str | None, qual: str) -> bool:
        return bpath is not None and (bpath, qual) not in self.taken

    def run(self) -> dict[Key, tuple[str, Unit]]:
        pending = [(hpath, q, u) for hpath, hf in sorted(self.head.items()) for q, u in hf.units.items()]
        for step in (self._same_unchanged, self._same_file_body, self._moved_body):
            pending = [(hpath, q, u) for hpath, q, u in pending if not step(hpath, q, u)]
        self.same_body = set(self.match)
        pending = [(hpath, q, u) for hpath, q, u in pending if not self._same_name(hpath, q, u)]
        return self.match

    def _same_unchanged(self, hpath: str, qual: str, unit: Unit) -> bool:
        bpath, bf = self._base_file(hpath)
        prior = bf.units.get(qual) if bf else None
        if bpath is None or prior is None or prior.body_hash != unit.body_hash:
            return False
        if not self._free(bpath, qual):
            return False
        self._claim((hpath, qual), (bpath, prior))
        return True

    def _same_file_body(self, hpath: str, qual: str, unit: Unit) -> bool:
        bpath, _ = self._base_file(hpath)
        for opath, origin in self.by_hash.get(unit.body_hash, []):
            if opath == bpath and self._free(opath, origin.qualname):
                self._claim((hpath, qual), (opath, origin))
                return True
        return False

    def _moved_body(self, hpath: str, qual: str, unit: Unit) -> bool:
        for opath, origin in self.by_hash.get(unit.body_hash, []):
            if self._free(opath, origin.qualname) and self._name_gone(opath, origin.qualname):
                self._claim((hpath, qual), (opath, origin))
                return True
        return False

    def _same_name(self, hpath: str, qual: str, unit: Unit) -> bool:
        bpath, bf = self._base_file(hpath)
        if bpath is None or bf is None or qual not in bf.units or not self._free(bpath, qual):
            return False
        self._claim((hpath, qual), (bpath, bf.units[qual]))
        return True


def _file_findings(path: str, hf: FileMeasure, bf: FileMeasure | None) -> list[Finding]:
    if hf.error:
        return [Finding(path, "MEASURE", MODULE_SCOPE, 1, f"could not be measured: {hf.error}")]
    lines = hf.metrics.get("FILE_LINES", 0)
    if lines <= TARGETS["FILE_LINES"]:
        return []
    base_lines = bf.metrics.get("FILE_LINES", 0) if bf else 0
    if lines <= max(TARGETS["FILE_LINES"], base_lines):
        return []
    return [Finding(path, "FILE_LINES", MODULE_SCOPE, 1, _detail(
        lines, TARGETS["FILE_LINES"], base_lines if bf else None, "lines"))]


def _unit_findings(path: str, hf: FileMeasure, match: dict[Key, tuple[str, Unit]]) -> list[Finding]:
    findings: list[Finding] = []
    for qual, unit in hf.units.items():
        origin = match.get((path, qual))
        prior = origin[1] if origin else None
        for metric, value in unit.metrics.items():
            target = TARGETS[metric]
            if value <= target:
                continue
            was = prior.metrics.get(metric) if prior else None
            if value > max(target, was or 0):
                findings.append(Finding(path, metric, qual, unit.line,
                                        _detail(value, target, was, metric)))
    return findings


def _detail(value: int, target: int, was: int | None, what: str) -> str:
    if was is None:
        return f"{what} {value} > target {target} (new code must meet the target)"
    if was <= target:
        return f"{what} {value} > target {target} (was {was})"
    return f"{what} {value} > {was}, its value on main (over target {target}: it may only go down)"


def _owned_base_hits(base: dict[str, FileMeasure], head_of: dict[str, str | None],
                     match: dict[Key, tuple[str, Unit]]) -> dict[str, Counter[Hit]]:
    """Each base hit, re-keyed once to the head (path, scope) that now owns it."""
    owner = {(opath, origin.qualname): key for key, (opath, origin) in match.items()}
    owned: dict[str, Counter[Hit]] = defaultdict(Counter)
    for bpath, bf in base.items():
        for hit, count in bf.hits.items():
            hpath, scope = owner.get((bpath, hit.scope), (head_of.get(bpath, bpath), hit.scope))
            if hpath is None:
                continue
            owned[hpath][Hit(hit.rule, scope, hit.text)] += count
    return owned


def _surviving_lines(hf: FileMeasure, bf: FileMeasure | None) -> set[int]:
    """Head lines that are unchanged base lines (code only: comments and whitespace ignored)."""
    if bf is None:
        return set()
    old = [bf.code_line(n) for n in range(1, len(bf.lines) + 1)]
    new = [hf.code_line(n) for n in range(1, len(hf.lines) + 1)]
    # autojunk off: `except Exception:` and `pass` are frequent lines, never noise here.
    blocks = SequenceMatcher(None, old, new, autojunk=False).get_matching_blocks()
    return {b.b + i + 1 for b in blocks for i in range(b.size)}


def _hit_findings(path: str, hf: FileMeasure, base_hits: Counter[Hit],
                  kept: set[int], unchanged_scopes: set[str]) -> list[Finding]:
    findings: list[Finding] = []
    for hit, lines in sorted(hf.hit_lines.items(), key=lambda kv: kv[1][0]):
        if hit.scope in unchanged_scopes:
            old, fresh = lines, []
        else:
            old = [n for n in lines if n in kept]
            fresh = [n for n in lines if n not in kept]
        extra = max(0, len(old) - base_hits.get(hit, 0))
        rule = RULES_BY_ID[hit.rule]
        for line in sorted(fresh + old[len(old) - extra:]):
            findings.append(Finding(path, hit.rule, hit.scope, line, rule.title,
                                    blocking=rule.blocking))
    return findings


def compare(base: dict[str, FileMeasure], head: dict[str, FileMeasure],
            changes: list[Change]) -> list[Finding]:
    base_of = {c.new: c.old for c in changes if c.new}
    head_of = {c.old: c.new for c in changes if c.old}
    matcher = _Matcher(base, head, base_of, head_of)
    match = matcher.run()
    owned = _owned_base_hits(base, head_of, match)
    findings: list[Finding] = []
    for path in sorted(head):
        hf = head[path]
        bpath = base_of.get(path, path)
        bf = base.get(bpath) if bpath else None
        unchanged = {qual for hpath, qual in matcher.same_body if hpath == path}
        findings += _file_findings(path, hf, bf)
        findings += _unit_findings(path, hf, match)
        findings += _hit_findings(path, hf, owned.get(path, Counter()),
                                  _surviving_lines(hf, bf), unchanged)
    return findings
