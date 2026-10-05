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

Splitting a file moves lines to ANOTHER file, where they cannot survive in place. A base
occurrence whose line left its file may pay, once, for an identical occurrence (same rule, same
code) that arrives in another file in the same diff: at module level (an import guard, a
module constant), or inside the unit its unit was matched to. A copy is never a move: the
origin line survives, so nothing departed.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
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


def _owners(head_of: dict[str, str | None],
            match: dict[Key, tuple[str, Unit]]) -> Callable[[str, str], tuple[str | None, str]]:
    """The head (path, scope) that owns a base (path, scope): its matched unit, else the same
    scope in the file's head path."""
    owner = {(opath, origin.qualname): key for key, (opath, origin) in match.items()}
    return lambda bpath, scope: owner.get((bpath, scope), (head_of.get(bpath, bpath), scope))


def _owned_base_hits(base: dict[str, FileMeasure],
                     owner_of: Callable[[str, str], tuple[str | None, str]]) -> dict[str, Counter[Hit]]:
    """Each base hit, re-keyed once to the head (path, scope) that now owns it."""
    owned: dict[str, Counter[Hit]] = defaultdict(Counter)
    for bpath, bf in base.items():
        for hit, count in bf.hits.items():
            hpath, scope = owner_of(bpath, hit.scope)
            if hpath is None:
                continue
            owned[hpath][Hit(hit.rule, scope, hit.text)] += count
    return owned


def _line_survival(hf: FileMeasure | None, bf: FileMeasure | None) -> tuple[set[int], set[int]]:
    """(head lines, base lines) that are the same unchanged lines (code only: comments and
    whitespace ignored)."""
    # Line matching only places hits; without any on either side it is pure cost.
    if hf is None or bf is None or not (hf.hit_lines or bf.hit_lines):
        return set(), set()
    old = [bf.code_line(n) for n in range(1, len(bf.lines) + 1)]
    new = [hf.code_line(n) for n in range(1, len(hf.lines) + 1)]
    # autojunk off: `except Exception:` and `pass` are frequent lines, never noise here.
    blocks = SequenceMatcher(None, old, new, autojunk=False).get_matching_blocks()
    kept = {b.b + i + 1 for b in blocks for i in range(b.size)}
    survived = {b.a + i + 1 for b in blocks for i in range(b.size)}
    return kept, survived


class _Departures:
    """Base occurrences whose line left its own file, each spendable once by an identical
    occurrence that arrives in ANOTHER file in the same diff: a split moves code, so moving a
    module-level guard (or a moved unit's body) carries its existing hits along. A copy is
    not a move (the origin line survives, so nothing departed), and a re-add in the same file
    is not one either (that stays new, like any identical violation on a new line)."""

    def __init__(self, base: dict[str, FileMeasure], survived: dict[str, set[int]]) -> None:
        self.pool: Counter[tuple[str, str, str, str]] = Counter()  # (path, scope, rule, text)
        self.module_paths: dict[tuple[str, str], list[str]] = defaultdict(list)
        for bpath, bf in sorted(base.items()):
            alive = survived.get(bpath, set())
            for hit, lines in bf.hit_lines.items():
                gone = sum(1 for n in lines if n not in alive)
                if not gone:
                    continue
                self.pool[(bpath, hit.scope, hit.rule, hit.text)] += gone
                if hit.scope == MODULE_SCOPE:
                    self.module_paths[(hit.rule, hit.text)].append(bpath)

    def take(self, hit: Hit, own_base: str | None, origin: tuple[str, Unit] | None) -> tuple[str, str] | None:
        """Spend one departed occurrence for ``hit``; returns its base (path, scope)."""
        if origin is not None and origin[0] != own_base:  # a unit matched across files
            candidates = [(origin[0], origin[1].qualname)]
        elif hit.scope == MODULE_SCOPE and origin is None:
            candidates = [(p, MODULE_SCOPE) for p in self.module_paths.get((hit.rule, hit.text), [])
                          if p != own_base]
        else:
            return None
        for bpath, scope in candidates:
            key = (bpath, scope, hit.rule, hit.text)
            if self.pool[key] > 0:
                self.pool[key] -= 1
                return bpath, scope
        return None


Split = dict[Hit, tuple[list[int], list[int]]]  # hit -> (old lines, new lines)


def _split_hits(hf: FileMeasure, kept: set[int], unchanged_scopes: set[str]) -> Split:
    split: Split = {}
    for hit, lines in hf.hit_lines.items():
        if hit.scope in unchanged_scopes:
            split[hit] = (list(lines), [])
        else:
            split[hit] = ([n for n in lines if n in kept], [n for n in lines if n not in kept])
    return split


def _hit_findings(path: str, split: Split, credit: Counter[Hit]) -> list[Finding]:
    findings: list[Finding] = []
    for hit, (old, fresh) in sorted(split.items(), key=lambda kv: min(kv[1][0] + kv[1][1], default=0)):
        extra = max(0, len(old) - credit.get(hit, 0))
        rule = RULES_BY_ID[hit.rule]
        for line in sorted(fresh + old[len(old) - extra:]):
            findings.append(Finding(path, hit.rule, hit.scope, line, rule.title,
                                    blocking=rule.blocking))
    return findings


def _spend_departures(splits: dict[str, Split], base_of: dict[str, str | None],
                      match: dict[Key, tuple[str, Unit]], departures: _Departures,
                      owner_of: Callable[[str, str], tuple[str | None, str]],
                      owned: dict[str, Counter[Hit]]) -> None:
    """New-looking occurrences that are moves pay with a departed occurrence, which then no
    longer counts as credit where its origin went (one base occurrence pays once)."""
    for path in sorted(splits):
        own_base = base_of.get(path, path)
        for hit, (old, fresh) in splits[path].items():
            remaining = []
            for line in fresh:
                source = departures.take(hit, own_base, match.get((path, hit.scope)))
                if source is None:
                    remaining.append(line)
                    continue
                hpath, scope = owner_of(*source)
                spent = Hit(hit.rule, scope, hit.text)
                if hpath is not None and owned[hpath][spent] > 0:
                    owned[hpath][spent] -= 1
            splits[path][hit] = (old, remaining)


def compare(base: dict[str, FileMeasure], head: dict[str, FileMeasure],
            changes: list[Change]) -> list[Finding]:
    base_of = {c.new: c.old for c in changes if c.new}
    head_of = {c.old: c.new for c in changes if c.old}
    matcher = _Matcher(base, head, base_of, head_of)
    match = matcher.run()
    owner_of = _owners(head_of, match)
    owned = _owned_base_hits(base, owner_of)
    splits: dict[str, Split] = {}
    survived: dict[str, set[int]] = {}
    for path, hf in head.items():
        bpath = base_of.get(path, path)
        kept, survived_lines = _line_survival(hf, base.get(bpath) if bpath else None)
        if bpath:
            survived[bpath] = survived_lines
        unchanged = {qual for hpath, qual in matcher.same_body if hpath == path}
        splits[path] = _split_hits(hf, kept, unchanged)
    _spend_departures(splits, base_of, match, _Departures(base, survived), owner_of, owned)
    findings: list[Finding] = []
    for path in sorted(head):
        hf = head[path]
        bpath = base_of.get(path, path)
        findings += _file_findings(path, hf, base.get(bpath) if bpath else None)
        findings += _unit_findings(path, hf, match)
        findings += _hit_findings(path, splits[path], owned.get(path, Counter()))
    return findings
