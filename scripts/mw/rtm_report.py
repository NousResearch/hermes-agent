#!/usr/bin/env python3
"""mw-rtm-report — 3-layer Requirements Traceability Matrix reporter.

Proves each requirement is traced through three layers and counts them live
(no hardcoded numbers):

  L1 REQUIREMENT EXISTS — REQ id is in the register.
  L2 HAS A TEST         — >=1 test case is mapped to that REQ.
  L3 VERIFIED           — >=1 mapped test PASSED and NO mapped test FAILED.

Fail closed: a REQ is never L3 without its own mapped tests really passing
with no failure. Mixed pass+fail is not verified. Ambiguous test ids are not
verified. Skip / missing results never grant L3. Suite-level JUnit errors
that are not attributable to test cases make the whole run unreliable.

stdlib-only core · optional PyYAML · Python 3.9+
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

EXIT_OK = 0
EXIT_BELOW = 1
EXIT_ERR = 2

DEFAULT_CONFIG_REL = Path(".work") / "rtm.yaml"
DEFAULT_ID_PATTERN = r"(?:REQ|R)-\d+"
DEFAULT_MARKER_PATTERN = r"REQ[:=\s]+((?:REQ|R)-\d+)"
DEFAULT_TEST_GLOB = "tests/**/*.py"
DEFAULT_MIN_VERIFIED_PCT = 100.0

# Tests may force the fallback YAML loader even when PyYAML is installed.
_FORCE_MINI_YAML = False

try:
    import yaml as _yaml  # type: ignore
except ImportError:  # pragma: no cover
    _yaml = None


# ---------------------------------------------------------------------------
# minimal YAML subset loader (self-contained copy; mirrors menu_gate approach)
# ---------------------------------------------------------------------------


def _scalar(v: str) -> Any:
    """Parse a YAML scalar: quoted string, bool, null, int/float, or raw str."""
    v = v.strip()
    if not v:
        return None
    if len(v) >= 2 and v[0] in "\"'" and v[-1] == v[0]:
        return v[1:-1]
    if " #" in v:
        v = v.split(" #", 1)[0].strip()
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if not inner:
            return []
        parts: List[str] = []
        buf = ""
        in_q: Optional[str] = None
        for ch in inner:
            if in_q:
                buf += ch
                if ch == in_q:
                    in_q = None
                continue
            if ch in "\"'":
                in_q = ch
                buf += ch
                continue
            if ch == ",":
                parts.append(buf.strip())
                buf = ""
                continue
            buf += ch
        if buf.strip():
            parts.append(buf.strip())
        return [_scalar(p) for p in parts]
    low = v.lower()
    if low in ("null", "~"):
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


class MiniYamlError(ValueError):
    """Raised when the mini YAML loader hits an ambiguous construct."""


def _mini_yaml(text: str) -> Any:
    """Minimal YAML-subset parser for RTM config / mapfile schemas.

    Handles nested mappings by 2-space indent, lists of scalars and mappings,
    quoted/unquoted scalars, booleans, null, inline ``[a, b]`` lists, ``#``
    comments, and blank lines. Not a full YAML 1.1 parser — fail closed on
    ambiguous constructs.
    """
    lines = [
        l
        for l in (raw.rstrip() for raw in text.splitlines())
        if l.strip() and not l.strip().startswith("#")
    ]
    pos = 0

    def indent_of(l: str) -> int:
        return len(l) - len(l.lstrip(" "))

    def _is_list_item(s: str) -> bool:
        return s == "-" or s.startswith("- ")

    def _list_item_body(s: str) -> str:
        if s == "-":
            return ""
        return s[2:].strip()

    def _parse_mapping_continuation(d: Dict[str, Any], base_indent: int) -> None:
        nonlocal pos
        while pos < len(lines):
            nline = lines[pos]
            ni = indent_of(nline)
            if ni <= base_indent:
                break
            ns = nline.strip()
            if _is_list_item(ns):
                break
            if ":" not in ns:
                raise MiniYamlError(
                    f"unrepresentable YAML at line content {ns!r}: "
                    "expected key: value under list item"
                )
            nk, _, nv = ns.partition(":")
            pos += 1
            if nv.strip():
                d[nk.strip()] = _scalar(nv)
            elif pos < len(lines) and indent_of(lines[pos]) > ni:
                d[nk.strip()] = parse_block(indent_of(lines[pos]))
            else:
                d[nk.strip()] = None

    def parse_block(indent: int) -> Any:
        nonlocal pos
        result: Any = None
        while pos < len(lines):
            line = lines[pos]
            cur = indent_of(line)
            if cur < indent:
                break
            if cur > indent:
                raise MiniYamlError(
                    f"orphan deeper indent without a parent key near: {line.strip()!r}"
                )
            s = line.strip()
            if _is_list_item(s):
                if result is None:
                    result = []
                if not isinstance(result, list):
                    raise MiniYamlError(
                        f"mixed mapping/list structure near: {s!r}"
                    )
                item = _list_item_body(s)
                pos += 1
                if item == "":
                    if pos < len(lines) and indent_of(lines[pos]) > indent:
                        nested_s = lines[pos].strip()
                        if not _is_list_item(nested_s) and ":" in nested_s:
                            d: Dict[str, Any] = {}
                            _parse_mapping_continuation(d, indent)
                            result.append(d)
                        else:
                            result.append(parse_block(indent_of(lines[pos])))
                    else:
                        result.append(None)
                elif ":" in item and item[0] not in "\"'":
                    k, _, v = item.partition(":")
                    d = {k.strip(): _scalar(v) if v.strip() else None}
                    _parse_mapping_continuation(d, indent)
                    if d and list(d.values()) == [None]:
                        only_k = next(iter(d))
                        if pos < len(lines) and indent_of(lines[pos]) > indent:
                            d[only_k] = parse_block(indent_of(lines[pos]))
                    result.append(d)
                else:
                    result.append(_scalar(item))
            else:
                if result is None:
                    result = {}
                if not isinstance(result, dict):
                    raise MiniYamlError(
                        f"mixed list/mapping structure near: {s!r}"
                    )
                if ":" not in s:
                    raise MiniYamlError(
                        f"unrepresentable YAML near: {s!r} (expected key: value)"
                    )
                k, _, v = s.partition(":")
                pos += 1
                if v.strip():
                    result[k.strip()] = _scalar(v)
                elif pos < len(lines) and indent_of(lines[pos]) > cur:
                    result[k.strip()] = parse_block(indent_of(lines[pos]))
                else:
                    result[k.strip()] = None
        return result

    return parse_block(0) if lines else {}


def load_yaml_text(text: str, force_mini: bool = False) -> Any:
    """Load YAML text: prefer PyYAML unless forced to mini loader."""
    use_mini = force_mini or _FORCE_MINI_YAML or _yaml is None
    if not use_mini:
        try:
            data = _yaml.safe_load(text)
        except Exception as e:  # PyYAML ScannerError / ParserError etc.
            # Fail closed: surface as ValueError so CLI exits 2.
            raise ValueError(f"YAML parse error: {e}") from e
        return data if data is not None else {}
    return _mini_yaml(text) or {}


def load_yaml_file(path: Path, force_mini: bool = False) -> Any:
    text = path.read_text(encoding="utf-8")
    return load_yaml_text(text, force_mini=force_mini)


# ---------------------------------------------------------------------------
# path helpers
# ---------------------------------------------------------------------------


def under_root(root: Path, rel: str) -> Path:
    """Join a project-relative path under root."""
    if rel is None:
        return root
    s = str(rel).replace("\\", "/").strip()
    while s.startswith("./"):
        s = s[2:]
    if s.startswith("/"):
        s = s.lstrip("/")
    return root / s


# ---------------------------------------------------------------------------
# L1 — requirement register
# ---------------------------------------------------------------------------


def parse_req_register(path: Path, id_pattern: str) -> List[str]:
    """Extract ordered unique REQ ids from the register file via id_pattern."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ValueError(f"cannot read req register {path}: {e}") from e
    try:
        cre = re.compile(id_pattern)
    except re.error as e:
        raise ValueError(f"invalid id_pattern {id_pattern!r}: {e}") from e
    seen: Set[str] = set()
    ordered: List[str] = []
    for m in cre.finditer(text):
        # Prefer capture group 1 when present, else full match.
        rid = m.group(1) if m.lastindex and m.group(1) else m.group(0)
        rid = rid.strip()
        if rid and rid not in seen:
            seen.add(rid)
            ordered.append(rid)
    return ordered


# ---------------------------------------------------------------------------
# L2 — test mapping
# ---------------------------------------------------------------------------


# Only functions whose name starts with ``test`` count as test defs.
_DEF_TEST_RE = re.compile(r"^(\s*)def\s+(test\w+)\s*\(", re.MULTILINE)


def _nearest_test_def(text: str, pos: int) -> Optional[str]:
    """Return the nearest PRECEDING ``def test...`` name before marker at pos.

    Scans only backward from the marker. Markers bind to the nearest earlier
    test function — never to a later one. Returns None when no test def
    precedes the marker (unbound / orphan marker).
    """
    best_name: Optional[str] = None
    for m in _DEF_TEST_RE.finditer(text):
        if m.start() >= pos:
            # Only preceding defs count; stop once past the marker.
            break
        best_name = m.group(2)
    return best_name


def scan_markers(
    root: Path,
    test_glob: str,
    marker_pattern: str,
    register: Set[str],
) -> Tuple[Dict[str, List[str]], List[Dict[str, str]]]:
    """Scan test files for REQ markers.

    Returns (req -> ordered unique test ids, orphan_mappings).
    """
    try:
        cre = re.compile(marker_pattern)
    except re.error as e:
        raise ValueError(f"invalid marker_pattern {marker_pattern!r}: {e}") from e

    mapping: Dict[str, List[str]] = {}
    orphans: List[Dict[str, str]] = []
    orphan_seen: Set[Tuple[str, str]] = set()

    # Path.glob supports ** patterns
    glob_pat = test_glob.replace("\\", "/")
    files = sorted(p for p in root.glob(glob_pat) if p.is_file())

    for fpath in files:
        try:
            text = fpath.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            rel = str(fpath.relative_to(root)).replace("\\", "/")
        except ValueError:
            rel = str(fpath)

        for m in cre.finditer(text):
            rid = (m.group(1) if m.lastindex and m.group(1) else m.group(0)).strip()
            if not rid:
                continue
            tname = _nearest_test_def(text, m.start())
            if not tname:
                # Unbound: no preceding test def — do not map to a later test.
                key = (rid, f"{rel}:unbound@{m.start()}")
                if key not in orphan_seen:
                    orphan_seen.add(key)
                    orphans.append(
                        {"req": rid, "test": "(unbound)", "file": rel}
                    )
                continue
            test_id = tname
            if rid not in register:
                key = (rid, test_id)
                if key not in orphan_seen:
                    orphan_seen.add(key)
                    orphans.append({"req": rid, "test": test_id, "file": rel})
                continue
            lst = mapping.setdefault(rid, [])
            if test_id not in lst:
                lst.append(test_id)

    return mapping, orphans


def load_mapfile(
    path: Path,
    register: Set[str],
) -> Tuple[Dict[str, List[str]], List[Dict[str, str]]]:
    """Load mapfile: REQ id -> list of test ids. Orphans recorded, no phantoms."""
    try:
        data = load_yaml_file(path)
    except (OSError, MiniYamlError, ValueError) as e:
        raise ValueError(f"cannot parse mapfile {path}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"mapfile must be a mapping of REQ id -> [test ids]: {path}")

    mapping: Dict[str, List[str]] = {}
    orphans: List[Dict[str, str]] = []
    for key, val in data.items():
        rid = str(key).strip()
        if isinstance(val, list):
            tests = [str(t).strip() for t in val if str(t).strip()]
        elif val is None:
            tests = []
        else:
            tests = [str(val).strip()] if str(val).strip() else []
        if rid not in register:
            for t in tests:
                orphans.append({"req": rid, "test": t, "file": str(path)})
            continue
        # preserve order, unique
        seen: Set[str] = set()
        ordered: List[str] = []
        for t in tests:
            if t not in seen:
                seen.add(t)
                ordered.append(t)
        mapping[rid] = ordered
    return mapping, orphans


# ---------------------------------------------------------------------------
# L3 — results
# ---------------------------------------------------------------------------

OUTCOME_PASS = "pass"
OUTCOME_FAIL = "fail"
OUTCOME_SKIP = "skip"
# Internal sentinel: bare name matches multiple distinct result identities.
OUTCOME_AMBIGUOUS = "__ambiguous__"


def _suite_reported_failures(suite: ET.Element) -> int:
    """Sum of suite-level errors + failures attributes (0 if absent/invalid)."""
    total = 0
    for attr in ("errors", "failures"):
        raw = suite.get(attr)
        if raw is None or raw == "":
            continue
        try:
            total += int(raw)
        except ValueError:
            # Non-integer attribute — treat as unreliable signal via ValueError upstream
            raise ValueError(
                f"junit suite has non-integer {attr}={raw!r} — results unreliable"
            ) from None
    return total


def _count_case_fail_error_nodes(suite: ET.Element) -> int:
    """Count <testcase> children (descendants) that contain failure or error."""
    n = 0
    for tc in suite.iter("testcase"):
        if tc.find("failure") is not None or tc.find("error") is not None:
            n += 1
    return n


def parse_junit(path: Path) -> Dict[str, str]:
    """Parse JUnit XML -> test id -> outcome {pass, fail, skip}.

    Test ids stored under multiple keys for flexible matching:
    name, classname.name, and classname::name when classname present.

    Fail closed on suite-level errors/failures that are not attributable to
    per-testcase failure/error nodes (e.g. collection/setup errors): raises
    ValueError so the CLI exits 2 and nothing can verify from a broken run.
    """
    try:
        tree = ET.parse(str(path))
    except (OSError, ET.ParseError) as e:
        raise ValueError(f"unparseable junit results {path}: {e}") from e
    root = tree.getroot()
    results: Dict[str, str] = {}

    # Collect testsuites for reliability check.
    if root.tag == "testsuite":
        suites = [root]
    elif root.tag == "testsuites":
        suites = list(root.findall("testsuite"))
    else:
        suites = list(root.iter("testsuite"))

    for suite in suites:
        reported = _suite_reported_failures(suite)
        if reported <= 0:
            continue
        attributable = _count_case_fail_error_nodes(suite)
        if reported > attributable:
            raise ValueError(
                f"junit suite reports {reported} failures/errors not "
                f"attributable to test cases — results unreliable"
            )

    # Handle both <testsuites><testsuite> and bare <testsuite>
    cases = list(root.iter("testcase"))
    if not cases and root.tag == "testcase":
        cases = [root]

    for tc in cases:
        name = (tc.get("name") or "").strip()
        classname = (tc.get("classname") or "").strip()
        if not name and not classname:
            continue

        if tc.find("failure") is not None or tc.find("error") is not None:
            outcome = OUTCOME_FAIL
        elif tc.find("skipped") is not None:
            outcome = OUTCOME_SKIP
        else:
            outcome = OUTCOME_PASS

        keys: List[str] = []
        if name:
            keys.append(name)
        if classname and name:
            keys.append(f"{classname}.{name}")
            keys.append(f"{classname}::{name}")
        if classname and not name:
            keys.append(classname)

        for k in keys:
            # Prefer fail over pass if the same key appears multiple times
            prev = results.get(k)
            if prev == OUTCOME_FAIL:
                continue
            if outcome == OUTCOME_FAIL or prev is None:
                results[k] = outcome
            elif prev == OUTCOME_SKIP and outcome == OUTCOME_PASS:
                results[k] = outcome
    return results


def parse_simple_results(path: Path) -> Dict[str, str]:
    """Parse simple results: lines ``<test_id> PASS|FAIL|SKIP``."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ValueError(f"cannot read results {path}: {e}") from e
    results: Dict[str, str] = {}
    line_re = re.compile(
        r"^\s*(\S.*?)\s+(PASS|FAIL|SKIP|pass|fail|skip)\s*$"
    )
    any_line = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = line_re.match(line)
        if not m:
            # tolerate free-form garbage as parse error only if nothing valid
            continue
        any_line = True
        tid = m.group(1).strip()
        outcome = m.group(2).lower()
        results[tid] = outcome
    if not any_line and text.strip():
        # Non-empty file with no valid lines
        raise ValueError(f"unparseable simple results {path}: no valid lines")
    return results


def _unique_bare_outcome(bare: str, results: Dict[str, str]) -> Optional[str]:
    """Resolve a bare test name against results only when the match is unique.

    Multiple fully-qualified keys sharing the same short name (e.g.
    ``pkgA::test_x`` and ``pkgB::test_x``) are ambiguous — return
    ``OUTCOME_AMBIGUOUS`` so L3 fails closed. Equivalent keys for one
    testcase (``name`` / ``classname.name`` / ``classname::name``) count
    as a single identity.
    """
    if not bare:
        return None

    # classname -> outcome for qualified keys ending with this bare name
    fq: Dict[str, str] = {}
    bare_direct: Optional[str] = None

    for key, outcome in results.items():
        if key == bare:
            # Prefer fail if bare key is written multiple times via merge
            if bare_direct == OUTCOME_FAIL:
                continue
            if outcome == OUTCOME_FAIL or bare_direct is None:
                bare_direct = outcome
            continue
        cls: Optional[str] = None
        if key.endswith("::" + bare):
            cls = key[: -len(bare) - 2]
        elif key.endswith("." + bare):
            cls = key[: -len(bare) - 1]
        if cls is None:
            continue
        prev = fq.get(cls)
        if prev == OUTCOME_FAIL:
            continue
        if outcome == OUTCOME_FAIL or prev is None:
            fq[cls] = outcome
        elif prev == OUTCOME_SKIP and outcome == OUTCOME_PASS:
            fq[cls] = outcome

    if len(fq) > 1:
        return OUTCOME_AMBIGUOUS
    if len(fq) == 1:
        return next(iter(fq.values()))
    if bare_direct is not None:
        return bare_direct
    return None


def lookup_outcome(test_id: str, results: Dict[str, str]) -> Optional[str]:
    """Resolve a mapped test id to an outcome (fail closed).

    1. Exact key match on the id as-is.
    2. Exact alternate form (``classname.name`` <-> ``classname::name``).
    3. Only if still unresolved: bare-name match **if and only if** it is
       unique among results. Multiple candidates -> ``OUTCOME_AMBIGUOUS``.
    4. No match -> None (missing result).

    Never lets an absent / colliding id borrow an unrelated pass.
    """
    if not test_id:
        return None

    # 1) Exact key as-is
    if test_id in results:
        # Bare short names may still be ambiguous when several FQ keys share
        # the same short name — detect that before trusting the short key.
        if ("::" not in test_id) and ("." not in test_id):
            bare_resolved = _unique_bare_outcome(test_id, results)
            if bare_resolved == OUTCOME_AMBIGUOUS:
                return OUTCOME_AMBIGUOUS
            # Prefer unique FQ identity over a merged short-key value.
            if bare_resolved is not None:
                return bare_resolved
        return results[test_id]

    # 2) Alternate separator form for qualified ids
    if "::" in test_id:
        left, right = test_id.rsplit("::", 1)
        alt = f"{left}.{right}"
        if alt in results:
            return results[alt]
    elif "." in test_id:
        left, right = test_id.rsplit(".", 1)
        alt = f"{left}::{right}"
        if alt in results:
            return results[alt]

    # 3) Unique bare-name match only
    if "::" in test_id:
        bare = test_id.rsplit("::", 1)[-1]
    elif "." in test_id:
        bare = test_id.rsplit(".", 1)[-1]
    else:
        bare = test_id
    return _unique_bare_outcome(bare, results)


# ---------------------------------------------------------------------------
# report model
# ---------------------------------------------------------------------------


@dataclass
class ReqRow:
    id: str
    l1: bool = True
    l2: bool = False
    l3: bool = False
    tests: List[str] = field(default_factory=list)
    reason: Optional[str] = None  # set when l2 and not l3


@dataclass
class RtmReport:
    total: int
    l2_count: int
    l3_count: int
    verified_pct: float
    threshold: float
    complete: bool
    reqs: List[ReqRow]
    gaps_no_test: List[str]
    gaps_not_verified: List[Dict[str, str]]
    orphan_mappings: List[Dict[str, str]]
    unmapped_result_tests: List[str] = field(default_factory=list)

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "l2_count": self.l2_count,
            "l3_count": self.l3_count,
            "verified_pct": self.verified_pct,
            "threshold": self.threshold,
            "complete": self.complete,
            "reqs": [
                {
                    "id": r.id,
                    "l1": r.l1,
                    "l2": r.l2,
                    "l3": r.l3,
                    "tests": list(r.tests),
                    "reason": r.reason,
                }
                for r in self.reqs
            ],
            "gaps": {
                "no_test": list(self.gaps_no_test),
                "not_verified": [
                    {"id": g["id"], "reason": g["reason"]}
                    for g in self.gaps_not_verified
                ],
                "orphan_mappings": list(self.orphan_mappings),
            },
        }


def compute_report(
    req_ids: List[str],
    mapping: Dict[str, List[str]],
    results: Dict[str, str],
    orphans: List[Dict[str, str]],
    threshold: float,
) -> RtmReport:
    """Compute live RTM counts from register + map + results (fail closed)."""
    rows: List[ReqRow] = []
    no_test: List[str] = []
    not_verified: List[Dict[str, str]] = []

    mapped_test_ids: Set[str] = set()
    for tests in mapping.values():
        mapped_test_ids.update(tests)

    for rid in req_ids:
        tests = list(mapping.get(rid, []))
        row = ReqRow(id=rid, l1=True, tests=tests)
        if not tests:
            row.l2 = False
            row.l3 = False
            row.reason = "no test"
            no_test.append(rid)
            rows.append(row)
            continue

        row.l2 = True
        outcomes: List[Optional[str]] = [
            lookup_outcome(t, results) for t in tests
        ]
        any_ambiguous = any(o == OUTCOME_AMBIGUOUS for o in outcomes)
        any_pass = any(o == OUTCOME_PASS for o in outcomes)
        any_fail = any(o == OUTCOME_FAIL for o in outcomes)

        # L3 only when >=1 real pass AND no mapped failure (and no ambiguity).
        # Skip / missing never grant L3; mixed pass+fail fails closed.
        if any_fail:
            row.l3 = False
            if any_pass:
                row.reason = "test failed (conflict)"
            else:
                row.reason = "test failed"
            not_verified.append({"id": rid, "reason": row.reason})
        elif any_ambiguous:
            row.l3 = False
            row.reason = "ambiguous test id"
            not_verified.append({"id": rid, "reason": row.reason})
        elif any_pass:
            row.l3 = True
            row.reason = None
        else:
            # missing results and/or only skips — fail closed
            row.l3 = False
            row.reason = "no result"
            not_verified.append({"id": rid, "reason": row.reason})
        rows.append(row)

    total = len(rows)
    l2_count = sum(1 for r in rows if r.l2)
    l3_count = sum(1 for r in rows if r.l3)
    # Display only — rounded for humans.
    verified_pct = round(100.0 * l3_count / total, 1) if total > 0 else 0.0
    # Completion uses exact integer math so 2000/2001 never counts as 100%.
    if total <= 0:
        complete = False
    else:
        complete = (l3_count * 100) >= (float(threshold) * total)

    # Informational: result test ids that map to no REQ
    unmapped: List[str] = []
    seen_u: Set[str] = set()
    for key in results:
        # skip composite keys if short form already listed
        short = key.rsplit(".", 1)[-1].rsplit("::", 1)[-1]
        hit = False
        for mt in mapped_test_ids:
            if lookup_outcome(mt, {key: results[key]}) is not None:
                # only counts if this key is the match vehicle — simpler check:
                pass
            if (
                mt == key
                or key.endswith("." + mt)
                or key.endswith("::" + mt)
                or mt.endswith("." + key)
                or mt.endswith("::" + key)
                or short == mt
            ):
                hit = True
                break
        if not hit and key not in seen_u and short not in mapped_test_ids:
            # de-dupe by short name
            if short not in seen_u:
                seen_u.add(short)
                unmapped.append(key)

    return RtmReport(
        total=total,
        l2_count=l2_count,
        l3_count=l3_count,
        verified_pct=verified_pct,
        threshold=float(threshold),
        complete=complete,
        reqs=rows,
        gaps_no_test=no_test,
        gaps_not_verified=not_verified,
        orphan_mappings=orphans,
        unmapped_result_tests=unmapped,
    )


# ---------------------------------------------------------------------------
# config load / orchestration
# ---------------------------------------------------------------------------


def resolve_config_path(root: Path, config_arg: Optional[str]) -> Path:
    if config_arg:
        p = Path(config_arg)
        if not p.is_absolute():
            p = Path.cwd() / p
        return p
    return root / DEFAULT_CONFIG_REL


def load_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"config not found: {path}")
    try:
        data = load_yaml_file(path)
    except MiniYamlError as e:
        raise ValueError(f"config parse error {path}: {e}") from e
    except OSError as e:
        raise ValueError(f"cannot read config {path}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data


def build_report_from_config(
    root: Path,
    cfg: Dict[str, Any],
    threshold_override: Optional[float] = None,
) -> RtmReport:
    """Full pipeline: register -> map -> results -> live counts."""
    # --- register ---
    reg = cfg.get("req_register") or {}
    if not isinstance(reg, dict):
        raise ValueError("req_register must be a mapping")
    reg_file = reg.get("file") or ".work/req-register.md"
    id_pattern = reg.get("id_pattern") or DEFAULT_ID_PATTERN
    reg_path = under_root(root, str(reg_file))
    if not reg_path.is_file():
        raise ValueError(f"req register not found: {reg_path}")
    req_ids = parse_req_register(reg_path, str(id_pattern))
    if not req_ids:
        raise ValueError("no requirements found in register")
    register_set = set(req_ids)

    # --- test map ---
    tmap = cfg.get("test_map") or {}
    if not isinstance(tmap, dict):
        raise ValueError("test_map must be a mapping")
    mode = str(tmap.get("mode") or "markers").lower()
    if mode == "markers":
        test_glob = str(tmap.get("test_glob") or DEFAULT_TEST_GLOB)
        marker_pattern = str(tmap.get("marker_pattern") or DEFAULT_MARKER_PATTERN)
        mapping, orphans = scan_markers(
            root, test_glob, marker_pattern, register_set
        )
    elif mode == "mapfile":
        map_file = tmap.get("file") or ".work/rtm-map.yaml"
        map_path = under_root(root, str(map_file))
        if not map_path.is_file():
            raise ValueError(f"mapfile not found: {map_path}")
        mapping, orphans = load_mapfile(map_path, register_set)
    else:
        raise ValueError(f"unknown test_map.mode: {mode!r} (use markers|mapfile)")

    # --- results ---
    res_cfg = cfg.get("results") or {}
    if not isinstance(res_cfg, dict):
        raise ValueError("results must be a mapping")
    rmode = str(res_cfg.get("mode") or "junit").lower()
    rpath_rel = res_cfg.get("path") or "test-results.xml"
    rpath = under_root(root, str(rpath_rel))
    if not rpath.is_file():
        raise ValueError(f"results file not found: {rpath}")
    if rmode == "junit":
        results = parse_junit(rpath)
    elif rmode == "simple":
        results = parse_simple_results(rpath)
    else:
        raise ValueError(f"unknown results.mode: {rmode!r} (use junit|simple)")

    # --- threshold ---
    thr_cfg = cfg.get("thresholds") or {}
    if not isinstance(thr_cfg, dict):
        thr_cfg = {}
    if threshold_override is not None:
        threshold = float(threshold_override)
    else:
        raw_t = thr_cfg.get("min_verified_pct", DEFAULT_MIN_VERIFIED_PCT)
        try:
            threshold = float(raw_t)
        except (TypeError, ValueError) as e:
            raise ValueError(f"invalid thresholds.min_verified_pct: {raw_t!r}") from e

    return compute_report(req_ids, mapping, results, orphans, threshold)


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


def format_human(report: RtmReport) -> str:
    lines: List[str] = []
    l2_pct = round(100.0 * report.l2_count / report.total, 1) if report.total else 0.0
    lines.append(
        f"RTM: {report.total} requirements · "
        f"L2 (has test) {report.l2_count}/{report.total} ({l2_pct}%) · "
        f"L3 (verified) {report.l3_count}/{report.total} ({report.verified_pct}%)"
    )
    if report.gaps_no_test:
        lines.append("no test: " + ", ".join(report.gaps_no_test))
    if report.gaps_not_verified:
        parts = [
            f"{g['id']} ({g['reason']})" for g in report.gaps_not_verified
        ]
        lines.append("tested but not verified: " + ", ".join(parts))
    if report.orphan_mappings:
        oparts = [
            f"{o.get('req', '?')}->{o.get('test', '?')}"
            for o in report.orphan_mappings
        ]
        lines.append("orphan test mappings: " + ", ".join(oparts))
    thr = report.threshold
    pct_disp = f"{report.verified_pct:.1f}"
    thr_disp = str(int(thr)) if thr == int(thr) else f"{thr:g}"
    if report.complete:
        lines.append(
            f"RTM verified {pct_disp}% >= threshold {thr_disp} -> COMPLETE"
        )
    else:
        lines.append(
            f"RTM verified {pct_disp}% < threshold {thr_disp} -> INCOMPLETE"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="rtm_report.py",
        description=(
            "3-layer Requirements Traceability Matrix reporter "
            "(L1 register · L2 has test · L3 verified by real pass)."
        ),
    )
    p.add_argument(
        "--config",
        default=None,
        help="Path to rtm.yaml (default: <root>/.work/rtm.yaml)",
    )
    p.add_argument(
        "--root",
        default=None,
        help="Project root (default: current working directory)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON report",
    )
    p.add_argument(
        "--min-verified-pct",
        type=float,
        default=None,
        dest="min_verified_pct",
        help="Override thresholds.min_verified_pct from config",
    )
    return p.parse_args(list(argv) if argv is not None else None)


def run(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry: returns exit code 0 / 1 / 2."""
    try:
        args = _parse_args(argv)
    except SystemExit as e:
        # argparse error
        code = e.code
        if code is None or code == 0:
            return EXIT_OK
        return EXIT_ERR

    root = Path(args.root).resolve() if args.root else Path.cwd().resolve()
    if not root.is_dir():
        print(f"error: root is not a directory: {root}", file=sys.stderr)
        return EXIT_ERR

    cfg_path = resolve_config_path(root, args.config)
    try:
        if not args.config and not cfg_path.is_file():
            print(
                f"error: config not found (pass --config or create {cfg_path})",
                file=sys.stderr,
            )
            return EXIT_ERR
        cfg = load_config(cfg_path)
        report = build_report_from_config(
            root, cfg, threshold_override=args.min_verified_pct
        )
    except (ValueError, MiniYamlError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_ERR

    if args.json:
        print(json.dumps(report.to_json_dict(), indent=2, sort_keys=False))
    else:
        print(format_human(report))

    return EXIT_OK if report.complete else EXIT_BELOW


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
