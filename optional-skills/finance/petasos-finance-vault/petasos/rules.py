"""petasos.rules — rules.md section-aware parser (C4, DG-3/DG-4/DG-14).

rules.md is the human-editable classification brain (P2: intelligence in
data). The parser binds every table to its OWNING `## N.` section — v1's
section-blind regex let §3a rows leak into the auto-post list (G4) — and a
malformed table (wrong column count, unparseable row) raises RulesError
naming the section and line, never silently swallowed (S-6).

Sections consumed:
  §1  auto-post patterns   | `Pattern` | Category | Type | FV | Notes? |
  §3a known-merchant       | `Pattern` | Category | Type | FV |
  §5  kind routing (DG-3)  | `Kind` | Tier default | Category | Type | FV |
  §6  confirmed-by-user    | Date | Item | Meaning |
  §2  aggregate rules      bullet lines → notes
`threshold_auto_post` backtick value anywhere is the default threshold
source (R4.4; v1 regex parity).

Resolution chain (DG-14): $PETASOS_RULES → <ledger_dir>/rules.md → bundled
templates/rules.md (read-only fallback). The loader names the file via
Rules.path/.source. v1 required the vault copy and crashed; v2 falls back.

Tier vocabulary for §5 (consumed by petasos.classify):
  expense               always auto-post as expense (category from the row)
  expense-under-thresh  expense only below the auto-post threshold
  balance-alert         direction-driven ladder incl. own-name guard and
                        mystery credits (v1 ACB behavior)
  plumbing              transfer, posted, reconcile-only, never an expense
  transfer-receipt      transfer needs-confirm (counterparty blind)
"""
import os
import re
from dataclasses import dataclass, field


class RulesError(Exception):
    """User-facing rules.md failure — callers print the message and exit nonzero."""


BUNDLED_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_HEADING = re.compile(r'^#{1,6}\s*(\S+?)\.?\s')
_THRESHOLD = re.compile(r'threshold_auto_post`:\s*\*{0,2}([\d,]+)')


@dataclass
class Rules:
    auto_post: list = field(default_factory=list)        # [(pattern, cat, type, fv)]
    known_route: list = field(default_factory=list)      # [(pattern, cat, type, fv)]
    kind_route: dict = field(default_factory=dict)       # kind -> (tier, cat, type, fv)
    aggregate_notes: list = field(default_factory=list)  # §2 bullet lines
    threshold_default: int | None = None
    confirmed: list = field(default_factory=list)        # §6 [(date, item, meaning)]
    path: str = ''
    source: str = 'bundled'                              # 'env' | 'vault' | 'bundled'


def _cells(line):
    """Split a table row on '|' — backticked spans (patterns like
    `COFFEE|HIGHLANDS`) are kept whole. Empty cells are preserved."""
    inner = line.strip()
    if inner.startswith('|'):
        inner = inner[1:]
    if inner.endswith('|'):
        inner = inner[:-1]
    cells, cur, in_bt = [], [], False
    for ch in inner:
        if ch == '`':
            in_bt = not in_bt
            cur.append(ch)
        elif ch == '|' and not in_bt:
            cells.append(''.join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    cells.append(''.join(cur).strip())
    return cells


def _pattern_from(cell):
    """'`PAT`' → 'PAT'; anything else is not a valid pattern cell."""
    if len(cell) >= 2 and cell.startswith('`') and cell.endswith('`'):
        return cell[1:-1]
    return None


def _is_separator(cells):
    return len(cells) >= 2 and all(re.fullmatch(r'[-: ]*', c) for c in cells)


def parse_rules(text, path='rules.md'):
    rules = Rules(path=path)
    section = ''
    in_table = False
    table_row_seen = False
    for lineno, line in enumerate(text.splitlines(), 1):
        m = _HEADING.match(line)
        if m:
            section = m.group(1)
            in_table = False
            table_row_seen = False
            continue
        if line.lstrip().startswith('|'):
            cells = _cells(line)
            if not in_table:                      # header row of a new table
                in_table = True
                table_row_seen = False
                continue
            if not table_row_seen and _is_separator(cells):
                continue
            table_row_seen = True
            _take_row(rules, section, cells, path, lineno, line)
            continue
        in_table = False
        table_row_seen = False
        if section == '2' and line.lstrip().startswith('- '):
            rules.aggregate_notes.append(line.lstrip()[2:].strip())
    tm = _THRESHOLD.search(text)
    if tm:
        rules.threshold_default = int(tm.group(1).replace(',', ''))
    return rules


def _check_regex(pat, section, path, lineno):
    """§1/§3a patterns are curated regexes — one that does not compile must
    fail LOUD at load time, naming the line, instead of crashing classify on
    the first real transaction (live finding 14/10)."""
    try:
        re.compile(pat)
    except re.error as e:
        raise RulesError(f'{path}: pattern in rules §{section} at line {lineno} '
                         f'does not compile ({e}): {pat!r}') from None


def _take_row(rules, section, cells, path, lineno, line):
    def bad():
        raise RulesError(f'{path}: malformed table row in rules §{section} '
                         f'at line {lineno}: {line.strip()!r}')

    if section == '1':
        if len(cells) < 4 or not _pattern_from(cells[0]):
            bad()
        pat = _pattern_from(cells[0])
        _check_regex(pat, section, path, lineno)
        rules.auto_post.append((pat, cells[1], cells[2], cells[3]))
    elif section == '3a':
        if len(cells) < 4 or not _pattern_from(cells[0]):
            bad()
        pat = _pattern_from(cells[0])
        _check_regex(pat, section, path, lineno)
        rules.known_route.append((pat, cells[1], cells[2], cells[3]))
    elif section == '5':
        if len(cells) < 5:
            bad()
        kind = cells[0]
        if len(kind) >= 2 and kind.startswith('`') and kind.endswith('`'):
            kind = kind[1:-1]
        rules.kind_route[kind] = (cells[1], cells[2], cells[3], cells[4])
    elif section == '6':
        if len(cells) < 3:
            bad()
        rules.confirmed.append((cells[0], cells[1], cells[2]))
    # §3/§4/§7 and unknown sections: prose/not consumed.


def _read(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


def load_rules(knobs) -> Rules:
    """Resolve + parse per the DG-14 chain; loud on malformed or missing."""
    cands = []
    if knobs.rules_path:
        cands.append((knobs.rules_path, 'env', True))
    cands.append((os.path.join(knobs.ledger_dir, 'rules.md'), 'vault', False))
    cands.append((os.path.join(BUNDLED_ROOT, 'templates', 'rules.md'), 'bundled', False))
    for path, source, explicit in cands:
        if os.path.exists(path):
            rules = parse_rules(_read(path), path=os.path.abspath(path))
            rules.source = source
            return rules
        if explicit:
            raise RulesError(
                f'$PETASOS_RULES names {path} but it does not exist (looked in: '
                + ', '.join(p for p, _, _ in cands) + ')')
    raise RulesError('rules.md not found (looked in: ' + ', '.join(p for p, _, _ in cands) + ')')
