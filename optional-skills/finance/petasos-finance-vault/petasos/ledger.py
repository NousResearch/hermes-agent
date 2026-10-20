"""petasos.ledger — the append-only JSONL ledger (C1, DG-11 frozen v1-verbatim).

One JSON object per line, UTF-8, trailing newline per object, append-only
(R1.1). Corrections relabel IN PLACE (R1.4) over a restricted field set —
the ONLY sanctioned mutation; evidence fields (id, date, dir, amount,
currency, old_ref, bal_after) are never relabelable. New rows carry full
thread-id old_refs (DG-7). Manual rows/corrections enter through the DG-16
module entry point (`python3 -m petasos.ledger relabel|add`) — never a
hand-edit of the JSONL.
"""
import argparse
import json
import os
import re
import sys

from .parse import normalize_date

STATUS_VOCAB = ('candidate', 'posted', 'needs-confirm', 'aggregated')
KIND_VOCAB = ('expense', 'transfer', 'salary', 'rental', 'oneoff')
_SRC_EMAIL = re.compile(r'email:\S+')
_SRC_USER = re.compile(r'user-confirm:\d{4}-\d{2}-\d{2}')
_SRC_IMPORT = re.compile(r'import:\S+')
# v1-compat provenance forms (live history 2026-07..10; v1 never enforced a
# closed vocabulary — these are legitimate evidence labels whose specifics
# live in old_ref). Widened 14/10: the closed vocabulary refused the ENTIRE
# live ledger at the write gate, blocking the production email pipeline.
_SRC_CHAT = re.compile(r'chat:\S+-\d{4}-\d{2}-\d{2}')
_SRC_SCREENSHOT = re.compile(r'\S+-screenshot:\d{4}-\d{2}-\d{2}')
_SRC_COUNT = re.compile(r'user-count:\d{4}-\d{2}-\d{2}')
RELABEL_OPEN = ('category', 'type', 'fv', 'desc', 'ask')
RELABEL_NEVER = ('id', 'date', 'dir', 'amount', 'currency', 'old_ref', 'bal_after')


class LedgerError(Exception):
    """User-facing ledger failure — callers print the message and exit nonzero."""


def _read_rows(ledger_path):
    rows = []
    if not os.path.exists(ledger_path):
        return rows
    with open(ledger_path, encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise LedgerError(
                    f'{ledger_path}: line {lineno} is not valid JSON: {e}') from None
    return rows


def _src_ok(src):
    return bool(_SRC_EMAIL.fullmatch(src or '') or _SRC_USER.fullmatch(src or '')
                or _SRC_IMPORT.fullmatch(src or '') or _SRC_CHAT.fullmatch(src or '')
                or _SRC_SCREENSHOT.fullmatch(src or '') or _SRC_COUNT.fullmatch(src or ''))


def next_id(rows, date_iso, pending_ids) -> str:
    """`YYYY-MM-DD.NN` with NN = MAX existing suffix for that date + 1 (R1.5 —
    `.01,.03` present → `.04`; gaps are normal and never renumbered).
    `pending_ids` feeds issued batch IDs back so one batch stays unique."""
    prefix = date_iso + '.'
    suffixes = [int(str(r.get('id', ''))[len(prefix):])
                for r in rows
                if str(r.get('id', '')).startswith(prefix)
                and str(r.get('id', ''))[len(prefix):].isdigit()]
    for pid in pending_ids:
        if str(pid).startswith(prefix) and str(pid)[len(prefix):].isdigit():
            suffixes.append(int(str(pid)[len(prefix):]))
    n = max(suffixes) + 1 if suffixes else 1
    return f'{prefix}{n:02d}'


def build_row(tx, decision, knobs) -> dict:
    """C1 row from a parsed tx + classify Decision (R1.2 provenance).
    `id` is left empty — the caller allocates it via next_id."""
    if tx.get('datetime'):
        date_iso = normalize_date(str(tx['datetime']).split(' ')[0])
    else:
        date_iso = normalize_date(str(tx.get('date') or ''))
    desc = (tx.get('content') or tx.get('merchant') or tx.get('beneficiary')
            or tx.get('kind') or '')
    row = {
        'id': '',
        'date': date_iso,
        'dir': 'in' if tx.get('dir') == 'IN' else 'out',
        'kind': decision.kind,
        'old_ref': f"mail:{tx['src_thread']}" if tx.get('src_thread') else '',
        'amount': tx.get('amount'),
        'currency': knobs.currency,
        'desc': str(desc)[:90],
        'status': decision.status,
        'src': f"email:{tx['kind']}",
    }
    if decision.category:
        row['category'] = decision.category
    if decision.type:
        row['type'] = decision.type
    if decision.fv:
        row['fv'] = decision.fv
    if decision.ask:
        row['ask'] = decision.ask
    if tx.get('bal_after') is not None:
        row['bal_after'] = tx['bal_after']
    return row


def validate_ledger(rows) -> list:
    """C1 validity: unique ids, closed status vocabulary, resolvable src."""
    problems, seen = [], set()
    for i, r in enumerate(rows):
        rid = r.get('id', f'<row {i}>')
        if rid in seen:
            problems.append(f'duplicate id {rid}')
        seen.add(rid)
        if r.get('status') not in STATUS_VOCAB:
            problems.append(f'{rid}: status {r.get("status")!r} outside vocabulary '
                            f'{STATUS_VOCAB}')
        if not _src_ok(r.get('src')):
            problems.append(f'{rid}: src {r.get("src")!r} does not resolve to a '
                            'provenance form (email:<KIND> | user-confirm:<date> | '
                            'import:<source>)')
    return problems


def append_rows(ledger_path, rows) -> None:
    """R1.1/R1.7 — validate the batch (and id collisions with the existing
    file) BEFORE touching disk, then append one JSON line per row."""
    existing = _read_rows(ledger_path)
    problems = validate_ledger(existing + rows)
    if problems:
        raise LedgerError('refusing to append: ' + '; '.join(problems))
    with open(ledger_path, 'a', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def verify_append(ledger_path, prior_count, batch) -> None:
    """R1.7 — re-read the whole file, parse every line, assert the count grew
    by exactly the batch size."""
    rows = _read_rows(ledger_path)
    if len(rows) != prior_count + len(batch):
        raise LedgerError(
            f'{ledger_path}: expected {prior_count + len(batch)} rows after append '
            f'({prior_count} + {len(batch)}), found {len(rows)}')


def relabel(ledger_path, row_id, fields) -> None:
    """R1.4 — in-place correction over the restricted C1 field set. Only the
    target line is rewritten; every other line stays byte-identical."""
    for k in fields:
        if k in RELABEL_NEVER:
            raise LedgerError(f'{k} is evidence and can never be relabeled')
    if 'status' in fields and fields['status'] != 'posted':
        raise LedgerError("relabel may only promote status to 'posted'")
    if 'src' in fields and not _SRC_USER.fullmatch(fields['src']):
        raise LedgerError("relabel src must be 'user-confirm:<YYYY-MM-DD>'")
    if 'kind' in fields and fields['kind'] not in KIND_VOCAB:
        raise LedgerError(f'kind must be one of {KIND_VOCAB}')

    with open(ledger_path, encoding='utf-8') as f:
        lines = f.read().splitlines()
    target = None
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get('id') == row_id:
            target = i
            break
    if target is None:
        raise LedgerError(f'{ledger_path}: no row {row_id!r}')
    row = json.loads(lines[target])
    if 'kind' in fields and row.get('status') != 'needs-confirm':
        raise LedgerError('kind may only be relabeled from the needs-confirm '
                          'confirmation path')
    row.update(fields)
    problems = validate_ledger([row])
    if problems:
        raise LedgerError('relabel would make the row invalid: ' + '; '.join(problems))
    lines[target] = json.dumps(row, ensure_ascii=False)
    with open(ledger_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')


def add_manual_row(ledger_path, *, date_iso, direction, kind, amount, desc,
                   category=None, type=None, fv=None, via=None,
                   confirmed=None) -> dict:
    """R4.6 — a chat-reported exact item: ONE posted row, src
    user-confirm:<date>. When `category` is omitted, a precedent row whose
    desc matches the same rules.md §1/§3a pattern supplies category/type/fv
    (never a new category for a known pattern)."""
    if kind not in KIND_VOCAB:
        raise LedgerError(f'kind must be one of {KIND_VOCAB}')
    if not isinstance(amount, int):
        raise LedgerError('amount must be a plain integer in ledger currency')
    confirmed = confirmed or date_iso
    rows = _read_rows(ledger_path)
    if category is None:
        precedent = _find_precedent(rows, desc)
        if precedent:
            category = precedent.get('category') or None
            type = type or precedent.get('type') or None
            fv = fv or precedent.get('fv') or None
    row = {
        'id': next_id(rows, date_iso, []),
        'date': date_iso,
        'dir': 'in' if str(direction).lower() in ('in', 'credit') else 'out',
        'kind': kind,
        'old_ref': '',
        'amount': amount,
        'currency': os.environ.get('PETASOS_CURRENCY', 'VND'),
        'desc': str(desc)[:90],
        'status': 'posted',
        'src': f'user-confirm:{confirmed}',
    }
    if category:
        row['category'] = category
    if type:
        row['type'] = type
    if fv:
        row['fv'] = fv
    if via:
        row['via'] = via
    append_rows(ledger_path, [row])
    verify_append(ledger_path, len(rows), [row])
    return row


def _find_precedent(rows, desc):
    """Latest row whose desc shares a rules.md §1/§3a pattern with `desc`."""
    try:
        from .rules import load_rules
        from .knob import load_knobs
        rules = load_rules(load_knobs(os.environ))
    except Exception:
        return None
    import re as _re
    for pat, *_ in list(rules.known_route) + list(rules.auto_post):
        try:
            if not _re.search(pat, desc, _re.I):
                continue
        except _re.error:
            continue
        for r in reversed(rows):
            try:
                if _re.search(pat, str(r.get('desc', '')), _re.I):
                    return r
            except _re.error:
                continue
    return None


# ---------------- DG-16 module entry point ----------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='petasos.ledger',
                                     description='Manual ledger operations (DG-16)')
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_rel = sub.add_parser('relabel', help='in-place correction (R1.4)')
    p_rel.add_argument('--ledger', required=True)
    p_rel.add_argument('--id', required=True)
    p_rel.add_argument('--set', action='append', default=[], metavar='K=V')

    p_add = sub.add_parser('add', help='chat-reported row (R4.6)')
    p_add.add_argument('--ledger', required=True)
    p_add.add_argument('--date', required=True)
    p_add.add_argument('--dir', required=True)
    p_add.add_argument('--kind', required=True)
    p_add.add_argument('--amount', required=True)
    p_add.add_argument('--desc', required=True)
    p_add.add_argument('--category')
    p_add.add_argument('--type')
    p_add.add_argument('--fv')
    p_add.add_argument('--via')

    args = parser.parse_args(argv)
    try:
        if args.cmd == 'relabel':
            fields = {}
            for kv in args.set:
                if '=' not in kv:
                    raise LedgerError(f'--set expects K=V, got {kv!r}')
                k, v = kv.split('=', 1)
                fields[k] = v
            relabel(args.ledger, args.id, fields)
            print(f'relabeled {args.id}: {fields}')
            return 0
        try:
            amount = int(str(args.amount).strip())
        except ValueError:
            raise LedgerError(f'amount must be an integer, got {args.amount!r}') from None
        row = add_manual_row(args.ledger, date_iso=args.date, direction=args.dir,
                             kind=args.kind, amount=amount, desc=args.desc,
                             category=args.category, type=args.type, fv=args.fv,
                             via=args.via)
        print(f'added {row["id"]}: {row["dir"]} {row["amount"]:,} '
              f'{row.get("category") or "-"} {row["desc"]}')
        return 0
    except LedgerError as e:
        print(str(e), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
