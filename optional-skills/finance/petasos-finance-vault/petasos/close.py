"""petasos.close — month freeze + archive (design §5, §6 C5; R3.5/R3.6/R3.8,
DG-10).

Semantics (all resolved REC/option A):
- Freeze covers EXPENSES only (month_expense_total); late income may append
  into a closed month and the frozen sum re-check still passes (DG-10).
- closed_months.json is IMMUTABLE once written: drift after close is a loud
  error — fix the ledger, never the freeze (R3.5).
- Preconditions (R3.8): close runs only with all gates green AND no
  needs-confirm rows open in the closing month.
- Archive (R3.6): full-ledger copy (intentional restore point, not just the
  month's rows) + generated Monthly Review carrying frozen_total.

v1 string parity for the close output lines (❄️ FROZEN / ✓ already frozen /
✗ FAIL / 📦 Snapshot); two v1 flaws repaired: the review builder crashed on
rows without the optional `category` (KeyError, C1 says category is optional)
and a malformed month argument was accepted silently.
"""
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import date, timedelta

from .gates import INCOME_KINDS, money, month_expense_total, run_gates
from .laneb_knobs import MissingKnobs, ledger_path, load_knobs as _default_load

MONTH_RE = re.compile(r'^\d{4}-\d{2}$')


def _load_rows(ledger_dir):
    with open(os.path.join(ledger_dir, 'transactions.jsonl'), encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def _monthly_review(month, mtxs, total, today):
    inc = sum(t.get('amount') or 0 for t in mtxs if t.get('kind') in INCOME_KINDS)
    cats = {}
    for t in mtxs:
        if t.get('kind') == 'expense':
            cat = t.get('category')
            cats[cat] = cats.get(cat, 0) + (t.get('amount') or 0)
    surplus = inc - total
    ncnt = sum(1 for t in mtxs if t.get('kind') == 'expense')
    icnt = sum(1 for t in mtxs if t.get('kind') in INCOME_KINDS)
    tcnt = sum(1 for t in mtxs if t.get('kind') == 'transfer')

    rv = (f"---\n"
          f"tags: [finance, monthly-review, archive]\n"
          f"month: {month}\n"
          f"frozen_total: {total}\n"
          f"generated: {today.isoformat()}\n"
          f"generator: _Ledger/monthly_close.py\n"
          f"---\n"
          f"\n"
          f"# 📋 Monthly Review — {month} (CLOSED)\n"
          f"\n"
          f"| Metric | Value |\n"
          f"|---|---|\n"
          f"| Income ({icnt} rows) | {money(inc)} |\n"
          f"| Tracked expenses ({ncnt} rows) | {money(total)} |\n"
          f"| **Surplus** | **{money(surplus)}**"
          f"{f' ({surplus / inc * 100:.1f}% of income)' if inc else ''} |\n"
          f"| Internal transfers | {tcnt} rows (excluded from totals) |\n"
          f"\n"
          f"## Spend by category\n"
          f"\n"
          f"| Category | Amount | % |\n"
          f"|---|---|---|\n")
    for cat, amt in sorted(cats.items(), key=lambda x: -x[1]):
        rv += f"| {cat or '—'} | {money(amt)} | {amt / total * 100:.0f}% |\n"
    rv += "\n## Largest transactions of the month\n\n"
    for t in sorted((t for t in mtxs if t.get('kind') == 'expense'),
                    key=lambda x: -(x.get('amount') or 0))[:5]:
        rv += (f"- {t.get('date', '')} · {money(t.get('amount') or 0)} · "
               f"{t.get('category') or '-'} · {(t.get('desc') or '')[:70]}\n")
    return rv


def _gate_script(ledger_dir):
    """C6 gate authority: the vault-local gen_views.py shadows the bundled
    one — a vault with tuned gates is judged by ITS gates, the same rule the
    post-write path applies. None = fall back to in-process package gates."""
    vault_copy = os.path.join(ledger_dir, 'gen_views.py')
    if os.path.exists(vault_copy):
        return vault_copy
    bundled = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'scripts', 'gen_views.py')
    return bundled if os.path.exists(bundled) else None


def _gates_green(knobs, ledger_dir, rows, balances, closed):
    """True when the AUTHORITATIVE gate source is green (C6 resolution)."""
    script = _gate_script(ledger_dir)
    if script:
        env = dict(os.environ)
        env['PETASOS_VAULT'] = knobs.vault
        env['PETASOS_LEDGER_DIR'] = os.path.relpath(ledger_dir, knobs.vault)
        proc = subprocess.run([sys.executable, script], capture_output=True,
                              text=True, env=env)
        out = (proc.stdout or '') + (proc.stderr or '')
        if proc.returncode != 0 or 'GATES FAILED' in out:
            print('✗ close aborted — gates are red (close runs only with all gates green):')
            for line in out.splitlines():
                if line.startswith('✗') or line == 'GATES FAILED':
                    print(f'   {line}')
            return False
        return True
    report = run_gates(rows, balances, closed)
    if not report.ok:
        print('✗ close aborted — gates are red (close runs only with all gates green):')
        for line in report.lines:
            if line.startswith('✗') or line == 'GATES FAILED':
                print(f'   {line}')
        return False
    return True


def close_month(knobs, month=None):
    """Freeze a month's expense total + archive. Default: previous month.

    Returns 0 on success, nonzero on any refusal (R3.8 preconditions failed,
    frozen-month drift). Never rewrites an existing freeze.
    """
    ledger_dir = ledger_path(knobs)
    rows = _load_rows(ledger_dir)
    with open(os.path.join(ledger_dir, 'balances.json'), encoding='utf-8') as f:
        balances = json.load(f)
    closed_path = os.path.join(ledger_dir, 'closed_months.json')
    closed = {}
    if os.path.exists(closed_path):
        with open(closed_path, encoding='utf-8') as f:
            closed = json.load(f)

    today = date.today()
    if month is None:
        month = f"{(today.replace(day=1) - timedelta(days=1)):%Y-%m}"

    # R3.8 preconditions — refuse loudly BEFORE freezing or archiving.
    # Gate authority follows C6: the vault that owns the ledger owns the
    # gate verdict (vault-local gen_views.py shadows the bundled one).
    if not _gates_green(knobs, ledger_dir, rows, balances, closed):
        return 1
    pending = [t for t in rows
               if t.get('date', '')[:7] == month
               and t.get('status') == 'needs-confirm']
    if pending:
        ids = ', '.join(t.get('id', '?') for t in pending[:3])
        print(f'✗ close aborted — {len(pending)} needs-confirm row(s) open '
              f'in {month}: {ids}')
        return 1

    total = month_expense_total(rows, month)
    if month not in closed:
        closed[month] = total
        with open(closed_path, 'w', encoding='utf-8') as f:
            json.dump(closed, f, indent=1)
        print(f'❄️ FROZEN {month}: expense total = {money(total)} (frozen)')
    else:
        # Defense-in-depth: drift normally aborts at the gates precondition
        # above (frozen-month gate); this branch guards the freeze itself in
        # case gate scope ever changes. Never rewrite an existing freeze.
        if closed[month] != total:
            print(f'✗ FAIL {month} already frozen at {money(closed[month])}; '
                  f'ledger now {money(total)}')
            print('   ⚠️ A FROZEN MONTH CHANGED — check the pipeline!')
            return 1
        print(f'✓ {month} already frozen at {money(closed[month])}; '
              f'ledger now {money(total)}')

    # Archive (R3.6): full-ledger restore point + generated review.
    arc = os.path.join(knobs.vault, '4-Archives', 'Finance', month)
    os.makedirs(arc, exist_ok=True)
    shutil.copyfile(os.path.join(ledger_dir, 'transactions.jsonl'),
                    os.path.join(arc, f'transactions-{month}.jsonl'))
    mtxs = [t for t in rows if t.get('date', '')[:7] == month]
    with open(os.path.join(arc, f'Monthly Review {month}.md'), 'w',
              encoding='utf-8') as f:
        f.write(_monthly_review(month, mtxs, total, today))
    inc = sum(t.get('amount') or 0 for t in mtxs if t.get('kind') in INCOME_KINDS)
    print(f'📦 Snapshot → 4-Archives/Finance/{month}/: '
          f'transactions-{month}.jsonl + Monthly Review {month}.md')
    print(f'   {len(mtxs)} rows in month {month} · income {money(inc)} · '
          f'expenses {money(total)} · surplus {money(inc - total)}')
    return 0


def main(argv=None, load=None):
    """CLI entry: monthly_close.py [YYYY-MM] (default: previous month).

    Exit codes: 0 closed/verified · 1 refused (preconditions, drift) ·
    2 config/usage fatal.
    """
    try:
        knobs = (load or _default_load)(os.environ)
    except MissingKnobs as e:
        print(f'[fatal] {e}', file=sys.stderr)
        return 2
    if not getattr(knobs, 'vault', ''):
        # loader-independent guard: petasos.knob is lenient on empty vault
        print('[fatal] set PETASOS_VAULT (or OBSIDIAN_VAULT_PATH)', file=sys.stderr)
        return 2
    argv = list(sys.argv[1:] if argv is None else argv)
    month = argv[0] if argv else None
    if month is not None and not MONTH_RE.match(month):
        print(f'[fatal] month must be YYYY-MM, got {month!r}', file=sys.stderr)
        return 2
    try:
        return close_month(knobs, month)
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f'[fatal] missing input: {e}', file=sys.stderr)
        return 2
