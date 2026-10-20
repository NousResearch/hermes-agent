"""petasos.views — view rendering + regeneration (design §5, §6 C6, DG-9).

The view side of the engine→view subprocess contract: reads transactions.jsonl
(C1), balances.json and closed_months.json (C5); writes Dashboards.md and
Salary History.md beside the ledger dir and persists the DERIVED
snapshot_totals into balances.json (DG-9 REC A — one writer of derived
totals). It NEVER writes transactions.jsonl, state.json, or closed_months.json.

Exit-code contract (C6/DG-15): main() returns 0 iff every gate passed; the
pipeline (Lane A, M3 wiring) MUST treat nonzero as fatal — that is the
enforcement behind "after every ledger write ALL gates pass or the write
pipeline aborts".

v1 string parity: section layout, money format, `generator: gen_views.py`
frontmatter, and the `ALL GATES PASS` / `GATES FAILED` verdicts are kept
byte-compatible with v1.1.0 gen_views.py (the production cron wrapper greps
them). Additive v2 section: ❓ Open confirmations — surfaces needs-confirm
rows so the dashboard carries the ask until the owner classifies it
(R2.5/R4.1 'never re-asked').
"""
import json
import os
import sys
from collections import defaultdict
from datetime import date

from .gates import (INCOME_KINDS, derive_totals, money, month_expense_total,
                    run_gates)
from .laneb_knobs import MissingKnobs, ledger_path, load_knobs as _default_load


def _load_rows(ledger_dir):
    path = os.path.join(ledger_dir, 'transactions.jsonl')
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _confirmations_section(rows):
    pending = [t for t in rows if t.get('status') == 'needs-confirm']
    if not pending:
        return []
    lines = ['', '## ❓ Open confirmations', '',
             '| Date | Amount | Ask |', '|---|---|---|']
    for t in sorted(pending, key=lambda t: (t.get('date', ''), t.get('id', ''))):
        ask = t.get('ask') or ''
        cell = f"{ask} — {t.get('desc', '')}".strip(' —') if ask else t.get('desc', '')
        lines.append(f"| {t.get('date', '')} | {money(t.get('amount') or 0)} | {cell} |")
    return lines


def _dashboard_text(rows, totals, today):
    cur = today.strftime('%Y-%m')
    months = sorted({t.get('date', '')[:7] for t in rows if t.get('date')})
    rev = cur if cur in months else (months[-1] if months else cur)

    flow = defaultdict(lambda: {'in': 0, 'out': 0})
    cat_rev = defaultdict(int)
    for t in rows:
        m = t.get('date', '')[:7]
        amt = t.get('amount') or 0
        kind = t.get('kind')
        if kind == 'transfer':
            continue
        if t.get('dir') == 'in' or kind in INCOME_KINDS:
            flow[m]['in'] += amt
        else:
            flow[m]['out'] += amt
            if m == rev:
                cat_rev[t.get('category') or 'Uncategorized'] += amt

    dash = [f'---\ntags: [finance, dashboard]\ngenerated: {today.isoformat()}'
            '\ngenerator: gen_views.py\n---',
            '', '# 📊 Dashboards', '',
            '> Auto-generated from the ledger — do not hand-edit numbers.', '',
            '## 🏦 Position', '',
            '| | Amount |', '|---|---|',
            f"| Liquid (bank + cash) | {money(totals['liquid_total'])} |",
            f"| Deposits | {money(totals['deposits_total'])} |",
            f"| Funds | {money(totals['funds_total'])} |",
            f"| Property | {money(totals['property_total'])} |",
            f"| **Total assets** | **{money(totals['assets_total'])}** |",
            f"| Loans / mortgage | {money(totals['debts_total'] - totals['cc_outstanding_total'])} |",
            f"| Credit cards outstanding | {money(totals['cc_outstanding_total'])} |",
            f"| **Total debts** | **{money(totals['debts_total'])}** |",
            f"| **NET WORTH** | **{money(totals['net_worth'])}** |",
            '', '## 📅 Monthly flow (transfers excluded)', '',
            '| Month | In | Out | Surplus |', '|---|---|---|---|']
    for m in months:
        r = flow[m]
        dash.append(f"| {m} | {money(r['in'])} | {money(r['out'])} |"
                    f" {money(r['in'] - r['out'])} |")

    if cat_rev:
        dash += ['', f'## 🗂 Spend by category — {rev}', '',
                 '| Category | Amount |', '|---|---|']
        for c, v in sorted(cat_rev.items(), key=lambda x: -x[1]):
            dash.append(f'| {c} | {money(v)} |')

    dash += _confirmations_section(rows)
    return '\n'.join(dash) + '\n'


def _salary_text(rows):
    salaries = defaultdict(lambda: defaultdict(int))  # year -> via -> total
    for t in rows:
        if t.get('kind') == 'salary':
            salaries[t.get('date', '')[:4]][t.get('via', '?')] += t.get('amount') or 0
    if not salaries:
        return None
    lines = ['---\ntags: [finance, salary, dashboard]\ngenerator: gen_views.py\n---',
             '', '# 💼 Salary History (auto-generated)', '',
             '| Year | Source | Total |', '|---|---|---|']
    for y in sorted(salaries):
        for via, v in sorted(salaries[y].items()):
            lines.append(f'| {y} | {via} | {money(v)} |')
    return '\n'.join(lines) + '\n'


def regenerate(knobs):
    """Gates → write views → persist snapshot totals (§5 sequence, DG-9).

    Reads the ledger/balances/closed files, evaluates all gates (pure), then —
    regardless of the verdict, v1 parity — regenerates the views and persists
    the repaired snapshot_totals (R3.4's repair must land on the drift run
    itself; the verdict travels in the returned report / exit code).
    """
    ledger_dir = ledger_path(knobs)
    rows = _load_rows(ledger_dir)
    balances = _load_json(os.path.join(ledger_dir, 'balances.json'), default={})
    closed = _load_json(os.path.join(ledger_dir, 'closed_months.json'), default={})

    report = run_gates(rows, balances, closed)

    # DG-9: snapshot_totals are DERIVED — recompute from components, persist.
    balances.setdefault('accounts', {})
    balances['snapshot_totals'] = derive_totals(balances['accounts'])

    today = date.today()
    views_dir = os.path.normpath(os.path.join(ledger_dir, '..'))
    with open(os.path.join(views_dir, 'Dashboards.md'), 'w', encoding='utf-8') as f:
        f.write(_dashboard_text(rows, balances['snapshot_totals'], today))
    salary = _salary_text(rows)
    if salary is not None:
        with open(os.path.join(views_dir, 'Salary History.md'), 'w',
                  encoding='utf-8') as f:
            f.write(salary)
    with open(os.path.join(ledger_dir, 'balances.json'), 'w', encoding='utf-8') as f:
        json.dump(balances, f, indent=1, ensure_ascii=False)
    return report


def main(load=None):
    """CLI entry: prints the gate report lines; 0 iff ALL GATES PASS.

    Exit codes: 0 all gates pass · 1 gate failure · 2 config/input fatal.
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
    try:
        report = regenerate(knobs)
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f'[fatal] missing input: {e}', file=sys.stderr)
        return 2
    print('\n'.join(report.lines))
    return 0 if report.ok else 1
