"""Lane-B (t_58139420) test helpers: scratch-vault factory + stub knobs.

Lane B builds strictly against the design doc's pinned FILE contracts (§6 C1,
C5, C6); the real Knobs dataclass belongs to Lane A's petasos.knob (§5). Tests
therefore pass stub objects exposing only the attributes lane B consumes:
.vault (vault root) and .ledger_dir (relative to the vault, C7 default).

All data here is synthetic round figures — zero private deployment details
(design §2 anti-goals, R5.1).
"""
import json
import os
import types

DEFAULT_LEDGER_DIR = '2-Areas/Finance/_Ledger'


def knobs_for(vault, ledger_dir=DEFAULT_LEDGER_DIR):
    """Stub knobs object with exactly the attributes lane B reads."""
    return types.SimpleNamespace(vault=str(vault), ledger_dir=ledger_dir)


def ledger_path(knobs):
    d = knobs.ledger_dir
    return d if os.path.isabs(d) else os.path.join(knobs.vault, d)


def row(**kw):
    """Synthetic ledger row (C1 shape) with lane-B-relevant defaults."""
    base = dict(
        id='2026-10-13.01', date='2026-10-13', dir='out', kind='expense',
        old_ref='mail:deadbeefcafe', amount=100000, currency='VND',
        desc='synthetic test row', status='posted', src='email:TEST',
    )
    base.update(kw)
    return base


def balances(accounts=None, snapshot_totals=None):
    """Minimal balances.json-shaped dict (C5)."""
    return {
        '_readme': 'synthetic test balances',
        'accounts': {} if accounts is None else accounts,
        'snapshot_totals': {} if snapshot_totals is None else snapshot_totals,
    }


def starter_accounts():
    """One account per kind, round synthetic figures. Kinds per C5."""
    return {
        'BANK-A': {'kind': 'checking', 'balance': 1000},
        'SAVE-A': {'kind': 'savings', 'balance': 200},
        'SAVE-B': {'kind': 'savings-joint', 'balance': 300},
        'CASH-DRAWER': {'kind': 'cash', 'balance': 50},
        'TERM-1': {'kind': 'deposit', 'balance': 5000},
        'FUND-1': {'kind': 'fund', 'balance': 700},
        'PROP-1': {'kind': 'property', 'balance': 9000},
        'MORT-1': {'kind': 'mortgage', 'balance': 4000},
        'LOAN-1': {'kind': 'loan', 'balance': 600},
        'CC-1': {'kind': 'cc', 'limit': 10000, 'outstanding': 250},
    }


def make_vault(root, rows=(), accounts=None, closed=None,
               snapshot_totals=None, ledger_dir=DEFAULT_LEDGER_DIR):
    """Create a scratch vault ledger tree under root; returns stub knobs.

    Writes transactions.jsonl, balances.json, closed_months.json (C1/C5).
    """
    knobs = knobs_for(root, ledger_dir)
    L = ledger_path(knobs)
    os.makedirs(L, exist_ok=True)
    with open(os.path.join(L, 'transactions.jsonl'), 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    bal = {
        '_readme': 'synthetic test balances',
        'accounts': starter_accounts() if accounts is None else accounts,
        'snapshot_totals': {} if snapshot_totals is None else snapshot_totals,
    }
    with open(os.path.join(L, 'balances.json'), 'w', encoding='utf-8') as f:
        json.dump(bal, f, indent=1, ensure_ascii=False)
    with open(os.path.join(L, 'closed_months.json'), 'w', encoding='utf-8') as f:
        json.dump({} if closed is None else closed, f, indent=1)
    return knobs
