"""petasos.scaffold — vault scaffolder (design §5; idempotent, synthetic
placeholders only, zero private deployment details R5.1).

Seeds the ledger tree with starter data the engine SHADOWS: the vault-side
rules.md (classification, C4) and banks.yaml (bank formats, C3) are copies of
the skill-bundled files — the user edits the vault copies and skill updates
never touch them (INV-6). Existing files are never overwritten, so re-running
is always safe.

Knob note (INTEGRATION SEAM, M3): unlike gen_views/monthly_close, this CLI is
argv-first (v1 usage: init_vault.py /path/to/vault); it resolves only
PETASOS_LEDGER_DIR (+ vault fallback) from the environment. Lane A's
petasos.knob is not required here.
"""
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

from .laneb_knobs import DEFAULT_LEDGER_DIR

SKILL_DIR = Path(__file__).resolve().parent.parent  # repo/skill root

STARTER_BALANCES = {
    '_readme': 'Account kinds: checking, savings*, cash (assets/liquid); deposit, '
               'fund, property (assets); mortgage, loan (debts, use balance); cc '
               '(debts, use "outstanding"). Keys are your own labels. '
               'snapshot_totals is DERIVED — gen_views.py recomputes it; do not '
               'hand-maintain.',
    'accounts': {
        'BANK-XXX': {'kind': 'checking', 'balance': 0},
        'SAVINGS-XXX': {'kind': 'savings', 'balance': 0},
        'CASH-DRAWER': {'kind': 'cash', 'balance': 0},
        'TERM-DEPOSIT-1': {'kind': 'deposit', 'balance': 0},
        'FUND-1': {'kind': 'fund', 'balance': 0},
        'PROPERTY-1': {'kind': 'property', 'balance': 0},
        'MORTGAGE-1': {'kind': 'mortgage', 'balance': 0},
        'LOAN-1': {'kind': 'loan', 'balance': 0},
        'CC-XXXX': {'kind': 'cc', 'limit': 0, 'outstanding': 0},
    },
    'snapshot_totals': {},
}


def scaffold(vault_path, knobs):
    """Create the vault ledger tree. Idempotent; existing files untouched.

    Returns 0 on success, nonzero on fatal (missing vault dir / no source).
    """
    vault = (vault_path or '').strip() or getattr(knobs, 'vault', '')
    if not vault:
        print('[fatal] pass the vault path as the first argument '
              '(or set PETASOS_VAULT)', file=sys.stderr)
        return 2
    if not os.path.isdir(vault):
        print(f'[fatal] {vault} does not exist — create the Obsidian vault '
              'first (or rerun with its parent created)', file=sys.stderr)
        return 2

    ledger_dir = getattr(knobs, 'ledger_dir', DEFAULT_LEDGER_DIR)
    L = ledger_dir if os.path.isabs(ledger_dir) else os.path.join(vault, ledger_dir)
    os.makedirs(L, exist_ok=True)

    created, kept = [], []

    def place(rel, text):
        path = os.path.join(L, rel)
        if os.path.exists(path):
            kept.append(rel)
        else:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(text)
            created.append(rel)

    def place_copy(rel, src):
        path = os.path.join(L, rel)
        if os.path.exists(path):
            kept.append(rel)
        elif os.path.exists(src):
            shutil.copyfile(src, path)
            created.append(rel)
        else:
            print(f'⚠️ no {rel} source found at {src} — create {rel} yourself')

    place('transactions.jsonl', '')
    place('state.json', json.dumps({'schema': 1, 'seen_threads': []},
                                   indent=1) + '\n')
    place('closed_months.json', '{}\n')
    place('balances.json',
          json.dumps(STARTER_BALANCES, indent=1, ensure_ascii=False) + '\n')
    place_copy('rules.md', os.path.join(SKILL_DIR, 'templates', 'rules.md'))
    place_copy('banks.yaml', os.path.join(SKILL_DIR, 'banks.yaml'))

    print(f'Petasos vault scaffold at {L}')
    for r in created:
        print(f'  + created  {r}')
    for r in kept:
        print(f'  = kept     {r} (already existed, untouched)')
    print(f'''
Next steps:
  1. Edit balances.json — real account keys/kinds/balances; delete placeholders you don't use
  2. Edit rules.md — your auto-post patterns, thresholds, known merchants
  3. Edit banks.yaml — your banks' email formats (verify: bank_adapters.py --probe <email.txt>,
     keep --test green by saving real emails under fixtures/ beside it)
  4. Set env: PETASOS_INBOX (+ AGENTMAIL_API_KEY), optionally PETASOS_ACCT_NUM/ACCT_KEY,
     PETASOS_OWN_NAMES, PETASOS_CURRENCY, PETASOS_THRESHOLD, PETASOS_BANKS
  5. Dry-run the pipeline:  python3 {os.path.join(SKILL_DIR, 'scripts', 'pipeline.py')} --dry
  6. Generate views + gates: python3 {os.path.join(SKILL_DIR, 'scripts', 'gen_views.py')}''')
    return 0


def main(argv=None):
    """CLI entry: init_vault.py [vault path] (env fallback per v1 usage)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    vault = argv[0] if argv else (os.environ.get('PETASOS_VAULT')
                                  or os.environ.get('OBSIDIAN_VAULT_PATH', ''))
    knobs = SimpleNamespace(
        vault=vault,
        ledger_dir=os.environ.get('PETASOS_LEDGER_DIR', DEFAULT_LEDGER_DIR),
    )
    return scaffold(argv[0] if argv else '', knobs)
