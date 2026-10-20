"""petasos.gates — the v2 gate evaluator (design §5, §6 C6 gate set; R3.3).

Pure function over (rows, balances, closed): no I/O, no mutation of its
arguments (§4 — snapshot-total repair + persistence is petasos.views' job,
DG-9). Emits the v1-observed gate lines (DG-15 string parity) plus the DG-1
chain gate:

  (a) ID uniqueness
  (b) per closed month: ledger expense sum == frozen total (R3.5, DG-10 —
      expenses-only freeze; late income may append and re-check passes)
  (c) balances arithmetic: assets = bank + cash + deposits + funds + property;
      debts = mortgage + loans + cc outstanding; net_worth = assets − debts;
      liquid = bank + cash. Stored snapshot_totals disagreeing with the
      derived values are flagged DRIFT (R3.4).
  (d) chain: bal_after(n) = bal_after(n−1) ∓ amount(n) over consecutive
      bal_after-bearing rows — "where evidence exists" (DG-1 REC A). An
      offsetting ± pair sharing one balance is ONE batched alert, not a
      break (F-1) — the same exception pipeline.check_chain applies.

Taxonomy (R3.7): CLOSED months are asserted; live periods are info-only —
no gate asserts the open month, so a new transaction never fails a gate.
"""
from dataclasses import dataclass, field
from datetime import date

BANK_KINDS = ('checking',)          # + any 'savings*' kind (prefix match)
ASSET_KINDS = BANK_KINDS + ('cash', 'deposit', 'fund', 'property')
DEBT_KINDS = ('mortgage', 'loan')
INCOME_KINDS = ('salary', 'rental', 'oneoff')


def money(n):
    """Dot-thousands integer format (v1 parity)."""
    return f"{n:,.0f}".replace(',', '.')


@dataclass
class GateReport:
    ok: bool
    lines: list = field(default_factory=list)


def _sum_kind(accounts, kinds, key='balance', prefix=False):
    return sum((a.get(key) or 0) for a in accounts.values()
               if (str(a.get('kind', '')).startswith(kinds[0]) if prefix
                   else a.get('kind') in kinds))


def _bank_total(accounts):
    """Bank accounts: kind 'checking' or any 'savings*' prefix (C5/R3.3).

    Repairs a v1 flaw: gen_views.py:66's prefix flag only matched kinds[0]
    ('checking'), so savings accounts silently vanished from bank_total,
    liquid and net_worth."""
    return sum((a.get('balance') or 0) for a in accounts.values()
               if str(a.get('kind', '')).startswith(('checking', 'savings')))


def derive_totals(accounts):
    """Recompute the ten snapshot_totals keys from account components (C5).

    Pure; used by run_gates for drift detection and by petasos.views for the
    DG-9 repair-and-persist step. Unused kinds sum to zero.
    """
    bank = _bank_total(accounts)
    cash = _sum_kind(accounts, ('cash',))
    deposits = _sum_kind(accounts, ('deposit',))
    funds = _sum_kind(accounts, ('fund',))
    property_v = _sum_kind(accounts, ('property',))
    assets = bank + cash + deposits + funds + property_v
    cc_out = sum((a.get('outstanding') or 0) for a in accounts.values()
                 if a.get('kind') == 'cc')
    debts = _sum_kind(accounts, DEBT_KINDS) + cc_out
    return {
        'bank_total': bank, 'cash_total': cash, 'deposits_total': deposits,
        'funds_total': funds, 'property_total': property_v,
        'assets_total': assets, 'cc_outstanding_total': cc_out,
        'debts_total': debts, 'net_worth': assets - debts,
        'liquid_total': bank + cash,
    }


def month_expense_total(rows, month):
    """Sum of expense amounts for a YYYY-MM (the DG-10 freeze scope)."""
    return sum((t.get('amount') or 0) for t in rows
               if t.get('date', '')[:7] == month and t.get('kind') == 'expense')


def _row_order_key(t):
    """(date, numeric suffix) — within a date the .NN suffix preserves the
    narrated-time order the ledger was appended in (R1.8)."""
    try:
        suffix = int(str(t.get('id', '')).rsplit('.', 1)[1])
    except (IndexError, ValueError):
        suffix = 0
    return (t.get('date', ''), suffix)


def offsetting_pair(prev, cur):
    """F-1 (t_d2548b02): an offsetting ± pair sharing one balance is ONE
    batched alert (13/10 incident log), not an unalerted transaction — the
    single definition of the exception, shared by this gate and
    pipeline.check_chain so the two chain checkers can never disagree again.
    Accepts ledger rows ('in'/'out') or parsed txs ('IN'/'OUT')."""
    return (prev.get('bal_after') == cur.get('bal_after')
            and abs(prev.get('amount') or 0) == abs(cur.get('amount') or 0)
            and str(prev.get('dir') or '').lower()
            != str(cur.get('dir') or '').lower())


def _chain_breaks(rows):
    """First-chain-break list over consecutive bal_after-bearing rows.

    F-5 (documented assumption): every bal_after-bearing row is treated as
    ONE chain — the C1/DG-11 frozen row schema has no account field. Today
    only ACB alerts carry bal_after so the assumption holds, but a SECOND
    bal_after-capturing bank added in banks.yaml (pure data) would interleave
    two accounts into this single chain and false-fail gates; resolving that
    needs an owner decision (schema additive key or account inference).

    F-1: an offsetting ± pair sharing one balance (one batched alert) is not
    a break — on a pair the chain reference stays on the first leg (both
    legs share its bal_after), mirroring check_chain exactly."""
    evidence = [t for t in sorted(rows, key=_row_order_key)
                if t.get('bal_after') is not None]
    breaks = []
    prev = None
    for cur in evidence:
        if prev is None:
            prev = cur
            continue
        if offsetting_pair(prev, cur):
            continue
        delta = (cur.get('amount') or 0) if cur.get('dir') == 'in' \
            else -(cur.get('amount') or 0)
        expected = prev['bal_after'] + delta
        if cur['bal_after'] != expected:
            breaks.append((prev.get('id'), cur.get('id'), expected,
                           cur['bal_after']))
        prev = cur
    return evidence, breaks


def run_gates(rows, balances, closed):
    """Evaluate all gates; return the report whose lines form the C6 stdout
    contract (final line `ALL GATES PASS` / `GATES FAILED`)."""
    results = []  # (ok, line)

    def gate(name, ok, detail=''):
        results.append((ok, ('✓' if ok else '✗ FAIL') + f' {name}'
                        + (f' — {detail}' if detail else '')))

    # (a) ID uniqueness (R1.5/R1.6 vocabulary guard is the pipeline's job)
    ids = [t.get('id', '') for t in rows]
    dups = sorted({i for i in ids if ids.count(i) > 1})
    gate('ID uniqueness', not dups,
         f'{len(ids)} rows' + (f' DUPES: {sorted(dups)[:3]}' if dups else ''))

    # (b) closed-month freeze (R3.5)
    for month, frozen in sorted((closed or {}).items()):
        live = month_expense_total(rows, month)
        gate(f'❄️  {month} closed = {money(frozen)} → ledger {money(live)}',
             live == frozen)

    # (c) balances arithmetic + snapshot drift flag (R3.3/R3.4)
    accounts = (balances or {}).get('accounts', {})
    totals = derive_totals(accounts)
    stored = (balances or {}).get('snapshot_totals') or {}
    drift = [k for k, v in totals.items() if k in stored and stored[k] != v]
    gate('Arithmetic: assets−debts=nw',
         not drift and totals['net_worth'] == totals['assets_total'] - totals['debts_total'],
         f"({money(totals['net_worth'])})" + (f' DRIFT: {drift}' if drift else ''))
    gate('Arithmetic: liquid = bank+cash',
         totals['liquid_total'] == totals['bank_total'] + totals['cash_total'],
         f"({money(totals['liquid_total'])})")

    # (d) chain gate (DG-1 REC A)
    evidence, breaks = _chain_breaks(rows)
    if breaks:
        pid, cid, expected, actual = breaks[0]
        gate('Chain: bal(n) = bal(n−1) ∓ amount(n)', False,
             f'break at {cid}: expected {money(expected)} ≠ {money(actual)} '
             f'({pid} → {cid})')
    else:
        gate('Chain: bal(n) = bal(n−1) ∓ amount(n)', True,
             f'{max(len(evidence) - 1, 0)} evidence pairs')

    lines = [line for _, line in results]
    cur = date.today().strftime('%Y-%m')
    lines.append(f'ℹ Live: {cur} expenses {money(month_expense_total(rows, cur))}'
                 f' · NW {money(totals["net_worth"])} · {len(rows)} rows')
    ok = all(ok for ok, _ in results)
    lines.append('ALL GATES PASS' if ok else 'GATES FAILED')
    return GateReport(ok=ok, lines=lines)
