"""petasos.balances — balances.json sync semantics (C5, S-2).

Bank alert `bal_after` is AUTHORITATIVE for its account; the latest alert by
recv_utc wins and an earlier alert never resurrects a balance. The pipeline
calls sync_from_alerts after the ledger append+verify (DG-6 write order);
`snapshot_totals` are DERIVED — recomputed from components on every sync
(C6 gate math: assets = bank + cash + deposits + funds + property;
debts = mortgage + loans + cc outstanding; net_worth = assets − debts;
liquid = bank + cash).

Account kinds (C5): checking, savings* (prefix), cash, deposit, fund,
property (assets via `balance`); mortgage, loan (debts via `balance`);
cc (debts via `outstanding`).
"""


def _is_bank(kind):
    return kind == 'checking' or str(kind).startswith('savings')


def recompute_totals(balances) -> dict:
    """Recompute the derived snapshot_totals from account components."""
    accounts = balances.get('accounts', {})
    bank = sum(a.get('balance', 0) for a in accounts.values() if _is_bank(a.get('kind')))
    cash = sum(a.get('balance', 0) for a in accounts.values() if a.get('kind') == 'cash')
    deposit = sum(a.get('balance', 0) for a in accounts.values()
                  if a.get('kind') == 'deposit')
    fund = sum(a.get('balance', 0) for a in accounts.values() if a.get('kind') == 'fund')
    property_ = sum(a.get('balance', 0) for a in accounts.values()
                    if a.get('kind') == 'property')
    mortgage = sum(a.get('balance', 0) for a in accounts.values()
                   if a.get('kind') == 'mortgage')
    loan = sum(a.get('balance', 0) for a in accounts.values() if a.get('kind') == 'loan')
    cc = sum(a.get('outstanding', 0) for a in accounts.values() if a.get('kind') == 'cc')
    assets = bank + cash + deposit + fund + property_
    debts = mortgage + loan + cc
    balances.setdefault('snapshot_totals', {}).update({
        'bank_total': bank,
        'cash_total': cash,
        'deposit_total': deposit,
        'fund_total': fund,
        'property_total': property_,
        'assets_total': assets,
        'debt_total': debts,
        'net_worth': assets - debts,
        'liquid_total': bank + cash,
    })
    return balances


def sync_from_alerts(balances, txs, knobs) -> dict:
    """Apply the latest alert for the primary account (env-driven mapping
    PETASOS_ACCT_NUM → PETASOS_ACCT_KEY), then recompute derived totals.

    Returns the same dict (mutated) so the caller controls file writes."""
    alerts = [t for t in txs
              if t.get('acct') == knobs.acct_num and t.get('bal_after') is not None]
    if alerts:
        latest = max(alerts, key=lambda t: str(t.get('recv_utc') or ''))
        accounts = balances.setdefault('accounts', {})
        acct = accounts.get(knobs.acct_key)
        if acct is None:
            acct = accounts[knobs.acct_key] = {'kind': 'checking', 'balance': 0}
        acct['balance'] = latest['bal_after']
        if latest.get('avail') is not None:
            acct['available'] = latest['avail']
        acct['verified'] = (f"email {latest.get('kind', 'bank')} "
                            f"{latest.get('recv_utc', '')} (auto-sync)")
    recompute_totals(balances)
    return balances
