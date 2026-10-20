"""petasos.classify — tier routing as a pure function over (tx, rules, knobs).

Precedence (§7, v1-observed order preserved; §6-first is the design's
PROPOSED correction so a confirmed item is never re-asked):
  1. rules.md §6 confirmed-by-user → posted, never re-asked (R2.5)
  2. §3a known-merchant routing → posted regardless of threshold, no ask
  3. §1 auto-post patterns → below threshold: silent post; at/above:
     post WITH ask 'above-thresh-but-pattern'
  4. §5 kind routing (DG-3 data) — tier vocabulary documented in
     templates/rules.md §5:
       expense / expense-under-thresh / balance-alert / plumbing /
       transfer-receipt
  5. own-name guard — narration matching PETASOS_OWN_NAMES → transfer
     needs-confirm 'destination unknown', NEVER an expense (R4.3)
  6. unknown ≥ threshold → needs-confirm; mystery credits (IN) hold kind
     'oneoff' — the only C1-vocabulary-valid holding kind for an
     unconfirmed credit (v1 wrote the invalid 'income?'); confirmation
     relabels it (R1.4 path).

All bank/merchant knowledge lives in data; this module knows only the tier
vocabulary above (P2). Effective threshold: env → rules.md
`threshold_auto_post` → 500000 (R4.4).
"""
import re
from dataclasses import dataclass

DEFAULT_THRESHOLD = 500000


@dataclass
class Decision:
    kind: str
    status: str
    category: str | None = None
    type: str | None = None
    fv: str | None = None
    ask: str | None = None


def effective_threshold(knobs, rules) -> int:
    if knobs.threshold is not None:
        return knobs.threshold
    if rules.threshold_default is not None:
        return rules.threshold_default
    return DEFAULT_THRESHOLD


def is_pending_hold(tx) -> bool:
    """S-3 — available < balance on the same alert is a pending hold:
    record to balances, NO ledger row."""
    bal = tx.get('bal_after')
    avail = tx.get('avail')
    return bal is not None and avail is not None and avail < bal


def _desc(tx) -> str:
    return ' '.join(str(tx.get(x) or '') for x in ('content', 'merchant', 'beneficiary'))


def _clean(v):
    return v if v else None


def _decision_from_row(kind, status, row, ask=None):
    _, cat, typ, fv = row if row else (None, None, None, None)
    return Decision(kind=kind, status=status, category=_clean(cat), type=_clean(typ),
                    fv=_clean(fv), ask=ask)


def _apply_tier(tx, row, desc, amount, thresh, own):
    tier = row[0]
    d = tx.get('dir')
    if tier == 'plumbing':                       # R4.1: reconcile-only transfer
        return Decision(kind='transfer', status='posted',
                        category=_clean(row[1]), type=_clean(row[2]), fv=_clean(row[3]))
    if tier == 'transfer-receipt':               # counterparty blind
        if own:
            return Decision(kind='transfer', status='needs-confirm',
                            ask='own-name transfer — destination unknown')
        return Decision(kind='transfer', status='needs-confirm')
    if tier == 'expense':                        # v1 HSBC_CC parity
        return _decision_from_row('expense', 'posted', row)
    if tier == 'expense-under-thresh':           # v1 BIDV_CARD parity
        if amount < thresh:
            return _decision_from_row('expense', 'posted', row)
        return Decision(kind='expense', status='needs-confirm', ask='matches no rule')
    if tier == 'balance-alert':                  # v1 ACB ladder
        if own:
            return Decision(kind='transfer', status='needs-confirm',
                            ask='own-name transfer — destination unknown')
        if d == 'OUT':
            if amount < thresh:
                return _decision_from_row('expense', 'posted', row)
            return Decision(kind='expense', status='needs-confirm', ask='matches no rule')
        if d == 'IN':
            if amount >= thresh:
                return Decision(kind='oneoff', status='needs-confirm', ask='source unknown')
            return Decision(kind='oneoff', status='posted')
    return Decision(kind='oneoff', status='needs-confirm', ask='matches no rule')


def _safe_search(pattern, text):
    """§6 items are human-written confirmations, not curated regexes — many
    begin with '+'/'*' or carry prose (live case 14/10: '+900,000 Zalo (…)').
    An invalid regex falls back to case-insensitive LITERAL substring match."""
    try:
        return re.search(pattern, text, re.I) is not None
    except re.error:
        return pattern.lower() in text.lower()


def classify(tx, rules, knobs) -> Decision:
    desc = _desc(tx)
    upper = desc.upper()
    amount = tx.get('amount') or 0
    own = any(n and n in upper for n in knobs.own_names)
    thresh = effective_threshold(knobs, rules)

    # 1. §6 confirmed-by-user — never re-asked (R2.5)
    for _date, item, _meaning in rules.confirmed:
        if item and _safe_search(item, desc):
            row = rules.kind_route.get(tx.get('kind'))
            if row and row[0] in ('expense', 'expense-under-thresh', 'balance-alert'):
                if row[0] == 'balance-alert':
                    d2 = _apply_tier(tx, row, desc, amount, 0, own=False)
                    if d2.status == 'posted':
                        return d2
                    return _decision_from_row(
                        'oneoff' if tx.get('dir') == 'IN' else 'expense', 'posted', row)
                return _decision_from_row('expense', 'posted', row)
            return Decision(kind='transfer' if row and row[0] in ('plumbing', 'transfer-receipt')
                            else 'oneoff', status='posted')

    # 2. §3a known-merchant routing — posts regardless of threshold, no ask
    for pat, cat, typ, fv in rules.known_route:
        if re.search(pat, desc, re.I):
            return Decision('expense', 'posted', _clean(cat), _clean(typ), _clean(fv))

    # 3. §1 auto-post patterns
    for pat, cat, typ, fv in rules.auto_post:
        if re.search(pat, desc, re.I):
            ask = 'above-thresh-but-pattern' if amount >= thresh else None
            return Decision('expense', 'posted', _clean(cat), _clean(typ), _clean(fv), ask)

    # 4. §5 kind routing (data-driven defaults per parsed kind)
    row = rules.kind_route.get(tx.get('kind'))
    if row:
        return _apply_tier(tx, row, desc, amount, thresh, own)

    # 5./6. own-name guard, then unknown (§7.6: an unknown IN ≥ threshold is a
    # mystery credit → 'source unknown', holding kind oneoff)
    if own and tx.get('dir') == 'OUT':
        return Decision(kind='transfer', status='needs-confirm',
                        ask='own-name transfer — destination unknown')
    if tx.get('dir') == 'IN':
        return Decision(kind='oneoff', status='needs-confirm', ask='source unknown')
    return Decision(kind='expense', status='needs-confirm', ask='matches no rule')
