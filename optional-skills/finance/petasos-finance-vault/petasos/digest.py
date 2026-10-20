"""petasos.digest — ops digest rendering (R4.5, DG-15).

The production cron wrapper greps these exact strings — keep verbatim from
v1 (pipeline.py:216–223 at 119eaab):
  'Petasos pipeline <ts> — N new transactions'   ([0-9]+ new transactions)
  '⚠️ NEEDS CONFIRMATION'                        (NEEDS CONFIRMATION)
and the gate line 'ALL GATES PASS' comes from the gen_views subprocess
stdout, which the pipeline passes through.

DG-2 addition: subject-keyword threads whose body failed to parse surface
as UNPARSED ask lines with a --probe hint — never a fabricated row (R1.2).
"""


def render_digest(rows, unparsed, ts) -> str:
    """rows: this batch's ledger rows; unparsed: [(thread_id, subject)] that
    matched a bank subject but produced no parse (DG-2)."""
    posted = [r for r in rows if r.get('status') == 'posted']
    asks = [r for r in rows if r.get('status') != 'posted' or r.get('ask')]
    lines = [
        f'Petasos pipeline {ts} — {len(rows)} new transactions',
        f'Auto-posted: {len(posted)} | Needs confirmation: {len(asks) + len(unparsed)}',
        '',
    ]
    for r in posted:
        lines.append(f"✓ {r['date']} {r['amount']:>12,} {r.get('category', '-')}: "
                     f"{r['desc'][:60]}")
    if asks or unparsed:
        lines.append('')
        lines.append('⚠️ NEEDS CONFIRMATION:')
    for r in asks:
        lines.append(f"? {r['date']} {'IN' if r['dir'] == 'in' else 'OUT'} "
                     f"{r['amount']:>12,} — {r.get('ask') or r['status']}")
        lines.append(f"    {r['desc'][:70]}")
    for tid, subject in unparsed:
        lines.append(f'? UNPARSED mail:{tid} — subject matched a bank keyword '
                     f'({subject[:60]!r}) but no format captured')
        lines.append('    probe it: python3 scripts/bank_adapters.py --probe '
                     '<email.txt> — a MISS line is the dead regex')
    return '\n'.join(lines) + '\n'
