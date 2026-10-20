"""petasos.pipeline — fetch → parse → classify → write → gates → digest.

This file grows in two TDD cycles: the PURE helpers first (narrated-time
ordering R1.8, per-account chain verification R1.3/DG-1), then the
orchestration `main()` (write order DG-6, --dry INV-1, dedupe R1.9/DG-7,
unparsed-email surfacing DG-2, fetch-skip C8, gates subprocess C6, digest
DG-15).

Chain rule (R1.3): bal(n) = bal(n−1) ∓ amount(n) per account, checked
BEFORE writing anything — a contradicting fetched sequence appends nothing
and reports the break. An offsetting ± pair sharing one balance is ONE
batched alert, not an unalerted transaction.

Orchestration (C8/DG-6): resolve registry+rules LOUD and BEFORE any fetch
(INV-2) → fetch (skip-notice on 401/403/404/URLError, touch nothing, exit 0;
read-timeout → one [fatal] line, exit 1)
→ dedupe (seen_threads + subject keywords + old_ref guard, DG-7 full-id with
v1-era 8-char prefix compat) → parse (DG-2: unparsed surface as asks) →
chain check → narrated-time order (R1.8) → classify/build rows → --dry
prints the plan and writes NOTHING (INV-1) → LIVE under a lockfile (S-5):
append → verify re-read (R1.7) → balances sync → state.json → gates
subprocess (C6; nonzero is fatal) → digest (DG-15 strings).

Exit codes: 0 ok/skip · 1 gate failure or chain break · 2 config errors ·
3 lock held by another run.
"""
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import datetime

from .balances import sync_from_alerts
from .classify import classify, is_pending_hold
from .digest import render_digest
from .gates import offsetting_pair
from .knob import load_knobs
from .ledger import append_rows, build_row, next_id, verify_append
from .parse import normalize_date

_TIME = re.compile(r'(\d{2}:\d{2}(?::\d{2})?)')


def narrated_time(tx) -> str:
    """Sortable 'YYYY-MM-DD HH:MM:SS' from the narration itself (R1.8):
    the datetime field when the bank narrates one, else the date at
    midnight. Receipt time is deliberately NOT used — same-day rows sit in
    true narrated-time order."""
    dt = tx.get('datetime')
    if dt:
        date = normalize_date(str(dt).split(' ')[0])
        m = _TIME.search(str(dt))
        if m:
            clock = m.group(1)
            if clock.count(':') == 1:
                clock += ':00'
        else:
            clock = '00:00:00'
        return f'{date} {clock}'
    date = tx.get('date')
    if date:
        return f'{normalize_date(str(date))} 00:00:00'
    return ''


def order_txs(txs) -> list:
    """Stable sort by narrated time — out-of-order arrivals become
    narrated-time order (R1.8)."""
    return sorted(txs, key=narrated_time)


def check_chain(txs) -> list:
    """Per-account bal_after consistency over the fetched batch (DG-1).
    Returns a list of human-readable breaks; empty = clean."""
    by_acct = {}
    for t in txs:
        acct = t.get('acct')
        bal = t.get('bal_after')
        if acct is None or bal is None or t.get('amount') is None:
            continue
        if t.get('dir') not in ('IN', 'OUT'):
            continue
        by_acct.setdefault(str(acct), []).append(t)
    breaks = []
    for acct, group in by_acct.items():
        prev = None
        for cur in order_txs(group):
            if prev is None:
                prev = cur
                continue
            if offsetting_pair(prev, cur):
                # offsetting ± pair sharing one balance = ONE batched alert
                # (single shared definition — gates._chain_breaks agrees)
                continue
            expected = (prev['bal_after'] - cur['amount']
                        if cur['dir'] == 'OUT' else prev['bal_after'] + cur['amount'])
            if cur['bal_after'] != expected:
                breaks.append(
                    f"chain break acct {acct}: bal {cur['bal_after']} after "
                    f"{cur['amount']} {cur['dir']} from {prev['bal_after']} "
                    f"(expected {expected}) at {narrated_time(cur)}")
            prev = cur
    return breaks


# ---------------- orchestration ----------------

LOCK_NAME = 'petasos.lock'
SKIP_NOTICE = ('⚠️ AgentMail inbox not visible to this credential (deleted or key '
               'rescoped). Skipping mail fetch — ledger untouched, state.json '
               'untouched. Catch-up resumes automatically once the inbox exists '
               'again.')
FETCH_TIMEOUT_FATAL = ('[fatal] mail fetch timed out after 30s — ledger '
                       'untouched, state.json untouched; catch-up resumes '
                       'automatically next run')


def _default_transport(path, api_base, api_key):
    req = urllib.request.Request(
        api_base + path, headers={'Authorization': f'Bearer {api_key}'})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def _strip_body(html_text):
    import html as _h
    text = re.sub(r'<[^>]+>', ' ', html_text or '')
    return re.sub(r'\s+', ' ', _h.unescape(text)).strip()


def _read_state(state_path):
    if not os.path.exists(state_path):
        return {}
    with open(state_path, encoding='utf-8') as f:
        try:
            state = json.load(f)
        except json.JSONDecodeError:
            return {}
    return state if isinstance(state, dict) else {}   # read-tolerant (C2)


def _read_rows(ledger_file):
    rows = []
    if not os.path.exists(ledger_file):
        return rows
    with open(ledger_file, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def resolve_views_script(knobs, bundled_root=None):
    """C6 — <ledger_dir>/gen_views.py (vault-local) shadows the skill's
    scripts/gen_views.py; None if neither exists (caller fails loudly)."""
    vault_copy = os.path.join(knobs.ledger_dir, 'gen_views.py')
    if os.path.exists(vault_copy):
        return vault_copy
    root = bundled_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bundled = os.path.join(root, 'scripts', 'gen_views.py')
    return bundled if os.path.exists(bundled) else None


def main(argv=None, environ=None, transport=None, now=None) -> int:
    argv = list(argv or [])
    dry = '--dry' in argv
    env = environ if environ is not None else os.environ
    now = now or (lambda: datetime.now())

    # 0. resolve config LOUD, before ANY fetch (INV-2)
    from .registry import RegistryError, resolve_registry, subject_keywords
    from .rules import RulesError, load_rules
    try:
        knobs = load_knobs(env)
        registry = resolve_registry(knobs)
        rules = load_rules(knobs)
    except (RegistryError, RulesError) as e:
        print(str(e), file=sys.stderr)
        return 2

    ledger_file = os.path.join(knobs.ledger_dir, 'transactions.jsonl')
    state_path = os.path.join(knobs.ledger_dir, 'state.json')
    balances_file = os.path.join(knobs.ledger_dir, 'balances.json')

    fetch = transport or (lambda path: _default_transport(
        path, knobs.api_base, env.get('AGENTMAIL_API_KEY')))

    # 1. fetch recent bank threads
    if not env.get('AGENTMAIL_API_KEY') or not knobs.inbox:
        missing = ('AGENTMAIL_API_KEY' if not env.get('AGENTMAIL_API_KEY')
                   else 'PETASOS_INBOX')
        print(f'{missing} is not set — cannot fetch bank mail.', file=sys.stderr)
        return 2
    try:
        d = fetch(f'/inboxes/{knobs.inbox}/threads?limit=50')
    except (TimeoutError, socket.timeout):          # F-4: read-timeout is not
        print(FETCH_TIMEOUT_FATAL, file=sys.stderr)  # a URLError — fail clean
        return 1
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        if getattr(e, 'code', None) in (401, 403, 404, None) or \
                isinstance(e, urllib.error.URLError):
            print(SKIP_NOTICE)
            return 0
        raise
    threads = sorted(d.get('threads', []),
                     key=lambda t: t.get('timestamp') or t.get('created_at') or '')

    # 2. dedupe (R1.9): seen_threads + subject keywords + old_ref guard (DG-7)
    state = _read_state(state_path)
    seen = list(state.get('seen_threads', []))
    known_refs = {r.get('old_ref') for r in _read_rows(ledger_file)}
    keywords = tuple(subject_keywords(registry))
    fresh = [t for t in threads
             if t.get('thread_id') not in seen
             and any(k in (t.get('subject') or '') for k in keywords)
             and f"mail:{t['thread_id']}" not in known_refs
             and f"mail:{t['thread_id'][:8]}" not in known_refs]
    print(f'[fetch] {len(threads)} threads, {len(fresh)} fresh bank emails')

    # 3. parse; DG-2 — subject-keyword threads that fail to parse surface
    new_txs, unparsed = [], []
    for t in fresh:
        tid = t['thread_id']
        try:
            tj = fetch(f'/inboxes/{knobs.inbox}/threads/{tid}')
        except (TimeoutError, socket.timeout):          # F-4, detail fetch
            print(FETCH_TIMEOUT_FATAL, file=sys.stderr)
            return 1
        msg0 = (tj.get('messages') or [{}])[0]
        body = _strip_body(msg0.get('extracted_html') or msg0.get('html') or '')
        subj = msg0.get('subject') or t.get('subject') or ''
        rcv = msg0.get('timestamp') or msg0.get('created_at') or ''
        from .parse import parse_any
        tx = {'recv_utc': rcv[:16], 'src_thread': tid}
        tx.update(parse_any(registry, subj, body))
        if 'kind' in tx:
            new_txs.append(tx)
        else:
            unparsed.append((tid, subj))
        seen.append(tid)                    # asked once; never re-asked blindly

    # 4. chain verification BEFORE writing (DG-1/R1.3)
    breaks = check_chain(new_txs)
    if breaks:
        for b in breaks:
            print(f'[fatal] {b} — appends NOTHING; fix the parse or the emails')
        return 1

    # 5. classify → rows, narrated-time order (R1.8), holds recorded not rowed (S-3)
    existing_rows = _read_rows(ledger_file)
    rows, pending_ids = [], []
    holds = [t for t in new_txs if is_pending_hold(t)]
    if holds:
        print(f'[hold] {len(holds)} pending-hold alert(s) recorded to balances '
              'only — no ledger row (S-3)')
    for tx in order_txs([t for t in new_txs if not is_pending_hold(t)]):
        decision = classify(tx, rules, knobs)
        row = build_row(tx, decision, knobs)
        row['id'] = next_id(existing_rows + rows, row['date'], pending_ids)
        pending_ids.append(row['id'])
        rows.append(row)

    ts = now().strftime('%Y-%m-%d %H:%M')
    digest_text = render_digest(rows, unparsed, ts)

    if dry:
        print(f'[dry] would append {len(rows)} rows:')
        for r in rows:
            print(f"  {r['id']} {r['dir']:>3} {r['amount']:>12,} "
                  f"{r['status']:<13} {r.get('category') or '':<12} {r['desc'][:50]}")
        n_asks = len([r for r in rows if r['status'] != 'posted' or r.get('ask')]) \
            + len(unparsed)
        print(f'[dry] digest would ask: {n_asks}')
        if unparsed:
            print('[dry] UNPARSED threads (probe hint):')
            for tid, subj in unparsed:
                print(f'    mail:{tid} {subj[:60]!r}')
        print('[dry] digest preview:')
        print(digest_text, end='')
        return 0                                      # INV-1: nothing written

    # 6. LIVE write phase under the run lock (DG-6/S-5)
    os.makedirs(knobs.ledger_dir, exist_ok=True)
    lock_path = os.path.join(knobs.ledger_dir, LOCK_NAME)
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f'[fatal] another petasos run holds {lock_path} — exiting loudly '
              '(no overlap on one ledger)')
        return 3
    try:
        os.write(lock_fd, f'{os.getpid()}\n'.encode())
        os.close(lock_fd)
        try:
            prior = len(existing_rows)
            if rows:
                append_rows(ledger_file, rows)            # R1.1 append-only
                verify_append(ledger_file, prior, rows)   # R1.7 re-read + count
            if rows or holds:
                with open(balances_file, encoding='utf-8') as f:
                    balances = json.load(f)
                sync_from_alerts(balances, new_txs, knobs)
                with open(balances_file, 'w', encoding='utf-8') as f:
                    json.dump(balances, f, indent=1, ensure_ascii=False)
            state['schema'] = state.get('schema', 1)
            state['seen_threads'] = list(dict.fromkeys(seen))   # full ids, deduped
            state['last_run'] = ts
            with open(state_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=1)
        except Exception as e:                # crash between steps: ledger is
            print(f'[fatal] write phase crashed after append — the ledger is '
                  f'authoritative: {e}')       # authoritative; balances self-heal
            return 1
    finally:
        if os.path.exists(lock_path):
            os.unlink(lock_path)

    # 7. gates subprocess (C6) — failure is fatal (R3.2)
    script = resolve_views_script(knobs)
    if not script:
        print('[fatal] no gen_views.py resolvable (vault-local or skill-bundled)',
              file=sys.stderr)
        return 1
    gen = subprocess.run([sys.executable, script], capture_output=True,
                         text=True, env=env)
    print(gen.stdout, end='')
    if gen.returncode != 0:
        print(gen.stderr, end='')
        print('[fatal] gen_views gates failed after append — exiting nonzero')
        return 1

    # 8. digest file for cron delivery (live only; DG-5 keeps dry clean)
    scratch = os.path.join(tempfile.gettempdir(), 'petasos-pipe')
    os.makedirs(scratch, exist_ok=True)
    digest_path = os.path.join(scratch, 'digest.txt')
    with open(digest_path, 'w', encoding='utf-8') as f:
        f.write(digest_text)
    print(digest_text, end='')
    print(f'[done] appended {len(rows)} rows, digest → {digest_path}')
    return 0
