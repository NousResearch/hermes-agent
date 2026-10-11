# Native Bot Mode delivery probe

Real Electron, production Python backend and tool execution, disposable HOME/HERMES_HOME,
loopback scripted inference (no paid model). Linux seat fixture; run from repository root:

```sh
npm ci --no-audit --no-fund
cp evals/botmode-dm-delivery/probe-dm-delivery.spec.ts apps/desktop/e2e/
git apply evals/botmode-dm-delivery/mock-trigger.patch
(cd apps/desktop && npm run build)
(cd apps/desktop && DISPLAY=:0 XAUTHORITY=/run/user/1000/xauth_cnpsqU \
  XDG_RUNTIME_DIR=/run/user/1000 VIRTUAL_ENV="$VIRTUAL_ENV" \
  HERMES_DESKTOP_CDP_PORT=off npx playwright test e2e/probe-dm-delivery.spec.ts --reporter=list)
git apply -R evals/botmode-dm-delivery/mock-trigger.patch
rm apps/desktop/e2e/probe-dm-delivery.spec.ts
```

Use the current seat's actual Xauthority path and an existing runtime venv.
Artifacts default to `/tmp/botmode-dm-review/native` (override with `BOT_DM_EVIDENCE`); sandbox path is printed. The
fixture's generated hermes shim pins every child to this checkout, not an installed launcher.

## Verified results

Base `cf35e7351e770`: nested beta quiet CLI executes real `message_agent` to the named
alpha Desktop owner. Admission, execution and reply succeed, and the quiet sender
receives its completion. A reload then shows the attributed incoming message. The
live pre-reload view showed the reply but omitted the incoming row in this fixture;
that renderer-refresh behavior is not fixed by this cron change.

Base CLI-owner case: separate cron producer returns exact `SESSION_NOT_OWNED` refusal.
Release owner, run scheduler tick, open Beta in Desktop: no cron output, assertion red.
Fixed case: queued receipt; release owner; real synchronous scheduler tick drains;
Beta Desktop renders `CLI_OWNER_CRON_SENTINEL` and its reply once. Final two-case
native run: **2 passed (1.2m)**. Supported-owner nested case also passed independently
on base (**1 passed (40.0s)**).

This does not establish native Windows parity, general retry after unowned CLI
failure, or correction of the issue #105460 `--in ~` premise. CLI title resolution
is profile-DB-based, not workspace selection; the named live-owner route is positive.

The queue is deliberately at-most-once after claim. A crash before spawning but
after claiming remains inspectable as claimed; it is not retried automatically.

## Independent review follow-up

The probe now admits the named Beta destination from the default scheduler, then
changes the ticker's `HOME` to a different existing directory before drain.
Published head `c91dfcbfe810c` fails with `Profile 'beta' does not exist`, leaving
zero sentinel inputs in Beta. The follow-up pins the admitted home and ID; Beta
renders one input and reply. Set `BOT_DM_CORRUPT=1` to add one malformed JSON
record alongside the valid admission: before the follow-up the real tick raises
`JSONDecodeError`; afterward the damaged record stays on disk while Beta delivers.

Fresh built native run with both root change and corruption: **2 passed (1.4m)**,
including the existing nested `message_agent` control. Receipts/screenshots:
`/tmp/botmode-dm-review/{native-red2,corrupt-red,final-native}` and matching `.log`
files. The first follow-up run also exposed a fixture mistake (the changed HOME
was not created, so `--in ~` correctly refused); that failed receipt is retained
in `native-green.log`, and both source legs were rerun with an existing HOME.
Prior `/tmp/botmode-dm-recovery*` evidence remains untouched.

## Per-record delivery exception isolation

Retired with the deferred `hermes chat` pending lane (`cron/bot_chat_delivery.py`): cron
output now reaches Bot Chat through the gateway authority door (`gateway/session_bot.py`),
which records a per-target receipt on the job and creates the Bot Chat when missing, so a
per-record file-scan exception has no counterpart. The regression case lives in
`tests/cron/test_cron_bot_chat_create_if_missing.py` and
`tests/gateway/test_cron_canonical_receipt_regression.py`.

## Ordinary custom-root fallback (#104066 / #104055)

`probe-cron-root.spec.ts` adds the never-deferred sibling: copy it to
`apps/desktop/e2e/` and run with the same native fixture (no mock-trigger patch
needed for this case). It keeps default's real Desktop Bot Chat lease, submits
ordinary cron output to unowned Alpha from a separate Python producer under a
custom Hermes root, and holds the real quiet CLI child at loopback inference.
The child shim PID must match Alpha's real CLI lease; default's lease is unchanged.
After release, Alpha has exactly one input and Desktop renders the output.
The same case removes unused Beta and verifies delivery neither recreates Beta nor
creates a second `.hermes` root under HOME.

Both `origin/main`'s scheduler and pre-follow-up `c827ae179d67c` fail with
`Profile 'alpha' does not exist` before any recipient turn. Fixed native run:
**1 passed (46.3s)**. Two invariant tests exercise the actual CLI startup resolver
across named/default/own destinations with a changed active profile, and refusal
when the destination is missing initially or disappears during discovery:
**5 failed before, 5 passed after**. The old env-clearing test is replaced by these
behavior checks rather than retaining the broken expectation.

Evidence: `/tmp/botmode-cron-root/{before2,origin-main,after}.log`,
`after/{owners.json,children.log,rows.json,result.json,missing.json,ordinary-recipient.png}`.
The first fixture attempt (`before.log`) used the wrong default row label; the
actual Desktop label is Hermes. No production failure is claimed for that attempt.
Full cron directory: **1346 passed, 1 skipped across 116 files**; sibling mailbox,
DM, gateway consumer and profile tests: **124 passed, 3 skipped across 4 files**.
Credit @fangliquanflq's #104066 for the root-boundary diagnosis and anchoring fix;
this combined branch reuses its already-resolved destination instead of repeating
name resolution. No retry or receipt semantics change.
