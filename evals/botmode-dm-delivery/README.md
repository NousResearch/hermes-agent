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
Artifacts are retained in `/tmp/botmode-dm-recovery`; sandbox path is printed. The
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
