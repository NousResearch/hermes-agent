# Desktop core suite (required CI lane)

A small, deterministic Electron suite that guards three issue classes end to end:

- **C2 transcript integrity** — `transcript-integrity.spec.ts`: one real app +
  one real `hermes serve`, only the LLM faked (`provider.ts`, scripted per turn
  by a unique marker, every streamed chunk recorded). After every transition —
  stream, tool-call turn, reasoning turn, steer, queued follow-up, session
  switch mid-stream, warm resume, reload, WebSocket drop + reconnect mid-stream,
  a non-default profile's chat, a forced second socket to the same backend,
  final reload — `oracle.ts` asserts:
  - every persisted user/assistant message is rendered exactly once, in order,
    and nothing unpersisted is rendered;
  - no marker is ever rendered twice, even transiently (in-page
    MutationObserver sampler — the #120005 garble healed on its own in the
    final DOM, so a final-state check alone misses it);
  - backend stream integrity: each turn's concatenated `message.delta` /
    `reasoning.delta` equals what the provider streamed, and
    `message.complete` equals the final completion.
  - one live socket per backend process (#120006).
    `switch-back-race.spec.ts` forces both orders of "reply completes" vs "the
    switch-back REST hydrate resolves" with gates (no sleeps) under the same
    oracle.
    `onboarding-first-chat.spec.ts` starts from a fresh home with no provider:
    the real onboarding (custom endpoint → the fake provider's URL), then the
    first chat, a second turn and a reload under the same oracle, plus
    persisted config == entered endpoint and one live socket afterwards.
- **C20 interactive prompts** — `interactive-prompts.spec.ts`: clarify (one
  card; the clicked choice is exactly what the model receives), approval
  Run once (the command runs only after the click) and Deny (never runs; the
  turn still completes), approvals in manual mode.
- **C5 boot / process lifecycle** — `boot-lifecycle.spec.ts`: interactive
  composer + first turn, exactly one backend; `kill -9` backend → exactly one
  supervised respawn and a working turn; quit mid-turn with a running tool
  child → zero sandbox processes (/proc census on the sandbox `HERMES_HOME`,
  so orphans reparented to init are counted); relaunch the same home 3× → one
  backend per boot, zero after each quit, transcript cold-hydrates once.

Rules the suite keeps (why the old lane was disabled): no fixed sleeps as
synchronisation (every wait is on a frame, pid, DOM state or persisted row
with a deadline), no shared mock state between scenarios (replies are keyed
by the turn's own marker), no visual baselines, `retries: 0`, one worker,
sandboxed `HOME`/`HERMES_HOME`/user-data per test, all `HERMES_*` and
credential env stripped from the spawned app.

Run locally (Linux, after `npm run build` in `apps/desktop`):

```sh
cd apps/desktop
xvfb-run -a npx playwright test -c e2e/core/playwright.config.ts
```

`HERMES_E2E_CORE_ROOT` picks the sandbox parent dir (default: OS tmpdir);
`HERMES_E2E_CORE_KEEP=1` keeps sandboxes for post-mortem.
