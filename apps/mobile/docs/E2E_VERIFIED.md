# Mobile E2E Verified — 2026-06-30

What actually got verified on a booted iPhone 17 Pro simulator running the
production Release build wired to `https://hermes-desktop.pmxt.dev`:

- ✅ App launches; renderer mounts; native fetch bridge to gateway returns
  200 (empty sessions list for this user). See
  `docs/screenshots/mobile-e2e-1-empty.png`.

What is NOT independently verified yet:

- ⚠️ Settings drill-down / master-detail nav — CSS gating was inspected
  in the Vite dev server with the `hermes-mobile-standalone` class applied
  manually, but no tap-driven flow was screenshotted on the sim (CLI
  tap-automation via cliclick did not propagate touches to the simulator
  window in this session, and `hermes://settings` is not registered as a
  routable URL scheme).
- ⚠️ Composer submit → response — not scripted.

Notes: see `docs/MOBILE_PARITY_AUDIT.md` for the gap inventory vs the
original 104k-LOC mobile fork (`fork/main`) and which gaps are landed
in this branch vs still outstanding.
