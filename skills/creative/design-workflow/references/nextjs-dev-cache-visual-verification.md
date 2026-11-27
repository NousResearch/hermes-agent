# Next.js Dev Cache During Visual Verification

Use this when a Next.js app suddenly renders as raw HTML, loses Tailwind styling, or reports missing `.next/server/chunks/*.js` while doing UI/design work.

## Symptom

- Page route returns 200 but CSS/JS assets under `/_next/static/...` return 404.
- Browser vision says the page is unstyled or looks like default HTML.
- Dev server logs show errors like `MODULE_NOT_FOUND`, `.next/server/app/.../page.js`, or missing `.next/server/chunks/<id>.js`.
- This often appears after running `next build` while `next dev` is already active.

## Interpretation

Treat this as a stale Next dev/build cache before critiquing the design. Do not conclude that the UI implementation is visually broken until fresh assets are served.

## Safe Recovery Pattern

1. Check whether a dev server is still listening:
   ```bash
   lsof -nP -iTCP:<port> -sTCP:LISTEN || true
   ```
2. Prefer starting on a fresh port if you only need visual verification:
   ```bash
   npm run dev -- --port <new-port>
   ```
3. If cleanup is necessary, split destructive commands into separate approval-friendly steps instead of chaining `kill && rm -rf .next && npm run dev`:
   ```bash
   lsof -tiTCP:<port> -sTCP:LISTEN
   kill <pid>
   rm -rf .next
   npm run dev -- --port <port>
   ```
4. Re-check one HTML route and the CSS asset response before browser vision:
   ```bash
   curl -sS -I http://localhost:<port> | sed -n '1,8p'
   ```

## Pitfalls

- A background process completion notice may be from an older `next dev` process, not the currently active server. Re-check the live port and route status.
- Do not retry an identical destructive command if the tool returns `BLOCKED: Command timed out. Do NOT retry this command.` Split the operation or ask the user for the next safe step.
- Feishu approval buttons can fail or expire; if approval is required, ask the user to use the text fallback (`/approve` or `/approve always`) rather than clicking an old card button.
