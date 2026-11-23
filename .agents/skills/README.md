# Project Skills Index

Project-specific skills tailored to the Hermes Desktop/Webapp (`apps/desktop/`) and authored across three subagent domains. Each skill lives in `.agents/skills/<name>/SKILL.md` with no external scripts or third-party installations.

## Code / Infra Subagent
- **[`hermes-webapp-build-test`](hermes-webapp-build-test/SKILL.md)** — Run `build-webapp.mjs --no-install` with pinned Node/pnpm runtimes, execute targeted Vitest suites, and enforce a verification-before-completion gate.
  - *When to use:* After any desktop/webapp code or theme change before declaring completion.
  - *Based on:* `https://skills.sh/obra/superpowers/verification-before-completion` (MIT), `https://skills.sh/antfu/skills/vitest` (inspired only), `https://skills.sh/antfu/skills/vite` (inspired only).
- **[`hermes-safe-backup-patch`](hermes-safe-backup-patch/SKILL.md)** — Create durable pre-edit backups with verified `SHA256SUMS` outside the checkout and commit atomically on a local feature branch.
  - *When to use:* Before editing source/test files and when preparing focused local commits.
  - *Based on:* Project-specific (Hermes Desktop/Webapp).

## Design Subagent
- **[`hermes-one-dark-theme`](hermes-one-dark-theme/SKILL.md)** — Apply or refine Atom One Dark Pro tokens (`one-dark.ts`) and unlayered Tailwind v4 renderer rules (`one-dark.css`) with runtime provenance and clean theme-switch teardown.
  - *When to use:* Adjusting One Dark Pro colors, markdown typography, selection highlights, or chrome borders.
  - *Based on:* `https://skills.sh/wshobson/agents/tailwind-design-system` (MIT).
- **[`hermes-wcag-contrast`](hermes-wcag-contrast/SKILL.md)** — Compute and verify WCAG 2.1 contrast ratios (>= 4.5:1) for UI surfaces and xterm ANSI `terminal`/`darkTerminal` palettes, plus focus/keyboard/reduced-motion checks.
  - *When to use:* Before committing any color, surface, or terminal palette update.
  - *Based on:* `https://skills.sh/vercel-labs/agent-skills/web-design-guidelines` (inspired only).
- **[`hermes-webapp-zoom-layout`](hermes-webapp-zoom-layout/SKILL.md)** — Diagnose and fix CSS `zoom` (`currentCSSZoom`) vs. `getBoundingClientRect()` drift in `PersistentTerminal` and `terminalSelectionAnchor`, stacking-context boundaries, and `TerminalRail` layout.
  - *When to use:* Debugging terminal overlay alignment, rail overlap/border height, or UI scale regressions.
  - *Based on:* Project-specific (Hermes Desktop/Webapp).

## Security / Performance Subagent
- **[`hermes-diff-security-audit`](hermes-diff-security-audit/SKILL.md)** — Audit staged/unstaged diffs before commit for leaked secrets, credentials, personal data, OWASP/XSS sinks in TSX/CSS, and compliance with `.ai/regras.md` and `.ai/teto.md`.
  - *When to use:* Before every commit or PR preparation on `apps/desktop/`.
  - *Based on:* `https://skills.sh/getsentry/skills/security-review` (Apache-2.0).
- **[`hermes-react-terminal-perf`](hermes-react-terminal-perf/SKILL.md)** — Maintain React/TSX + Vite + xterm.js rendering performance by avoiding unnecessary re-renders around WebGL hosts, coalescing observer bursts with `requestAnimationFrame`, and keeping bundles lean.
  - *When to use:* Editing React components or hooks around `PersistentTerminal`, `KeepAlivePanes`, or hot chat/sidebar paths.
  - *Based on:* `https://skills.sh/vercel-labs/agent-skills/vercel-react-best-practices` (inspired only).
