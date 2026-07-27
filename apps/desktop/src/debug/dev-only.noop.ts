// Production stand-in for `dev-only.ts`. `vite.config.ts` aliases the dev
// entry to this module for any build that isn't the dev server, so the
// diagnostics graph — and bippy with it — never reaches a shipped renderer.
//
// Keep this file free of imports. It exists precisely so the production
// bundle contains nothing from `debug/`.

export {}
