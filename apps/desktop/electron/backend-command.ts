// Backend subcommand routing for the desktop-managed Hermes process.
//
// The desktop app launches its own headless backend via `hermes serve` — it
// must NEVER depend on or launch the browser `dashboard`. But `serve` is a
// newer subcommand: a runtime that predates it (an older managed install the
// app hasn't updated yet, or an older `hermes` resolved from PATH) only knows
// `dashboard --no-open`. To avoid bricking those users mid-upgrade we detect
// whether the resolved runtime understands `serve` and, only when it does not,
// fall back to the legacy `dashboard --no-open` invocation. Both produce the
// exact same headless gateway; `serve` is just the decoupled name.
//
// `--ws-only` is an even newer flag: it bypasses the FastAPI/uvicorn dashboard
// entirely and runs a bare `websockets` server that calls `handle_ws` directly.
// When a runtime understands `serve` but NOT `--ws-only`, the flag is silently
// dropped by the runtime's argparse (it would fail "unrecognized arguments" on
// older versions). The `sourceDeclaresWsOnly` check gates this: only emit
// `--ws-only` when the runtime's source actually registers it.
//
// These helpers are pure so they can be unit-tested without Electron.

/**
 * Build the canonical headless backend argv (always `serve`).
 * @param {string} [profile] optional Hermes profile to pin via `--profile`.
 * @param {object} [opts] runtime capability flags.
 * @param {boolean} [opts.wsOnly=false] emit `--ws-only` (the slim WS server).
 *   OPT-IN: the WS-only server has no HTTP routes, and the desktop's
 *   `hermes:api` REST plane still speaks http — flipping this on by default
 *   would break every REST consumer (see PR #94245 review F1). Stays off
 *   until the REST consumers migrate to JSON-RPC (#94484 phase 3).
 */
export function serveBackendArgs(profile?: string, opts?: { wsOnly?: boolean }) {
  const head = profile ? ['--profile', profile] : []
  const wsOnly = opts?.wsOnly === true // default false — see docstring

  const tail = wsOnly ? ['--ws-only'] : []
  return [...head, 'serve', '--host', '127.0.0.1', '--port', '0', ...tail]
}

/**
 * Rewrite a resolved backend argv from `serve` to the legacy
 * `dashboard --no-open` form, preserving every other argument (incl. a leading
 * `-m hermes_cli.main` and any `--profile <name>`). Returns a copy; if there is
 * no `serve` token the argv is returned unchanged.
 */
export function dashboardFallbackArgs(args) {
  const i = args.indexOf('serve')

  if (i === -1) {
    return args.slice()
  }

  // `--ws-only` is serve-only; strip it when falling back to `dashboard`.
  const rest = args.slice(i + 1).filter(a => a !== '--ws-only')
  return [...args.slice(0, i), 'dashboard', '--no-open', ...rest]
}

/**
 * True when a runtime's `hermes_cli/subcommands/dashboard.py` source registers
 * the `serve` subcommand. Matches `add_parser("serve"` / `add_parser('serve'`
 * specifically so the substring "server" (e.g. "start_server", "web server")
 * never produces a false positive.
 */
export function sourceDeclaresServe(dashboardPySource) {
  return /add_parser\(\s*["']serve["']/.test(String(dashboardPySource || ''))
}

/**
 * True when a runtime's `hermes_cli/subcommands/dashboard.py` source registers
 * the `--ws-only` flag on the `serve` subcommand. Used to gate emitting the
 * flag: an older runtime that knows `serve` but not `--ws-only` would reject
 * it as an unrecognized argument.
 */
export function sourceDeclaresWsOnly(dashboardPySource) {
  return /["']--ws-only["']/.test(String(dashboardPySource || ''))
}
