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
// These helpers are pure so they can be unit-tested without Electron.

/**
 * The two install shapes the desktop app can run against.
 *
 * - `'slot'` — a managed install under `$HERMES_HOME`. The stable launcher
 *   (`$HERMES_HOME/bin/hermes`) resolves `current.txt` and the env, so the
 *   backend is ALWAYS `hermes serve …` — no source sniffing or venv-path
 *   assembly needed for the child.
 * - `'checkout'` — a source tree (dev run or a git checkout install). The
 *   runtime may predate `serve`, so we still sniff `sourceDeclaresServe` and
 *   fall back to `dashboard --no-open` when needed.
 */
export type BackendInstallType = 'slot' | 'checkout'

/**
 * The resolved spawn configuration: whether to probe the runtime for `serve`
 * support before launching, or trust it unconditionally.
 */
export interface BackendSpawnConfig {
  /** When true, always use `serve` — skip the `sourceDeclaresServe` sniff. */
  alwaysServe: boolean
}

/**
 * Pure routing decision: given the install type, decide whether the backend
 * spawn needs the legacy `serve`-support sniff (checkout) or can always
 * launch `serve` directly (slot).
 *
 * Slot installs use the stable launcher which resolves `current.txt` and the
 * venv — the child is always `hermes serve …`, so sniffing is unnecessary and
 * the `dashboardFallbackArgs` rewrite never applies on this path. Checkout
 * installs may still run an older runtime, so keep the sniff (legacy checkouts
 * exist until sunset).
 */
export function routeBackendSpawn(installType: BackendInstallType): BackendSpawnConfig {
  return { alwaysServe: installType === 'slot' }
}

/**
 * Build the canonical headless backend argv (always `serve`).
 * @param {string} [profile] optional Hermes profile to pin via `--profile`.
 */
export function serveBackendArgs(profile?: string) {
  const head = profile ? ['--profile', profile] : []

  return [...head, 'serve', '--host', '127.0.0.1', '--port', '0']
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

  return [...args.slice(0, i), 'dashboard', '--no-open', ...args.slice(i + 1)]
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
