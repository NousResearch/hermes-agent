/**
 * Dev Chrome DevTools Protocol exposure for the desktop renderer.
 *
 * The renderer is a Chromium page, so `--remote-debugging-port` turns it into
 * something the repo's existing CDP tooling (`scripts/eval.mjs`,
 * `scripts/perf/lib/cdp.mjs`, the `diag-*` / `probe-*` family) can attach to
 * and read the live DOM from. Every one of those scripts already defaults to
 * 9222, so a dev-server run opens 9222 and they just work.
 *
 * If you are running a dev server you are already executing arbitrary local
 * JS — vite's module graph and every postinstall in node_modules — so a
 * loopback debugging port does not meaningfully widen that. `perf:serve`
 * already opens one unconditionally. What must never happen is a *packaged*
 * app exposing it, which is the one hard gate here.
 *
 *  - packaged build              → always closed, whatever the env says.
 *  - no HERMES_DESKTOP_DEV_SERVER → closed (an unpackaged `electron .` against
 *    dist/ is how the packaged app gets smoke tested; it should behave like
 *    the packaged app).
 *  - otherwise                    → open on 9222, or HERMES_DESKTOP_CDP_PORT.
 *
 * `HERMES_DESKTOP_CDP_PORT=off` (or `0` / `false`) opts out for anyone who
 * wants the port closed on a dev run.
 *
 * The port binds to loopback (Chromium's default) and the address is
 * deliberately not configurable: there is no reason to expose a renderer
 * debugger off-host, and offering the knob invites someone to try.
 */

/** Why the port is closed, for a one-line log the developer can act on. */
type ClosedReason = 'packaged' | 'no-dev-server' | 'opted-out' | 'invalid-port'

type DevCdpDecision = { port: number; reason: null } | { port: null; reason: ClosedReason }

type DevCdpInput = {
  env: Record<string, string | undefined>
  isPackaged: boolean
  devServer: string | undefined
}

/** What every script under scripts/ already reaches for. */
const DEFAULT_PORT = 9222

// Below 1024 needs privileges on most platforms; 65535 is the ceiling.
const MIN_PORT = 1024
const MAX_PORT = 65535

const OPT_OUT = new Set(['0', 'off', 'false', 'no'])

/**
 * Decide whether this run may expose a renderer debugging port, and on which
 * port. Pure: every input is passed in, so the gate is testable without an
 * Electron app or a real environment.
 */
function resolveDevCdpPort({ env, isPackaged, devServer }: DevCdpInput): DevCdpDecision {
  // Packaged wins over everything. Checked first so no combination of
  // environment variables can talk a shipped build into opening the port.
  if (isPackaged) {
    return { port: null, reason: 'packaged' }
  }

  // A dev server means a source-tree run (`npm run dev` / `hgui`).
  if (!devServer) {
    return { port: null, reason: 'no-dev-server' }
  }

  const requested = (env.HERMES_DESKTOP_CDP_PORT ?? '').trim()

  if (!requested) {
    return { port: DEFAULT_PORT, reason: null }
  }

  if (OPT_OUT.has(requested.toLowerCase())) {
    return { port: null, reason: 'opted-out' }
  }

  const port = Number(requested)

  if (!Number.isInteger(port) || port < MIN_PORT || port > MAX_PORT) {
    return { port: null, reason: 'invalid-port' }
  }

  return { port, reason: null }
}

/** One-line explanation for a closed port, or null when it opened. */
function describeDevCdpDecision(decision: DevCdpDecision): string | null {
  switch (decision.reason) {
    case null:
      return null

    case 'invalid-port':
      return `HERMES_DESKTOP_CDP_PORT is not a valid port (expected an integer ${MIN_PORT}-${MAX_PORT}, or "off"); renderer debugging is disabled.`

    case 'opted-out':
      return 'renderer debugging disabled by HERMES_DESKTOP_CDP_PORT.'

    // Packaged and dist-run builds are closed by design — the common case, not
    // worth a line of startup noise.
    case 'packaged':

    case 'no-dev-server':
      return null
  }
}

export { DEFAULT_PORT, describeDevCdpDecision, resolveDevCdpPort }
export type { DevCdpDecision }
