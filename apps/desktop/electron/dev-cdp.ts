/**
 * Dev-only Chrome DevTools Protocol exposure for the desktop renderer.
 *
 * The renderer is a Chromium page, so `--remote-debugging-port` turns it into
 * something the repo's existing CDP tooling (`scripts/eval.mjs`,
 * `scripts/perf/lib/cdp.mjs`, the `diag-*` / `probe-*` family) can attach to
 * and read the live DOM from. That is genuinely useful while iterating on the
 * UI — and it is also arbitrary code execution against whatever the running
 * app can reach, so it stays off unless three independent conditions all hold:
 *
 *  1. The build is NOT packaged. A shipped app never opens this port, whatever
 *     the environment says.
 *  2. A dev server is wired up (`HERMES_DESKTOP_DEV_SERVER`). That is the
 *     signature of `npm run dev` / `hgui`; a packaged or `dist`-loading run
 *     has no dev server and does not qualify.
 *  3. The developer opted in explicitly with a valid `HERMES_DESKTOP_CDP_PORT`.
 *     Absent that, a plain `npm run dev` behaves exactly as it does today —
 *     nobody gets a debugging port they did not ask for.
 *
 * The port binds to loopback (Chromium's default) and the address is
 * deliberately not configurable: there is no reason to expose a renderer
 * debugger off-host, and offering the knob invites someone to try.
 */

/** Why the port is closed, for a one-line log the developer can act on. */
type ClosedReason = 'packaged' | 'no-dev-server' | 'not-requested' | 'invalid-port'

type DevCdpDecision = { port: number; reason: null } | { port: null; reason: ClosedReason }

type DevCdpInput = {
  env: Record<string, string | undefined>
  isPackaged: boolean
  devServer: string | undefined
}

// Below 1024 needs privileges; the ephemeral range is fair game but the
// well-known CDP port (9222) is what every script in scripts/ defaults to.
const MIN_PORT = 1024
const MAX_PORT = 65535

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

  const requested = (env.HERMES_DESKTOP_CDP_PORT ?? '').trim()

  if (!requested) {
    return { port: null, reason: 'not-requested' }
  }

  // A dev server means a source-tree run (`npm run dev` / `hgui`). An
  // unpackaged `electron .` against dist/ is how the packaged app is smoke
  // tested, and it should behave like the packaged app here.
  if (!devServer) {
    return { port: null, reason: 'no-dev-server' }
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

    case 'packaged':
      return 'HERMES_DESKTOP_CDP_PORT ignored: renderer debugging is dev-only and this is a packaged build.'

    case 'no-dev-server':
      return 'HERMES_DESKTOP_CDP_PORT ignored: no HERMES_DESKTOP_DEV_SERVER, so this is not a dev-server run.'

    case 'invalid-port':
      return `HERMES_DESKTOP_CDP_PORT ignored: not a valid port (expected an integer ${MIN_PORT}-${MAX_PORT}).`

    case 'not-requested':
      return null
  }
}

export { describeDevCdpDecision, resolveDevCdpPort }
export type { DevCdpDecision }
