/**
 * `hermes://` deep-link registration. Every launch rewrites the OS handler (on Windows,
 * HKCU\Software\Classes\<protocol>), so a test or dev run of an unpackaged checkout would
 * repoint the user's installed handler at that checkout. `HERMES_DESKTOP_SKIP_PROTOCOL_REGISTRATION=1`
 * (set by the E2E harnesses) leaves the OS registration untouched; deep links that arrive
 * through argv / second-instance still route normally.
 */

export interface ProtocolRegistrar {
  setAsDefaultProtocolClient: (protocol: string, path?: string, args?: string[]) => boolean
}

export interface DeepLinkProtocolOptions {
  app: ProtocolRegistrar
  protocol: string
  env: NodeJS.ProcessEnv
  /** `process.defaultApp`: running as `electron <dir>` rather than a packaged binary. */
  defaultApp: boolean
  argv: string[]
  execPath: string
  resolve: (entry: string) => string
  log: (line: string) => void
}

/**
 * `hermes://launch` opens and focuses the app and does nothing else: nothing it carries
 * (path, query) is read or reaches the renderer. Checked on the raw URL before any path
 * decoding, so a malformed tail cannot stop the launch. A window still booting is left to
 * its own reveal (focusing it early shows it before first paint); the launch itself opened it.
 * Returns true when the link was a launch link (handled), false for every other link.
 */
export function handleLaunchDeepLink(
  url: string,
  schemes: readonly string[],
  window: { ready: boolean; activate: () => void }
): boolean {
  let parsed: URL

  try {
    parsed = new URL(url)
  } catch {
    return false
  }

  if (!schemes.includes(parsed.protocol.replace(/:$/, '')) || parsed.hostname !== 'launch') {
    return false
  }

  if (window.ready) {
    window.activate()
  }

  return true
}

/** The first `<scheme>://` URL in a launch's argv, if any. */
export function extractDeepLink(argv: unknown, schemes: readonly string[]): string | null {
  if (!Array.isArray(argv)) {
    return null
  }

  return argv.find(a => typeof a === 'string' && schemes.some(s => a.startsWith(`${s}://`))) || null
}

export function registerDeepLinkProtocol(options: DeepLinkProtocolOptions): void {
  const { app, protocol, env, defaultApp, argv, execPath, resolve, log } = options

  if (env.HERMES_DESKTOP_SKIP_PROTOCOL_REGISTRATION === '1') {
    log(`[deeplink] ${protocol}:// registration skipped (HERMES_DESKTOP_SKIP_PROTOCOL_REGISTRATION)`)

    return
  }

  try {
    if (defaultApp && argv.length >= 2) {
      // Dev: register with the electron exec path + entry script so the OS can
      // relaunch us with the URL. argv[1] is usually "." when launched via
      // `electron .` from apps/desktop — resolve against cwd.
      app.setAsDefaultProtocolClient(protocol, execPath, [resolve(argv[1])])
    } else {
      app.setAsDefaultProtocolClient(protocol)
    }

    log(`[deeplink] registered ${protocol}:// handler`)
  } catch (err) {
    log(`[deeplink] protocol registration failed: ${(err as Error).message}`)
  }
}
