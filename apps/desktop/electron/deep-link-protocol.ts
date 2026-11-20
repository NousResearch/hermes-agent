/**
 * `hermes://` deep links: OS registration and the gate every inbound link passes.
 *
 * Any web page can fire a `hermes://` URL, so a link only opens and focuses the app. Its
 * content never reaches the renderer: no prompt, install, navigation or prefill comes from a
 * URL. The two exceptions carry nothing an attacker can author into the app:
 *  - `close-preview`: the out-of-band exit from a preview pane that fullscreened itself.
 *  - `connections/done?op=<id>`: the browser leg of a connector sign-in came back; the op id
 *    only names an operation the app itself started, and the link's status is never trusted.
 *
 * Every launch rewrites the OS handler (on Windows, HKCU\Software\Classes\<protocol>), so a test
 * or dev run of an unpackaged checkout would repoint the user's installed handler at that
 * checkout. `HERMES_DESKTOP_SKIP_PROTOCOL_REGISTRATION=1` (set by the E2E harnesses) leaves the
 * OS registration untouched; links that arrive through argv / second-instance still pass the gate.
 */

export type DeepLinkVerdict =
  | { type: 'close-preview' }
  | { type: 'connection-done'; op: string }
  | { type: 'open' }
  | { type: 'ignore'; reason: string }

/** Classify an inbound URL. Anything that parses under an accepted scheme at least opens the app. */
export function classifyDeepLink(url: unknown, schemes: readonly string[]): DeepLinkVerdict {
  if (typeof url !== 'string' || !url) {
    return { type: 'ignore', reason: 'empty' }
  }

  let parsed: URL

  try {
    parsed = new URL(url)
  } catch {
    return { type: 'ignore', reason: 'malformed' }
  }

  if (!schemes.includes(parsed.protocol.replace(/:$/, ''))) {
    return { type: 'ignore', reason: `scheme ${parsed.protocol}` }
  }

  const kind = parsed.hostname
  const path = parsed.pathname.replace(/^\/+|\/+$/g, '')

  if (kind === 'close-preview') {
    return { type: 'close-preview' }
  }

  // The Windows Copilot key fires start and stop nearly together; only start summons the app,
  // or the stop would read as a second activation.
  if (kind === 'copilot-key' && path !== 'start') {
    return { type: 'ignore', reason: `copilot-key ${path}` }
  }

  const op = (parsed.searchParams.get('op') || '').trim()

  if (kind === 'connections' && path === 'done' && op) {
    return { type: 'connection-done', op }
  }

  return { type: 'open' }
}

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
