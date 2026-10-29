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
