/**
 * Deep-link protocol registration (plus the package.json desktop identity the
 * Linux association depends on).
 *
 * Extracted from main.ts (which cannot be imported in tests — it boots
 * Electron) so the contract that matters is provable in isolation: the
 * boolean app.setAsDefaultProtocolClient actually reports, never an assumed
 * success. Linux OS delivery depends on the installed `hermes.desktop` entry
 * (Exec `%u`, MimeType x-scheme-handler/hermes) written by the `hermes desktop`
 * launcher — see hermes_cli/linux_desktop_entry.py. The window's desktop-file
 * identity is pinned through the `desktopName` field in package.json (Electron
 * 40 has no app.setDesktopName).
 */

export interface ProtocolRegistrarApp {
  setAsDefaultProtocolClient(scheme: string, execPath?: string, args?: string[]): boolean
}

export interface ProtocolRegistrationContext {
  /** Scheme to claim: hermes in packaged builds, hermes-dev under DEV_SERVER. */
  protocol: string
  /** True under `electron .` (unpackaged) — the OS must relaunch via execPath + entry. */
  defaultApp: boolean
  /** process.argv; argv[1] is the entry script in the defaultApp case. */
  argv: string[]
  /** process.execPath. */
  execPath: string
  /** path.resolve, injectable for tests. */
  resolve: (p: string) => string
}

/**
 * Claim `protocol` as the OS handler and return whether the registration
 * actually took (the API's boolean, never assumed). Callers log the outcome.
 */
export function registerDeepLinkProtocol(
  app: ProtocolRegistrarApp,
  { protocol, defaultApp, argv, execPath, resolve }: ProtocolRegistrationContext
): boolean {
  if (defaultApp && argv.length >= 2) {
    // Dev: register with the electron exec path + entry script so the OS can
    // relaunch us with the URL. argv[1] is usually "." when launched via
    // `electron .` from apps/desktop — resolve against cwd.
    return app.setAsDefaultProtocolClient(protocol, execPath, [resolve(argv[1])])
  }

  return app.setAsDefaultProtocolClient(protocol)
}
