/** Plugins-tab copy kept beside the locale catalogs: the desktop-half and MCP server states a
 *  plugin row shows, and the fleet-wide update (header button, row caret, review dialog). */
export interface PluginsTabCopy {
  desktopHalfPending: string
  desktopHalfPendingTip: string
  desktopHalfRemote: string
  desktopHalfRemoteTip: string
  serverStates: {
    connected: string
    app_not_running: string
    hermes_not_connected: string
    endpoint_unavailable: string
    no_interactive_session: string
    version_too_old: string
    missing_app: string
    unsupported_gpu: string
    unknown: string
  }
  fleet: {
    updateEverywhere: (count: number) => string
    updateHereOnly: (profile: string) => string
    updateAll: (count: number) => string
    updateAllTip: (installs: number, plugins: number) => string
    titleOne: (name: string) => string
    titleAll: string
    summary: (installs: number, gateways: number) => string
    summaryDone: (installs: number) => string
    summaryFailed: (ok: number, failed: number) => string
    summaryDeclined: (ok: number, declined: number) => string
    thisProfile: string
    failed: string
    declined: string
    skippedOffline: string
    skippedSignIn: string
    skippedIdle: string
    skippedUnreadable: string
    confirm: (installs: number) => string
    updating: string
    done: string
    consentBody: (names: string, installs: number) => string
    finished: (installs: number) => string
  }
}
