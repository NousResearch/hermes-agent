import type { PluginsTabCopy } from './types_plugins_tab'

export const enPluginsTab: PluginsTabCopy = {
  desktopHalfPending: 'copying…',
  desktopHalfPendingTip:
    'This package ships a desktop half that has not been copied into the app yet. Use Rescan, or restart the app.',
  desktopHalfRemote: 'unavailable (remote backend)',
  desktopHalfRemoteTip:
    "This package's desktop half is on the remote backend's disk, which this app cannot read. To use it here, run Install from Git with the package's repo URL and the Desktop target checked — that clones the desktop half onto this machine.",
  serverStates: {
    connected: 'connected',
    app_not_running: 'app not running',
    hermes_not_connected: 'MCP connection missing',
    endpoint_unavailable: 'endpoint unavailable',
    no_interactive_session: 'no interactive session',
    version_too_old: 'version too old',
    missing_app: 'app missing',
    unsupported_gpu: 'GPU not supported',
    unknown: 'status unknown'
  },
  fleet: {
    updateEverywhere: (count: number) => `Update everywhere · ${count}`,
    updateHereOnly: (profile: string) => `Only ${profile}`,
    updateAll: (count: number) => `Update all · ${count}`,
    updateAllTip: (installs: number, plugins: number) =>
      `${plugins === 1 ? '1 plugin' : `${plugins} plugins`} out of date in ${installs === 1 ? '1 profile' : `${installs} profiles`} across your gateways`,
    titleOne: (name: string) => `Update ${name} everywhere`,
    titleAll: 'Update plugins everywhere',
    summary: (installs: number, gateways: number) =>
      `${installs === 1 ? '1 profile' : `${installs} profiles`} on ${gateways === 1 ? '1 gateway' : `${gateways} gateways`} run an older version.`,
    summaryDone: (installs: number) =>
      `Updated ${installs === 1 ? '1 profile' : `${installs} profiles`}. Restart those gateways to load the new version.`,
    summaryFailed: (ok: number, failed: number) => `${ok} updated, ${failed} failed. Hover a failure for the reason.`,
    summaryDeclined: (ok: number, declined: number) =>
      `${ok} updated. ${declined} not applied because the new permissions were declined.`,
    thisProfile: '· this profile',
    failed: 'Failed',
    declined: 'Not applied',
    skippedOffline: 'Offline · skipped',
    skippedSignIn: 'Signed out · skipped',
    skippedIdle: 'Not connected · skipped',
    skippedUnreadable: 'Could not check',
    confirm: (installs: number) => (installs === 1 ? 'Update 1' : `Update ${installs}`),
    updating: 'Updating…',
    done: 'Done',
    consentBody: (names: string, installs: number) =>
      `The new catalog pin of ${names} adds surfaces the installed version does not have. Approving applies it to ${installs === 1 ? '1 profile' : `${installs} profiles`}:`,
    finished: (installs: number) =>
      `Updated ${installs === 1 ? '1 profile' : `${installs} profiles`}. Restart those gateways to apply.`
  }
}
