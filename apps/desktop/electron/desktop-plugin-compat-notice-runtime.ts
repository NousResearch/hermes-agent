// Report plugin imports of old desktop module paths once per distinct report.
// The protocol is late-initialized in main, so resolve it only on click.
export function createDesktopPluginCompatNoticeRuntime(deps: {
  HERMES_HOME: string
  app: any
  dialog: any
  getHermesProtocol: () => string
  getMainWindow: () => Electron.BrowserWindow | null
  handleDeepLink: (url: string) => void
  pendingPluginCompatNotice: (...args: any[]) => any
  recordPluginCompatDismissed: (...args: any[]) => any
  rememberLog: (message: string) => void
}) {
  const {
    HERMES_HOME,
    app,
    dialog,
    getHermesProtocol,
    getMainWindow,
    handleDeepLink,
    pendingPluginCompatNotice,
    recordPluginCompatDismissed,
    rememberLog
  } = deps

  // One-time modal for plugins importing pre-decomposition module paths (see
  // electron/plugin-compat-notice.ts). The backend writes the report during plugin
  // discovery; we show each distinct report exactly once and remember the dismissal
  // in userData so the user is never nagged twice about the same set of plugins.
  let pluginCompatNoticeShown = false

  async function showPluginCompatNoticeOnce() {
    if (pluginCompatNoticeShown) {
      return
    }

    const mainWindow = getMainWindow()

    if (!mainWindow || mainWindow.isDestroyed()) {
      return
    }

    let notice

    try {
      notice = pendingPluginCompatNotice(HERMES_HOME, app.getPath('userData'))
    } catch (err) {
      rememberLog(`[plugins] compat notice check failed: ${err.message}`)

      return
    }

    if (!notice) {
      return
    }

    pluginCompatNoticeShown = true
    rememberLog(`[plugins] compat notice shown (${notice.key})`)

    try {
      // 'OK' is the default and cancel so a stray Enter/Escape never navigates;
      // 'Open Plugins' rides the existing deep-link channel (hermes://open/…),
      // which the renderer already maps to its hash router.
      const { response } = await dialog.showMessageBox(mainWindow, {
        type: 'warning',
        title: notice.title,
        message: notice.message,
        detail: notice.detail,
        buttons: ['Open Plugins', 'OK'],
        defaultId: 1,
        cancelId: 1,
        noLink: true
      })

      if (response === 0) {
        handleDeepLink(`${getHermesProtocol()}://open/capabilities?tab=plugins`)
      }
    } finally {
      try {
        recordPluginCompatDismissed(app.getPath('userData'), notice.key)
      } catch (err) {
        rememberLog(`[plugins] could not persist compat notice dismissal: ${err.message}`)
      }
    }
  }

  return { showPluginCompatNoticeOnce }
}
