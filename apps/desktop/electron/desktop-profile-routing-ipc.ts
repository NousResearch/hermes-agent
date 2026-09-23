export interface DesktopProfileRoutingIpcDeps {
  ipcMain: any
  desktopProfilePreferences: any
  readActiveDesktopProfile: any
  writeActiveDesktopProfile: any
  assertCanMutateManagedPrimaryRouting: any
  teardownPrimaryBackendAndWait: any
  getMainWindow: any
}

export function registerDesktopProfileRoutingIpc(deps: DesktopProfileRoutingIpcDeps) {
  const {
    ipcMain,
    desktopProfilePreferences,
    readActiveDesktopProfile,
    writeActiveDesktopProfile,
    assertCanMutateManagedPrimaryRouting,
    teardownPrimaryBackendAndWait,
    getMainWindow
  } = deps

  ipcMain.handle('hermes:profile:default:get', async () => desktopProfilePreferences.getDefault())
  ipcMain.handle('hermes:profile:default:set', async (_event, route) => desktopProfilePreferences.setDefault(route))
  ipcMain.handle('hermes:profile:get', async () => ({ profile: readActiveDesktopProfile() }))
  // Persistence-only sibling of hermes:profile:set: records the profile the
  // Desktop last used WITHOUT tearing down the backend or reloading the window.
  // An explicit default route wins at launch and is never replaced here.
  ipcMain.handle('hermes:profile:remember', async (_event, name) => ({
    profile: writeActiveDesktopProfile(name)
  }))
  ipcMain.handle('hermes:profile:set', async (_event, name) => {
    assertCanMutateManagedPrimaryRouting()
    const next = writeActiveDesktopProfile(name)

    // Switching profiles is a backend re-home: relaunch the dashboard under the
    // new HERMES_HOME. Pool backends keep their own homes, so only the primary
    // is torn down.
    await teardownPrimaryBackendAndWait()
    getMainWindow()?.reload()

    return { profile: next }
  })
}
