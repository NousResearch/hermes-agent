// PowerMonitor registration is intentionally late (after app ready) and
// idempotent. The battery query handler remains available from module setup.
export function createDesktopPowerRuntime(deps: {
  BrowserWindow: any
  attachPowerResumeRemoteRevalidation: (...args: any[]) => any
  getMainWindow: () => Electron.BrowserWindow | null
  ipcMain: any
  powerMonitor: any
  rememberLog: (message: string) => void
  revalidateSuspectPoolAfterResume: () => any
}) {
  const {
    BrowserWindow,
    attachPowerResumeRemoteRevalidation,
    getMainWindow,
    ipcMain,
    powerMonitor,
    rememberLog,
    revalidateSuspectPoolAfterResume
  } = deps

  // Tell the renderer the machine just woke. Sleep silently drops the
  // renderer's WebSocket to the local backend; the renderer reconnects on this
  // signal so the chat composer doesn't stay stuck on "Starting Hermes...".
  function sendPowerResume() {
    const mainWindow = getMainWindow()

    if (!mainWindow || mainWindow.isDestroyed()) {
      return
    }

    const { webContents } = mainWindow

    if (!webContents || webContents.isDestroyed()) {
      return
    }

    webContents.send('hermes:power-resume')
  }

  let powerResumeRegistered = false

  // Mirror of powerMonitor's AC/battery state, broadcast to every window so
  // renderer backstop polls can slow down on battery (see store/power.ts).
  // `null` until the first powerMonitor read after app ready.
  let onBatteryPower: boolean | null = null

  // Renderer-side battery gating seeds from this and stays current via the
  // 'hermes:power-battery' push below.
  ipcMain.handle('hermes:power-battery:get', () => onBatteryPower === true)

  function broadcastBatteryState(next: boolean) {
    if (onBatteryPower === next) {
      return
    }

    onBatteryPower = next

    for (const win of BrowserWindow.getAllWindows()) {
      const { webContents } = win

      if (webContents && !webContents.isDestroyed()) {
        webContents.send('hermes:power-battery', next)
      }
    }
  }

  function registerPowerResumeListeners() {
    if (powerResumeRegistered) {
      return
    }

    powerResumeRegistered = true

    try {
      // 'resume' covers sleep/wake; 'unlock-screen' covers lock/unlock without a
      // full suspend. Either can drop an idle socket.
      powerMonitor.on('resume', sendPowerResume)
      powerMonitor.on('unlock-screen', sendPowerResume)
      powerMonitor.on('on-battery', () => broadcastBatteryState(true))
      powerMonitor.on('on-ac', () => broadcastBatteryState(false))
      onBatteryPower = powerMonitor.isOnBatteryPower()
      // Pooled remote/SSH backends are also suspect after a wake (#93910): the
      // renderer nudge above only re-drives the PRIMARY socket, while pooled
      // tunnels have no renderer loop of their own. Bounded + coalesced inside;
      // never a hot loop.
      attachPowerResumeRemoteRevalidation({
        log: rememberLog,
        powerMonitor,
        revalidate: () => revalidateSuspectPoolAfterResume()
      })
    } catch {
      // powerMonitor is unavailable before app 'ready' on some platforms; the
      // caller registers after 'ready', so this should not normally throw.
    }
  }

  return { registerPowerResumeListeners }
}
