// Shared input setup for the install drivers. Only the selected app window
// is changed; helper windows retain their own coordinate system.
async function prepareWindowForInput(app, page) {
  const window = await app.browserWindow(page)
  // Use the same persistent setting as Appearance. A bare setZoomLevel is
  // overwritten by the app's focus/navigation handlers restoring saved zoom.
  const persistent = await page.evaluate(() => {
    const zoom = globalThis.hermesDesktop?.zoom
    if (!zoom?.setPercent || !zoom?.get) return false
    zoom.setPercent(100)
    return true
  })
  if (persistent) {
    // Playwright 1.58 treats an async waitForFunction predicate's Promise as
    // truthy even when it resolves false. Await each IPC read on the driver.
    const deadline = Date.now() + 15_000
    for (;;) {
      const state = await page.evaluate(() => {
        // Cold-start restoration can overwrite the first preference write.
        // Reapply through its owner until a subsequent read observes it.
        globalThis.hermesDesktop.zoom.setPercent(100)
        return globalThis.hermesDesktop.zoom.get()
      })
      // The renderer IPC and BrowserWindow can observe different moments of
      // startup restoration. Both must agree before the driver sends input.
      const factor = await window.evaluate(win => win.webContents.getZoomFactor())
      if (state.percent === 100 && Math.abs(factor - 1) < 0.001) return
      if (Date.now() >= deadline) {
        throw new Error(`timed out waiting for 100% app window zoom (IPC ${state.percent}%, factor ${factor})`)
      }
      await page.waitForTimeout(100)
    }
  } else {
    // Older sampled releases have no zoom preference bridge.
    await window.evaluate(win => win.webContents.setZoomLevel(0))
  }
  // DPR includes OS display scaling; 100% page zoom is not always DPR 1.
  const factor = await window.evaluate(win => win.webContents.getZoomFactor())
  if (Math.abs(factor - 1) > 0.001) {
    throw new Error(`could not set app window zoom to 100% (factor ${factor})`)
  }
}

module.exports = { prepareWindowForInput }
