import { type ActiveWork, mergeActiveWork, normalizeActiveWork } from './quit-guard'
import { createStreamThrottle } from './stream-throttle'

// Main owns the merged active-turn snapshot used by both the quit guard and
// background throttling. Sender destruction clears its report at the same
// point as the original IPC listener.
export function registerDesktopActiveWorkRuntime(ipcMain: any) {
  // Each renderer reports the turns it has in flight; the quit guard reads the
  // merged picture. Keyed by webContents id so a closed window stops counting.
  const activeWorkByWebContents = new Map<number, ActiveWork>()

  // The same merged picture drives background throttling: chat windows run
  // unthrottled while any turn is in flight (streaming must paint while hidden)
  // and fall back to Chromium's default throttling at idle. See stream-throttle.ts.
  const streamThrottle = createStreamThrottle()

  function updateStreamThrottleFromActiveWork() {
    streamThrottle.update(mergeActiveWork(activeWorkByWebContents.values()).count > 0)
  }

  ipcMain.on('hermes:active-work', (event, payload) => {
    const id = event.sender.id

    if (!activeWorkByWebContents.has(id)) {
      event.sender.once('destroyed', () => {
        activeWorkByWebContents.delete(id)
        updateStreamThrottleFromActiveWork()
      })
    }

    activeWorkByWebContents.set(id, normalizeActiveWork(payload))
    updateStreamThrottleFromActiveWork()
  })

  return { activeWorkByWebContents, streamThrottle }
}
