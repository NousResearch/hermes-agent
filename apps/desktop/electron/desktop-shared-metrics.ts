// Main-process shared-metrics wiring, kept out of main.ts: the once-per-launch startup-latency
// claim and the packaged self-update recorder the renderer drains once a backend is attached.

import { app, BrowserWindow, ipcMain } from 'electron'

import { INSTALL_STAMP } from './install-stamp'
import { registerStartupLatencyIpc } from './startup-latency-ipc'
import type { UpdaterApplyResultWire, UpdaterStrategy } from './updater/index'
import { registerUpdateMetricsIpc, UpdateRunRecorder } from './updater/update-metrics'

export interface DesktopSharedMetrics {
  noteUpdateProgress(stage: string): void
  /** Run `strategy.apply()`; recorded only when `packaged` (a checkout hand-off is counted by
   *  `hermes update`'s own receipt). */
  trackUpdateApply(packaged: UpdaterStrategy | null, strategy: UpdaterStrategy): Promise<UpdaterApplyResultWire>
}

export function registerDesktopSharedMetrics(): DesktopSharedMetrics {
  registerStartupLatencyIpc()

  const recorder = new UpdateRunRecorder({
    dir: () => app.getPath('userData'),
    appVersion: () => app.getVersion(),
    onRecorded: () => {
      for (const window of BrowserWindow.getAllWindows()) {
        window.webContents.send('hermes:updates:metric:pending')
      }
    }
  })

  registerUpdateMetricsIpc(ipcMain, recorder)

  return {
    noteUpdateProgress: stage => recorder.noteProgress(stage),
    trackUpdateApply: (packaged, strategy) =>
      recorder.track(packaged?.mechanism, INSTALL_STAMP?.commitDate, () => strategy.apply())
  }
}
