// Main-process shared-metrics wiring, kept out of main.ts: the once-per-launch startup-latency
// claim, the packaged self-update recorder and the consent-gated renderer-crash recorder the
// renderer drains once a backend is attached.

import { app, BrowserWindow, ipcMain } from 'electron'

import { INSTALL_STAMP } from './install-stamp'
import { registerRendererCrashIpc, RendererCrashRecorder } from './renderer-crash-metrics'
import { registerStartupLatencyIpc } from './startup-latency-ipc'
import type { UpdaterApplyResultWire, UpdaterStrategy } from './updater/index'
import { registerUpdateMetricsIpc, UpdateRunRecorder } from './updater/update-metrics'

export interface DesktopSharedMetrics {
  noteUpdateProgress(stage: string): void
  /** Run `strategy.apply()`; recorded only when `packaged` (a checkout hand-off is counted by
   *  `hermes update`'s own receipt). */
  trackUpdateApply(packaged: UpdaterStrategy | null, strategy: UpdaterStrategy): Promise<UpdaterApplyResultWire>
  /** `installWindowRendererLifecycle` hook: counts a live window's renderer loss (no-op unless that
   *  window's focused profile collects). */
  recordRendererGone(windowId: number, reason: unknown): void
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

  const crashes = new RendererCrashRecorder({ dir: app.getPath('userData') })

  registerRendererCrashIpc(ipcMain, crashes)

  return {
    recordRendererGone: (windowId, reason) => crashes.record(windowId, reason),
    noteUpdateProgress: stage => recorder.noteProgress(stage),
    trackUpdateApply: (packaged, strategy) =>
      recorder.track(packaged?.mechanism, INSTALL_STAMP?.commitDate, () => strategy.apply())
  }
}
