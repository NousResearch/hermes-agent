// Report a detached update's result once, where boot passes the update gate.

import path from 'node:path'

import type { Dialog, Shell } from 'electron'

import { readAndConsumeHandoffResult } from './handoff-result'

export interface HandoffReportHost {
  hermesHome: string
  /** Marker line 2 of the run this boot parked on (C2 started_at match), else null. */
  expectedStartedAt: number | null
  /** Stable run id of that marker (survives heartbeat refreshes), else null. */
  expectedRunId: string | null
  log: (line: string) => void
  dialog: Pick<Dialog, 'showMessageBox'>
  shell: Pick<Shell, 'showItemInFolder'>
  /** The menu's open-updates path (queued until the renderer is ready). */
  openUpdates: () => void
}

/**
 * The detached hand-off script (scripts/desktop-update/windows.ps1) runs hidden;
 * its result file is the ONLY way the user learns a detached update
 * failed. Consume it exactly once, here, right where boot passes the
 * update gate — success gets a log line, failure gets a real dialog
 * (previously a failed detached update was indistinguishable from
 * "nothing happened").
 */
export function reportHandoffResult(host: HandoffReportHost): void {
  try {
    const result = readAndConsumeHandoffResult(host.hermesHome, {
      expectedStartedAt: host.expectedStartedAt,
      expectedRunId: host.expectedRunId,
      log: host.log
    })

    if (result && result.ok && result.warnings.length && !result.manual) {
      // Committed, but follow-up work failed (C2/C3): the user IS on the new
      // version, so this is a non-blocking notice, never "previous version".
      host.log(`[updates] detached update finished with warnings: ${result.warnings.join(' | ')}`)
      void host.dialog.showMessageBox({
        type: 'info',
        title: 'Hermes update',
        message: 'Hermes updated, but some follow-up steps need another try',
        detail: `${result.warnings.join('\n')}\n\nHermes retries them on the next launch or the next update.`
      })
    } else if (result && result.ok && result.manual) {
      // Update landed but the user must act (reopen/reinstall/sandbox). On
      // machines with no shim browser and no notifier this dialog is the
      // FIRST time the message is visible — it must not be a log line.
      host.log(`[updates] detached update finished with manual action (branch ${result.branch}): ${result.message}`)
      host.dialog.showMessageBox({
        type: 'warning',
        title: 'Hermes update',
        message: 'The update finished, but needs one more step',
        detail: result.message
      })
    } else if (result && result.ok) {
      host.log(`[updates] detached update finished OK (branch ${result.branch})`)
    } else if (result) {
      host.log(`[updates] detached update FAILED (exit ${result.exitCode}): ${result.message}`)
      const handoffLogPath = path.join(host.hermesHome, 'logs', 'desktop-update-handoff.log')

      // Async so boot is not blocked behind the dialog; the response handlers
      // reuse the menu's open-updates path (queued until the renderer is ready)
      // and the same reveal primitive as 'hermes:logs:reveal'.
      void host.dialog
        .showMessageBox({
          type: 'error',
          title: 'Hermes update',
          message: "Hermes couldn't finish updating",
          detail:
            "You're still on the previous version and can keep using it. Try the update again, or open the update log to report the problem.\n\n" +
            `Details: ${result.message}`,
          buttons: ['Try again', 'Open log', 'Close'],
          defaultId: 0,
          cancelId: 2,
          noLink: true
        })
        .then(({ response }) => {
          if (response === 0) {
            host.openUpdates()
          } else if (response === 1) {
            host.shell.showItemInFolder(handoffLogPath)
          }
        })
    }
  } catch (err) {
    host.log(`[updates] could not read hand-off result: ${(err as Error).message}`)
  }
}
