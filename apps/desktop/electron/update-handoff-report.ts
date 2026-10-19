// Report a detached update's result once, where boot passes the update gate.

import path from 'node:path'

import type { Dialog, Shell } from 'electron'

import { readAndConsumeHandoffResult, type ReleaseHolder } from './handoff-result'
import { closableReleaseHolders } from './release-holder-consent'
import { explainUpdateStop, readStoppedRun } from './update-stop-remedy'

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
  /** The user consented to closing these programs: record it and run the update again. */
  closeHoldersAndRetry?: (holders: ReleaseHolder[]) => void
}

/** The run identity a live marker carried when the boot gate saw it. */
export interface ParkedRun {
  startedAt: number | null
  runId: string | null
}

/**
 * The run a boot wait parked on: the result reported when the wait ends is
 * that run's, never an older one's. The run id is stable across the scripts'
 * line-2 heartbeat, including when the wait first saw the marker mid-update;
 * line 2 is kept only for older result producers that write no run id.
 */
export function parkedRunReport() {
  let parked: ParkedRun = { startedAt: null, runId: null }

  return {
    park: (marker: ParkedRun) => {
      parked = marker
    },
    report: (host: Omit<HandoffReportHost, 'expectedStartedAt' | 'expectedRunId'>) =>
      reportHandoffResult({ ...host, expectedStartedAt: parked.startedAt, expectedRunId: parked.runId })
  }
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
    } else if (result && result.ok && result.manual && result.releaseHolders.length && host.closeHoldersAndRetry) {
      reportReleaseHolders(host, result.releaseHolders, result.message)
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
      // The run's own receipt names the stop the updater took (it is written
      // before this dialog renders). Matched by correlation id, so an older
      // run's stop can never explain this one.
      const stop: ReturnType<typeof readStoppedRun> = readStoppedRun(host.hermesHome, result.runId)
      const explained: ReturnType<typeof explainUpdateStop> = explainUpdateStop(stop?.stopClass ?? null)
      const handoffLogPath = path.join(host.hermesHome, 'logs', 'desktop-update-handoff.log')

      host.log(
        `[updates] detached update FAILED (exit ${result.exitCode}): ${result.message}` +
          (stop === null ? '' : ` (stop_class ${stop.stopClass})`)
      )

      // Async so boot is not blocked behind the dialog; the response handlers
      // reuse the menu's open-updates path (queued until the renderer is ready)
      // and the same reveal primitive as 'hermes:logs:reveal'.
      void host.dialog
        .showMessageBox({
          type: 'error',
          title: 'Hermes update',
          message: explained?.message ?? "Hermes couldn't finish updating",
          detail: explained === null
            ? "You're still on the previous version and can keep using it. Try the update again, or open the update log to report the problem.\n\n" +
              `Details: ${result.message}`
            : `${explained.detail}\n\nDetails: ${result.message}`,
          // A refusal is not retried: the same offer fails the same way. Its way
          // out is the channel the update settings own, so lead with that.
          buttons: explained === null ? ['Try again', 'Open log', 'Close'] : ['Update settings', 'Open log', 'Close'],
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

/**
 * #101789: the update landed, but a program outside Hermes holds the Desktop app's folder. Name it
 * and offer to close it, only on an explicit click; dismissing leaves it running. A holder whose
 * program path Windows would not reveal cannot be closed from here: it is named for the user to
 * close, with no Close button that could never work.
 */
function reportReleaseHolders(host: HandoffReportHost, holders: ReleaseHolder[], message: string): void {
  const names = holders.map(holder => holder.name).join(', ')
  const closable = closableReleaseHolders(holders)
  const closableNames = closable.map(holder => holder.name).join(', ')

  const listed = holders
    .map(
      holder => `• ${holder.name} (pid ${holder.pid})\n   ${holder.exe || 'Close this one yourself (Task Manager).'}`
    )
    .join('\n')

  const action = closable.length
    ? 'Hermes did not close it. Close it and retry the update now, or keep using the current app and update later.'
    : 'Hermes cannot close it. Close it yourself (Task Manager), then run the update again.'

  host.log(`[updates] Desktop app swap blocked by programs outside Hermes: ${names}`)
  void host.dialog
    .showMessageBox({
      type: 'warning',
      title: 'Hermes update',
      message: `${names} is blocking the Desktop app update`,
      detail:
        `Hermes updated, but Windows would not replace the Desktop app because this program, which is not part of Hermes, is using its folder:\n\n${listed}\n\n` +
        `${action}\n\nDetails: ${message}`,
      buttons: closable.length ? [`Close ${closableNames} and retry`, 'Not now'] : ['OK'],
      defaultId: closable.length ? 1 : 0,
      cancelId: closable.length ? 1 : 0,
      noLink: true
    })
    .then(({ response }) => {
      if (closable.length && response === 0) {
        host.log(`[updates] user consented to closing ${closableNames}; retrying the update`)
        host.closeHoldersAndRetry?.(closable)
      }
    })
}
