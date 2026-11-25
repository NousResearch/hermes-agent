import path from 'node:path'

// Every Chromium CHECK/LOG(FATAL) in the shell ends at the same instruction —
// base::ImmediateCrash at the tail of logging::LogMessage::HandleFatal — so a
// core dump alone says "something fatal happened" and nothing about what
// (#100573: ten Linux reports, one shared trap address, zero fatal messages).
// The message itself goes to stderr a moment before the trap, and every Linux
// launcher (.desktop entry, Omarchy's hermes-desktop wrapper) discards stderr.
// Route Chromium's own log to a file next to desktop.log and let Crashpad keep
// local minidumps, so the next crash carries its FATAL line with it.

export interface LinuxCrashDiagnostics {
  /** Chromium command-line switches, applied before `app` is ready. */
  switches: ReadonlyArray<readonly [name: string, value: string]>
  /** `crashReporter.start` options; local database only, nothing leaves the machine. */
  crashReporter: { uploadToServer: false; compress: false }
}

// Chromium log severities: 0 INFO, 1 WARNING, 2 ERROR, 3 FATAL. ERROR keeps the
// file quiet during normal use (INFO would mirror every renderer console line).
const CHROMIUM_LOG_LEVEL_ERROR = '2'

export const CHROMIUM_LOG_FILENAME = 'desktop-chromium.log'

export function linuxCrashDiagnostics(
  logsDir: string,
  platform: NodeJS.Platform = process.platform
): LinuxCrashDiagnostics | null {
  if (platform !== 'linux') {
    return null
  }

  return {
    switches: [
      ['enable-logging', 'file'],
      ['log-file', path.join(logsDir, CHROMIUM_LOG_FILENAME)],
      ['log-level', CHROMIUM_LOG_LEVEL_ERROR]
    ],
    crashReporter: { uploadToServer: false, compress: false }
  }
}

/** The side effects the plan needs, injected so the failure paths are provable. */
export interface CrashDiagnosticsHost {
  /** Create the logs directory. May throw (read-only or invalid HERMES_HOME). */
  ensureLogsDir(dir: string): void
  /** Bound the Chromium log before Chromium appends to it (APPEND_TO_OLD_LOG_FILE). */
  reclaimChromiumLog(file: string): void
  appendSwitch(name: string, value: string): void
  startCrashReporter(options: LinuxCrashDiagnostics['crashReporter']): void
}

// Diagnostics are optional; startup is not. Every step is best-effort, because
// a read-only or invalid HERMES_HOME/logs must degrade to "no crash log", never
// to a desktop that dies before app readiness. The existing desktop log path
// swallows the same failures for the same reason.
export function enableLinuxCrashDiagnostics(
  plan: LinuxCrashDiagnostics | null,
  logsDir: string,
  host: CrashDiagnosticsHost
): void {
  if (!plan) {
    return
  }

  let logsDirReady = true

  try {
    host.ensureLogsDir(logsDir)
  } catch {
    // No writable logs dir: Chromium could not open the file anyway. Skip the
    // logging switches and keep the crash reporter, which writes elsewhere.
    logsDirReady = false
  }

  if (logsDirReady) {
    for (const [name, value] of plan.switches) {
      if (name === 'log-file') {
        try {
          host.reclaimChromiumLog(value)
        } catch {
          // Best-effort — an unbounded log beats no app, but try every launch.
        }
      }

      try {
        host.appendSwitch(name, value)
      } catch {
        // Ignore: a switch we cannot set only costs us the diagnostic.
      }
    }
  }

  try {
    host.startCrashReporter(plan.crashReporter)
  } catch {
    // Crashpad unavailable (sandbox, missing helper) — not a startup failure.
  }
}
