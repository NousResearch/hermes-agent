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
