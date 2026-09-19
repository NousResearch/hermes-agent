import { execFileSync } from 'node:child_process'

import { hiddenWindowsChildOptions } from '../windows-child-options'

interface StateDbPreflight {
  python: string | null
  script: string
  home: string
  log: (message: string) => void
}

// Synchronous by design: the caller must not stop the backend before the snapshot.
export function preflightStateDb({ python, script, home, log }: StateDbPreflight): void {
  try {
    if (!python) {
      throw new Error('Python not found')
    }

    const result: string = execFileSync(
      python,
      ['-I', '-S', script, home],
      hiddenWindowsChildOptions({ encoding: 'utf8', timeout: 30_000, stdio: ['ignore', 'pipe', 'pipe'] })
    )

    log(`[updates] state.db pre-flight: ${result.trim()}`)
  } catch (error: unknown) {
    const message =
      `state.db pre-flight failed: ${error instanceof Error ? error.message : String(error)}. ` +
      'Update cancelled before backend shutdown. Update the selected installation with its hermes update command, then retry.'

    log(`[updates] ${message}`)
    throw new Error(message, { cause: error })
  }
}
