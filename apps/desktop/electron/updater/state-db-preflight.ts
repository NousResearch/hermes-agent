import { execFile } from 'node:child_process'

import { hiddenWindowsChildOptions } from '../windows-child-options'

interface StateDbPreflight {
  python: string | null
  script: string
  home: string
  log: (message: string) => void
}

// Await the snapshot before backend shutdown without freezing Electron's event loop.
export async function preflightStateDb({ python, script, home, log }: StateDbPreflight): Promise<void> {
  try {
    if (!python) {
      throw new Error('Python not found')
    }

    const result = await new Promise<string>((resolve, reject) => {
      execFile(
        python,
        ['-I', '-S', script, home],
        hiddenWindowsChildOptions({ encoding: 'utf8', timeout: 900_000 }),
        (error, stdout) => error ? reject(error) : resolve(String(stdout))
      )
    })

    log(`[updates] state.db pre-flight: ${result.trim()}`)
  } catch (error: unknown) {
    const message =
      `state.db pre-flight failed: ${error instanceof Error ? error.message : String(error)}. ` +
      'Update cancelled before backend shutdown. Update the selected installation with its hermes update command, then retry.'

    log(`[updates] ${message}`)
    throw new Error(message, { cause: error })
  }
}
