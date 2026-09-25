import { execFileSync } from 'node:child_process'
import { statSync } from 'node:fs'
import path from 'node:path'

import { hiddenWindowsChildOptions } from '../windows-child-options'

interface StateDbPreflight {
  python: string | null
  launcher?: string | null
  script: string
  home: string
  log: (message: string) => void
}

function managedPython(launcher: string): string | null {
  // The installation launcher exposes its committed store interpreter before
  // importing Hermes or activating a dependency generation. The snapshot
  // itself still runs stdlib-only with -I -S, even if application imports fail.
  const viaCmd: boolean = process.platform === 'win32' && /\.cmd$/i.test(launcher)

  if (viaCmd && /["%&|<>^\r\n]/.test(launcher)) {
    return null
  }

  const output: string = execFileSync(
    viaCmd ? (process.env.ComSpec ?? 'cmd.exe') : launcher,
    viaCmd
      ? ['/d', '/v:off', '/s', '/c', `""${launcher}" --print-runtime-command"`]
      : ['--print-runtime-command'],
    hiddenWindowsChildOptions({ encoding: 'utf8', timeout: 15_000, windowsVerbatimArguments: viaCmd })
  )

  const command: unknown = JSON.parse(output)
  const candidate: unknown = Array.isArray(command) ? command[0] : null

  if (typeof candidate !== 'string' || !path.isAbsolute(candidate)) {
    return null
  }

  try {
    return statSync(candidate).isFile() ? candidate : null
  } catch {
    return null
  }
}

// Synchronous by design: the caller must not stop the backend before the snapshot.
export function preflightStateDb({ python, launcher, script, home, log }: StateDbPreflight): void {
  try {
    const selectedPython: string | null = python || (launcher ? managedPython(launcher) : null)

    if (!selectedPython) {
      throw new Error('Python not found')
    }

    const result: string = execFileSync(
      selectedPython,
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
