/**
 * Guard Electron main-process stdout/stderr against broken-pipe crashes.
 *
 * When the parent console is closed (Windows updater handoff, detached launch,
 * redirected stdio), writes to standard pipes raise EPIPE / ERR_STREAM_DESTROYED.
 * Those are expected and must not take down the Desktop main process.
 *
 * Only structural `error.code` values are ignorable — never match on message
 * text (see review on #61894).
 */

const INSTALLED_PIPE_GUARD = '__hermesDesktopStdioPipeGuardInstalled'

type ErrorStream = {
  on: (event: 'error', listener: (error: NodeJS.ErrnoException) => void) => unknown
  [INSTALLED_PIPE_GUARD]?: boolean
}

export function isIgnorablePipeError(error: NodeJS.ErrnoException | null | undefined): boolean {
  const code = error?.code

  return code === 'EPIPE' || code === 'ERR_STREAM_DESTROYED'
}

function attachPipeGuard(stream: ErrorStream | null | undefined): boolean {
  if (!stream || stream[INSTALLED_PIPE_GUARD]) {
    return false
  }

  stream[INSTALLED_PIPE_GUARD] = true
  stream.on('error', error => {
    if (!isIgnorablePipeError(error)) {
      throw error
    }
  })

  return true
}

export function installStdioPipeErrorGuards({
  stdout = process.stdout,
  stderr = process.stderr
}: {
  stdout?: ErrorStream | null
  stderr?: ErrorStream | null
} = {}): number {
  return [stdout, stderr].filter(attachPipeGuard).length
}
