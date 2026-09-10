const INSTALLED_PIPE_GUARD = '__hermesDesktopStdioPipeGuardInstalled'

type ErrorStream = {
  on: (event: 'error', listener: (error: NodeJS.ErrnoException) => void) => unknown
  [INSTALLED_PIPE_GUARD]?: boolean
}

export function isIgnorablePipeError(error: NodeJS.ErrnoException | null | undefined): boolean {
  const code = error?.code || error?.errno

  if (code === 'EPIPE' || code === 'ERR_STREAM_DESTROYED') {
    return true
  }

  return false
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
