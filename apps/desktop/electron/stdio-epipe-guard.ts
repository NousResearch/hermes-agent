export interface StandardStreamErrorSource {
  on: (event: 'error', listener: (error: unknown) => void) => unknown
}

export interface StdioEpipeGuardOptions {
  onUnexpected?: (error: unknown) => void
  stderr?: StandardStreamErrorSource
  stdout?: StandardStreamErrorSource
}

function isBrokenPipeError(error: unknown): boolean {
  return Boolean(
    error &&
      typeof error === 'object' &&
      'code' in error &&
      (error as { code?: unknown }).code === 'EPIPE'
  )
}

/**
 * Renderer console messages can hit a closed inherited console pipe after a
 * detached Windows launch. Keep that expected EPIPE from becoming a main-process
 * exception while preserving every other stream failure for crash forensics.
 */
export function installStdioEpipeGuard({
  onUnexpected = error => {
    throw error
  },
  stderr = process.stderr,
  stdout = process.stdout
}: StdioEpipeGuardOptions = {}): void {
  for (const stream of new Set([stdout, stderr])) {
    stream.on('error', error => {
      if (isBrokenPipeError(error)) {
        return
      }

      onUnexpected(error)
    })
  }
}
