/**
 * A local stand-in for `SshConnection.exec` used by the remote-lifecycle
 * suites: runs the command through a POSIX shell and, like the real mux exec,
 * writes `stdinData` to the remote process's stdin. A double that drops it
 * leaves the programs the lifecycle ships on stdin (`python3 -`) reading the
 * test runner's own stdin instead of the program they were handed.
 */

import { spawn } from 'node:child_process'

export interface ShellExecOptions {
  shell?: string
  env?: NodeJS.ProcessEnv
  timeoutMs?: number
}

export interface ShellExecResult {
  stdout: string
  stderr: string
}

/** Rejection shape of the promisified `child_process.exec` these doubles replaced. */
export interface ShellExecFailure extends Error, ShellExecResult {
  code: number | string
}

export function shellExec(
  command: string,
  { shell = 'sh', env, timeoutMs = 10_000, stdinData }: ShellExecOptions & { stdinData?: string } = {}
): Promise<ShellExecResult> {
  return new Promise((resolve, reject) => {
    const child = spawn(shell, ['-c', command], { env, stdio: 'pipe', timeout: timeoutMs })
    let stdout = ''
    let stderr = ''

    child.stdout.setEncoding('utf8').on('data', chunk => (stdout += chunk))
    child.stderr.setEncoding('utf8').on('data', chunk => (stderr += chunk))
    child.once('error', reject)
    child.once('close', (code, signal) => {
      if (code === 0) {
        resolve({ stdout, stderr })

        return
      }

      const failure = new Error(stderr.trim() || `${shell} -c exited ${code ?? signal}`) as ShellExecFailure
      failure.code = code ?? (signal as string)
      failure.stdout = stdout
      failure.stderr = stderr
      reject(failure)
    })
    child.stdin.end(stdinData ?? '')
  })
}

/** `{ exec }` with the real exec's `(command, { stdinData })` signature. */
export function shellSshDouble(options: ShellExecOptions = {}) {
  return {
    exec: async (command: string, { stdinData }: { stdinData?: string } = {}): Promise<string> =>
      (await shellExec(command, { ...options, stdinData })).stdout
  }
}
