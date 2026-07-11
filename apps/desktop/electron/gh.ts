import { execFile } from 'node:child_process'
import path from 'node:path'

export const GH_TIMEOUT_MS = 30_000
export const GH_MAX_BUFFER = 8 * 1024 * 1024

export interface GhResult {
  ok: boolean
  kind: 'failure' | 'missing' | 'success' | 'timeout'
  stdout?: string
  stderr?: string
  exitCode?: number | null
}
export interface GhRunOptions {
  cwd?: string
  ghBin?: string
  timeout?: number
  maxBuffer?: number
  exec?: typeof execFile
}

export function ghEnv(ghBin) {
  const extra = [ghBin ? path.dirname(ghBin) : '', '/opt/homebrew/bin', '/usr/local/bin', '/usr/bin'].filter(
    dir => dir && dir !== '.'
  )

  return { ...process.env, PATH: [...extra, process.env.PATH].filter(Boolean).join(path.delimiter) }
}

export function runGh(args: string[], options: GhRunOptions = {}): Promise<GhResult> {
  const { cwd, ghBin = 'gh', timeout = GH_TIMEOUT_MS, maxBuffer = GH_MAX_BUFFER, exec = execFile } = options

  return new Promise<GhResult>(resolve => {
    exec(
      ghBin,
      args,
      { cwd, env: ghEnv(ghBin), windowsHide: true, timeout, maxBuffer },
      (error, stdout = '', stderr = '') => {
        const missing = error?.code === 'ENOENT'
        const timedOut = Boolean(error?.killed) || error?.code === 'ETIMEDOUT'
        resolve({
          ok: !error,
          kind: missing ? 'missing' : timedOut ? 'timeout' : error ? 'failure' : 'success',
          stdout: String(stdout),
          stderr: String(stderr),
          exitCode: typeof error?.code === 'number' ? error.code : null
        })
      }
    )
  })
}
