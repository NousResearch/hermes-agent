import { execFile } from 'node:child_process'
import { promisify } from 'node:util'

const DISABLED_BACKUP_MODES = new Set(['off', 'false', 'none', 'disabled'])
const execFileAsync = promisify(execFile)

export function preUpdateBackupEnabled(value: unknown): boolean {
  if (value === false || value === null) {
    return false
  }

  return typeof value !== 'string' || !DISABLED_BACKUP_MODES.has(value.trim().toLowerCase())
}

export interface HermesConfigRuntime {
  command?: string
  args?: string[]
  env?: NodeJS.ProcessEnv
  shell?: boolean
}

export async function readPreUpdateBackupEnabled(
  runtime: HermesConfigRuntime,
  hermesHome: string,
  run = execFileAsync
): Promise<boolean> {
  if (!runtime.command || !runtime.args) {
    return true
  }

  try {
    const result = await run(runtime.command, runtime.args, {
      encoding: 'utf8',
      env: { ...process.env, ...runtime.env, HERMES_HOME: hermesHome },
      shell: Boolean(runtime.shell),
      timeout: 15_000,
      windowsHide: true
    })

    return preUpdateBackupEnabled(JSON.parse(String(result.stdout).trim()))
  } catch {
    // A missing runtime, probe failure, or malformed response must not weaken
    // the emergency recovery path.
    return true
  }
}
