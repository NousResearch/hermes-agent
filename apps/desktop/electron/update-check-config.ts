import { execFile } from 'node:child_process'
import { promisify } from 'node:util'

const execFileAsync = promisify(execFile)

export interface HermesConfigRuntime {
  command?: string | null
  args?: string[]
  env?: NodeJS.ProcessEnv
  shell?: boolean
}

/** Read config.yaml's `updates.check` through the Hermes runtime. Default
 *  (fail-open) is true: a missing runtime, probe failure, or malformed
 *  response must never silently disable update checks — the same policy the
 *  pre-update backup probe applies to the emergency snapshot. */
export async function readUpdatesCheckEnabled(
  runtime: HermesConfigRuntime | Promise<HermesConfigRuntime>,
  run = execFileAsync
): Promise<boolean> {
  try {
    const resolved = await runtime

    if (!resolved.command || !resolved.args) {
      return true
    }

    const result = await run(resolved.command, resolved.args, {
      encoding: 'utf8',
      env: { ...process.env, ...resolved.env },
      shell: Boolean(resolved.shell),
      timeout: 15_000,
      windowsHide: true
    })

    const parsed = JSON.parse(String(result.stdout).trim())

    return parsed !== false && parsed !== 'false'
  } catch {
    return true
  }
}
