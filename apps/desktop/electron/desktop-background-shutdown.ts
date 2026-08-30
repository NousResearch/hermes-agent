import type { ChildProcess, SpawnOptions } from 'node:child_process'

interface ResolvedHermesCommand {
  command?: string
  args?: string[]
  root?: string
  env?: NodeJS.ProcessEnv
  shell?: boolean
  kind?: string
}

interface StoppableChild {
  once: ChildProcess['once']
  kill: ChildProcess['kill']
}

interface StopDesktopBackgroundServicesOptions {
  resolveBackend: (args: string[]) => ResolvedHermesCommand | null | undefined
  spawnFn: (command: string, args: string[], options: SpawnOptions) => StoppableChild
  env: NodeJS.ProcessEnv
  timeoutMs?: number
  onError?: (message: string) => void
}

/**
 * Stop every local supervised gateway before Desktop exits. The gateway owns
 * cron and Kanban dispatch, so stopping only Desktop's `serve` child is not
 * sufficient when launchd/systemd is also supervising a gateway profile.
 */
export function stopDesktopBackgroundServices({
  resolveBackend,
  spawnFn,
  env,
  timeoutMs = 20_000,
  onError = () => undefined
}: StopDesktopBackgroundServicesOptions): Promise<boolean> {
  const backend = resolveBackend(['gateway', 'stop', '--all'])

  if (!backend?.command || backend.kind === 'bootstrap-needed') {
    onError('No runnable local Hermes command was available for gateway stop --all')
    return Promise.resolve(false)
  }

  return new Promise(resolve => {
    let settled = false
    let child: StoppableChild
    let timer: ReturnType<typeof setTimeout> | undefined

    const finish = (ok: boolean) => {
      if (settled) return
      settled = true
      if (timer) clearTimeout(timer)
      resolve(ok)
    }

    try {
      child = spawnFn(backend.command!, backend.args || ['gateway', 'stop', '--all'], {
        cwd: backend.root || undefined,
        env: { ...env, ...(backend.env || {}) },
        shell: Boolean(backend.shell),
        windowsHide: true,
        stdio: 'ignore'
      })
    } catch (error) {
      onError(`Failed to start gateway stop --all: ${String(error)}`)
      resolve(false)
      return
    }

    timer = setTimeout(() => {
      onError(`gateway stop --all exceeded ${timeoutMs}ms; terminating the stop helper`)
      try {
        child.kill('SIGTERM')
      } catch {
        // The helper may have exited between the timeout and kill.
      }
      finish(false)
    }, timeoutMs)

    child.once('error', error => {
      onError(`gateway stop --all failed: ${String(error)}`)
      finish(false)
    })
    child.once('exit', code => finish(code === 0))
  })
}
