import type { SpawnOptions } from 'node:child_process'

interface ResolvedHermesCommand {
  command?: string
  args?: string[]
  root?: string
  env?: NodeJS.ProcessEnv
  shell?: boolean
  kind?: string
}

interface StoppableChild {
  once: (event: string, listener: (...args: any[]) => void) => unknown
  kill: (signal?: NodeJS.Signals | number) => boolean
}

interface StopDesktopBackgroundServicesOptions {
  resolveBackend: (args: string[]) => ResolvedHermesCommand | null | undefined
  spawnFn: (command: string, args: string[], options: SpawnOptions) => StoppableChild
  env: NodeJS.ProcessEnv
  timeoutMs?: number
  platform?: NodeJS.Platform
  uid?: number
  onError?: (message: string) => void
}

interface StopCommand {
  command: string
  args: string[]
  options: SpawnOptions
  label: string
  tolerateNonzero?: boolean
}

function runStopCommand(
  spawnFn: StopDesktopBackgroundServicesOptions['spawnFn'],
  command: StopCommand,
  timeoutMs: number,
  onError: (message: string) => void
): Promise<boolean> {
  return new Promise(resolve => {
    let settled = false
    let child: StoppableChild
    let timer: ReturnType<typeof setTimeout> | undefined

    const finish = (ok: boolean) => {
      if (settled) {
        return
      }

      settled = true

      if (timer) {
        clearTimeout(timer)
      }

      resolve(ok)
    }

    try {
      child = spawnFn(command.command, command.args, command.options)
    } catch (error) {
      onError(`Failed to start ${command.label}: ${String(error)}`)
      resolve(false)

      return
    }

    if (timeoutMs > 0) {
      timer = setTimeout(() => {
        onError(`${command.label} exceeded ${timeoutMs}ms; terminating the stop helper`)

        try {
          child.kill('SIGTERM')
        } catch {
          // The helper may have exited between the timeout and kill.
        }

        finish(false)
      }, timeoutMs)
    }

    child.once('error', error => {
      onError(`${command.label} failed: ${String(error)}`)
      finish(false)
    })
    child.once('exit', code => finish(code === 0 || Boolean(command.tolerateNonzero)))
  })
}

/**
 * Stop Desktop-owned background supervision before Desktop exits. Messaging
 * gateways are user-owned long-lived services and must survive a UI quit;
 * only the companion backend launchd job is unloaded here.
 */
export function stopDesktopBackgroundServices({
  resolveBackend,
  spawnFn,
  env,
  timeoutMs = 20_000,
  platform = process.platform,
  uid = typeof process.getuid === 'function' ? process.getuid() : -1,
  onError = () => undefined
}: StopDesktopBackgroundServicesOptions): Promise<boolean> {
  const commands: StopCommand[] = []

  if (platform === 'darwin' && uid >= 0) {
    commands.push({
      command: '/bin/launchctl',
      args: ['bootout', `gui/${uid}/com.local.hermes.companion-backend`],
      label: 'Hermes companion launchd stop',
      tolerateNonzero: true,
      options: {
        env: { ...env },
        shell: false,
        windowsHide: true,
        stdio: 'ignore'
      }
    })
  }

  if (commands.length === 0) {
    return Promise.resolve(true)
  }

  return Promise.all(
    commands.map(command => runStopCommand(spawnFn, command, timeoutMs, onError))
  ).then(results => results.every(Boolean))
}
