/**
 * installs-ipc.ts
 *
 * Electron IPC for the desktop `hermes installs` UI. Three channels mirror
 * the CLI subcommands:
 *   - `hermes:installs:list`    → `hermes installs list --json`
 *   - `hermes:installs:remove`  → `hermes installs remove <id> --yes`
 *   - `hermes:installs:dismiss` → `hermes installs dismiss`
 *
 * The handler never builds a shell string: it validates the id against a
 * strict pattern (installs-cli.ts), builds an argv array, and hands it to the
 * injected runner. main.ts supplies the venv python spawn, so `current` is the
 * install the desktop app is running from — the same backend the uninstaller
 * uses for `hermes uninstall --gui-summary`.
 *
 * The boot notice lives here too: it runs the list once, decides from the pure
 * predicate, and keeps the notice until a renderer takes it. Main only pings
 * the windows: the renderer may not have mounted yet, so it pulls the notice
 * when it mounts and again on every ping. Taking consumes it, so it shows at
 * most once per app launch. Failures are swallowed with a debug log.
 */

import {
  dismissArgs,
  type InstallsListResult,
  listArgs,
  parseInstallsList,
  removeArgs,
  runTimeoutMs,
  shouldShowBootNotice
} from './installs-cli'

export interface InstallsRunOutcome {
  code: null | number
  stdout: string
  stderr: string
}

export interface InstallsRemoveResult {
  ok: boolean
  error?: string
  message?: string
}

/** The runtime the app itself runs, as `resolveHermesBackend(args)` reports it. */
export interface InstallsBackend {
  /** Null when no runtime is installed yet (the bootstrap-needed case). */
  command: null | string
  /** The full argv after `command`, including the `installs ...` args. */
  args: string[]
  env: NodeJS.ProcessEnv
  shell?: boolean
  /** The runtime's source tree. The command runs from here. */
  root?: string
  label?: string
}

export interface InstallsChild {
  stdout: null | { on: (event: 'data', listener: (chunk: Buffer | string) => void) => unknown }
  stderr: null | { on: (event: 'data', listener: (chunk: Buffer | string) => void) => unknown }
  on: ((event: 'error', listener: (error: Error) => void) => unknown) &
    ((event: 'exit', listener: (code: null | number) => void) => unknown)
  kill: () => unknown
}

export interface InstallsRunnerDeps {
  /** The backend the app runs, so `current` is the install the user is looking at. */
  resolveBackend: (args: string[]) => Promise<InstallsBackend>
  spawn: (command: string, args: string[], options: { cwd?: string; env: NodeJS.ProcessEnv; shell: boolean }) => InstallsChild
  hermesHome: string
}

/**
 * Runs one `hermes installs ...` command through the app's own backend runtime: the bundled
 * payload Python on a packaged app, the checkout under `npm run dev`, the pinned Hermes on Nix,
 * or the managed install. The CLI decides which install is "current" from the interpreter that
 * runs it, so a different Python would mark the wrong install as running and enable Remove on it.
 * A timeout kills the child, because the UI would otherwise report a failure for a removal that
 * keeps running.
 */
export function createInstallsRunner({
  resolveBackend,
  spawn,
  hermesHome
}: InstallsRunnerDeps): (args: string[]) => Promise<InstallsRunOutcome> {
  return async (args: string[]): Promise<InstallsRunOutcome> => {
    const backend = await resolveBackend(args)

    if (!backend.command) {
      return { code: null, stdout: '', stderr: backend.label ?? 'no Hermes runtime is installed' }
    }

    const command = backend.command

    return new Promise<InstallsRunOutcome>(resolve => {
      let stdout = ''
      let stderr = ''
      let settled = false
      let timer: ReturnType<typeof setTimeout> | undefined

      const done = (outcome: InstallsRunOutcome): void => {
        if (!settled) {
          settled = true
          clearTimeout(timer)
          resolve(outcome)
        }
      }

      try {
        const child = spawn(command, backend.args, {
          cwd: backend.root,
          env: { ...process.env, ...backend.env, HERMES_HOME: hermesHome, NO_COLOR: '1' },
          shell: Boolean(backend.shell)
        })

        child.stdout?.on('data', (chunk: Buffer | string): void => {
          stdout += chunk.toString()
        })
        child.stderr?.on('data', (chunk: Buffer | string): void => {
          stderr += chunk.toString()
        })
        child.on('error', (error: Error): void => done({ code: null, stdout, stderr: error.message }))
        child.on('exit', (code: null | number): void => done({ code, stdout, stderr }))
        timer = setTimeout((): void => {
          child.kill()
          done({ code: null, stdout, stderr: 'timeout' })
        }, runTimeoutMs(args))
      } catch (error) {
        done({ code: null, stdout, stderr: error instanceof Error ? error.message : String(error) })
      }
    })
  }
}

export interface InstallsIpcDeps {
  ipcMain: {
    handle: (channel: string, handler: (event: unknown, payload?: unknown) => Promise<unknown>) => void
  }
  /** Spawn `python -m hermes_cli.main <args…>` through the backend python. */
  runInstalls: (args: string[]) => Promise<InstallsRunOutcome>
  /** Where the renderer pulls the boot notice from. */
  notice: Pick<InstallsNotice, 'take'>
  /** Debug sink for swallowed failures (desktop.log). */
  logDebug: (message: string) => void
}

/** The id the renderer sent: `{ id }` from the preload, or a bare value. Validation happens in `removeArgs`. */
function requestedInstallId(payload: unknown): unknown {
  return payload !== null && typeof payload === 'object' && 'id' in payload ? payload.id : payload
}

export function registerInstallsIpc({ ipcMain, runInstalls, notice, logDebug }: InstallsIpcDeps): void {
  ipcMain.handle('hermes:installs:list', async (): Promise<null | InstallsListResult> => {
    try {
      const outcome = await runInstalls(listArgs())

      if (outcome.code !== 0) {
        return null
      }

      return parseInstallsList(outcome.stdout)
    } catch (error) {
      logDebug(`[installs] list failed: ${error instanceof Error ? error.message : String(error)}`)

      return null
    }
  })

  ipcMain.handle(
    'hermes:installs:remove',
    async (_event: unknown, payload?: unknown): Promise<InstallsRemoveResult> => {
      const rawId = requestedInstallId(payload)
      const args = removeArgs(rawId)

      if (!args) {
        return { ok: false, error: 'invalid-id', message: `Not a valid install id: ${String(rawId ?? '')}` }
      }

      try {
        const outcome = await runInstalls(args)

        if (outcome.code !== 0) {
          const detail = outcome.stderr.trim() || outcome.stdout.trim() || `exit code ${String(outcome.code)}`

          return { ok: false, error: 'remove-failed', message: detail }
        }

        return { ok: true }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        logDebug(`[installs] remove failed: ${message}`)

        return { ok: false, error: 'spawn-failed', message }
      }
    }
  )

  ipcMain.handle('hermes:installs:take-notice', (): Promise<InstallsNoticePayload | null> => {
    return Promise.resolve(notice.take())
  })

  ipcMain.handle('hermes:installs:dismiss', async (): Promise<{ ok: boolean }> => {
    try {
      const outcome = await runInstalls(dismissArgs())

      return { ok: outcome.code === 0 }
    } catch (error) {
      logDebug(`[installs] dismiss failed: ${error instanceof Error ? error.message : String(error)}`)

      return { ok: false }
    }
  })
}

export interface InstallsNoticePayload {
  /** Other installs the user can remove from the Settings page. */
  count: number
  /** The count can be low (Windows Store packages are not counted at launch). */
  partial: boolean
}

export interface InstallsNoticeDeps {
  runInstalls: (args: string[]) => Promise<InstallsRunOutcome>
  /** Tell the windows a notice is waiting. They pull it with `take`. */
  signalNotice: () => void
  logDebug: (message: string) => void
}

export interface InstallsNotice {
  /**
   * One background check: list once after the backend is ready and, when other
   * installs exist and the user has not dismissed the notice, keep a notice
   * for the renderer and signal the windows. Stops checking once it has found
   * one. Never delays startup: the caller fires this without awaiting.
   */
  check: (deps: InstallsNoticeDeps) => void
  /** The waiting notice, once. The renderer cannot be listening when main finds it, so it pulls. */
  take: () => InstallsNoticePayload | null
}

/** State lives in the returned object, so each app launch (and each test) gets its own. */
export function createInstallsNotice(): InstallsNotice {
  let found = false
  let pending: InstallsNoticePayload | null = null

  return {
    check: ({ runInstalls, signalNotice, logDebug }: InstallsNoticeDeps): void => {
      if (found) {
        return
      }

      void runInstalls(listArgs())
        .then(outcome => {
          if (outcome.code !== 0) {
            logDebug(`[installs] boot notice list exited ${String(outcome.code)}`)

            return
          }

          const list = parseInstallsList(outcome.stdout)

          if (!list || !shouldShowBootNotice(list)) {
            return
          }

          found = true
          pending = { count: list.notice.count, partial: list.notice.partial }
          signalNotice()
        })
        .catch((error: unknown) => {
          logDebug(`[installs] boot notice check failed: ${error instanceof Error ? error.message : String(error)}`)
        })
    },
    take: (): InstallsNoticePayload | null => {
      const notice = pending
      pending = null

      return notice
    }
  }
}
