// The embedded terminal's PTY host: shell resolution, env scrubbing, session
// registry, and the hermes:terminal:* IPC surface. Extracted from main.ts; the
// factory owns the session map and returns the dispose helpers main.ts needs
// for SSH teardown. findOnPath / logging / connection routing stay injected.
import { execFile } from 'node:child_process'
import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

import { app, ipcMain } from 'electron'
import nodePty from 'node-pty'

import { resolveTerminalConnectionForSender } from './connection-apply'
import { ensureSpawnHelperExecutable } from './spawn-helper-perms'
import { buildInteractiveSshArgs } from './ssh-connection'
import { buildWindowsInteractiveCommand } from './windows-remote-lifecycle'

export interface TerminalIpcDeps {
  isWindows: boolean
  findOnPath: (command: string) => null | string
  rememberLog: (line: string) => void
  activeSshTerminalTarget: (webContentsId: number) => unknown
  ensureBackend: (webContentsId: number) => Promise<unknown>
  getSshConnectionState: (scope: string) => undefined | { remotePlatform?: string }
}

export interface TerminalIpcApi {
  disposeTerminalSession: (id: string) => boolean
  disposeTerminalSessionsForSshScope: (scope: string) => void
  disposeAllTerminalSessions: () => void
}

export function registerTerminalIpc({
  isWindows,
  findOnPath,
  rememberLog,
  activeSshTerminalTarget,
  ensureBackend,
  getSshConnectionState
}: TerminalIpcDeps): TerminalIpcApi {
  const terminalSessions = new Map()

  function isExecutableFile(filePath) {
    if (!filePath || !path.isAbsolute(filePath)) {
      return false
    }

    try {
      fs.accessSync(filePath, fs.constants.X_OK)

      return true
    } catch {
      return false
    }
  }

  function posixShellSpec(shellPath) {
    const shellName = path.basename(shellPath)
    const interactiveArgs = shellName.includes('zsh') || shellName.includes('bash') ? ['-il'] : ['-i']

    return { args: interactiveArgs, command: shellPath, name: shellName }
  }

  // Windows PowerShell 5.1 ships at a fixed System32 path on every Windows box;
  // prefer it only after PowerShell 7+ (`pwsh`).
  function windowsPowerShellPath() {
    const systemRoot = process.env.SystemRoot || process.env.windir || 'C:\\Windows'
    const builtin = path.join(systemRoot, 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe')

    return isExecutableFile(builtin) ? builtin : findOnPath('powershell.exe')
  }

  // Map a resolved shell path to its spawn spec, picking interactive flags by
  // family: PowerShell drops its logo banner (so the prompt sits flush like the
  // POSIX shells), cmd needs nothing, and everything else (zsh/bash/fish/sh…)
  // gets POSIX interactive-login flags.
  function shellSpecFor(shellPath) {
    const name = path.basename(shellPath).toLowerCase()

    if (name.startsWith('pwsh') || name.startsWith('powershell')) {
      return { args: ['-NoLogo'], command: shellPath, name }
    }

    if (name.startsWith('cmd')) {
      return { args: [], command: shellPath, name }
    }

    return posixShellSpec(shellPath)
  }

  // Best installed Windows shell: PowerShell 7+ (`pwsh`), then Windows PowerShell
  // 5.1, then comspec/cmd.exe as the universal fallback.
  function windowsShellSpec() {
    const command =
      findOnPath('pwsh.exe') || findOnPath('pwsh') || windowsPowerShellPath() || process.env.COMSPEC || 'cmd.exe'

    return shellSpecFor(command)
  }

  // Resolve the interactive shell for the embedded terminal: an explicit user
  // override wins, otherwise auto-detect the best one installed for the platform.
  function terminalShellCommand() {
    // HERMES_DESKTOP_SHELL is the cross-platform escape hatch (a path or a bare
    // name on PATH); $SHELL is honored on POSIX, where it's the user's canonical
    // choice, but ignored on Windows, where it's usually a stray MSYS/Git path
    // node-pty can't spawn natively.
    const override = (process.env.HERMES_DESKTOP_SHELL || (isWindows ? '' : process.env.SHELL) || '').trim()

    if (override) {
      const resolved = isExecutableFile(override) ? override : findOnPath(override)

      if (resolved) {
        return shellSpecFor(resolved)
      }
    }

    if (isWindows) {
      return windowsShellSpec()
    }

    const shellPath = ['/bin/zsh', '/bin/bash', '/bin/sh'].find(candidate => isExecutableFile(candidate))

    return posixShellSpec(shellPath || '/bin/sh')
  }

  function safeTerminalCwd(cwd) {
    const candidate = path.resolve(String(cwd || app.getPath('home')))

    try {
      const stat = fs.statSync(candidate)

      return stat.isDirectory() ? candidate : path.dirname(candidate)
    } catch {
      return app.getPath('home')
    }
  }

  function terminalShellEnv() {
    const env = { ...process.env }

    // Electron is commonly launched through `npm run dev`; do not leak npm's
    // managed prefix into a user's interactive shell (nvm/proto warn loudly).
    for (const key of Object.keys(env)) {
      if (key === 'npm_config_prefix' || key.startsWith('npm_config_') || key.startsWith('npm_package_')) {
        delete env[key]
      }
    }

    // Strip color/theme-detection vars that ride along when Electron is launched
    // from a non-tty agent shell (Cursor's runner sets NO_COLOR/FORCE_COLOR=0
    // /TERM=dumb; some terminals set COLORFGBG which would flip Hermes' TUI into
    // light-mode). Our PTY is a real xterm-compat terminal — force truecolor.
    delete env.NO_COLOR
    delete env.FORCE_COLOR
    delete env.COLORFGBG

    env.COLORTERM = 'truecolor'
    env.LC_CTYPE = env.LC_CTYPE || 'UTF-8'
    env.TERM = 'xterm-256color'
    env.TERM_PROGRAM = 'Hermes'
    env.TERM_PROGRAM_VERSION = app.getVersion()

    // Let a hermes/--tui launched in this pane know it's embedded in the desktop
    // GUI (build_environment_hints surfaces this). Distinct from HERMES_DESKTOP,
    // which marks the agent *backend* and gates cron/gateway behavior.
    env.HERMES_DESKTOP_TERMINAL = '1'

    return env
  }

  function terminalChannel(id, suffix) {
    return `hermes:terminal:${id}:${suffix}`
  }

  // Best-effort read of a live PTY child's current working directory so a
  // reopened tab can restart the shell where the user last `cd`'d, instead of the
  // tab's original launch dir. Shell-agnostic (no prompt/OSC config needed) on
  // POSIX; Windows has no cheap per-process cwd query without a native module, so
  // it returns null and the caller falls back to the launch cwd.
  function readProcessCwd(pid) {
    return new Promise(resolve => {
      if (!Number.isInteger(pid) || pid <= 0) {
        resolve(null)

        return
      }

      if (process.platform === 'linux') {
        fs.promises
          .readlink(`/proc/${pid}/cwd`)
          .then(target => resolve(target || null))
          .catch(() => resolve(null))

        return
      }

      if (process.platform === 'darwin') {
        // lsof ships with macOS; -Fn emits the cwd fd's path on an `n<path>` line.
        execFile('lsof', ['-a', '-p', String(pid), '-d', 'cwd', '-Fn'], { timeout: 2000 }, (err, stdout) => {
          if (err) {
            resolve(null)

            return
          }

          const line = String(stdout || '')
            .split('\n')
            .find(entry => entry.startsWith('n'))

          resolve(line ? line.slice(1) : null)
        })

        return
      }

      resolve(null)
    })
  }

  function disposeTerminalSession(id: string) {
    const sessionInfo = terminalSessions.get(id)

    if (!sessionInfo) {
      return false
    }

    terminalSessions.delete(id)

    try {
      sessionInfo.pty.kill()
    } catch {
      // Process may already be gone.
    }

    return true
  }

  // SSH teardown: close every pane whose PTY rode the disconnected tunnel.
  function disposeTerminalSessionsForSshScope(scope: string) {
    for (const [id, info] of [...terminalSessions.entries()]) {
      if (info.sshScope === scope) {
        disposeTerminalSession(id)
      }
    }
  }

  // App shutdown: kill every open PTY before environment teardown.
  function disposeAllTerminalSessions() {
    for (const id of [...terminalSessions.keys()]) {
      disposeTerminalSession(id)
    }
  }

  // node-pty's published tarball ships the POSIX `spawn-helper` without an exec
  // bit; the dev flow resolves node-pty straight from node_modules (nothing
  // chmods it there), so the first terminal spawn dies with `posix_spawnp
  // failed`. Restore the bit once, lazily, right before the first spawn. Packaged
  // builds already stage an executable copy, so this is a no-op there.
  let _spawnHelperEnsured = false

  function ensureNodePtySpawnHelper() {
    if (_spawnHelperEnsured || isWindows) {
      return
    }

    _spawnHelperEnsured = true

    try {
      const nodePtyRoot = path.dirname(require.resolve('node-pty/package.json'))
      const { fixed, errors } = ensureSpawnHelperExecutable(nodePtyRoot)

      for (const helperPath of fixed) {
        rememberLog(`[terminal] restored +x on node-pty spawn-helper: ${helperPath}`)
      }

      for (const failure of errors) {
        rememberLog(`[terminal] could not chmod spawn-helper ${failure.path}: ${failure.error}`)
      }
    } catch (error) {
      rememberLog(
        `[terminal] spawn-helper exec check skipped: ${error instanceof Error ? error.message : String(error)}`
      )
    }
  }

  ipcMain.handle('hermes:terminal:start', async (event, payload = {}) => {
    ensureNodePtySpawnHelper()

    const id = crypto.randomUUID()
    const { args, command, name } = terminalShellCommand()
    const cwd = safeTerminalCwd(payload?.cwd)
    const cols = Math.max(2, Number.parseInt(String(payload?.cols || 80), 10) || 80)
    const rows = Math.max(2, Number.parseInt(String(payload?.rows || 24), 10) || 24)

    const sshTarget = await resolveTerminalConnectionForSender(
      event.sender.id,
      activeSshTerminalTarget,
      ensureBackend
    )
    const remote = Boolean(sshTarget)
    const remoteState = remote ? getSshConnectionState(sshTarget.scope) : null

    const remoteCommand =
      remoteState?.remotePlatform === 'Windows'
        ? buildWindowsInteractiveCommand(String(payload?.cwd || '').trim())
        : undefined

    const ptyProcess = remote
      ? nodePty.spawn(
          process.platform === 'win32'
            ? path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'OpenSSH', 'ssh.exe')
            : 'ssh',
          buildInteractiveSshArgs(sshTarget.ssh, String(payload?.cwd || '').trim(), undefined, remoteCommand),
          { cols, cwd: app.getPath('home'), env: terminalShellEnv(), name: 'xterm-256color', rows }
        )
      : nodePty.spawn(command, args, { cols, cwd, env: terminalShellEnv(), name: 'xterm-256color', rows })

    terminalSessions.set(id, {
      pty: ptyProcess,
      webContentsId: event.sender.id,
      ...(remote ? { sshScope: sshTarget.scope, remoteCwd: String(payload?.cwd || '') } : {})
    })

    const send = (suffix, payload) => {
      if (event.sender.isDestroyed()) {
        return
      }

      event.sender.send(terminalChannel(id, suffix), payload)
    }

    ptyProcess.onData(data => send('data', data))
    ptyProcess.onExit(({ exitCode, signal }) => {
      terminalSessions.delete(id)
      send('exit', { code: exitCode, signal: signal || null })
    })
    event.sender.once('destroyed', () => disposeTerminalSession(id))

    return { cwd: remote ? null : cwd, id, shell: remote ? 'ssh' : name }
  })

  ipcMain.handle('hermes:terminal:write', (_event, id, data) => {
    const sessionInfo = terminalSessions.get(String(id || ''))

    if (!sessionInfo) {
      return false
    }

    sessionInfo.pty.write(String(data || ''))

    return true
  })

  ipcMain.handle('hermes:terminal:resize', (_event, id, size = {}) => {
    const sessionInfo = terminalSessions.get(String(id || ''))

    if (!sessionInfo) {
      return false
    }

    const cols = Math.max(2, Number.parseInt(String(size?.cols || 80), 10) || 80)
    const rows = Math.max(2, Number.parseInt(String(size?.rows || 24), 10) || 24)

    sessionInfo.pty.resize(cols, rows)

    return true
  })
  ipcMain.handle('hermes:terminal:cwd', async (_event, id) => {
    const sessionInfo = terminalSessions.get(String(id || ''))

    if (!sessionInfo) {
      return null
    }

    return sessionInfo.sshScope !== undefined ? null : readProcessCwd(sessionInfo.pty.pid)
  })

  ipcMain.handle('hermes:terminal:dispose', (_event, id) => disposeTerminalSession(String(id || '')))

  return { disposeTerminalSession, disposeTerminalSessionsForSshScope, disposeAllTerminalSessions }
}
