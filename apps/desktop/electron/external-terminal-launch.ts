// The side effects of handing Hermes to the user's OWN terminal emulator:
// write the launcher script, find an emulator, spawn it detached. The pure
// pieces (argv, quoting, emulator choice) live in external-terminal.ts.
//
// The desktop's runtime is usually a venv Python invoked as
// `python -m hermes_cli.main`, so the caller resolves the SAME backend the app
// itself launches and we carry its argv + PYTHONPATH into a launcher script
// rather than hoping a `hermes` exists on the user's interactive PATH.
// Resolution only — never ensureRuntime(), which would kick off a first-run
// install from a click; an unresolved runtime is reported instead.

import { spawn } from 'node:child_process'
import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

import {
  buildTerminalScript,
  resolveTerminalLaunch,
  terminalScriptEnv,
  terminalScriptExtension
} from './external-terminal'

export interface ExternalTerminalDeps {
  findOnPath: (command: string) => null | string
  hermesHome: string
  log: (line: string) => void
  resolveBackend: (
    backendArgs: string[]
  ) => Promise<{ args: string[]; command: null | string; env: Record<string, string | undefined> }>
  sanitizeCwd: (cwd: unknown) => { cwd: string }
  scriptDir: () => string
}

/** Run Hermes with `backendArgs` in a new terminal window; `label` only names it in the log. */
export async function openHermesInTerminal(
  backendArgs: string[],
  {
    cwd,
    holdOnFailure,
    label,
    prepareArgs
  }: { cwd?: unknown; holdOnFailure?: boolean; label: string; prepareArgs?: null | string[] },
  deps: ExternalTerminalDeps
): Promise<{ ok: boolean; error?: string }> {
  try {
    const backend = await deps.resolveBackend(backendArgs)
    const prepare = prepareArgs ? await deps.resolveBackend(prepareArgs) : null

    if (!backend.command) {
      return { ok: false, error: 'Hermes is not installed yet' }
    }

    const scriptCwd = deps.sanitizeCwd(cwd).cwd
    const scriptDir = deps.scriptDir()
    fs.mkdirSync(scriptDir, { recursive: true })

    const scriptPath = path.join(
      scriptDir,
      `hermes-${crypto.randomBytes(6).toString('hex')}${terminalScriptExtension()}`
    )

    fs.writeFileSync(
      scriptPath,
      buildTerminalScript({
        args: backend.args,
        before: prepare?.command ? { args: prepare.args, command: prepare.command } : undefined,
        command: backend.command,
        cwd: scriptCwd,
        env: terminalScriptEnv(backend.env, deps.hermesHome),
        holdOnFailure
      }),
      { mode: 0o700 }
    )

    const launch = resolveTerminalLaunch({ findOnPath: deps.findOnPath, scriptPath })

    if (!launch) {
      return { ok: false, error: 'No terminal emulator found' }
    }

    deps.log(`[terminal] opening ${label} via ${launch.command}`)

    // Detached + unref'd: the terminal window outlives the desktop app, and
    // never inherits our stdio (a closed pipe would kill the TUI).
    const child = spawn(launch.command, launch.args, { detached: true, stdio: 'ignore' })
    child.unref()

    return { ok: true }
  } catch (error) {
    deps.log(`[terminal] open in terminal failed: ${error.message}`)

    return { ok: false, error: error.message }
  }
}
