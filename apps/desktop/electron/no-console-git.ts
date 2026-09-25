// Electron is a GUI-subsystem process, so a direct git.exe spawn allocates its
// own console even when Node's windowsHide is set. windowsHide is SW_HIDE and,
// from a console-less parent, does not stop that flash (hermes_cli/_subprocess_compat.py).
//
// The flag that does is CREATE_NO_WINDOW (0x08000000). A console-subsystem
// python.exe started with that flag owns one hidden console; git is then
// spawned from that host with the same flag, and the git argv is forwarded
// unchanged. simple-git cannot take a creation flag, so its binary tuple is
// [python, this host script].

import { spawn, type SpawnOptions } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

export const CREATE_NO_WINDOW = 0x08000000

export const NO_CONSOLE_GIT_SCRIPT = `import json, os, subprocess, sys
git = json.loads(os.environ["HERMES_GIT_ARGV0"])
argv = [git, *sys.argv[1:]]
kwargs = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE}
if sys.platform == "win32" or os.environ.get("HERMES_GIT_NO_CONSOLE") == "1":
    kwargs["creationflags"] = 0x08000000
if os.environ.get("HERMES_GIT_DRY_RUN") == "1":
    sys.stdout.write(json.dumps({"argv": argv, "creationflags": kwargs.get("creationflags", 0)}))
    raise SystemExit(0)
proc = subprocess.run(argv, **kwargs)
sys.stdout.buffer.write(proc.stdout or b"")
sys.stderr.buffer.write(proc.stderr or b"")
code = proc.returncode
raise SystemExit(code if isinstance(code, int) and code >= 0 else 1)
`

const PIPED_STDIO = ['ignore', 'pipe', 'pipe'] as const

export interface NoConsoleGitPlan {
  command: string
  args: string[]
  env: NodeJS.ProcessEnv
  windowsHide: boolean
  stdio: ['ignore', 'pipe', 'pipe']
  creationFlags: number
}

export interface NoConsoleGitHost {
  isWindows: true
  pythonBin: string
  scriptPath: string
}

let configuredRoots: string[] = []

export function setNoConsoleGitRoots(roots: Array<string | null | undefined | false>) {
  configuredRoots = roots.filter((root): root is string => Boolean(root))
}

function isPythonW(candidate: string) {
  return path.win32.basename(candidate).toLowerCase() === 'pythonw.exe'
}

function isWindowsAppsStub(candidate: string) {
  return candidate.replace(/\\/g, '/').toLowerCase().includes('/windowsapps/')
}

export function resolveNoConsolePython({
  isWindows,
  env = process.env,
  roots = configuredRoots,
  fileExists = fs.existsSync
}: {
  isWindows: boolean
  env?: NodeJS.ProcessEnv
  roots?: string[]
  fileExists?: (candidate: string) => boolean
}): string | null {
  if (!isWindows) {
    return null
  }

  const candidates: string[] = []
  const override = env.HERMES_DESKTOP_PYTHON

  if (override) {
    candidates.push(override)
  }

  const hermesRoot = env.HERMES_DESKTOP_HERMES_ROOT
  const searchRoots = hermesRoot ? [hermesRoot, ...roots] : roots

  for (const root of searchRoots) {
    if (!root) {
      continue
    }

    candidates.push(path.win32.join(root, '.venv', 'Scripts', 'python.exe'))
    candidates.push(path.win32.join(root, 'venv', 'Scripts', 'python.exe'))
  }

  for (const candidate of candidates) {
    if (!candidate || isPythonW(candidate) || isWindowsAppsStub(candidate)) {
      continue
    }

    try {
      if (fileExists(candidate)) {
        return candidate
      }
    } catch {
      continue
    }
  }

  return null
}

export function ensureNoConsoleGitScript(dir = os.tmpdir()) {
  const scriptPath = path.join(dir, 'hermes-no-console-git.py')

  try {
    if (fs.readFileSync(scriptPath, 'utf8') === NO_CONSOLE_GIT_SCRIPT) {
      return scriptPath
    }
  } catch {
    // Missing or unreadable: rewrite below.
  }

  fs.writeFileSync(scriptPath, NO_CONSOLE_GIT_SCRIPT, { encoding: 'utf8', mode: 0o600 })

  return scriptPath
}

export function windowsGitHost(isWindows = process.platform === 'win32'): NoConsoleGitHost | null {
  if (!isWindows) {
    return null
  }

  const pythonBin = resolveNoConsolePython({ isWindows: true })

  if (!pythonBin) {
    return null
  }

  try {
    return { isWindows: true, pythonBin, scriptPath: ensureNoConsoleGitScript() }
  } catch {
    return null
  }
}

export function noConsoleGitEnv(base: NodeJS.ProcessEnv | undefined, gitBin: string): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = {}

  for (const [key, value] of Object.entries(base || {})) {
    if (value !== undefined) {
      env[key] = value
    }
  }

  env.HERMES_GIT_ARGV0 = JSON.stringify(gitBin || 'git')

  if (!env.GIT_TERMINAL_PROMPT) {
    env.GIT_TERMINAL_PROMPT = '0'
  }

  return env
}

export function planNoConsoleGitSpawn({
  gitBin,
  args,
  isWindows = false,
  pythonBin = null,
  scriptPath = null,
  env
}: {
  gitBin: string
  args: string[]
  isWindows?: boolean
  pythonBin?: string | null
  scriptPath?: string | null
  env?: NodeJS.ProcessEnv
}): NoConsoleGitPlan {
  if (isWindows && pythonBin && scriptPath) {
    return {
      command: pythonBin,
      args: [scriptPath, ...args],
      env: noConsoleGitEnv(env, gitBin),
      windowsHide: true,
      stdio: [...PIPED_STDIO],
      creationFlags: CREATE_NO_WINDOW
    }
  }

  return {
    command: gitBin || 'git',
    args,
    env: { ...(env || {}) },
    windowsHide: Boolean(isWindows),
    stdio: [...PIPED_STDIO],
    creationFlags: 0
  }
}

export function simpleGitBinary(
  gitBin: string | undefined,
  host?: { isWindows: boolean; pythonBin?: string | null; scriptPath?: string | null } | null
): string | [string, string] {
  if (host?.isWindows && host.pythonBin && host.scriptPath) {
    return [host.pythonBin, host.scriptPath]
  }

  return gitBin || 'git'
}

export function hiddenGitSpawnSpec(
  gitBin: string,
  args: string[],
  options: SpawnOptions & { isWindows?: boolean } = {}
) {
  const isWindows = options.isWindows ?? process.platform === 'win32'
  const host = isWindows ? windowsGitHost(true) : null

  const plan = planNoConsoleGitSpawn({
    gitBin,
    args,
    isWindows,
    pythonBin: host?.pythonBin ?? null,
    scriptPath: host?.scriptPath ?? null,
    env: (options.env as NodeJS.ProcessEnv | undefined) || process.env
  })

  const { isWindows: _ignored, ...rest } = options

  return {
    command: plan.command,
    args: plan.args,
    creationFlags: plan.creationFlags,
    options: {
      ...rest,
      env: plan.env,
      windowsHide: plan.windowsHide || rest.windowsHide,
      stdio: rest.stdio ?? plan.stdio
    }
  }
}

export function execGit(
  gitBin: string,
  args: string[],
  options: { cwd?: string; env?: NodeJS.ProcessEnv; timeoutMs?: number } = {}
): Promise<{ code: number | null; stdout: string; stderr: string }> {
  const spec = hiddenGitSpawnSpec(gitBin, args, {
    cwd: options.cwd,
    env: options.env,
    stdio: ['ignore', 'pipe', 'pipe']
  })

  return new Promise((resolve, reject) => {
    const child = spawn(spec.command, spec.args, spec.options)
    let stdout = ''
    let stderr = ''
    let settled = false

    const finish = (error?: Error, code: number | null = child.exitCode) => {
      if (settled) {
        return
      }

      settled = true

      if (error) {
        reject(error)

        return
      }

      resolve({ code, stdout, stderr })
    }

    const timer = options.timeoutMs
      ? setTimeout(() => {
          child.kill()
          const error = new Error('git timed out') as NodeJS.ErrnoException & { stderr?: string }

          error.stderr = stderr
          finish(error)
        }, options.timeoutMs)
      : null

    child.stdout?.on('data', chunk => {
      stdout += chunk.toString()
    })
    child.stderr?.on('data', chunk => {
      stderr += chunk.toString()
    })
    child.once('error', error => {
      if (timer) {
        clearTimeout(timer)
      }

      finish(error)
    })
    child.once('close', code => {
      if (timer) {
        clearTimeout(timer)
      }

      finish(undefined, code)
    })
  })
}
