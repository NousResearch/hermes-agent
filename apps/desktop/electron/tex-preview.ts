import { type ChildProcess, spawn } from 'node:child_process'
import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

const MAX_LOG_BYTES = 1024 * 1024
const DEFAULT_TIMEOUT_MS = 60_000
const ALLOWED_PROGRAMS = new Set(['lualatex', 'pdflatex', 'xelatex'])

export interface TexDiagnostic {
  file?: string
  line?: number
  message: string
}

export interface TexCompileResult {
  diagnostics: TexDiagnostic[]
  durationMs: number
  engine?: string
  log: string
  pdfPath?: string
  rootPath: string
  status: 'error' | 'missing-engine' | 'success'
}

interface RunResult {
  code: number | null
  log: string
  timedOut: boolean
}

const TEX_ENV_KEYS = [
  'HOME',
  'LANG',
  'LC_ALL',
  'LC_CTYPE',
  'LOCALAPPDATA',
  'PATH',
  'PATHEXT',
  'Path',
  'ProgramFiles',
  'SystemRoot',
  'TEMP',
  'TMP',
  'TMPDIR',
  'USERPROFILE',
  'windir'
] as const

export function texSubprocessEnvironment(env: NodeJS.ProcessEnv = process.env): NodeJS.ProcessEnv {
  const safe: NodeJS.ProcessEnv = {}

  for (const key of TEX_ENV_KEYS) {
    if (env[key] !== undefined) {
      safe[key] = env[key]
    }
  }

  for (const [key, value] of Object.entries(env)) {
    if (key.startsWith('TEXMF') && value !== undefined) {
      safe[key] = value
    }
  }

  return { ...safe, max_print_line: '1000', openin_any: 'p', openout_any: 'p' }
}

function executableExtensions() {
  return process.platform === 'win32'
    ? String(process.env.PATHEXT || '.EXE;.CMD;.BAT')
        .split(';')
        .filter(Boolean)
    : ['']
}

export function findTexExecutable(name: string, env = process.env): string | null {
  const suffixes = executableExtensions()
  const home = env.HOME || env.USERPROFILE || ''

  const common =
    process.platform === 'win32'
      ? [
          path.join(env.LOCALAPPDATA || '', 'Programs', 'MiKTeX', 'miktex', 'bin', 'x64'),
          path.join(env.ProgramFiles || '', 'MiKTeX', 'miktex', 'bin', 'x64')
        ]
      : ['/Library/TeX/texbin', '/opt/homebrew/bin', '/usr/local/bin', '/usr/bin', path.join(home, '.local', 'bin')]

  const dirs = [...String(env.PATH || env.Path || '').split(path.delimiter), ...common].filter(Boolean)

  for (const dir of dirs) {
    for (const suffix of suffixes) {
      const candidate = path.join(dir, `${name}${suffix}`)

      try {
        fs.accessSync(candidate, fs.constants.X_OK)

        return candidate
      } catch {
        // Continue through the explicit resolver ladder.
      }
    }
  }

  return null
}

export function texDirectives(source: string, sourcePath: string) {
  const header = source.split(/\r?\n/, 24).join('\n')
  const rootValue = /^\s*%\s*!TeX\s+root\s*=\s*(.+?)\s*$/im.exec(header)?.[1]
  const programValue = /^\s*%\s*!TeX\s+program\s*=\s*([\w-]+)\s*$/im.exec(header)?.[1]?.toLowerCase()

  return {
    program: programValue && ALLOWED_PROGRAMS.has(programValue) ? programValue : undefined,
    rootPath: rootValue ? path.resolve(path.dirname(sourcePath), rootValue.trim()) : sourcePath
  }
}

export function parseTexDiagnostics(log: string): TexDiagnostic[] {
  const diagnostics: TexDiagnostic[] = []

  for (const line of log.split(/\r?\n/)) {
    const located = /^(.+?):(\d+):\s*(.+)$/.exec(line)

    if (located) {
      diagnostics.push({ file: located[1], line: Number(located[2]), message: located[3].trim() })
    } else if (/^!\s+/.test(line)) {
      diagnostics.push({ message: line.replace(/^!\s+/, '').trim() })
    }

    if (diagnostics.length >= 50) {
      break
    }
  }

  return diagnostics
}

function missingEngineGuidance() {
  if (process.platform === 'darwin') {
    return 'No supported TeX engine was found. Install MacTeX (for example: brew install --cask mactex-no-gui) and restart Hermes.'
  }

  if (process.platform === 'win32') {
    return 'No supported TeX engine was found. Install MiKTeX or TeX Live, then restart Hermes so the desktop can refresh PATH.'
  }

  return 'No supported TeX engine was found. Install latexmk plus XeLaTeX/LuaLaTeX/pdfLaTeX (from TeX Live), or install Tectonic, then restart Hermes.'
}

function appendBounded(current: string, chunk: unknown) {
  if (Buffer.byteLength(current) >= MAX_LOG_BYTES) {
    return current
  }

  const remaining = MAX_LOG_BYTES - Buffer.byteLength(current)

  return current + Buffer.from(String(chunk)).subarray(0, remaining).toString()
}

function terminateProcess(child: ChildProcess) {
  if (!child.pid) {
    return
  }

  if (process.platform === 'win32') {
    spawn('taskkill', ['/pid', String(child.pid), '/t', '/f'], { stdio: 'ignore', windowsHide: true }).unref()
  } else {
    try {
      process.kill(-child.pid, 'SIGKILL')
    } catch {
      child.kill('SIGKILL')
    }
  }
}

async function run(
  command: string,
  args: string[],
  cwd: string,
  signal: AbortSignal,
  timeoutMs: number
): Promise<RunResult> {
  return new Promise((resolve, reject) => {
    let log = ''
    let timedOut = false
    let settled = false

    const child = spawn(command, args, {
      cwd,
      detached: process.platform !== 'win32',
      env: texSubprocessEnvironment(),
      stdio: ['ignore', 'pipe', 'pipe'],
      windowsHide: true
    })

    const finish = (result: RunResult) => {
      if (settled) {
        return
      }

      settled = true
      clearTimeout(timer)
      signal.removeEventListener('abort', abort)
      resolve(result)
    }

    const abort = () => {
      terminateProcess(child)
      finish({ code: null, log, timedOut: false })
    }

    const timer = setTimeout(() => {
      timedOut = true
      terminateProcess(child)
    }, timeoutMs)

    child.stdout?.on('data', chunk => {
      log = appendBounded(log, chunk)
    })
    child.stderr?.on('data', chunk => {
      log = appendBounded(log, chunk)
    })
    child.once('error', reject)
    child.once('close', code => finish({ code, log, timedOut }))
    signal.addEventListener('abort', abort, { once: true })

    if (signal.aborted) {
      abort()
    }
  })
}

export async function compileTexPreview({
  outputRoot,
  requestId,
  signal,
  program: requestedProgram,
  sourcePath,
  timeoutMs = DEFAULT_TIMEOUT_MS
}: {
  outputRoot: string
  requestId: string
  signal: AbortSignal
  program?: string
  sourcePath: string
  timeoutMs?: number
}): Promise<TexCompileResult> {
  const started = Date.now()
  const rootPath = sourcePath

  const outputDir = path.join(
    outputRoot,
    crypto.createHash('sha256').update(`${sourcePath}\0${requestId}`).digest('hex')
  )

  await fs.promises.mkdir(outputDir, { recursive: true })
  const requested = requestedProgram && ALLOWED_PROGRAMS.has(requestedProgram) ? requestedProgram : undefined
  const latexmk = findTexExecutable('latexmk')
  const preferredPrograms = requested ? [requested] : ['xelatex', 'lualatex', 'pdflatex']
  const program = preferredPrograms.find(name => findTexExecutable(name))
  const tectonic = requested ? null : findTexExecutable('tectonic')
  let engine: string | undefined
  let result: RunResult | undefined

  if (latexmk && program) {
    engine = `latexmk/${program}`
    const mode = program === 'xelatex' ? '-xelatex' : program === 'lualatex' ? '-lualatex' : '-pdf'
    result = await run(
      latexmk,
      [
        '-norc',
        mode,
        '-recorder',
        `-outdir=${outputDir}`,
        '-latexoption=-no-shell-escape',
        '-latexoption=-interaction=nonstopmode',
        '-latexoption=-halt-on-error',
        '-latexoption=-file-line-error',
        rootPath
      ],
      path.dirname(rootPath),
      signal,
      timeoutMs
    )
  } else if (tectonic) {
    engine = 'tectonic'
    result = await run(
      tectonic,
      ['--keep-logs', '--outdir', outputDir, '--untrusted', rootPath],
      path.dirname(rootPath),
      signal,
      timeoutMs
    )
  } else if (program) {
    engine = program
    const executable = findTexExecutable(program)!

    const args = [
      '-no-shell-escape',
      '-interaction=nonstopmode',
      '-halt-on-error',
      '-file-line-error',
      `-output-directory=${outputDir}`,
      rootPath
    ]

    result = await run(executable, args, path.dirname(rootPath), signal, timeoutMs)

    if (result.code === 0 && !signal.aborted) {
      const second = await run(executable, args, path.dirname(rootPath), signal, timeoutMs)
      result = { ...second, log: appendBounded(result.log, second.log) }
    }
  }

  if (!result) {
    await fs.promises.rm(outputDir, { force: true, recursive: true }).catch(() => {})

    return {
      diagnostics: [],
      durationMs: Date.now() - started,
      log: missingEngineGuidance(),
      rootPath,
      status: 'missing-engine'
    }
  }

  const pdfPath = path.join(outputDir, `${path.basename(rootPath, path.extname(rootPath))}.pdf`)
  const success = result.code === 0 && !result.timedOut && !signal.aborted && fs.existsSync(pdfPath)
  const log = result.timedOut ? appendBounded(result.log, '\nCompilation timed out after 60 seconds.\n') : result.log

  if (!success) {
    await fs.promises.rm(outputDir, { force: true, recursive: true }).catch(() => {})
  }

  return {
    diagnostics: parseTexDiagnostics(log),
    durationMs: Date.now() - started,
    engine,
    log,
    pdfPath: success ? pdfPath : undefined,
    rootPath,
    status: success ? 'success' : 'error'
  }
}
