/**
 * bootstrap-runner.ts
 *
 * Drives apps/desktop's first-launch install of Hermes Agent by spawning
 * scripts/install.ps1 stage-by-stage and streaming progress events back to
 * the renderer.
 *
 * Wired from electron/main.ts:
 *   import { runBootstrap }from './bootstrap-runner'
 *   const result = await runBootstrap({
 *     installStamp,        // INSTALL_STAMP from main.ts (may be null in dev)
 *     activeRoot,          // ACTIVE_HERMES_ROOT
 *     sourceRepoRoot,      // SOURCE_REPO_ROOT (for dev install.ps1 lookup)
 *     hermesHome,          // HERMES_HOME
 *     logRoot,             // HERMES_HOME/logs
 *     emit: ev => {...}    // event sink (sender.send or similar)
 *   })
 *
 * Emits events with shape:
 *   { type: 'manifest',  stages: [{name, title, category, needs_user_input}, ...] }
 *   { type: 'stage',     name, state: 'running'|'succeeded'|'skipped'|'failed',
 *                        json?, durationMs?, error? }
 *   { type: 'log',       stage?, line, stream: 'stdout'|'stderr' } // one installer line, escapes stripped
 *   { type: 'complete',  marker: <written marker payload> }
 *   { type: 'failed',    stage?, error }     // bootstrap aborted
 *
 * Resolves with the same shape as the final 'complete' or 'failed' event so
 * callers can await either way.
 *
 * NOT implemented yet (deferred to Phase 1E / 1F):
 *   - User-facing retry / cancel from the renderer (event channels exist;
 *     no UI consumes them yet)
 */

import { execFileSync, spawn } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import fs from 'node:fs'
import type { IncomingMessage } from 'node:http'
import https from 'node:https'
import path from 'node:path'
import { Transform } from 'node:stream'
import { pipeline } from 'node:stream/promises'

// Relative, not `@hermes/shared/ansi`: the electron bundle is built by esbuild
// with no tsconfig path resolution (see scripts/bundle-electron-main.mjs).
import { stripAnsi } from '../../shared/src/ansi'

import { pathEnvKey, storeFirstPath } from './backend-env'
import { hiddenWindowsChildOptions } from './windows-child-options'

const IS_WINDOWS = process.platform === 'win32'

const STAMP_COMMIT_RE = /^[0-9a-f]{7,40}$/i
const FALLBACK_COMMIT_RE = /^0{7,40}$/
const FALLBACK_BRANCH = 'main'

function isPinnedCommit(commit) {
  return typeof commit === 'string' && STAMP_COMMIT_RE.test(commit) && !FALLBACK_COMMIT_RE.test(commit)
}

type ExecGitFn = (args: string[], cwd: string) => string
type ResolveHeadFn = (activeRoot: string | null | undefined) => string | null

/**
 * Read HEAD from a managed checkout. Used after bootstrap so fallback
 * (all-zero) install stamps still produce a marker that
 * isBootstrapComplete() accepts (pinnedCommit length >= 7).
 */
function resolveCheckoutHead(
  activeRoot: string | null | undefined,
  opts: { execGit?: ExecGitFn; gitBinary?: string } = {}
): string | null {
  if (!activeRoot) {
    return null
  }

  // Bare 'git' takes the first PATH hit, which can exist yet be unlaunchable
  // (Intel-only build on Apple Silicon); main.ts passes its probed binary.
  const run: ExecGitFn =
    opts.execGit ||
    ((args, cwd) =>
      execFileSync(opts.gitBinary || 'git', args, {
        cwd,
        encoding: 'utf8',
        stdio: ['ignore', 'pipe', 'ignore'],
        timeout: 15_000,
        ...hiddenWindowsChildOptions()
      }).trim())

  try {
    const sha = run(['-c', 'windows.appendAtomically=false', 'rev-parse', 'HEAD'], activeRoot)

    return isPinnedCommit(sha) ? sha : null
  } catch {
    return null
  }
}

/** Prefer a real pin already written by install.ps1's bootstrap-marker stage. */
function readExistingPinnedCommit(activeRoot: string | null | undefined): string | null {
  if (!activeRoot) {
    return null
  }

  try {
    const raw = fs.readFileSync(path.join(activeRoot, '.hermes-bootstrap-complete'), 'utf8')
    const parsed = JSON.parse(raw)

    return parsed && isPinnedCommit(parsed.pinnedCommit) ? parsed.pinnedCommit : null
  } catch {
    return null
  }
}

/**
 * Pick the commit to store on the bootstrap-complete marker.
 * The installed checkout owns source runtime identity: its live HEAD wins, so
 * a repair/update bootstrap reports the commit the checkout is actually at,
 * never the older commit baked into the packaged app. Packaged fallback stamps
 * (all-zero) are not real pins and never win.
 */
function resolveMarkerPinnedCommit(
  installStamp: { commit?: string; branch?: string | null } | null | undefined,
  activeRoot: string | null | undefined,
  opts: { resolveHead?: ResolveHeadFn } = {}
): string | null {
  const resolveHead = opts.resolveHead || resolveCheckoutHead

  const head = resolveHead(activeRoot)

  if (head) {
    return head
  }

  if (installStamp && isPinnedCommit(installStamp.commit)) {
    return installStamp.commit
  }

  return readExistingPinnedCommit(activeRoot)
}

/**
 * Map an install stamp to the same source identity the installer stages use.
 * Fresh installs use the packaged immutable SHA. Existing checkouts
 * intentionally ignore that old app pin and follow the branch, so the script
 * must follow the branch too or old installer code can drive a newer tree.
 * Non-git fallback stamps also follow the branch (#50823).
 */
function installRefForStamp(installStamp, { pinCommit = true } = {}) {
  if (pinCommit && installStamp && isPinnedCommit(installStamp.commit)) {
    return {
      ref: installStamp.commit,
      cacheKey: installStamp.commit,
      pinned: true
    }
  }

  const validStamp =
    installStamp &&
    typeof installStamp.commit === 'string' &&
    (isPinnedCommit(installStamp.commit) || FALLBACK_COMMIT_RE.test(installStamp.commit))

  if (validStamp) {
    const ref = installStamp.branch || FALLBACK_BRANCH

    return {
      ref,
      cacheKey: `branch-${String(ref).replace(/[^0-9A-Za-z._-]/g, '_')}`,
      pinned: false
    }
  }

  return null
}

// Stages flagged needs_user_input=true in the manifest are skipped by the
// runner (passed -NonInteractive to install.ps1, which the install script
// itself handles by emitting skipped=true frames). The renderer / 1E onboarding
// overlay takes over for those concerns (API keys, model, persona, gateway).
// We let install.ps1's own -NonInteractive logic drive this rather than
// filtering client-side -- single source of truth.

// ---------------------------------------------------------------------------
// install.ps1 source resolution
// ---------------------------------------------------------------------------

function installScriptName() {
  return process.platform === 'win32' ? 'install.ps1' : 'install.sh'
}

function installScriptKind(): 'powershell' | 'posix' {
  return process.platform === 'win32' ? 'powershell' : 'posix'
}

function resolveLocalInstallScript(sourceRepoRoot) {
  if (!sourceRepoRoot) {
    return null
  }

  const candidate = path.join(sourceRepoRoot, 'scripts', installScriptName())

  try {
    fs.accessSync(candidate, fs.constants.R_OK)

    return candidate
  } catch {
    return null
  }
}

function bootstrapCacheDir(hermesHome) {
  return path.join(hermesHome, 'bootstrap-cache')
}

function hasExistingGitCheckout(activeRoot) {
  if (!activeRoot) {
    return false
  }

  try {
    return fs.existsSync(path.join(activeRoot, '.git'))
  } catch {
    return false
  }
}

function cachedScriptPath(hermesHome, cacheKey) {
  return path.join(bootstrapCacheDir(hermesHome), `install-${cacheKey}.${process.platform === 'win32' ? 'ps1' : 'sh'}`)
}

// The site publishes main only: using it for a stamped commit or another
// branch would execute installer code belonging to a different checkout.
const SCRIPT_FALLBACK_STATUSES = new Set([403, 429])
const SCRIPT_REDIRECT_STATUSES = new Set([301, 302, 303, 307, 308])
const SCRIPT_DOWNLOAD_TIMEOUT_MS = 30_000
const SCRIPT_DOWNLOAD_IDLE_TIMEOUT_MS = 10_000
const SCRIPT_MAX_REDIRECTS = 5
const MAX_SCRIPT_BYTES = 2 * 1024 * 1024
const UTF8_BOM = Buffer.from([0xef, 0xbb, 0xbf])

interface ScriptDownloadOptions {
  timeoutMs?: number
  idleTimeoutMs?: number
  onFallback?: (url: string) => void
  signal?: AbortSignal
}

class ScriptDownloadError extends Error {
  constructor(
    message: string,
    readonly origin: 'remote' | 'local',
    readonly statusCode?: number,
    options?: ErrorOptions
  ) {
    super(message, options)
    this.name = 'ScriptDownloadError'
  }
}

function scriptUrls(ref: string): string[] {
  const encodedRef = ref.split('/').map(encodeURIComponent).join('/')
  const raw = `https://raw.githubusercontent.com/NousResearch/hermes-agent/${encodedRef}/scripts/${installScriptName()}`

  return ref === FALLBACK_BRANCH ? [raw, `https://hermes-agent.nousresearch.com/${installScriptName()}`] : [raw]
}

// Cached PowerShell files are executed with -File: Windows PowerShell 5.1
// needs a BOM to recognize UTF-8 (#67193). A shell shebang must stay BOM-free.
function prepareCachedScriptBytes(kind: 'powershell' | 'posix', bytes: Buffer): Buffer {
  return kind === 'powershell' && !bytes.subarray(0, UTF8_BOM.length).equals(UTF8_BOM)
    ? Buffer.concat([UTF8_BOM, bytes])
    : bytes
}

function validateScriptBody(bytes: Buffer, contentType: string): void {
  if (bytes.includes(0)) {
    throw new ScriptDownloadError('Installer body contains NUL bytes', 'remote')
  }

  let text: string

  try {
    text = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(bytes)
  } catch (err) {
    throw new ScriptDownloadError('Installer body is not UTF-8', 'remote', undefined, { cause: err })
  }

  const prefix = text.replace(/^\uFEFF/, '').trimStart().toLowerCase()

  if (!prefix.trim()) {
    throw new ScriptDownloadError('Empty installer body', 'remote')
  }

  if (/html|json/i.test(contentType) || prefix.startsWith('<!doctype html') || prefix.startsWith('<html')) {
    throw new ScriptDownloadError('Installer response is an error document', 'remote')
  }
}

async function fetchScriptOnce(url: string, destPath: string, opts: ScriptDownloadOptions): Promise<void> {
  // One deadline spans DNS, TLS, every redirect and the entire body. An idle
  // timer alone can be kept alive forever by a response that trickles bytes.
  const controller = new AbortController()
  const onAbort = () => controller.abort(opts.signal?.reason)

  if (opts.signal?.aborted) {
    onAbort()
  } else {
    opts.signal?.addEventListener('abort', onAbort, { once: true })
  }

  const deadline = setTimeout(() => controller.abort(), opts.timeoutMs ?? SCRIPT_DOWNLOAD_TIMEOUT_MS)
  let tempPath: string | undefined
  let tempCreated = false
  let response: IncomingMessage | undefined

  try {
    try {
      fs.mkdirSync(path.dirname(destPath), { recursive: true })
      tempPath = path.join(path.dirname(destPath), `.install-download-${randomUUID()}`)
    } catch (err) {
      throw new ScriptDownloadError(`Cannot prepare installer download: ${err.message}`, 'local', undefined, { cause: err })
    }

    let currentUrl = new URL(url)

    for (let redirects = 0; ; redirects++) {
      if (currentUrl.protocol !== 'https:') {
        throw new ScriptDownloadError(`Installer download requires HTTPS: ${currentUrl}`, 'remote')
      }

      response = await new Promise<IncomingMessage>((resolve, reject) => {
        const req = https.get(currentUrl, { signal: controller.signal }, resolve)
        req.setTimeout(opts.idleTimeoutMs ?? SCRIPT_DOWNLOAD_IDLE_TIMEOUT_MS, () => {
          req.destroy(new Error('Installer download timed out while waiting for data'))
        })
        req.on('error', reject)
      })

      if (!SCRIPT_REDIRECT_STATUSES.has(response.statusCode ?? 0)) {
        break
      }

      const location = response.headers.location
      response.destroy()

      if (!location || redirects >= SCRIPT_MAX_REDIRECTS) {
        throw new ScriptDownloadError(`Invalid or excessive installer redirects from ${currentUrl}`, 'remote')
      }

      currentUrl = new URL(location, currentUrl)
    }

    if (response.statusCode !== 200) {
      throw new ScriptDownloadError(
        `Failed to download ${installScriptName()}: HTTP ${response.statusCode} from ${currentUrl}`,
        'remote',
        response.statusCode
      )
    }

    if (Number(response.headers['content-length']) > MAX_SCRIPT_BYTES) {
      throw new ScriptDownloadError(`Installer body exceeds ${MAX_SCRIPT_BYTES} bytes`, 'remote')
    }

    // Buffer only the bounded script body. Nothing reaches the executable cache
    // until UTF-8/content checks pass; the streaming cap also covers chunked data.
    const chunks: Buffer[] = []
    let bodyBytes = 0
    const validatedBody = new Transform({
      transform(chunk: Buffer, _encoding, callback) {
        if (chunk.length > MAX_SCRIPT_BYTES - bodyBytes) {
          callback(new ScriptDownloadError(`Installer body exceeds ${MAX_SCRIPT_BYTES} bytes`, 'remote'))

          return
        }

        bodyBytes += chunk.length
        chunks.push(chunk)
        callback()
      },
      flush(callback) {
        try {
          const bytes = Buffer.concat(chunks, bodyBytes)
          validateScriptBody(bytes, response.headers['content-type'] || '')
          this.push(prepareCachedScriptBytes(installScriptKind(), bytes))
          callback()
        } catch (err) {
          callback(err)
        }
      }
    })

    let remoteError: Error | undefined
    let localError: Error | undefined
    let out: fs.WriteStream

    try {
      out = fs.createWriteStream(tempPath, { flags: 'wx', mode: 0o600 })
    } catch (err) {
      throw new ScriptDownloadError(`Cannot create installer download: ${err.message}`, 'local', undefined, { cause: err })
    }

    out.once('open', () => {
      tempCreated = true
    })
    response.on('error', err => {
      remoteError = err
    })
    validatedBody.on('error', err => {
      remoteError = err
    })
    out.on('error', err => {
      // pipeline also destroys its writer on a failed response or abort; those
      // propagated errors are transport failures, not a disk failure.
      if (!remoteError && !controller.signal.aborted) {
        localError = err
      }
    })

    try {
      await pipeline(response, validatedBody, out, { signal: controller.signal })
    } catch (err) {
      throw new ScriptDownloadError(
        `Failed to save ${installScriptName()}: ${(localError || remoteError || err).message}`,
        localError ? 'local' : 'remote',
        undefined,
        { cause: localError || remoteError || err }
      )
    }

    if (controller.signal.aborted) {
      throw new ScriptDownloadError('Installer download deadline exceeded', 'remote')
    }

    try {
      fs.renameSync(tempPath, destPath)
      tempPath = undefined
    } catch (err) {
      throw new ScriptDownloadError(`Cannot publish installer download: ${err.message}`, 'local', undefined, { cause: err })
    }
  } catch (err) {
    if (err instanceof ScriptDownloadError) {
      throw err
    }

    throw new ScriptDownloadError(
      controller.signal.aborted ? 'Installer download deadline exceeded' : `Installer download failed: ${err.message}`,
      'remote',
      undefined,
      { cause: err }
    )
  } finally {
    clearTimeout(deadline)
    opts.signal?.removeEventListener('abort', onAbort)
    response?.destroy()

    if (tempPath && tempCreated) {
      // pipeline waits for the writer to close before cleanup, so a late open
      // or response event cannot recreate a deleted partial file.
      fs.rmSync(tempPath, { force: true })
    }
  }
}

async function downloadInstallScript(ref: string, destPath: string, opts: ScriptDownloadOptions = {}): Promise<string> {
  const failures: string[] = []

  for (const [index, url] of scriptUrls(ref).entries()) {
    try {
      await fetchScriptOnce(url, destPath, opts)
    } catch (err) {
      failures.push(err.message)

      if (opts.signal?.aborted) {
        break
      }

      const retryable =
        err instanceof ScriptDownloadError && err.origin === 'remote' &&
        (err.statusCode === undefined || SCRIPT_FALLBACK_STATUSES.has(err.statusCode) || err.statusCode >= 500)

      if (!retryable) {
        break
      }

      continue
    }

    if (index > 0) {
      opts.onFallback?.(url)
    }

    return destPath
  }

  throw new Error(failures.join('\n'))
}

async function resolveInstallScript({
  installStamp,
  sourceRepoRoot,
  hermesHome,
  emit,
  pinCommit = true,
  abortSignal = null,
  _download = downloadInstallScript
}) {
  // 1. Dev shortcut: prefer a local checkout's installer so we can iterate
  //    without pushing. SOURCE_REPO_ROOT comes from main.ts (path.resolve
  //    of APP_ROOT/../..).
  const localScript = resolveLocalInstallScript(sourceRepoRoot)

  if (localScript) {
    emit({ type: 'log', line: `[bootstrap] using local ${installScriptName()} at ${localScript}` })

    return { path: localScript, source: 'local', kind: installScriptKind() }
  }

  // 2. Packaged path: download using the same identity policy as the stages.
  // Fresh installs use the packaged commit; existing checkouts and non-git
  // fallback builds follow the branch.
  const installRef = installRefForStamp(installStamp, { pinCommit })

  if (!installRef) {
    throw new Error(
      `Cannot resolve ${installScriptName()}: no SOURCE_REPO_ROOT and no install stamp. ` +
        'This packaged build was produced without a valid build-time stamp.'
    )
  }

  // Separate Desktop instances may use distinct userData while sharing this
  // home. Manifest and stages must keep the exact bytes this resolve selected.
  const cached = cachedScriptPath(hermesHome, `${installRef.cacheKey}-${randomUUID()}`)
  const resolvedCommit = installRef.pinned ? installRef.ref : null

  // The cache is only this run's -File target, never a source of truth.
  // Refreshing also makes old Desktop builds pick up installer fixes instead
  // of repeatedly executing bytes cached before those fixes existed.
  emit({
    type: 'log',
    line:
      `[bootstrap] fetching ${installScriptName()} for ${installRef.ref.slice(0, 12)} from GitHub` +
      (installRef.pinned ? '' : ' (unpinned branch)')
  })

  try {
    await _download(installRef.ref, cached, {
      signal: abortSignal,
      onFallback: url => emit({ type: 'log', line: `[bootstrap] downloaded ${installScriptName()} from fallback ${url}` })
    })
  } catch (err) {
    fs.rmSync(cached, { force: true })
    throw err
  }
  emit({ type: 'log', line: `[bootstrap] saved to ${cached}` })

  return { path: cached, source: 'download', commit: resolvedCommit, kind: installScriptKind() }
}

// ---------------------------------------------------------------------------
// powershell wrapper
// ---------------------------------------------------------------------------

// Canonical PowerShell 5.1 location under a Windows root (%SystemRoot%).
function powershellUnderRoot(root) {
  return path.join(root, 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe')
}

// Resolve the PowerShell interpreter to spawn.
//
// Spawning bare 'powershell.exe' trusts PATH to contain
// %SystemRoot%\System32\WindowsPowerShell\v1.0. On machines whose PATH was
// trimmed, truncated, or stored as a non-expanding REG_SZ (so %SystemRoot%
// never expands), that lookup fails and the spawn dies with ENOENT before
// install.ps1 ever runs — the installer stalls at "0 of 0 steps". Resolve by
// absolute path first, then fall back to PATH (powershell 5.1, then pwsh 7),
// then a bare name as a last resort.
function resolveWindowsPowerShell() {
  for (const v of ['SystemRoot', 'windir']) {
    const root = process.env[v]

    if (root) {
      const candidate = powershellUnderRoot(root)

      try {
        if (fs.statSync(candidate).isFile()) {
          return candidate
        }
      } catch {
        void 0
      }
    }
  }

  const pathDirs = (process.env.PATH || process.env.Path || '').split(path.delimiter).filter(Boolean)

  for (const exe of ['powershell.exe', 'pwsh.exe']) {
    for (const dir of pathDirs) {
      const candidate = path.join(dir, exe)

      try {
        if (fs.statSync(candidate).isFile()) {
          return candidate
        }
      } catch {
        void 0
      }
    }
  }

  return 'powershell.exe'
}

// install.sh (and the git/curl/uv children it drives) writes SGR colours,
// cursor/erase sequences, OSC titles and \r progress redraws into the pipe as
// if it were a TTY. The install overlay renders each line as plain text, so
// strip them ONCE here, at the emitter: the main-process log ring, the
// renderer's Details panel and "Copy output" all read the same clean line
// (#112675). \r redraws collapse to the last frame a terminal would show.
function cleanInstallerLogLine(raw: string): string {
  const frames = raw.split('\r').map(stripAnsi).filter(Boolean)

  return frames.length ? frames[frames.length - 1] : ''
}

// The installer drives Hermes's own toolchain (install.sh takes a uv from PATH
// when it is new enough), so store dirs already on PATH stay ahead of the
// login-shell entries shell-path.ts merged in front of them.
function installerEnv(hermesHome) {
  const env = { ...process.env, HERMES_HOME: hermesHome || process.env.HERMES_HOME || '' }
  const key = pathEnvKey(env)

  env[key] = storeFirstPath(env[key] || '', { currentEnv: env })

  return env
}

function spawnPowerShell(scriptPath, args, { emit, stageName, abortSignal, hermesHome }: any = {}) {
  return new Promise<any>((resolve, reject) => {
    const ps = process.platform === 'win32' ? resolveWindowsPowerShell() : 'pwsh'
    const fullArgs = ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', scriptPath, ...args]

    const child = spawn(
      ps,
      fullArgs,
      hiddenWindowsChildOptions({
        stdio: ['ignore', 'pipe', 'pipe'],
        // Pass HERMES_HOME through so install.ps1 respects the caller's
        // choice rather than re-computing the default.
        env: installerEnv(hermesHome)
      })
    )

    let stdout = ''
    let stderr = ''
    let killed = false

    const onAbort = () => {
      killed = true

      try {
        child.kill('SIGTERM')
      } catch {
        void 0
      }
    }

    if (abortSignal) {
      if (abortSignal.aborted) {
        onAbort()
      } else {
        abortSignal.addEventListener('abort', onAbort, { once: true })
      }
    }

    child.stdout.setEncoding('utf8')
    child.stderr.setEncoding('utf8')

    // Stream stdout line-by-line so the renderer sees progress in real time.
    let stdoutBuf = ''
    child.stdout.on('data', chunk => {
      stdout += chunk
      stdoutBuf += chunk
      let nl

      while ((nl = stdoutBuf.indexOf('\n')) !== -1) {
        const line = cleanInstallerLogLine(stdoutBuf.slice(0, nl))
        stdoutBuf = stdoutBuf.slice(nl + 1)

        if (line) {
          emit && emit({ type: 'log', stage: stageName, line, stream: 'stdout' })
        }
      }
    })

    let stderrBuf = ''
    child.stderr.on('data', chunk => {
      stderr += chunk
      stderrBuf += chunk
      let nl

      while ((nl = stderrBuf.indexOf('\n')) !== -1) {
        const line = cleanInstallerLogLine(stderrBuf.slice(0, nl))
        stderrBuf = stderrBuf.slice(nl + 1)

        if (line) {
          emit && emit({ type: 'log', stage: stageName, line, stream: 'stderr' })
        }
      }
    })

    child.on('error', err => {
      if (abortSignal) {
        abortSignal.removeEventListener('abort', onAbort)
      }

      reject(err)
    })

    child.on('close', (code, signal) => {
      if (abortSignal) {
        abortSignal.removeEventListener('abort', onAbort)
      }

      // Flush any trailing bytes
      const stdoutTail = cleanInstallerLogLine(stdoutBuf)
      const stderrTail = cleanInstallerLogLine(stderrBuf)

      if (stdoutTail) {
        emit && emit({ type: 'log', stage: stageName, line: stdoutTail, stream: 'stdout' } as any)
      }

      if (stderrTail) {
        emit && emit({ type: 'log', stage: stageName, line: stderrTail, stream: 'stderr' } as any)
      }

      resolve({ stdout, stderr, code, signal, killed } as any)
    })
  })
}

function spawnBash(scriptPath, args, { emit, stageName, abortSignal, hermesHome }: any = {}) {
  return new Promise<any>((resolve, reject) => {
    const child = spawn('bash', [scriptPath, ...args], {
      stdio: ['ignore', 'pipe', 'pipe'],
      env: installerEnv(hermesHome)
    })

    let stdout = ''
    let stderr = ''
    let killed = false

    const onAbort = () => {
      killed = true

      try {
        child.kill('SIGTERM')
      } catch {
        void 0
      }
    }

    if (abortSignal) {
      if (abortSignal.aborted) {
        onAbort()
      } else {
        abortSignal.addEventListener('abort', onAbort, { once: true })
      }
    }

    child.stdout.setEncoding('utf8')
    child.stderr.setEncoding('utf8')

    let stdoutBuf = ''
    child.stdout.on('data', chunk => {
      stdout += chunk
      stdoutBuf += chunk
      let nl

      while ((nl = stdoutBuf.indexOf('\n')) !== -1) {
        const line = cleanInstallerLogLine(stdoutBuf.slice(0, nl))
        stdoutBuf = stdoutBuf.slice(nl + 1)

        if (line) {
          emit && emit({ type: 'log', stage: stageName, line, stream: 'stdout' })
        }
      }
    })

    let stderrBuf = ''
    child.stderr.on('data', chunk => {
      stderr += chunk
      stderrBuf += chunk
      let nl

      while ((nl = stderrBuf.indexOf('\n')) !== -1) {
        const line = cleanInstallerLogLine(stderrBuf.slice(0, nl))
        stderrBuf = stderrBuf.slice(nl + 1)

        if (line) {
          emit && emit({ type: 'log', stage: stageName, line, stream: 'stderr' })
        }
      }
    })

    child.on('error', err => {
      if (abortSignal) {
        abortSignal.removeEventListener('abort', onAbort)
      }

      reject(err)
    })

    child.on('close', (code, signal) => {
      if (abortSignal) {
        abortSignal.removeEventListener('abort', onAbort)
      }

      const stdoutTail = cleanInstallerLogLine(stdoutBuf)
      const stderrTail = cleanInstallerLogLine(stderrBuf)

      if (stdoutTail) {
        emit && emit({ type: 'log', stage: stageName, line: stdoutTail, stream: 'stdout' })
      }

      if (stderrTail) {
        emit && emit({ type: 'log', stage: stageName, line: stderrTail, stream: 'stderr' })
      }

      resolve({ stdout, stderr, code, signal, killed })
    })
  })
}

// ---------------------------------------------------------------------------
// Manifest + stage dispatch
// ---------------------------------------------------------------------------

// Build the installer branch/pin args from the install stamp. The commit pin
// is fresh-install only: once a managed checkout already exists, bootstrap is
// a repair/update path and must not let an old packaged app detach the checkout
// back to the commit baked into that app. All-zero fallback stamps are never
// passed as -Commit/--commit — only the branch is used (#50823 / #50864 review).
function buildPinArgs(installStamp, { pinCommit = true } = {}) {
  const args = []

  if (pinCommit && installStamp && isPinnedCommit(installStamp.commit)) {
    args.push('-Commit', installStamp.commit)
  }

  if (installStamp && installStamp.branch) {
    args.push('-Branch', installStamp.branch)
  }

  return args
}

function buildPosixPinArgs({ installStamp, activeRoot, hermesHome, pinCommit = true }) {
  const args = ['--dir', activeRoot, '--hermes-home', hermesHome]

  if (installStamp && installStamp.branch) {
    args.push('--branch', installStamp.branch)
  }

  if (pinCommit && installStamp && isPinnedCommit(installStamp.commit)) {
    args.push('--commit', installStamp.commit)
  }

  return args
}

async function fetchManifest({
  scriptPath,
  installerKind,
  emit,
  hermesHome,
  activeRoot,
  installStamp,
  pinCommit,
  abortSignal
}) {
  abortSignal?.throwIfAborted()
  const isPosix = installerKind === 'posix'

  const args = isPosix
    ? ['--manifest', ...buildPosixPinArgs({ installStamp, activeRoot, hermesHome, pinCommit })]
    : ['-Manifest', ...buildPinArgs(installStamp, { pinCommit })]

  const result = await (isPosix ? spawnBash : spawnPowerShell)(scriptPath, args, {
    emit,
    stageName: '__manifest__',
    abortSignal,
    hermesHome
  })

  if (result.code !== 0) {
    // The tail lands in the Setup failure banner, not the log ring, so strip
    // the installer's colour/OSC bytes here too (#112675).
    const tail = stripAnsi(result.stderr || result.stdout).trim()

    throw new Error(
      `${isPosix ? 'install.sh --manifest' : 'install.ps1 -Manifest'} failed: exit ${result.code}\n${tail}`
    )
  }

  // The manifest is the LAST JSON line on stdout (install.ps1 may print
  // banner / info lines first depending on Console.OutputEncoding effects).
  // Find the last line that parses as JSON with a `stages` field.
  const lines = result.stdout.split(/\r?\n/).filter(Boolean)

  for (let i = lines.length - 1; i >= 0; i--) {
    try {
      const parsed = JSON.parse(lines[i])

      if (parsed && Array.isArray(parsed.stages)) {
        return parsed
      }
    } catch {
      void 0
    }
  }

  throw new Error(
    `${isPosix ? 'install.sh --manifest' : 'install.ps1 -Manifest'} produced no parseable JSON payload\n${result.stdout}`
  )
}

// Parse the JSON result frame from a stage run. The protocol guarantees
// exactly one JSON line per stage in -Json or -Stage mode (post #27224 fix
// for the double-emit bug we addressed in the install.ps1 PR).
function parseStageResult(stdout) {
  const lines = stdout.split(/\r?\n/).filter(Boolean)

  for (let i = lines.length - 1; i >= 0; i--) {
    try {
      const parsed = JSON.parse(lines[i])

      if (parsed && typeof parsed.ok === 'boolean' && typeof parsed.stage === 'string') {
        return parsed
      }
    } catch {
      void 0
    }
  }

  return null
}

async function runStage({
  scriptPath,
  installerKind,
  stage,
  emit,
  hermesHome,
  activeRoot,
  abortSignal,
  installStamp,
  pinCommit
}) {
  const startedAt = Date.now()
  emit({ type: 'stage', name: stage.name, state: 'running' })

  const isPosix = installerKind === 'posix'

  const args = isPosix
    ? [
        '--stage',
        stage.name,
        '--non-interactive',
        '--json',
        ...buildPosixPinArgs({ installStamp, activeRoot, hermesHome, pinCommit })
      ]
    : ['-Stage', stage.name, '-NonInteractive', '-Json', ...buildPinArgs(installStamp, { pinCommit })]

  const result = await (isPosix ? spawnBash : spawnPowerShell)(scriptPath, args, {
    emit,
    stageName: stage.name,
    abortSignal,
    hermesHome
  })

  const durationMs = Date.now() - startedAt

  if (result.killed) {
    const ev = { type: 'stage', name: stage.name, state: 'failed', durationMs, error: 'cancelled by user' }
    emit(ev)

    return ev
  }

  const json = parseStageResult(result.stdout)

  if (!json) {
    const ev = {
      type: 'stage',
      name: stage.name,
      state: 'failed',
      durationMs,
      error: `${isPosix ? 'install.sh --stage' : 'install.ps1 -Stage'} ${stage.name} produced no JSON result frame (exit=${result.code})`,
      json: null
    }

    emit(ev)

    return ev
  }

  if (json.ok && json.skipped) {
    const ev = { type: 'stage', name: stage.name, state: 'skipped', durationMs, json }
    emit(ev)

    return ev
  }

  if (json.ok) {
    const ev = { type: 'stage', name: stage.name, state: 'succeeded', durationMs, json }
    emit(ev)

    return ev
  }

  const ev = {
    type: 'stage',
    name: stage.name,
    state: 'failed',
    durationMs,
    json,
    error: json.reason || `exit code ${result.code}`
  }

  emit(ev)

  return ev
}

// ---------------------------------------------------------------------------
// Per-run log file
// ---------------------------------------------------------------------------

function openRunLog(logRoot) {
  fs.mkdirSync(logRoot, { recursive: true })
  const ts = new Date().toISOString().replace(/[:.]/g, '-')
  const logPath = path.join(logRoot, `bootstrap-${ts}.log`)
  const stream = fs.createWriteStream(logPath, { flags: 'a' })

  return { path: logPath, stream }
}

// ---------------------------------------------------------------------------
// Public entrypoint
// ---------------------------------------------------------------------------

async function runBootstrap(opts) {
  const {
    installStamp,
    activeRoot,
    sourceRepoRoot,
    hermesHome,
    logRoot,
    onEvent,
    abortSignal,
    writeMarker, // callback to write the bootstrap-complete marker; main.ts provides
    gitBinary // probed git path from main.ts; bare 'git' when absent
  } = opts

  // Bail before spawning anything if the user already cancelled — otherwise an
  // already-aborted signal would still fetch the manifest (a spawn) before the
  // in-loop abort check fires.
  if (abortSignal && abortSignal.aborted) {
    if (typeof onEvent === 'function') {
      try {
        onEvent({ type: 'failed', error: 'bootstrap cancelled by user' })
      } catch {
        void 0
      }
    }

    return { ok: false, cancelled: true }
  }

  const runLog = openRunLog(logRoot || path.join(hermesHome, 'logs'))
  let downloadedScript: string | undefined

  // Tee every event to the runLog AND the caller's onEvent. This gives us a
  // forensic trail per bootstrap run AND lets the renderer subscribe live.
  const emit = ev => {
    try {
      runLog.stream.write(JSON.stringify(ev) + '\n')
    } catch {
      void 0
    }

    try {
      if (typeof onEvent === 'function') {
        onEvent(ev)
      }
    } catch (err) {
      // Don't let a subscriber bug crash the bootstrap
      runLog.stream.write(`emit error: ${err && err.message}\n`)
    }
  }

  emit({
    type: 'log',
    line:
      `[bootstrap] starting at ${new Date().toISOString()}; ` +
      `activeRoot=${activeRoot}; ` +
      `stamp=${installStamp ? installStamp.commit.slice(0, 12) : '<none>'}; ` +
      `runLog=${runLog.path}`
  })

  try {
    const existingCheckout = hasExistingGitCheckout(activeRoot)
    const pinCommit = !existingCheckout

    if (existingCheckout && installStamp && installStamp.commit) {
      emit({
        type: 'log',
        line:
          `[bootstrap] existing checkout detected at ${activeRoot}; ` +
          `not pinning to packaged install stamp ${installStamp.commit.slice(0, 12)}`
      })
    }

    // 1. Resolve the platform installer.
    const scriptInfo = await resolveInstallScript({ installStamp, sourceRepoRoot, hermesHome, emit, pinCommit, abortSignal })

    if (scriptInfo.source === 'download') {
      downloadedScript = scriptInfo.path
    }

    abortSignal?.throwIfAborted()

    const installerKind = scriptInfo.kind || 'powershell'

    // 2. Fetch manifest
    const manifest = await fetchManifest({
      scriptPath: scriptInfo.path,
      installerKind,
      emit,
      hermesHome,
      activeRoot,
      installStamp,
      pinCommit,
      abortSignal
    })

    abortSignal?.throwIfAborted()

    emit({
      type: 'manifest',
      stages: manifest.stages,
      protocolVersion: manifest.protocol_version || manifest.protocolVersion || null
    })

    // 3. Iterate stages in order. Stages flagged needs_user_input are still
    //    invoked -- install.ps1's own -NonInteractive handler in those stages
    //    emits skipped=true. We trust the protocol rather than filtering
    //    client-side.
    for (const stage of manifest.stages) {
      if (abortSignal && abortSignal.aborted) {
        emit({ type: 'failed', error: 'bootstrap cancelled by user' })

        return { ok: false, cancelled: true }
      }

      const ev = await runStage({
        scriptPath: scriptInfo.path,
        installerKind,
        stage,
        emit,
        hermesHome,
        activeRoot,
        abortSignal,
        installStamp,
        pinCommit
      })

      if (ev.state === 'failed') {
        emit({ type: 'failed', stage: stage.name, error: (ev as any).error || 'stage failed' })

        return { ok: false, failedStage: stage.name, error: (ev as any).error }
      }
    }

    // 4. Write the bootstrap-complete marker. Fallback (all-zero) stamps are
    // not real pins -- resolve HEAD from the checkout we just installed so
    // isBootstrapComplete() (pinnedCommit.length >= 7) accepts the marker
    // instead of re-running bootstrap on every launch (#50823 review).
    const pinnedCommit = resolveMarkerPinnedCommit(installStamp, activeRoot, {
      resolveHead: root => resolveCheckoutHead(root, { gitBinary })
    })

    if (!pinnedCommit) {
      emit({
        type: 'log',
        line:
          '[bootstrap] WARNING: could not resolve a real pinnedCommit for the ' +
          'bootstrap-complete marker; subsequent launches may re-run bootstrap'
      })
    } else if (installStamp && !isPinnedCommit(installStamp.commit)) {
      emit({
        type: 'log',
        line: `[bootstrap] fallback stamp resolved marker pin to ${pinnedCommit.slice(0, 12)} from checkout`
      })
    }

    const markerPayload = {
      pinnedCommit,
      pinnedBranch: installStamp ? installStamp.branch : null
    }

    const marker = typeof writeMarker === 'function' ? writeMarker(markerPayload) : markerPayload
    emit({ type: 'complete', marker })

    return { ok: true, marker }
  } catch (err) {
    if (abortSignal?.aborted) {
      emit({ type: 'failed', error: 'bootstrap cancelled by user' })

      return { ok: false, cancelled: true }
    }

    emit({ type: 'failed', error: err.message || String(err) })

    return { ok: false, error: err.message || String(err) }
  } finally {
    if (downloadedScript) {
      try {
        fs.rmSync(downloadedScript, { force: true })
      } catch (err) {
        emit({ type: 'log', line: `[bootstrap] could not remove run installer ${downloadedScript}: ${err.message}` })
      }
    }

    try {
      await new Promise<void>(resolve => runLog.stream.end(resolve))
    } catch {
      void 0
    }
  }
}

export {
  buildPinArgs,
  buildPosixPinArgs,
  cachedScriptPath,
  cleanInstallerLogLine,
  downloadInstallScript,
  hasExistingGitCheckout,
  installRefForStamp,
  isPinnedCommit,
  // Exposed for testability
  parseStageResult,
  prepareCachedScriptBytes,
  resolveCheckoutHead,
  resolveInstallScript,
  resolveLocalInstallScript,
  resolveMarkerPinnedCommit,
  runBootstrap,
  scriptUrls
}
