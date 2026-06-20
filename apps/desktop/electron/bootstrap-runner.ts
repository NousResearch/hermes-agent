/**
 * bootstrap-runner.ts
 *
 * Drives apps/desktop's first-launch install of Hermes Agent by spawning
 * scripts/install.ps1 stage-by-stage and streaming progress events back to
 * the renderer.
 *
 * Wired from electron/main.ts:
 *   import { runBootstrap }from './bootstrap-runner.ts'
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
 *   { type: 'log',       stage?, line, stream: 'stdout'|'stderr' } // raw line from install.ps1
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

import { spawn } from 'node:child_process'
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import https from 'node:https'
import path from 'node:path'

const IS_WINDOWS = process.platform === 'win32'

function hiddenWindowsChildOptions(options = {}) {
  if (!IS_WINDOWS || Object.prototype.hasOwnProperty.call(options, 'windowsHide')) {
    return options
  }

  return { ...options, windowsHide: true }
}

const STAMP_COMMIT_RE = /^[0-9a-f]{7,40}$/i

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

function installScriptKind() {
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

// The install.sh / install.ps1 that ships inside the already-installed agent
// checkout under ~/.hermes/hermes-agent. Used as a last-resort fallback when
// the pinned commit can't be fetched from GitHub (e.g. a locally-built desktop
// app stamped to an unpushed HEAD).
function installedAgentInstallScript(hermesHome) {
  if (!hermesHome) {
    return null
  }
  const candidate = path.join(hermesHome, 'hermes-agent', 'scripts', installScriptName())

  try {
    fs.accessSync(candidate, fs.constants.R_OK)

    return candidate
  } catch {
    return null
  }
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

function cachedScriptPath(hermesHome, commit) {
  return path.join(bootstrapCacheDir(hermesHome), `install-${commit}.${process.platform === 'win32' ? 'ps1' : 'sh'}`)
}

function downloadInstallScript(commit, destPath) {
  // Fetch from GitHub raw at the pinned commit. The raw URL with a SHA
  // is immutable (unlike a branch ref), so we don't need integrity
  // verification beyond "did the file we wrote pass a syntax probe."
  const scriptName = installScriptName()
  const url = `https://raw.githubusercontent.com/NousResearch/hermes-agent/${commit}/scripts/${scriptName}`

  return new Promise((resolve, reject) => {
    fs.mkdirSync(path.dirname(destPath), { recursive: true })
    const tmpPath = destPath + '.tmp'
    const out = fs.createWriteStream(tmpPath)
    https
      .get(url, res => {
        if (res.statusCode === 301 || res.statusCode === 302) {
          // GitHub raw shouldn't redirect for a SHA URL, but follow once
          // defensively. Validate the redirect target is https and on
          // raw.githubusercontent.com before following.
          out.close()
          fs.unlinkSync(tmpPath)
          const redirectUrl = res.headers.location || ''
          try {
            const parsed = new URL(redirectUrl)
            if (parsed.protocol !== 'https:' || parsed.hostname !== 'raw.githubusercontent.com') {
              reject(new Error('Redirect to unexpected host: ' + redirectUrl))
              return
            }
          } catch {
            reject(new Error('Invalid redirect URL: ' + redirectUrl))
            return
          }
          https
            .get(redirectUrl, res2 => {
              if (res2.statusCode !== 200) {
                reject(
                  new Error(
                    `Failed to download ${scriptName}: HTTP ${res2.statusCode} from redirect ${res.headers.location}`
                  )
                )

                return
              }

              const out2 = fs.createWriteStream(tmpPath)
              res2.pipe(out2)
              out2.on('finish', () => {
                out2.close()
                fs.renameSync(tmpPath, destPath)
                resolve(destPath)
              })
              out2.on('error', reject)
            })
            .on('error', reject)

          return
        }

        if (res.statusCode !== 200) {
          out.close()

          try {
            fs.unlinkSync(tmpPath)
          } catch {
            void 0
          }

          reject(new Error(`Failed to download ${scriptName}: HTTP ${res.statusCode} from ${url}`))

          return
        }

        res.pipe(out)
        out.on('finish', () => {
          out.close()
          fs.renameSync(tmpPath, destPath)
          resolve(destPath)
        })
        out.on('error', err => {
          try {
            fs.unlinkSync(tmpPath)
          } catch {
            void 0
          }

          reject(err)
        })
      })
      .on('error', err => {
        try {
          fs.unlinkSync(tmpPath)
        } catch {
          void 0
        }

        reject(err)
      })
  })
}

async function resolveInstallScript({
  installStamp,
  sourceRepoRoot,
  hermesHome,
  emit,
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

  // 2. Packaged path: download from GitHub at the pinned commit (1B's stamp).
  if (!installStamp || !installStamp.commit || !STAMP_COMMIT_RE.test(installStamp.commit)) {
    throw new Error(
      `Cannot resolve ${installScriptName()}: no SOURCE_REPO_ROOT and no install stamp. ` +
        'This packaged build was produced without a valid build-time stamp.'
    )
  }

  const cached = cachedScriptPath(hermesHome, installStamp.commit)

  try {
    await fsp.access(cached, fs.constants.R_OK)
    emit({
      type: 'log',
      line: `[bootstrap] using cached ${installScriptName()} for ${installStamp.commit.slice(0, 12)}`
    })

    return { path: cached, source: 'cache', commit: installStamp.commit, kind: installScriptKind() }
  } catch {
    // not cached; download
  }

  emit({
    type: 'log',
    line: `[bootstrap] fetching ${installScriptName()} for ${installStamp.commit.slice(0, 12)} from GitHub`
  })

  try {
    await _download(installStamp.commit, cached)
    emit({ type: 'log', line: `[bootstrap] saved to ${cached}` })

    return { path: cached, source: 'download', commit: installStamp.commit, kind: installScriptKind() }
  } catch (err) {
    // The pinned commit may not be fetchable from GitHub -- most commonly a
    // locally-built desktop app stamped to an unpushed HEAD (see
    // write-build-stamp.mjs fromLocalGit). Fall back to the installer that
    // ships inside the already-installed agent checkout so dev/self-builds can
    // still bootstrap instead of dying with a fatal 404.
    const installed = installedAgentInstallScript(hermesHome)

    if (installed) {
      emit({
        type: 'log',
        line:
          `[bootstrap] GitHub fetch failed (${err.message}); ` +
          `falling back to installed agent ${installScriptName()} at ${installed}`
      })

      try {
        fs.mkdirSync(path.dirname(cached), { recursive: true })
        fs.copyFileSync(installed, cached)

        return { path: cached, source: 'installed-agent', commit: installStamp.commit, kind: installScriptKind() }
      } catch {
        // Cache copy failed (read-only FS, etc.) -- use the source path directly.
        return { path: installed, source: 'installed-agent', commit: installStamp.commit, kind: installScriptKind() }
      }
    }

    throw err
  }
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

function spawnPowerShell(scriptPath, args, { emit, stageName, abortSignal, hermesHome }: any = {}) {
  return new Promise<any>((resolve, reject) => {
    const ps = process.platform === 'win32' ? resolveWindowsPowerShell() : 'pwsh'
    const fullArgs = ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', scriptPath, ...args]

    const child = spawn(
      ps,
      fullArgs,
      hiddenWindowsChildOptions({
        stdio: ['ignore', 'pipe', 'pipe'],
        env: {
          ...process.env,
          // Pass HERMES_HOME through so install.ps1 respects the caller's
          // choice rather than re-computing the default.
          HERMES_HOME: hermesHome || process.env.HERMES_HOME || ''
        }
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
        const line = stdoutBuf.slice(0, nl).replace(/\r$/, '')
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
        const line = stderrBuf.slice(0, nl).replace(/\r$/, '')
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
      if (stdoutBuf) {
        emit && emit({ type: 'log', stage: stageName, line: stdoutBuf, stream: 'stdout' } as any)
      }

      if (stderrBuf) {
        emit && emit({ type: 'log', stage: stageName, line: stderrBuf, stream: 'stderr' } as any)
      }
      resolve({ stdout, stderr, code, signal, killed } as any)
    })
  })
}

function spawnBash(scriptPath, args, { emit, stageName, abortSignal, hermesHome }: any = {}) {
  return new Promise<any>((resolve, reject) => {
    const child = spawn('bash', [scriptPath, ...args], {
      stdio: ['ignore', 'pipe', 'pipe'],
      env: {
        ...process.env,
        HERMES_HOME: hermesHome || process.env.HERMES_HOME || ''
      }
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
        const line = stdoutBuf.slice(0, nl).replace(/\r$/, '')
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
        const line = stderrBuf.slice(0, nl).replace(/\r$/, '')
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

      if (stdoutBuf) {
        emit && emit({ type: 'log', stage: stageName, line: stdoutBuf, stream: 'stdout' })
      }

      if (stderrBuf) {
        emit && emit({ type: 'log', stage: stageName, line: stderrBuf, stream: 'stderr' })
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
// back to the commit baked into that app.
function buildPinArgs(installStamp, { pinCommit = true } = {}) {
  const args = []

  if (pinCommit && installStamp && installStamp.commit) {
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

  if (pinCommit && installStamp && installStamp.commit) {
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
  pinCommit
}) {
  const isPosix = installerKind === 'posix'

  const args = isPosix
    ? ['--manifest', ...buildPosixPinArgs({ installStamp, activeRoot, hermesHome, pinCommit })]
    : ['-Manifest', ...buildPinArgs(installStamp, { pinCommit })]

  const result = await (isPosix ? spawnBash : spawnPowerShell)(scriptPath, args, {
    emit,
    stageName: '__manifest__',
    hermesHome
  })

  if (result.code !== 0) {
    throw new Error(
      `${isPosix ? 'install.sh --manifest' : 'install.ps1 -Manifest'} failed: exit ${result.code}\n${result.stderr || result.stdout}`
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
    writeMarker // callback to write the bootstrap-complete marker; main.ts provides
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
    const scriptInfo = await resolveInstallScript({ installStamp, sourceRepoRoot, hermesHome, emit })
    const installerKind = scriptInfo.kind || 'powershell'

    // 2. Fetch manifest
    const manifest = await fetchManifest({
      scriptPath: scriptInfo.path,
      installerKind,
      emit,
      hermesHome,
      activeRoot,
      installStamp,
      pinCommit
    })

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

    // 4. Write the bootstrap-complete marker.
    const markerPayload = {
      pinnedCommit: installStamp ? installStamp.commit : null,
      pinnedBranch: installStamp ? installStamp.branch : null
    }

    const marker = typeof writeMarker === 'function' ? writeMarker(markerPayload) : markerPayload
    emit({ type: 'complete', marker })

    return { ok: true, marker }
  } catch (err) {
    emit({ type: 'failed', error: err.message || String(err) })

    return { ok: false, error: err.message || String(err) }
  } finally {
    try {
      runLog.stream.end()
    } catch {
      void 0
    }
  }
}

export {
  buildPinArgs,
  buildPosixPinArgs,
  cachedScriptPath,
  hasExistingGitCheckout,
  installedAgentInstallScript,
  // Exposed for testability
  parseStageResult,
  resolveInstallScript,
  resolveLocalInstallScript,
  runBootstrap
}
