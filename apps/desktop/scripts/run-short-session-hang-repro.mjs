#!/usr/bin/env node

import { spawn, spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import {
  appendFileSync,
  cpSync,
  copyFileSync,
  createWriteStream,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  symlinkSync,
  writeFileSync
} from 'node:fs'
import http from 'node:http'
import { createRequire } from 'node:module'
import { createServer } from 'node:net'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { finished } from 'node:stream/promises'
import { fileURLToPath } from 'node:url'

import { CDP, discoverTarget, sleep } from './perf/lib/cdp.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const DESKTOP_ROOT = resolve(HERE, '..')
const REPO_ROOT = resolve(DESKTOP_ROOT, '..', '..')
const HARNESS_SOURCE = resolve(DESKTOP_ROOT, 'src/app/chat/short-session-hang-repro.tsx')
const UPSTREAM_URL = 'https://github.com/NousResearch/hermes-agent.git'
const DEFAULT_BASELINE = '3651627d88858912e8460e6f949b7125725600c3'
const DEFAULT_CANDIDATE = '3651627d88858912e8460e6f949b7125725600c3'
const FREEZE_MS = 5_000
const JOURNAL_SEED_TIMEOUT_MS = 30_000
const STREAM_RESPONSE_TIMEOUT_MS = 30_000
const STREAM_RESPONSE_EVALUATION_TIMEOUT_MS = 30_000
const STREAM_PAYLOAD_BYTES = 512 * 1024
const LEGACY_STORAGE_KEY = 'hermes.desktop.inflightTurnJournal.v1'
const LEGACY_MIGRATION_KEY = 'hermes.desktop.inflightTurnJournal.v2.migrated'
const LEGACY_SEED_ENTRY_COUNT = 15
const MAX_LEGACY_SEED_BYTES = 1.9 * 1024 * 1024
// Reserve space for the JSON envelope and per-entry metadata so the generated
// fixture remains below the guard even when the entry count or shape changes.
const LEGACY_SEED_METADATA_BUDGET_BYTES = 16 * 1024
const LEGACY_SEED_PAYLOAD_CHARS = Math.floor(
  (MAX_LEGACY_SEED_BYTES - LEGACY_SEED_METADATA_BUDGET_BYTES) / LEGACY_SEED_ENTRY_COUNT
)
const EXCHANGE_COUNT = 5
const NATIVE_VISIBILITY_PAUSE_MS = 750
const EXCHANGE_FREEZE_TIMEOUTS = 9
const POST_EXCHANGE_FREEZE_TIMEOUTS = 13
const WATCHDOG_OVERHEAD_MS = 120_000
// The watchdog is a last-resort guard around the complete five-exchange run.
// Inner timeouts still identify actual renderer hangs; this budget prevents a
// slow but valid stream/response sequence from being misclassified by the old
// fixed 120s outer limit. The overhead covers renderer startup/cleanup and
// native visibility calls not represented by the per-operation timers.
const OUTER_WATCHDOG_MS =
  EXCHANGE_COUNT *
    (EXCHANGE_FREEZE_TIMEOUTS * FREEZE_MS +
      STREAM_RESPONSE_TIMEOUT_MS +
      STREAM_RESPONSE_EVALUATION_TIMEOUT_MS +
      NATIVE_VISIBILITY_PAUSE_MS) +
  POST_EXCHANGE_FREEZE_TIMEOUTS * FREEZE_MS +
  2_000 +
  WATCHDOG_OVERHEAD_MS
const SECRET_ENV_RE = /(credential|token|secret|password|(^|_)key($|_)|auth|cookie)/i
const HEX_OBJECT_RE = /^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$/
const PULL_REF_RE = /^refs\/pull\/[1-9][0-9]*\/(?:head|merge)$/
const HEAD_REF_RE = /^refs\/heads\/[A-Za-z0-9][A-Za-z0-9._/-]*$/
const HEAD_COMPONENT_RE = /^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$/
const require = createRequire(import.meta.url)

class ReproductionError extends Error {
  name = 'ReproductionError'
}

function usage() {
  console.log(`Usage: node scripts/run-short-session-hang-repro.mjs [options]

Options:
  --baseline <ref>       baseline ref (default: ${DEFAULT_BASELINE})
  --candidate <ref>      candidate ref (default: ${DEFAULT_CANDIDATE})
  --repetitions <n>      measured repetitions per ref (default: 5)
  --output <dir>         artifact directory (default: short-session-hang-artifacts)
  --keep-worktrees       retain ephemeral source copies
  --dry-run              validate refs, lockfile/Electron parity, and print the plan
  --help                 show this help

Each ref gets one warm-up plus N measured fresh-app runs. Measured A/B order is
counterbalanced. A run is a reproduction only when a renderer/main operation,
heartbeat, or event-loop gap exceeds ${FREEZE_MS}ms, or Electron becomes
unresponsive, loses the renderer, or exits unexpectedly.`)
}

function parseArgs(argv) {
  const out = {
    baseline: DEFAULT_BASELINE,
    candidate: DEFAULT_CANDIDATE,
    repetitions: 5,
    output: resolve(process.cwd(), 'short-session-hang-artifacts'),
    dryRun: false,
    keepWorktrees: false
  }

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i]
    const next = argv[i + 1]

    if (arg === '--help') {
      usage()
      process.exit(0)
    } else if (arg === '--dry-run') {
      out.dryRun = true
    } else if (arg === '--keep-worktrees') {
      out.keepWorktrees = true
    } else if (arg === '--baseline' || arg === '--candidate' || arg === '--repetitions' || arg === '--output') {
      if (!next || next.startsWith('--')) {
        throw new Error(`${arg} requires a value`)
      }

      const key = arg.slice(2)
      out[key] = key === 'repetitions' ? Number(next) : key === 'output' ? resolve(next) : next
      i += 1
    } else {
      throw new Error(`unknown option: ${arg}`)
    }
  }

  if (!Number.isInteger(out.repetitions) || out.repetitions < 1 || out.repetitions > 20) {
    throw new Error('--repetitions must be an integer from 1 to 20')
  }

  return out
}

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: options.cwd ?? REPO_ROOT,
    encoding: 'utf8',
    env: options.env ?? process.env,
    maxBuffer: 64 * 1024 * 1024
  })

  if (options.logPath) {
    writeFileSync(options.logPath, `${result.stdout ?? ''}${result.stderr ?? ''}`)
  }

  if (result.error) {
    throw new Error(`${command} ${args.join(' ')} failed to run: ${result.error.message}`)
  }

  if (result.status !== 0) {
    throw new Error(
      `${command} ${args.join(' ')} exited ${result.status ?? `signal ${result.signal}`}:\n${result.stderr || result.stdout}`
    )
  }

  return String(result.stdout ?? '').trim()
}

function sha256(value) {
  return createHash('sha256').update(value).digest('hex')
}

function validateRef(ref) {
  const headParts = ref.startsWith('refs/heads/') ? ref.slice('refs/heads/'.length).split('/') : []
  const invalidHead =
    ref.includes('..') ||
    headParts.length === 0 ||
    headParts.some(part => !HEAD_COMPONENT_RE.test(part) || part.endsWith('.lock'))

  if (HEX_OBJECT_RE.test(ref) || PULL_REF_RE.test(ref) || (HEAD_REF_RE.test(ref) && !invalidHead)) {
    return ref
  }

  throw new Error(
    `unsafe ref ${JSON.stringify(ref)}; use a 40/64-digit hex object ID, refs/heads/<safe>, or refs/pull/<number>/(head|merge)`
  )
}

function sanitizedEnv(extra = {}) {
  const clean = Object.fromEntries(Object.entries(process.env).filter(([name]) => !SECRET_ENV_RE.test(name)))

  return { ...clean, ...extra }
}

function resolveRef(rawRef) {
  const ref = validateRef(rawRef)
  const tryResolve = () => {
    const result = spawnSync('git', ['rev-parse', '--verify', '--end-of-options', `${ref}^{commit}`], {
      cwd: REPO_ROOT,
      encoding: 'utf8'
    })

    return result.status === 0 ? result.stdout.trim() : null
  }

  const local = HEX_OBJECT_RE.test(ref) ? tryResolve() : null

  if (local) {
    return local
  }

  run('git', ['fetch', '--no-tags', UPSTREAM_URL, ref], {
    env: sanitizedEnv({ GCM_INTERACTIVE: 'never', GIT_TERMINAL_PROMPT: '0' })
  })

  if (ref.startsWith('refs/')) {
    const fetched = run('git', ['rev-parse', '--verify', '--end-of-options', 'FETCH_HEAD^{commit}'])

    return fetched
  }

  const resolved = tryResolve()

  if (!resolved) {
    throw new Error(`cannot resolve ref ${ref}`)
  }

  return resolved
}

function readTargetMetadata(sha) {
  const lock = run('git', ['show', '--end-of-options', `${sha}:package-lock.json`])
  const pyproject = run('git', ['show', '--end-of-options', `${sha}:pyproject.toml`])
  const uvLock = run('git', ['show', '--end-of-options', `${sha}:uv.lock`])
  const pkg = JSON.parse(run('git', ['show', '--end-of-options', `${sha}:apps/desktop/package.json`]))

  return {
    electron: pkg.devDependencies?.electron,
    electronBuild: pkg.build?.electronVersion,
    lockSha256: sha256(lock),
    pyprojectSha256: sha256(pyproject),
    uvLockSha256: sha256(uvLock)
  }
}

function readJournalContract(sha) {
  const source = run('git', ['show', '--end-of-options', `${sha}:apps/desktop/src/lib/inflight-turn-journal.ts`])
  const storageKey = source.match(/const (?:LEGACY_STORAGE_KEY|STORAGE_KEY) = '([^']+)'/)?.[1] ?? null
  const migrationKey = source.match(/const LEGACY_MIGRATION_KEY = '([^']+)'/)?.[1] ?? null
  const legacyStoreLimit = source.match(/const MAX_LEGACY_STORE_CHARS = (\d+) \* 1024 \* 1024/)?.[1]

  return {
    legacyStoreLimit: legacyStoreLimit ? Number(legacyStoreLimit) * 1024 * 1024 : null,
    migrationKey,
    storageKey
  }
}

function injectHarness(targetRoot) {
  const targetHarness = join(targetRoot, 'apps/desktop/src/app/chat/short-session-hang-repro.tsx')
  const targetMain = join(targetRoot, 'apps/desktop/src/main.tsx')
  const main = readFileSync(targetMain, 'utf8')

  mkdirSync(dirname(targetHarness), { recursive: true })
  copyFileSync(HARNESS_SOURCE, targetHarness)

  if (!main.includes("import('./app/chat/short-session-hang-repro')")) {
    writeFileSync(
      targetMain,
      `${main}\nif (import.meta.env.VITE_SHORT_SESSION_HANG_REPRO === '1') {\n  import('./app/chat/short-session-hang-repro')\n}\n`
    )
  }
}

function linkShared(targetRoot, relativePath) {
  const source = join(REPO_ROOT, relativePath)
  const target = join(targetRoot, relativePath)

  if (existsSync(source) && !existsSync(target)) {
    mkdirSync(dirname(target), { recursive: true })
    symlinkSync(source, target, 'dir')
  }
}

function prepareTarget(label, sha, root, output) {
  const targetRoot = join(root, label)
  run('git', ['worktree', 'add', '--detach', targetRoot, sha])

  try {
    injectHarness(targetRoot)
    linkShared(targetRoot, 'node_modules')
    linkShared(targetRoot, 'apps/desktop/node_modules')
    linkShared(targetRoot, '.venv')

    const targetDesktop = join(targetRoot, 'apps/desktop')
    const buildLog = join(output, `${label}-build.log`)
    run('npm', ['run', '--prefix', 'apps/desktop', 'build'], {
      cwd: targetRoot,
      env: sanitizedEnv({ VITE_SHORT_SESSION_HANG_REPRO: '1' }),
      logPath: buildLog
    })

    return { journalContract: readJournalContract(sha), label, sha, targetDesktop, targetRoot }
  } catch (error) {
    try {
      run('git', ['worktree', 'remove', '--force', targetRoot])
    } catch {
      // The original build error is more useful; cleanup is retried manually.
    }

    throw error
  }
}

function startMockInference() {
  const reply = `deterministic-tool-heavy-output:${'x'.repeat(STREAM_PAYLOAD_BYTES)}`
  let streamingCompletionRequests = 0
  let activeStreamingRequests = 0
  let listening = false
  const transportErrors = []
  const requests = []
  const recordTransportError = (source, error) => {
    transportErrors.push({
      at: new Date().toISOString(),
      message: error instanceof Error ? error.message : String(error),
      source
    })
  }
  const server = http.createServer((request, response) => {
    requests.push({ method: request.method, url: request.url ?? null })
    request.on('error', error => recordTransportError('request', error))
    response.on('error', error => recordTransportError('response', error))

    if (request.method === 'GET' && request.url === '/v1/models') {
      response.writeHead(200, { 'content-type': 'application/json' })
      response.end(JSON.stringify({ data: [{ id: 'short-session-model', object: 'model' }], object: 'list' }))

      return
    }

    if (request.method === 'POST' && request.url?.startsWith('/v1/chat/completions')) {
      let body = ''
      request.on('data', chunk => {
        body += String(chunk)
      })
      request.on('end', () => {
        let stream = false

        try {
          stream = JSON.parse(body).stream === true
        } catch {
          stream = false
        }

        if (stream) {
          streamingCompletionRequests += 1
          activeStreamingRequests += 1
          let finishedStream = false
          const finishStream = () => {
            if (finishedStream) return
            finishedStream = true
            activeStreamingRequests = Math.max(0, activeStreamingRequests - 1)
          }
          response.once('close', finishStream)
          response.writeHead(200, { 'content-type': 'text/event-stream' })
          let offset = 0
          const writeNext = () => {
            if (response.destroyed) {
              finishStream()
              return
            }
            if (offset >= reply.length) {
              response.write(
                `data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: 'stop', index: 0 }], id: 'short-session', object: 'chat.completion.chunk' })}\n\n`
              )
              response.end('data: [DONE]\n\n')
              finishStream()
              return
            }
            const chunk = reply.slice(offset, offset + 16 * 1024)
            offset += chunk.length
            response.write(
              `data: ${JSON.stringify({ choices: [{ delta: { content: chunk }, finish_reason: null, index: 0 }], id: 'short-session', object: 'chat.completion.chunk' })}\n\n`
            )
            setTimeout(writeNext, 10)
          }
          writeNext()
        } else {
          response.writeHead(200, { 'content-type': 'application/json' })
          response.end(
            JSON.stringify({
              choices: [{ finish_reason: 'stop', index: 0, message: { content: reply, role: 'assistant' } }],
              id: 'short-session',
              object: 'chat.completion'
            })
          )
        }
      })

      return
    }

    response.writeHead(404, { 'content-type': 'application/json' })
    response.end('{"error":"not found"}')
  })

  return new Promise((resolveStart, reject) => {
    server.on('error', error => {
      recordTransportError('server', error)

      if (!listening) reject(error)
    })
    server.listen(0, '127.0.0.1', () => {
      const address = server.address()

      if (!address || typeof address === 'string') {
        reject(new Error('mock inference server has no TCP address'))
        return
      }

      listening = true
      resolveStart({
        activeStreamingRequests: () => activeStreamingRequests,
        close: () =>
          new Promise(resolveClose => {
            let settled = false
            const settle = () => {
              if (settled) return
              settled = true
              resolveClose()
            }
            const timeout = setTimeout(() => {
              server.closeAllConnections?.()
              settle()
            }, 2_000)

            server.close(() => {
              clearTimeout(timeout)
              settle()
            })
            server.closeAllConnections?.()
          }),
        requestSummary: () => requests.map(request => ({ ...request })),
        streamingCompletionRequests: () => streamingCompletionRequests,
        transportErrors: () => transportErrors.map(error => ({ ...error })),
        waitForStreaming: async (timeoutMs = 30_000, baseline = 0) => {
          const deadline = Date.now() + timeoutMs
          while (activeStreamingRequests === 0 && streamingCompletionRequests <= baseline && Date.now() < deadline) {
            await sleep(20)
          }
          if (activeStreamingRequests === 0 && streamingCompletionRequests <= baseline) {
            throw new Error('streaming response did not start')
          }
        },
        url: `http://127.0.0.1:${address.port}`
      })
    })
  })
}
function writeSandboxConfig(home, mockUrl) {
  mkdirSync(home, { recursive: true })
  writeFileSync(
    join(home, 'config.yaml'),
    `model:\n  default: short-session-model\n  provider: custom:short-session\nauxiliary:\n  title_generation:\n    enabled: false\nproviders:\n  short-session:\n    api: ${mockUrl}/v1\n    transport: chat_completions\n    default_model: short-session-model\n    key_env: SHORT_SESSION_API_KEY\n`
  )
  writeFileSync(join(home, '.env'), 'SHORT_SESSION_API_KEY=local-diagnostic-only\n')
}

async function waitFor(cdp, expression, timeoutMs, label, ErrorType = Error) {
  const deadline = Date.now() + timeoutMs

  while (Date.now() < deadline) {
    try {
      if (await cdp.eval(expression)) {
        return
      }
    } catch {
      // Renderer is still loading.
    }

    await sleep(Math.min(250, Math.max(1, deadline - Date.now())))
  }

  throw new ErrorType(`timed out waiting for ${label}`)
}

async function screenshot(cdp, path) {
  try {
    await cdp.send('Page.enable')
    const shot = await cdp.send('Page.captureScreenshot', { format: 'png', fromSurface: true })
    writeFileSync(path, Buffer.from(shot.data, 'base64'))
  } catch {
    // A frozen renderer may not service screenshot capture.
  }
}

function reliablePid(pid) {
  return Number.isSafeInteger(pid) && pid > 1 ? pid : null
}

function processRows() {
  const result = spawnSync('ps', ['-axo', 'pid=,ppid=,%cpu=,%mem=,state=,etime=,command='], { encoding: 'utf8' })

  if (result.error || result.status !== 0) {
    const detail =
      result.error?.message ?? (String(result.stderr ?? '').trim() || `exit status ${result.status ?? 'unknown'}`)
    throw new Error(`process discovery failed: ${detail}`)
  }

  return String(result.stdout ?? '')
    .split(/\r?\n/)
    .map(line => {
      const match = line.match(/^\s*(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)\s+(\S+)\s+(.*)$/)

      return match
        ? {
            command: match[7],
            cpu: Number(match[3]),
            elapsed: match[6],
            memory: Number(match[4]),
            pid: Number(match[1]),
            ppid: Number(match[2]),
            state: match[5]
          }
        : null
    })
    .filter(Boolean)
}

function processTree(rootPid) {
  if (!reliablePid(rootPid)) {
    return []
  }

  let rows
  let discoveryError = null

  try {
    rows = processRows()
  } catch (error) {
    rows = []
    discoveryError = error instanceof Error ? error.message : String(error)
  }

  const selected = new Set([rootPid])
  let changed = true

  while (changed) {
    changed = false

    for (const row of rows) {
      if (!selected.has(row.pid) && selected.has(row.ppid)) {
        selected.add(row.pid)
        changed = true
      }
    }
  }

  const tree = rows.filter(row => selected.has(row.pid))
  const rootMissing = !tree.some(row => row.pid === rootPid)

  if (rootMissing) {
    tree.unshift({
      command: '<synthetic-root>',
      cpu: 0,
      elapsed: '',
      memory: 0,
      pid: rootPid,
      ppid: 0,
      state: '?',
      synthetic: true
    })
  }

  if (discoveryError) {
    tree.discoveryError = discoveryError
  }

  if (rootMissing) {
    tree.rootMissing = true
  }

  return tree
}

function redactCommand(command) {
  return command
    .replace(/([a-z][a-z0-9+.-]*:\/\/)[^\s/@]+(?::[^\s/@]*)?@/gi, '$1[REDACTED]@')
    .replace(/\b([A-Za-z_][A-Za-z0-9_]*(?:TOKEN|SECRET|PASSWORD|KEY|AUTH|COOKIE)[A-Za-z0-9_]*)=\S+/gi, '$1=[REDACTED]')
    .replace(/(--?\S*(?:token|secret|password|key|auth|cookie)\S*)(?:=|\s+)\S+/gi, '$1=[REDACTED]')
}

function processSnapshot(path, rootPid) {
  const tree = processTree(rootPid)
  const rows = tree.map(row => ({ ...row, command: redactCommand(row.command) }))
  writeFileSync(
    path,
    `${JSON.stringify({ rootPid, rows, discoveryError: tree.discoveryError ?? null, rootMissing: tree.rootMissing ?? false }, null, 2)}\n`
  )

  return rows
}

function sampleOne(path, pid) {
  if (process.platform !== 'darwin' || !reliablePid(pid)) {
    return
  }

  const result = spawnSync('sample', [String(pid), '5', '1'], { encoding: 'utf8', timeout: 8_000 })
  writeFileSync(path, `${result.stdout ?? ''}${result.stderr ?? ''}`)
}

function sampleTree(runDir, rootPid, prefix) {
  const rows = processTree(rootPid)
  sampleOne(join(runDir, `${prefix}-main.sample.txt`), rootPid)
  const renderer = rows
    .filter(row => row.pid !== rootPid && /(?:^|\s)--type=renderer(?:\s|$)/.test(row.command))
    .sort((a, b) => b.cpu - a.cpu)[0]

  if (renderer) {
    sampleOne(join(runDir, `${prefix}-renderer-${renderer.pid}.sample.txt`), renderer.pid)
  }
}

function captureDiagnostics(runDir, rootPid, prefix) {
  try {
    processSnapshot(join(runDir, `${prefix}-processes.txt`), rootPid)
  } catch {
    // Diagnostics are best effort and must not replace the original result.
  }

  try {
    sampleTree(runDir, rootPid, prefix)
  } catch {
    // Diagnostics are best effort and must not replace the original result.
  }
}

function liveCaptured(captured) {
  let current

  try {
    current = new Map(processRows().map(row => [row.pid, row]))
  } catch (error) {
    const live = captured.filter(original => pidIsLive(original.pid))
    live.discoveryError = error instanceof Error ? error.message : String(error)
    return live
  }

  return captured.filter(original => {
    if (original.synthetic) {
      return pidIsLive(original.pid)
    }

    const row = current.get(original.pid)

    return row && !row.state.startsWith('Z') && row.command === original.command
  })
}

function pidIsLive(pid) {
  try {
    process.kill(pid, 0)
    return true
  } catch (error) {
    return error?.code === 'EPERM'
  }
}

async function stopProcessTree(rootPid) {
  const captured = processTree(rootPid)
  const processDiscoveryErrors = new Set()

  if (captured.discoveryError) {
    processDiscoveryErrors.add(captured.discoveryError)
  }

  const currentLive = () => {
    const live = liveCaptured(captured)

    if (live.discoveryError) {
      processDiscoveryErrors.add(live.discoveryError)
    }

    return live
  }

  for (const { pid } of [...captured].reverse()) {
    try {
      process.kill(pid, 'SIGTERM')
    } catch {
      // It already exited.
    }
  }

  const deadline = Date.now() + 3_000

  while (Date.now() < deadline && currentLive().length > 0) {
    await sleep(100)
  }

  const remaining = currentLive()

  for (const { pid } of remaining.reverse()) {
    try {
      process.kill(pid, 'SIGKILL')
    } catch {
      // It exited between the liveness check and signal.
    }
  }

  const killDeadline = Date.now() + 2_000

  while (Date.now() < killDeadline && currentLive().length > 0) {
    await sleep(100)
  }

  return {
    captured: captured.map(row => row.pid),
    processDiscoveryErrors: [...processDiscoveryErrors],
    remainingAfterKill: currentLive().map(row => row.pid)
  }
}

function withWatchdog(task, onTimeout) {
  let timer

  return Promise.race([
    task,
    new Promise((_, reject) => {
      timer = setTimeout(() => {
        onTimeout()
        reject(new ReproductionError(`outer watchdog exceeded ${OUTER_WATCHDOG_MS}ms`))
      }, OUTER_WATCHDOG_MS)
    })
  ]).finally(() => clearTimeout(timer))
}

function withTimeout(task, timeoutMs, label, ErrorType = Error) {
  let timer

  return Promise.race([
    task,
    new Promise((_, reject) => {
      timer = setTimeout(() => reject(new ErrorType(`${label} exceeded ${timeoutMs}ms`)), timeoutMs)
    })
  ]).finally(() => clearTimeout(timer))
}

async function waitForResponsive(cdp, expression, timeoutMs, label, evaluationTimeoutMs = FREEZE_MS) {
  const deadline = Date.now() + timeoutMs

  while (Date.now() < deadline) {
    try {
      if (
        await withTimeout(cdp.eval(expression), evaluationTimeoutMs, `${label} renderer evaluation`, ReproductionError)
      ) {
        return
      }
    } catch (error) {
      if (error instanceof ReproductionError) {
        throw error
      }

      // Match the initial renderer-readiness polling: CDP can fail transiently while a document is replaced.
    }

    await sleep(Math.min(250, Math.max(1, deadline - Date.now())))
  }

  throw new Error(`timed out waiting for ${label} while the renderer remained responsive`)
}

async function waitForPredicate(predicate, timeoutMs, label) {
  const deadline = Date.now() + timeoutMs

  while (Date.now() < deadline) {
    if (await predicate()) {
      return
    }

    await sleep(Math.min(100, Math.max(1, deadline - Date.now())))
  }

  throw new Error(`timed out waiting for ${label}`)
}

async function verifyInteractiveSurfaces(cdp, timed, measure, label) {
  const sentinel = `short-session-sentinel-${label}`
  const composerPainted = await timed(`composer.paint.${label}`, async () => {
    const focused = await cdp.eval(
      `(() => { const el = document.querySelector('[data-slot="composer-rich-input"]'); if (!el || el.contentEditable !== 'true') return false; el.focus(); return true })()`
    )

    if (!focused) {
      return false
    }

    await cdp.send('Input.insertText', { text: sentinel })

    return cdp.eval(
      `document.querySelector('[data-slot="composer-rich-input"]')?.textContent?.includes(${JSON.stringify(sentinel)}) === true`
    )
  })

  if (!composerPainted) {
    throw new Error(`composer did not paint sentinel at ${label}`)
  }

  const version = await timed(`version.ipc.${label}`, () => cdp.eval('window.hermesDesktop.getVersion()'))

  if (typeof version?.appVersion !== 'string' || version.appVersion.length === 0) {
    throw new Error(`version IPC returned no appVersion at ${label}: ${JSON.stringify(version)}`)
  }

  await timed(`about.open.${label}`, () => cdp.eval("location.hash = '#/settings?tab=about'; true"))
  await measure(`about.ready.${label}`, () =>
    waitForResponsive(
      cdp,
      `document.body.textContent.includes(${JSON.stringify(version.appVersion)}) && !!document.querySelector('button[aria-label]')`,
      FREEZE_MS,
      `About settings at ${label}`
    )
  )
  const aboutClosed = await timed(`about.close.${label}`, () =>
    cdp.eval(
      `(() => { const close = document.querySelector('div[role="presentation"] > div > div:first-child button[aria-label]'); if (!close) return false; close.click(); return true })()`
    )
  )

  if (!aboutClosed) {
    throw new Error(`About settings close control unavailable at ${label}`)
  }

  await measure(`about.closed.${label}`, () =>
    waitForResponsive(cdp, "!location.hash.includes('/settings')", FREEZE_MS, `About settings close at ${label}`)
  )
  const interactive = await timed(`transcript.interactive.${label}`, () =>
    cdp.eval(
      `(() => { const viewport = document.querySelector('[data-slot="aui_thread-viewport"]'); const row = document.querySelector('[data-message-id]'); if (!viewport || !row) return false; const before = viewport.scrollTop; viewport.scrollTop = Math.min(viewport.scrollHeight, before + 40); viewport.dispatchEvent(new Event('scroll', { bubbles: true })); return getComputedStyle(row).pointerEvents !== 'none' })()`
    )
  )

  if (!interactive) {
    throw new Error(`transcript was not interactive at ${label}`)
  }

  return { sentinel, version }
}

function nativeWindowVisibilitySource() {
  return [
    'import AppKit',
    'import CoreGraphics',
    'import Foundation',
    'import Darwin',
    'guard CommandLine.arguments.count >= 3,',
    '      let processId = Int32(CommandLine.arguments[1]),',
    '      let application = NSRunningApplication(processIdentifier: pid_t(processId)) else {',
    '  fputs("target application was not found", stderr)',
    '  exit(2)',
    '}',
    'let process = pid_t(processId)',
    'let visible = CommandLine.arguments[2] == "visible"',
    'var changed: Bool',
    'if visible {',
    '  changed = application.unhide() || application.activate(options: [.activateIgnoringOtherApps, .activateAllWindows])',
    '} else {',
    '  let _ = application.activate(options: [.activateIgnoringOtherApps, .activateAllWindows])',
    '  usleep(100_000)',
    '  changed = application.hide()',
    '  if !changed, let source = CGEventSource(stateID: .hidSystemState),',
    '     let down = CGEvent(keyboardEventSource: source, virtualKey: 4, keyDown: true),',
    '     let up = CGEvent(keyboardEventSource: source, virtualKey: 4, keyDown: false) {',
    '    down.flags = .maskCommand',
    '    up.flags = .maskCommand',
    '    down.postToPid(process)',
    '    up.postToPid(process)',
    '    changed = true',
    '  }',
    '}',
    'if !changed {',
    '  fputs("native visibility request was rejected", stderr)',
    '  exit(3)',
    '}'
  ].join(String.fromCharCode(10))
}

function compileNativeWindowVisibilityHelper(helperPath) {
  const sourcePath = `${helperPath}.swift`
  writeFileSync(sourcePath, nativeWindowVisibilitySource())

  try {
    const result = spawnSync('/usr/bin/swiftc', [sourcePath, '-o', helperPath], {
      encoding: 'utf8',
      timeout: 30_000
    })

    if (result.status !== 0) {
      const detail = result.error?.message || result.stderr || result.stdout || String(result.status)
      throw new Error(`native macOS visibility helper compilation failed: ${detail}`)
    }
  } finally {
    rmSync(sourcePath, { force: true })
  }

  return helperPath
}

function setNativeWindowVisibility(helperPath, pid, visible) {
  if (process.platform !== 'darwin' || !helperPath || !Number.isSafeInteger(pid) || pid <= 1) {
    throw new Error(`native macOS visibility requires a helper and valid app pid, got ${pid}`)
  }

  const result = spawnSync(helperPath, [String(pid), visible ? 'visible' : 'hidden'], {
    encoding: 'utf8',
    timeout: 10_000
  })

  if (result.status !== 0) {
    const detail = result.error?.message || result.stderr || result.stdout || String(result.status)
    throw new Error(`native macOS visibility change failed: ${detail}`)
  }
}

async function runRealChatChecks(cdp, timed, measure, mock, runDir, appPid, nativeVisibilityHelper, getOperationCount) {
  const requestCountBefore = mock.streamingCompletionRequests()

  for (let exchange = 1; exchange <= EXCHANGE_COUNT; exchange += 1) {
    const streamingRequestsBefore = mock.streamingCompletionRequests()
    const beforeAssistant = await timed(`real-chat.assistant-count.${exchange}`, () =>
      cdp.eval(
        // Settled = no streaming marker anywhere in the row's subtree. The
        // marker moved off the message root onto a hidden leaf inside it — a
        // per-flip attribute write on the root widened style recalc across the
        // whole message subtree — so this counts the descendant instead of the
        // root's own attribute. Same rows, same gate.
        //
        // Counted by subtraction rather than with
        // `:not(:has([data-message-streaming="true"]))`, which makes the engine
        // walk every row's subtree on each evaluation — and this probe runs
        // inside the very latency window it is measuring, so that cost lands in
        // the number. At most one marker per row carries the attribute, so
        // roots - streaming is exactly the settled count.
        `document.querySelectorAll('[data-slot="aui_assistant-message-root"]').length - document.querySelectorAll('[data-message-streaming="true"]').length`
      )
    )
    const composer = await timed(`real-chat.composer-focus.${exchange}`, () =>
      cdp.eval(
        `(() => { const el = document.querySelector('[data-slot="composer-rich-input"]'); if (!el || el.contentEditable !== 'true') return false; el.focus(); return true })()`
      )
    )

    if (!composer) {
      throw new Error(`real chat composer unavailable at exchange ${exchange}`)
    }

    const prompt = `Deterministic real chat exchange ${exchange}`
    await timed(`real-chat.insert.${exchange}`, () => cdp.send('Input.insertText', { text: prompt }))
    const inserted = await timed(`real-chat.inserted.${exchange}`, () =>
      cdp.eval(
        `document.querySelector('[data-slot="composer-rich-input"]')?.textContent?.includes(${JSON.stringify(prompt)}) === true`
      )
    )

    if (!inserted) {
      throw new Error(`real chat prompt did not reach the composer at exchange ${exchange}`)
    }

    await measure(`real-chat.submit-ready.${exchange}`, () =>
      waitForResponsive(
        cdp,
        `!!document.querySelector('[data-slot="composer-root"] button[type="submit"]:not(:disabled)')`,
        FREEZE_MS,
        `real chat submit control ${exchange}`
      )
    )
    const submitted = await timed(`real-chat.submit.${exchange}`, () =>
      cdp.eval(
        `(() => { const button = document.querySelector('[data-slot="composer-root"] button[type="submit"]:not(:disabled)'); if (!button) return false; window.setTimeout(() => button.click(), 0); return true })()`
      )
    )

    if (!submitted) {
      throw new Error(`real chat submit control unavailable at exchange ${exchange}`)
    }

    try {
      await measure(`real-chat.stream-start.${exchange}`, () => mock.waitForStreaming(30_000, streamingRequestsBefore))
    } catch (error) {
      const state = await timed(`real-chat.dispatch-probe.${exchange}`, () =>
        cdp.eval(`({
          assistantMessages: document.querySelectorAll('[data-slot="aui_assistant-message-root"]').length,
          assistantText: document.querySelector('[data-slot="aui_assistant-message-root"]')?.textContent ?? null,
          composerText: document.querySelector('[data-slot="composer-rich-input"]')?.textContent ?? null,
          harness: window.__SHORT_SESSION_HANG_REPRO__.summary(),
          submitDisabled: document.querySelector('[data-slot="composer-root"] button[type="submit"]')?.disabled ?? null,
          userMessages: document.querySelectorAll('[data-slot="aui_user-message-root"]').length
        })`)
      )

      throw new Error(
        `${error instanceof Error ? error.message : String(error)}; mock=${JSON.stringify(mock.requestSummary())}; dispatch probe responded with state: ${JSON.stringify(state)}`
      )
    }
    await measure(`real-chat.native-hide.${exchange}`, () =>
      setNativeWindowVisibility(nativeVisibilityHelper, appPid, false)
    )
    await sleep(NATIVE_VISIBILITY_PAUSE_MS)
    await measure(`real-chat.native-restore.${exchange}`, () =>
      setNativeWindowVisibility(nativeVisibilityHelper, appPid, true)
    )

    try {
      await measure(`real-chat.mock-request.${exchange}`, () =>
        waitForPredicate(
          () => mock.streamingCompletionRequests() >= requestCountBefore + exchange,
          FREEZE_MS,
          `mock inference request ${exchange}`
        )
      )
    } catch (error) {
      const state = await timed(`real-chat.dispatch-probe.${exchange}`, () =>
        cdp.eval(`({
          assistantMessages: document.querySelectorAll('[data-slot="aui_assistant-message-root"]').length,
          composerText: document.querySelector('[data-slot="composer-rich-input"]')?.textContent ?? null,
          harness: window.__SHORT_SESSION_HANG_REPRO__.summary(),
          userMessages: document.querySelectorAll('[data-slot="aui_user-message-root"]').length
        })`)
      )

      throw new Error(
        `${error instanceof Error ? error.message : String(error)}; dispatch probe responded with state: ${JSON.stringify(state)}`
      )
    }

    await measure(`real-chat.assistant-response.${exchange}`, () =>
      waitForResponsive(
        cdp,
        // Same count-by-subtraction as the pre-send probe above: no `:has()`
        // subtree walk inside the responsiveness measurement window.
        `document.querySelectorAll('[data-slot="aui_assistant-message-root"]').length - document.querySelectorAll('[data-message-streaming="true"]').length > ${beforeAssistant}`,
        STREAM_RESPONSE_TIMEOUT_MS,
        `real assistant response ${exchange}`,
        STREAM_RESPONSE_EVALUATION_TIMEOUT_MS
      )
    )

    if (mock.streamingCompletionRequests() < requestCountBefore + exchange) {
      throw new Error(`mock inference request count did not advance for exchange ${exchange}`)
    }
  }

  const requestDelta = mock.streamingCompletionRequests() - requestCountBefore

  if (requestDelta !== EXCHANGE_COUNT) {
    throw new Error(`expected exactly ${EXCHANGE_COUNT} mock completion requests, observed ${requestDelta}`)
  }

  const streamHeartbeat = await timed('renderer.heartbeat.stream', () =>
    cdp.eval('window.__SHORT_SESSION_HANG_REPRO__.summary()')
  )
  await timed('renderer.heartbeat.checkpoint', () => cdp.eval('window.__SHORT_SESSION_HANG_REPRO__.checkpoint()'))
  const postStreamOperationStart = getOperationCount()
  const surfaces = await verifyInteractiveSurfaces(cdp, timed, measure, 'real-chat-exchange-5')
  await withTimeout(screenshot(cdp, join(runDir, 'real-chat-exchange-5.png')), 2_000, 'real chat screenshot').catch(
    () => {}
  )

  return {
    assistantResponses: EXCHANGE_COUNT,
    exchanges: EXCHANGE_COUNT,
    messageRecords: EXCHANGE_COUNT * 2,
    mockCompletionRequests: requestDelta,
    postStreamOperationStart,
    surfaces,
    streamMaxGapMs: Number(streamHeartbeat.maxGapMs || 0)
  }
}

async function runRendererChecks(cdp, label, runDir, mock, appPid, journalContract) {
  const operations = []
  const measure = async (name, body) => {
    const started = performance.now()
    const value = await Promise.resolve().then(body)
    const latencyMs = performance.now() - started
    operations.push({ name, latencyMs })

    return value
  }
  const timed = (name, body) =>
    measure(name, () => withTimeout(Promise.resolve().then(body), FREEZE_MS, name, ReproductionError))

  await cdp.send('Runtime.enable')
  await cdp.send('Profiler.enable')
  await cdp.send('Profiler.start')

  if (journalContract.storageKey !== LEGACY_STORAGE_KEY) {
    throw new Error(
      `journal storage-key contract mismatch: target=${journalContract.storageKey ?? 'missing'} harness=${LEGACY_STORAGE_KEY}`
    )
  }

  if (journalContract.migrationKey !== null && journalContract.migrationKey !== LEGACY_MIGRATION_KEY) {
    throw new Error(
      `journal migration-key contract mismatch: target=${journalContract.migrationKey} harness=${LEGACY_MIGRATION_KEY}`
    )
  }

  const checkpoints = []
  const fixtureManifest = null
  const nativeVisibilityHelper = compileNativeWindowVisibilityHelper(join(runDir, 'native-window-visibility'))
  const journalSeed = await measure('journal.seed.legacy', () =>
    withTimeout(
      cdp.eval(`(() => {
      const payload = 'j'.repeat(${LEGACY_SEED_PAYLOAD_CHARS})
      const entries = Object.fromEntries(Array.from({ length: ${LEGACY_SEED_ENTRY_COUNT} }, (_, index) => {
        const assistantId = 'legacy-seed-a-' + index
        return [
          'legacy-seed-' + index,
          {
            messages: [
              { id: 'legacy-seed-u-' + index, role: 'user', parts: [{ type: 'text', text: 'legacy seed prompt ' + index }] },
              { id: assistantId, role: 'assistant', parts: [{ type: 'text', text: payload }], pending: true }
            ],
            streamId: assistantId,
            turnStartedAt: Date.now(),
            updatedAt: Date.now()
          }
        ]
      }))
      const raw = JSON.stringify({ entries, version: 1 })
      localStorage.removeItem(${JSON.stringify(LEGACY_MIGRATION_KEY)})
      localStorage.setItem(${JSON.stringify(LEGACY_STORAGE_KEY)}, raw)
      return { bytes: raw.length, entries: Object.keys(entries).length }
    })()`),
      JOURNAL_SEED_TIMEOUT_MS,
      'journal.seed.legacy'
    )
  )

  if (journalSeed.bytes >= MAX_LEGACY_SEED_BYTES) {
    throw new Error(`legacy journal seed exceeded migration limit: ${journalSeed.bytes} bytes`)
  }

  if (journalContract.legacyStoreLimit !== null && journalSeed.bytes >= journalContract.legacyStoreLimit) {
    throw new Error(
      `legacy journal seed exceeds target migration limit: seed=${journalSeed.bytes} target=${journalContract.legacyStoreLimit}`
    )
  }

  await timed('harness.reset', () => cdp.eval('window.__SHORT_SESSION_HANG_REPRO__.reset()'))
  const realChat = await runRealChatChecks(
    cdp,
    timed,
    measure,
    mock,
    runDir,
    appPid,
    nativeVisibilityHelper,
    () => operations.length
  )

  const heartbeat = await timed('renderer.heartbeat.summary', () =>
    cdp.eval('window.__SHORT_SESSION_HANG_REPRO__.summary()')
  )
  const profile = await cdp.send('Profiler.stop')
  writeFileSync(join(runDir, 'renderer.cpuprofile'), JSON.stringify(profile.profile))

  const maxOperationMs = operations
    .slice(realChat.postStreamOperationStart)
    .reduce((max, operation) => Math.max(max, operation.latencyMs), 0)
  const maxGapMs = Number(heartbeat.maxGapMs || 0)
  const reproduced = maxOperationMs > FREEZE_MS || maxGapMs > FREEZE_MS

  return {
    checkpoints,
    fixtureManifest,
    hardFailure: reproduced,
    maxGapMs,
    maxOperationMs,
    operations,
    outcome: reproduced ? 'reproduced' : 'not-reproduced',
    journalSeed,
    realChat,
    syntheticScenario: 'disabled; real-chat-only'
  }
}

async function withTemporarySandbox(label, body) {
  const sandbox = mkdtempSync(join(tmpdir(), `hermes-short-session-${label}-`))

  try {
    return await body(sandbox)
  } finally {
    rmSync(sandbox, { force: true, recursive: true })
  }
}

async function allocateDebugPort() {
  const server = createServer()

  try {
    await new Promise((resolve, reject) => {
      server.once('error', reject)
      server.listen(0, '127.0.0.1', resolve)
    })

    const address = server.address()

    if (!address || typeof address === 'string') {
      throw new Error('temporary debug-port allocation returned no numeric address')
    }

    return address.port
  } finally {
    if (server.listening) {
      await new Promise(resolve => server.close(() => resolve()))
    }
  }
}

function resultForError(error) {
  const reproduced = error instanceof ReproductionError

  return {
    error: error instanceof Error ? (error.stack ?? error.message) : String(error),
    hardFailure: true,
    lifecycleSignals: [],
    maxGapMs: null,
    maxOperationMs: null,
    operations: [],
    outcome: reproduced ? 'reproduced' : 'harness-error'
  }
}

async function executeRun(target, index, warmup, mock, output) {
  return withTemporarySandbox(target.label, sandbox =>
    executeRunInSandbox(target, index, warmup, mock, output, sandbox)
  )
}

function runDirForAttempt(output, label, index, warmup, attempt = 1) {
  const base = join(output, label, warmup ? 'warmup' : `run-${index + 1}`)

  return attempt > 1 ? `${base}-attempt-${attempt}` : base
}

function promoteAttemptArtifacts(output, label, index, warmup, attempt) {
  if (attempt <= 1) {
    return
  }

  cpSync(runDirForAttempt(output, label, index, warmup, attempt), runDirForAttempt(output, label, index, warmup), {
    force: true,
    recursive: true
  })
}

async function executeRunInSandboxAttempt(target, index, warmup, mock, output, sandbox, attempt) {
  const runDir = runDirForAttempt(output, target.label, index, warmup, attempt)
  const hermesHome = join(sandbox, 'hermes-home')
  const userData = join(sandbox, 'electron-user-data')
  const desktopLog = join(hermesHome, 'logs', 'desktop.log')
  const stdoutPath = join(runDir, 'electron.stdout.log')
  const stderrPath = join(runDir, 'electron.stderr.log')
  const eventsPath = join(runDir, 'events.jsonl')
  const port = await allocateDebugPort()

  mkdirSync(runDir, { recursive: true })
  mkdirSync(userData, { recursive: true })
  writeSandboxConfig(hermesHome, mock.url)

  const electron = require('electron')
  const stdoutLog = createWriteStream(stdoutPath)
  const stderrLog = createWriteStream(stderrPath)
  const stdoutFlushed = finished(stdoutLog).then(
    () => null,
    error => error
  )
  const stderrFlushed = finished(stderrLog).then(
    () => null,
    error => error
  )
  const child = spawn(
    electron,
    [target.targetDesktop, `--user-data-dir=${userData}`, `--remote-debugging-port=${port}`],
    {
      cwd: target.targetDesktop,
      env: sanitizedEnv({
        HERMES_DESKTOP_APP_NAME: `HermesShortSession-${target.label}-${process.pid}-${index}-${warmup ? 'w' : 'm'}-${attempt}`,
        HERMES_DESKTOP_HERMES_ROOT: target.targetRoot,
        HERMES_DESKTOP_IGNORE_EXISTING: '1',
        HERMES_DESKTOP_USER_DATA_DIR: userData,
        HERMES_HOME: hermesHome,
        SHORT_SESSION_API_KEY: 'local-diagnostic-only'
      }),
      stdio: ['ignore', 'pipe', 'pipe']
    }
  )

  let exited = null
  let spawnFailed = false
  let stopping = false
  let resolveSpawn
  let rejectSpawn
  const spawnReady = new Promise((resolve, reject) => {
    resolveSpawn = resolve
    rejectSpawn = reject
  })
  child.once('spawn', () => resolveSpawn())
  child.once('error', error => {
    spawnFailed = true
    exited = exited ?? { code: null, signal: null }
    appendFileSync(
      eventsPath,
      `${JSON.stringify({ at: new Date().toISOString(), error: error.message, type: 'spawn-error' })}\n`
    )
    rejectSpawn(error)
  })
  if (child.stdout) {
    child.stdout.pipe(stdoutLog)
  } else {
    stdoutLog.end()
  }

  if (child.stderr) {
    child.stderr.pipe(stderrLog)
  } else {
    stderrLog.end()
  }
  child.once('exit', (code, signal) => {
    exited = { code, signal }
    appendFileSync(
      eventsPath,
      `${JSON.stringify({ at: new Date().toISOString(), code, signal, type: stopping ? 'teardown-exit' : 'unexpected-exit' })}\n`
    )
  })

  let cdp
  let result

  try {
    await withTimeout(spawnReady, FREEZE_MS, 'Electron spawn')
    const targetInfo = await withTimeout(discoverTarget({ port, timeoutMs: 60_000 }), 65_000, 'CDP target discovery')
    cdp = await withTimeout(CDP.open(targetInfo.webSocketDebuggerUrl), 10_000, 'CDP connection')
    await waitFor(cdp, '!!window.__SHORT_SESSION_HANG_REPRO__', 60_000, 'short-session renderer harness')
    await waitFor(
      cdp,
      `document.querySelector('[data-slot="composer-rich-input"]')?.contentEditable === 'true'`,
      60_000,
      'interactive composer'
    )

    result = await withWatchdog(
      runRendererChecks(cdp, `${target.label}-${index + 1}`, runDir, mock, child.pid, target.journalContract),
      () => {
        captureDiagnostics(runDir, child.pid, 'hard-timeout')
      }
    )
  } catch (error) {
    result = resultForError(error)
    captureDiagnostics(runDir, child.pid, 'failure')
    if (cdp) {
      await withTimeout(screenshot(cdp, join(runDir, 'failure.png')), 2_000, 'failure screenshot').catch(() => {})
    }
  } finally {
    cdp?.close()
    const logText = existsSync(desktopLog) ? readFileSync(desktopLog, 'utf8') : ''
    const lifecycleSignals = logText
      .split(/\r?\n/)
      .filter(line => /webContents became unresponsive|render-process-gone/i.test(line))
    const lifecycleReproduced = lifecycleSignals.length > 0 || Boolean(exited && !spawnFailed)

    result = { ...(result ?? resultForError(new Error('run produced no result'))), lifecycleSignals }

    if (lifecycleReproduced) {
      result.hardFailure = true
      result.outcome = 'reproduced'
    }

    stopping = true
    const teardown = await stopProcessTree(child.pid)
    const logFlushErrors = await withTimeout(
      Promise.all([stdoutFlushed, stderrFlushed]),
      FREEZE_MS,
      'Electron log flush'
    ).catch(error => [error])
    writeFileSync(join(runDir, 'teardown.json'), `${JSON.stringify(teardown, null, 2)}\n`)

    if (existsSync(desktopLog)) {
      copyFileSync(desktopLog, join(runDir, 'desktop.log'))
    }

    if (teardown.remainingAfterKill.length > 0) {
      result = {
        ...(result ?? {}),
        error: `${result?.error ? `${result.error}\n` : ''}process teardown leaked PIDs ${teardown.remainingAfterKill.join(', ')}`,
        hardFailure: true,
        outcome: result?.outcome === 'reproduced' ? 'reproduced' : 'harness-error'
      }
    }

    if (teardown.processDiscoveryErrors.length > 0) {
      result = {
        ...(result ?? {}),
        error: `${result?.error ? `${result.error}\n` : ''}${teardown.processDiscoveryErrors.join('\n')}`,
        hardFailure: true,
        outcome: result?.outcome === 'reproduced' ? 'reproduced' : 'harness-error'
      }
    }

    const logFlushError = logFlushErrors.find(Boolean)

    if (logFlushError) {
      result = {
        ...(result ?? {}),
        error: `${result?.error ? `${result.error}\n` : ''}Electron log flush failed: ${logFlushError instanceof Error ? logFlushError.message : String(logFlushError)}`,
        hardFailure: true,
        outcome: result?.outcome === 'reproduced' ? 'reproduced' : 'harness-error'
      }
    }
  }

  writeFileSync(join(runDir, 'result.json'), `${JSON.stringify(result, null, 2)}\n`)
  appendFileSync(eventsPath, `${JSON.stringify({ at: new Date().toISOString(), result, type: 'run-complete' })}\n`)

  return result
}

function isDebugStartupFailure(result) {
  return !result.lifecycleSignals?.length && /CDP (?:target discovery|connection)/i.test(result.error ?? '')
}

async function executeRunInSandbox(target, index, warmup, mock, output, sandbox) {
  for (let attempt = 1; attempt <= 2; attempt += 1) {
    const result = await executeRunInSandboxAttempt(target, index, warmup, mock, output, sandbox, attempt)

    if (attempt === 1 && isDebugStartupFailure(result)) {
      rmSync(join(sandbox, 'electron-user-data'), { force: true, recursive: true })
      continue
    }

    promoteAttemptArtifacts(output, target.label, index, warmup, attempt)

    return result
  }

  throw new Error('unreachable debug startup retry state')
}

function classify(results, warmup) {
  const invalid = [warmup, ...results].some(result => result.outcome === 'harness-error')
  const reproduced = results.filter(result => result.outcome === 'reproduced').length
  const reproducedThreshold = Math.ceil(results.length * 0.8)

  return {
    classification: invalid
      ? 'invalid'
      : reproduced >= reproducedThreshold
        ? 'reproduced'
        : reproduced === 0
          ? 'not-reproduced'
          : 'intermittent',
    invalid,
    reproduced,
    reproducedThreshold,
    total: results.length
  }
}

function pairedSoftSignal(baseline, candidate) {
  if (baseline.length === 0 || candidate.length === 0) {
    return { material: false, materialPairs: 0, materialThreshold: 0, reason: 'insufficient-runs', thresholdPct: 30 }
  }

  if ([...baseline, ...candidate].some(result => result.outcome !== 'not-reproduced')) {
    return { material: false, materialPairs: 0, materialThreshold: 0, reason: 'hard-or-invalid-run', thresholdPct: 30 }
  }

  let materialPairs = 0

  for (let i = 0; i < Math.min(baseline.length, candidate.length); i += 1) {
    const base = Math.max(1, baseline[i].maxGapMs || 0, baseline[i].maxOperationMs || 0)
    const next = Math.max(candidate[i].maxGapMs || 0, candidate[i].maxOperationMs || 0)

    if (next >= base * 1.3) {
      materialPairs += 1
    }
  }

  const materialThreshold = Math.ceil(Math.min(baseline.length, candidate.length) * 0.8)

  return { material: materialPairs >= materialThreshold, materialPairs, materialThreshold, thresholdPct: 30 }
}

function validateArtifactBundle(output, repetitions) {
  const required = ['environment.json', 'summary.json', 'baseline-build.log', 'candidate-build.log']

  for (const label of ['baseline', 'candidate']) {
    for (const runName of ['warmup', ...Array.from({ length: repetitions }, (_, index) => `run-${index + 1}`)]) {
      for (const artifact of ['events.jsonl', 'result.json', 'teardown.json']) {
        required.push(join(label, runName, artifact))
      }
    }
  }

  for (const relativePath of required) {
    const path = join(output, relativePath)

    if (!existsSync(path) || readFileSync(path).byteLength === 0) {
      throw new Error(`missing or empty required diagnostic artifact: ${relativePath}`)
    }
  }

  const summary = JSON.parse(readFileSync(join(output, 'summary.json'), 'utf8'))

  validateSummary(summary, repetitions)
}

function validateSummary(summary, repetitions) {
  if (!Number.isInteger(repetitions) || repetitions < 1 || repetitions > 20) {
    throw new Error('invalid summary repetitions')
  }

  let invalid = false

  for (const label of ['baseline', 'candidate']) {
    if (!summary[label]?.warmup || summary[label].runs?.length !== repetitions) {
      throw new Error(`invalid ${label} summary shape`)
    }

    for (const result of [summary[label].warmup, ...summary[label].runs]) {
      if (!['harness-error', 'not-reproduced', 'reproduced'].includes(result.outcome)) {
        throw new Error(`invalid ${label} run outcome`)
      }

      const expectedHardFailure = result.outcome !== 'not-reproduced'

      if (result.hardFailure !== expectedHardFailure) {
        throw new Error(`inconsistent ${label} run outcome and hardFailure`)
      }
    }

    const derived = classify(summary[label].runs, summary[label].warmup)

    for (const field of ['classification', 'invalid', 'reproduced', 'reproducedThreshold', 'total']) {
      if (summary[label][field] !== derived[field]) {
        throw new Error(`inconsistent ${label} summary ${field}`)
      }
    }

    invalid ||= derived.invalid
  }

  if (summary.invalid !== invalid) {
    throw new Error('inconsistent summary invalid state')
  }
}

async function main() {
  const options = parseArgs(process.argv.slice(2))

  if (!existsSync(HARNESS_SOURCE)) {
    throw new Error(`renderer harness missing: ${HARNESS_SOURCE}`)
  }

  validateRef(options.baseline)
  validateRef(options.candidate)
  const baselineSha = resolveRef(options.baseline)
  const candidateSha = resolveRef(options.candidate)
  const mergeBase = run('git', ['merge-base', '--', baselineSha, candidateSha])
  const ancestor = spawnSync('git', ['merge-base', '--is-ancestor', '--', baselineSha, candidateSha], {
    cwd: REPO_ROOT,
    encoding: 'utf8'
  })

  if (mergeBase !== baselineSha || ancestor.status !== 0) {
    throw new Error(
      `baseline must be the exact merge-base and an ancestor of candidate; baseline=${baselineSha} merge-base=${mergeBase} candidate=${candidateSha}`
    )
  }

  const baselineMetadata = readTargetMetadata(baselineSha)
  const candidateMetadata = readTargetMetadata(candidateSha)
  const harnessHead = run('git', ['rev-parse', '--verify', '--end-of-options', 'HEAD^{commit}'])
  const harnessMetadata = readTargetMetadata(harnessHead)

  if (
    JSON.stringify(baselineMetadata) !== JSON.stringify(candidateMetadata) ||
    JSON.stringify(baselineMetadata) !== JSON.stringify(harnessMetadata)
  ) {
    throw new Error(
      `shared dependency mismatch; harness checkout, baseline, and candidate Electron/lockfile metadata must match:\nharness ${JSON.stringify(harnessMetadata)}\nbaseline ${JSON.stringify(baselineMetadata)}\ncandidate ${JSON.stringify(candidateMetadata)}`
    )
  }

  const plan = {
    baseline: { ref: options.baseline, sha: baselineSha },
    candidate: { ref: options.candidate, sha: candidateSha },
    harness: { head: harnessHead, metadata: harnessMetadata },
    harnessSha256: sha256(readFileSync(HARNESS_SOURCE)),
    mergeBase,
    metadata: baselineMetadata,
    repetitions: options.repetitions,
    runner: { arch: process.arch, platform: process.platform, versions: process.versions }
  }

  if (options.dryRun) {
    console.log(JSON.stringify(plan, null, 2))

    return
  }

  if (process.platform !== 'darwin' || process.arch !== 'arm64') {
    throw new Error(
      `the short-session hang diagnostic requires macOS arm64; got ${process.platform}-${process.arch} (use --dry-run for preflight)`
    )
  }

  mkdirSync(options.output, { recursive: true })
  writeFileSync(join(options.output, 'environment.json'), `${JSON.stringify(plan, null, 2)}\n`)

  const ephemeralRoot = mkdtempSync(join(tmpdir(), 'hermes-short-session-ab-'))
  const prepared = []
  const mock = await startMockInference()

  try {
    prepared.push(prepareTarget('baseline', baselineSha, ephemeralRoot, options.output))
    prepared.push(prepareTarget('candidate', candidateSha, ephemeralRoot, options.output))
    const byLabel = Object.fromEntries(prepared.map(target => [target.label, target]))

    const warmups = {}

    for (const label of ['baseline', 'candidate']) {
      warmups[label] = await executeRun(byLabel[label], 0, true, mock, options.output)
    }

    const measured = { baseline: [], candidate: [] }

    for (let index = 0; index < options.repetitions; index += 1) {
      const order = index % 2 === 0 ? ['baseline', 'candidate'] : ['candidate', 'baseline']

      for (const label of order) {
        measured[label].push(await executeRun(byLabel[label], index, false, mock, options.output))
      }
    }

    const baselineClassification = classify(measured.baseline, warmups.baseline)
    const candidateClassification = classify(measured.candidate, warmups.candidate)
    const invalid = baselineClassification.invalid || candidateClassification.invalid
    const summary = {
      ...plan,
      invalid,
      baseline: { ...plan.baseline, ...baselineClassification, runs: measured.baseline, warmup: warmups.baseline },
      candidate: {
        ...plan.candidate,
        ...candidateClassification,
        runs: measured.candidate,
        warmup: warmups.candidate
      },
      softSignal: invalid
        ? { material: false, materialPairs: 0, materialThreshold: 0, reason: 'invalid-run', thresholdPct: 30 }
        : pairedSoftSignal(measured.baseline, measured.candidate),
      mockRequests: mock.requestSummary(),
      mockTransportErrors: mock.transportErrors()
    }
    writeFileSync(join(options.output, 'summary.json'), `${JSON.stringify(summary, null, 2)}\n`)
    validateArtifactBundle(options.output, options.repetitions)
    console.log(JSON.stringify(summary, null, 2))

    // A warm-up reproduction can be the one-shot v1 migration or first-paint
    // cost. It is retained in the summary, but only a warm-up harness error or
    // a measured hard failure makes the A/B job fail.
    if (
      warmups.baseline.outcome === 'harness-error' ||
      warmups.candidate.outcome === 'harness-error' ||
      measured.baseline.some(result => result.hardFailure) ||
      measured.candidate.some(result => result.hardFailure)
    ) {
      process.exitCode = 1
    }
  } finally {
    await mock.close()

    if (!options.keepWorktrees) {
      for (const target of prepared.reverse()) {
        try {
          run('git', ['worktree', 'remove', '--force', target.targetRoot])
        } catch (error) {
          console.error(`warning: failed to remove ${target.targetRoot}: ${error.message}`)
        }
      }

      rmSync(ephemeralRoot, { force: true, recursive: true })
    } else {
      console.log(`kept ephemeral worktrees at ${ephemeralRoot}`)
    }
  }
}

if (resolve(process.argv[1] ?? '') === fileURLToPath(import.meta.url)) {
  main().catch(error => {
    console.error(error instanceof Error ? (error.stack ?? error.message) : String(error))
    process.exitCode = 2
  })
}

export {
  ReproductionError,
  classify,
  pairedSoftSignal,
  resultForError,
  validateArtifactBundle,
  validateSummary,
  waitFor,
  waitForPredicate,
  waitForResponsive,
  withTemporarySandbox,
  withTimeout
}
