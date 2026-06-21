'use strict'

const fs = require('node:fs')
const http = require('node:http')
const https = require('node:https')
const path = require('node:path')
const { execFileSync, spawn: defaultSpawn } = require('node:child_process')

const DEFAULT_LANGFLOW_ROOT = process.env.HERMES_DESKTOP_LANGFLOW_ROOT
  ? path.resolve(process.env.HERMES_DESKTOP_LANGFLOW_ROOT)
  : ''
const DEFAULT_LANGFLOW_HOST = process.env.HERMES_DESKTOP_LANGFLOW_HOST || '127.0.0.1'
const DEFAULT_LANGFLOW_PORT = Number(process.env.HERMES_DESKTOP_LANGFLOW_PORT || 7860)
const DEFAULT_READY_TIMEOUT_MS = 10 * 60 * 1000
const DEFAULT_POLL_INTERVAL_MS = 1_000
const DEFAULT_PROBE_TIMEOUT_MS = 1_500
const DEFAULT_STOP_TIMEOUT_MS = 8_000
const DEFAULT_KARI_CLOUD_BASE_URL = process.env.VITE_KARI_CLOUD_URL || 'https://lotjc.com/hermes'
const LANGFLOW_FRONTEND_RELATIVE_PATH = 'src/backend/base/langflow/frontend'

function workflowUrl(host = DEFAULT_LANGFLOW_HOST, port = DEFAULT_LANGFLOW_PORT) {
  return `http://${host}:${port}`
}

function isMissingFileError(error) {
  return error?.code === 'ENOENT' || error?.code === 'ENOTDIR'
}

function assignStringEnv(target, key, value) {
  const normalizedKey = String(key || '').trim()

  if (!normalizedKey || value === undefined || value === null) {
    return
  }

  const normalizedValue = String(value).trim()

  if (!normalizedValue) {
    return
  }

  target[normalizedKey] = normalizedValue
}

function readWorkflowSecrets(secretsPath = '', readFile = fs.readFileSync) {
  if (!secretsPath) {
    return {}
  }

  try {
    const raw = readFile(secretsPath, 'utf8')
    const parsed = JSON.parse(String(raw || '{}'))

    return parsed && typeof parsed === 'object' ? parsed : {}
  } catch (error) {
    if (isMissingFileError(error)) {
      return {}
    }

    throw error
  }
}

function normalizeBaseUrl(value) {
  return String(value || '').trim().replace(/\/+$/, '')
}

async function getWorkflowAuthStatus({
  probe = probeHttpUrl,
  readFile = fs.readFileSync,
  secretsPath = ''
} = {}) {
  const secrets = readWorkflowSecrets(secretsPath, readFile)
  const kari = secrets?.kari && typeof secrets.kari === 'object' ? secrets.kari : {}
  const token = String(kari.token || '').trim()
  const cloudBaseUrl = normalizeBaseUrl(kari.cloudBaseURL || kari.cloudBaseUrl || '')
  const fallbackCloudBaseUrl = normalizeBaseUrl(DEFAULT_KARI_CLOUD_BASE_URL)

  if (!token) {
    return {
      cloudBaseUrl: cloudBaseUrl || fallbackCloudBaseUrl,
      cloudReachable: false,
      error: null,
      loggedIn: false
    }
  }

  if (!cloudBaseUrl) {
    return {
      cloudBaseUrl: fallbackCloudBaseUrl,
      cloudReachable: false,
      error: 'Kari hub is not configured. Please log in to Workflow again.',
      loggedIn: false
    }
  }

  let reachable = false
  try {
    reachable = Boolean(await probe(cloudBaseUrl))
  } catch {
    reachable = false
  }

  if (!reachable) {
    return {
      cloudBaseUrl,
      cloudReachable: false,
      error: `Kari hub is not reachable: ${cloudBaseUrl}. Please log in to Workflow again or start the local hub.`,
      loggedIn: false
    }
  }

  return {
    cloudBaseUrl,
    cloudReachable: true,
    error: null,
    loggedIn: true
  }
}

function workflowSecretsEnv(secrets) {
  const result = {}

  if (!secrets || typeof secrets !== 'object') {
    return result
  }

  if (secrets.env && typeof secrets.env === 'object') {
    for (const [key, value] of Object.entries(secrets.env)) {
      assignStringEnv(result, key, value)
    }
  }

  if (secrets.kari && typeof secrets.kari === 'object') {
    // 桌面端只注入云端中枢地址 + per-user token;真实 KIE_API_KEY 只在云端(kari-cloud),
    // 绝不下发到本地。节点经 KARI_HUB_URL 凭 token relay 出图/扣费(见 lfx.kari_media / kari_billing)。
    assignStringEnv(result, 'KARI_HUB_URL', secrets.kari.cloudBaseURL || secrets.kari.cloudBaseUrl)
    assignStringEnv(result, 'KARI_WORKSPACE_TOKEN', secrets.kari.token)
  }

  if (secrets.openai && typeof secrets.openai === 'object') {
    assignStringEnv(result, 'OPENAI_API_KEY', secrets.openai.apiKey)
    assignStringEnv(result, 'OPENAI_BASE_URL', secrets.openai.baseURL || secrets.openai.baseUrl)
    assignStringEnv(result, 'KARI_LLM_PERFORMANCE_API_KEY', secrets.openai.apiKey)
    assignStringEnv(result, 'KARI_LLM_PERFORMANCE_BASE_URL', secrets.openai.baseURL || secrets.openai.baseUrl)
    assignStringEnv(result, 'KARI_LLM_PERFORMANCE_MODEL', secrets.openai.model || '性能')
  }

  if (secrets.anthropic && typeof secrets.anthropic === 'object') {
    assignStringEnv(result, 'ANTHROPIC_API_KEY', secrets.anthropic.apiKey)
    assignStringEnv(result, 'ANTHROPIC_BASE_URL', secrets.anthropic.baseURL || secrets.anthropic.baseUrl)
    assignStringEnv(result, 'KARI_LLM_EXTREME_API_KEY', secrets.anthropic.apiKey)
    assignStringEnv(result, 'KARI_LLM_EXTREME_BASE_URL', secrets.anthropic.baseURL || secrets.anthropic.baseUrl)
    assignStringEnv(result, 'KARI_LLM_EXTREME_MODEL', secrets.anthropic.model || '极致')
  }

  return result
}

// macOS/Linux GUI apps launched via Finder/`open` inherit a minimal PATH
// (/usr/bin:/bin:...) that omits where uv/node/npm actually live. Without this
// augmentation the embedded Langflow launch fails with `spawn uv ENOENT` and the
// make fallback dies in install_frontend (Error 127). We prepend the common dev
// tool dirs so the spawned backend can find its toolchain.
const POSIX_TOOL_PATH_DIRS = ['/opt/homebrew/bin', '/usr/local/bin', '/usr/bin', '/bin', '/usr/sbin', '/sbin']
const HOME_TOOL_PATH_DIRS = ['.local/bin', '.cargo/bin']

function buildAugmentedPath({ env = process.env, exists = fs.existsSync, home = '', platform = process.platform } = {}) {
  const existing = String(env.PATH || env.Path || '')

  if (platform === 'win32') {
    return existing
  }

  const homeDir = home || env.HOME || ''
  const candidates = []

  if (homeDir) {
    for (const rel of HOME_TOOL_PATH_DIRS) {
      candidates.push(path.join(homeDir, rel))
    }
  }

  candidates.push(...POSIX_TOOL_PATH_DIRS)

  const existingDirs = existing ? existing.split(path.delimiter) : []
  const seen = new Set(existingDirs.map(dir => dir.trim()).filter(Boolean))
  const prepend = []

  for (const dir of candidates) {
    if (seen.has(dir) || !exists(dir)) {
      continue
    }

    seen.add(dir)
    prepend.push(dir)
  }

  return [...prepend, ...existingDirs].filter(Boolean).join(path.delimiter)
}

function buildLangflowRuntimeEnv(
  configDir = '',
  env = process.env,
  { exists = fs.existsSync, platform = process.platform, readFile = fs.readFileSync, secretsPath = '' } = {}
) {
  const resolvedConfigDir = configDir ? path.resolve(configDir) : ''
  const secretsEnv = workflowSecretsEnv(readWorkflowSecrets(secretsPath, readFile))
  const runtimeEnv = {
    ...env,
    ...secretsEnv,
    LANGFLOW_AUTO_LOGIN: 'true',
    LANGFLOW_SKIP_AUTH_AUTO_LOGIN: 'true',
    // 本地为主:嵌入的 langflow 以 workspace 模式跑 —— 跳过 hub 的"创建办公空间"门槛,
    // 直接进画布(/status 返回 hasWorkspace:true、无子域名)。
    KARI_MODE: 'workspace',
    PATH: buildAugmentedPath({ env, exists, platform }),
    PYTHONUNBUFFERED: '1'
  }

  if (resolvedConfigDir) {
    runtimeEnv.LANGFLOW_CONFIG_DIR = resolvedConfigDir
    runtimeEnv.KARI_PERMS_DB = path.join(resolvedConfigDir, '.kari_perms.sqlite')
  }

  return runtimeEnv
}

function buildLangflowLaunchOptions(root, runtimeEnv) {
  return {
    cwd: root,
    env: runtimeEnv,
    detached: true,
    shell: false,
    windowsHide: true
  }
}

function hasBuiltLangflowFrontend(root, exists = fs.existsSync) {
  if (!root) {
    return false
  }

  return exists(path.join(root, ...LANGFLOW_FRONTEND_RELATIVE_PATH.split('/'), 'index.html'))
}

function buildDirectLangflowLaunch({ configDir = '', env = process.env, exists, host, port, readFile, root, secretsPath }) {
  const runtimeEnv = buildLangflowRuntimeEnv(configDir, env, { exists, readFile, secretsPath })
  // Kari 自定义节点(图片/视频/大模型/网站/任务…)进画布组件面板。
  if (root) runtimeEnv.LANGFLOW_COMPONENTS_PATH = path.join(root, 'kari_components')

  return {
    mode: 'direct',
    command: 'uv',
    args: [
      'run',
      'langflow',
      'run',
      '--frontend-path',
      LANGFLOW_FRONTEND_RELATIVE_PATH,
      '--log-level',
      'debug',
      '--host',
      host,
      '--port',
      String(port),
      '--env-file',
      '.env',
      '--no-open-browser'
    ],
    options: buildLangflowLaunchOptions(root, runtimeEnv)
  }
}

function buildMakeLangflowLaunch({ configDir = '', env = process.env, exists, host, port, readFile, root, secretsPath }) {
  const runtimeEnv = buildLangflowRuntimeEnv(configDir, env, { exists, readFile, secretsPath })
  // Kari 自定义节点进画布组件面板。
  if (root) runtimeEnv.LANGFLOW_COMPONENTS_PATH = path.join(root, 'kari_components')

  return {
    mode: 'make',
    command: 'make',
    args: ['run_cli', `host=${host}`, `port=${port}`, 'open_browser=false'],
    options: buildLangflowLaunchOptions(root, runtimeEnv)
  }
}

function buildLangflowLaunches({
  configDir = '',
  env = process.env,
  exists = fs.existsSync,
  host = DEFAULT_LANGFLOW_HOST,
  port = DEFAULT_LANGFLOW_PORT,
  readFile = fs.readFileSync,
  root = DEFAULT_LANGFLOW_ROOT,
  secretsPath = ''
} = {}) {
  const makeLaunch = buildMakeLangflowLaunch({ configDir, env, exists, host, port, readFile, root, secretsPath })

  if (!hasBuiltLangflowFrontend(root, exists)) {
    return [makeLaunch]
  }

  return [buildDirectLangflowLaunch({ configDir, env, exists, host, port, readFile, root, secretsPath }), makeLaunch]
}

function buildLangflowLaunch(options = {}) {
  return buildLangflowLaunches(options)[0]
}

function isLangflowRoot(root, exists = fs.existsSync) {
  if (!root) {
    return false
  }

  return exists(path.join(root, 'Makefile')) && exists(path.join(root, 'pyproject.toml'))
}

function resolveLangflowRoot({ candidates = [], env = process.env, exists = fs.existsSync } = {}) {
  const configured = String(env?.HERMES_DESKTOP_LANGFLOW_ROOT || '').trim()

  if (configured) {
    return path.resolve(configured)
  }

  for (const candidate of candidates) {
    const resolved = path.resolve(String(candidate || ''))

    if (isLangflowRoot(resolved, exists)) {
      return resolved
    }
  }

  return ''
}

function killProcessTree(
  child,
  { execFileSync: execSync = execFileSync, kill = process.kill, platform = process.platform } = {}
) {
  if (!child?.pid || child.killed) {
    return false
  }

  if (platform === 'win32') {
    try {
      execSync('taskkill', ['/PID', String(child.pid), '/T', '/F'], {
        stdio: 'ignore',
        windowsHide: true
      })
      return true
    } catch {
      // Fall through to the direct child as a best-effort fallback.
    }
  } else {
    try {
      kill(-child.pid, 'SIGTERM')
      return true
    } catch {
      // Fall through to the direct child as a best-effort fallback.
    }
  }

  if (typeof child.kill === 'function') {
    try {
      return child.kill('SIGTERM')
    } catch {
      return false
    }
  }

  return false
}

function probeHttpUrl(rawUrl, timeoutMs = DEFAULT_PROBE_TIMEOUT_MS) {
  return new Promise(resolve => {
    let settled = false
    const parsed = new URL(rawUrl)
    const client = parsed.protocol === 'https:' ? https : http

    const finish = value => {
      if (settled) return
      settled = true
      resolve(value)
    }

    const req = client.request(
      parsed,
      {
        method: 'GET',
        timeout: timeoutMs
      },
      res => {
        res.resume()
        finish((res.statusCode || 500) < 500)
      }
    )

    req.on('error', () => finish(false))
    req.on('timeout', () => {
      req.destroy()
      finish(false)
    })
    req.end()
  })
}

function wireLogStream(stream, emitLine) {
  if (!stream || typeof stream.on !== 'function') {
    return
  }

  let buffer = ''
  stream.on('data', chunk => {
    buffer += chunk.toString()
    let newlineIndex

    while ((newlineIndex = buffer.indexOf('\n')) !== -1) {
      const line = buffer.slice(0, newlineIndex).trimEnd()
      buffer = buffer.slice(newlineIndex + 1)

      if (line) {
        emitLine(line)
      }
    }
  })
}

function createWorkflowBackendManager({
  configDir = '',
  env = process.env,
  exists = fs.existsSync,
  host = DEFAULT_LANGFLOW_HOST,
  killTree = killProcessTree,
  log = () => {},
  pollIntervalMs = DEFAULT_POLL_INTERVAL_MS,
  port = DEFAULT_LANGFLOW_PORT,
  probe = probeHttpUrl,
  readFile = fs.readFileSync,
  readyTimeoutMs = DEFAULT_READY_TIMEOUT_MS,
  root = DEFAULT_LANGFLOW_ROOT,
  secretsPath = '',
  spawn = defaultSpawn,
  stopTimeoutMs = DEFAULT_STOP_TIMEOUT_MS
} = {}) {
  const url = workflowUrl(host, port)
  let child = null
  let error = null
  let external = false
  let readyTimer = null
  let state = 'stopped'
  let stopping = false

  function clearReadyTimer() {
    if (!readyTimer) {
      return
    }

    clearTimeout(readyTimer)
    readyTimer = null
  }

  function snapshot() {
    return {
      error,
      external,
      pid: child?.pid || null,
      root,
      state,
      url
    }
  }

  function setError(message) {
    error = message
    external = false
    state = 'error'
  }

  function markReady(isExternal) {
    clearReadyTimer()
    error = null
    external = Boolean(isExternal)
    state = 'ready'
  }

  function scheduleReadyProbe(activeChild, deadline = Date.now() + readyTimeoutMs) {
    clearReadyTimer()

    const tick = async () => {
      readyTimer = null

      if (child !== activeChild || !child || stopping || !['error', 'starting'].includes(state)) {
        return
      }

      if (await probe(url)) {
        markReady(false)
        return
      }

      if (Date.now() >= deadline) {
        setError(`Timed out waiting for Langflow at ${url}`)
        return
      }

      readyTimer = setTimeout(tick, pollIntervalMs)
    }

    readyTimer = setTimeout(tick, 0)
  }

  async function status() {
    if (state === 'stopped' || state === 'exited' || state === 'error') {
      if (await probe(url)) {
        child = null
        markReady(true)
      }
    }

    return snapshot()
  }

  async function start() {
    if (state === 'starting' || (child && !child.killed && state === 'ready')) {
      return snapshot()
    }

    if (await probe(url)) {
      child = null
      markReady(true)
      return snapshot()
    }

    if (child && !child.killed) {
      error = null
      external = false
      state = 'starting'
      scheduleReadyProbe(child)
      return snapshot()
    }

    if (!isLangflowRoot(root, exists)) {
      setError(root ? `Langflow checkout not found or incomplete: ${root}` : 'Langflow checkout is not configured.')
      return snapshot()
    }

    const launches = buildLangflowLaunches({ configDir, env, exists, host, port, readFile, root, secretsPath })

    function spawnLaunch(launchIndex, fallbackReason = '') {
      const launch = launches[launchIndex]

      if (!launch) {
        setError(fallbackReason || 'No Langflow launch command is available.')
        return false
      }

      if (fallbackReason) {
        log(`${fallbackReason}; falling back to ${launch.command} ${launch.args.join(' ')}`)
      }

      stopping = false

      try {
        child = spawn(launch.command, launch.args, launch.options)
      } catch (err) {
        const message = err?.message || String(err)

        if (launchIndex + 1 < launches.length) {
          return spawnLaunch(launchIndex + 1, `Langflow ${launch.mode} launch failed before ready (${message})`)
        }

        child = null
        setError(message)
        return false
      }

      const activeChild = child

      function fallbackOrSetError(message) {
        if (child !== activeChild || stopping) {
          return
        }

        clearReadyTimer()
        external = false

        if (launchIndex + 1 < launches.length) {
          child = null
          error = null
          state = 'starting'
          spawnLaunch(launchIndex + 1, message)
          return
        }

        child = null
        setError(message)
      }

      state = 'starting'
      error = null
      external = false
      log(`spawned ${launch.command} ${launch.args.join(' ')} cwd=${root}`)

      wireLogStream(activeChild.stdout, line => log(`[stdout] ${line}`))
      wireLogStream(activeChild.stderr, line => log(`[stderr] ${line}`))

      activeChild.once('error', err => {
        fallbackOrSetError(`Langflow ${launch.mode} launch failed before ready (${err?.message || String(err)})`)
      })

      activeChild.once('exit', (code, signal) => {
        if (child !== activeChild) {
          return
        }

        clearReadyTimer()

        if (stopping) {
          return
        }

        if (state === 'ready') {
          child = null
          external = false
          state = 'exited'
          error = signal || code === 0 ? null : `Langflow exited (${code})`
          return
        }

        fallbackOrSetError(`Langflow ${launch.mode} exited before ready (${signal || code})`)
      })

      scheduleReadyProbe(child)
      return true
    }

    spawnLaunch(0)

    return snapshot()
  }

  async function stop() {
    clearReadyTimer()
    stopping = true

    const dying = child && !child.killed ? child : null

    if (dying) {
      // Wait for the process to actually exit before returning. Otherwise a
      // follow-up start() (e.g. the account login/logout restart) probes the
      // old server while its port is still listening, falsely attaches to it as
      // "external/ready", and never respawns with the updated credentials — so
      // login appears to succeed yet the Kari nodes stay unavailable.
      const exited = new Promise(resolve => {
        let settled = false
        let killTimer = null
        let giveUpTimer = null

        const finish = () => {
          if (settled) {
            return
          }

          settled = true

          if (killTimer) {
            clearTimeout(killTimer)
          }

          if (giveUpTimer) {
            clearTimeout(giveUpTimer)
          }

          resolve()
        }

        // fakeChild.kill() in tests emits 'exit' synchronously, so attach the
        // listener before killTree() runs.
        dying.once('exit', finish)

        // Escalate to SIGKILL if SIGTERM is ignored, then give up so a wedged
        // process can never block login/logout indefinitely.
        killTimer = setTimeout(() => {
          try {
            if (dying.pid) {
              process.kill(-dying.pid, 'SIGKILL')
            }
          } catch {
            /* process group already gone */
          }
        }, Math.max(0, stopTimeoutMs - 1_000))
        giveUpTimer = setTimeout(finish, stopTimeoutMs)
      })

      killTree(dying)
      await exited
    }

    child = null
    error = null
    external = false
    state = 'stopped'
    stopping = false

    return snapshot()
  }

  return {
    start,
    status,
    stop
  }
}

module.exports = {
  DEFAULT_KARI_CLOUD_BASE_URL,
  DEFAULT_LANGFLOW_HOST,
  DEFAULT_LANGFLOW_PORT,
  DEFAULT_LANGFLOW_ROOT,
  buildAugmentedPath,
  buildLangflowLaunch,
  buildLangflowLaunches,
  buildLangflowRuntimeEnv,
  createWorkflowBackendManager,
  getWorkflowAuthStatus,
  hasBuiltLangflowFrontend,
  isLangflowRoot,
  killProcessTree,
  probeHttpUrl,
  readWorkflowSecrets,
  resolveLangflowRoot,
  workflowSecretsEnv,
  workflowUrl
}
