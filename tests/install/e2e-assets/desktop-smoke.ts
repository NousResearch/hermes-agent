import fs from 'node:fs'
import { createRequire } from 'node:module'
import path from 'node:path'
import { pathToFileURL } from 'node:url'
import { parseArgs } from 'node:util'

import { _electron, type ElectronApplication, type Page } from '@playwright/test'
import { z } from 'zod'

import { readChatIdentity, runDesktopChatSmoke, waitForChatReady } from '../../../tests-js/scripts/desktop-chat-smoke.ts'
import { assertBackendOrigin, localBackendProcess, readInstallationCommit, within } from '../../../tests-js/scripts/desktop-smoke-process.ts'
import { validateMockUrl, writeEnvFile, writeMockProviderConfig } from '../../../tests-js/scripts/mock-provider-config.ts'
import { type MockServer, startMockServer } from '../../../tests-js/scripts/mock-server.ts'

const require = createRequire(import.meta.url)
const { pickAppWindow }: { pickAppWindow: (app: ElectronApplication, log: (message: string) => void) => Promise<Page> } = require('./update-ui.cjs')
const { prepareWindowForInput }: { prepareWindowForInput: (app: ElectronApplication, page: Page) => Promise<void> } = require('./window-input.cjs')

const absolutePath = z.string().refine(path.isAbsolute, 'Expected an absolute path')
const optionsSchema = z.object({
  exe: absolutePath, root: absolutePath, origin: z.enum(['source', 'bundled']),
  home: absolutePath, 'user-data': absolutePath, out: absolutePath,
  phase: z.enum(['old', 'new', 'installed']), 'expect-commit': z.string().regex(/^[0-9a-f]{40}$/),
  'launch-spec': absolutePath.optional(), 'mock-url': z.string().optional(),
})
type SmokeOptions = z.infer<typeof optionsSchema>

const launchSpecSchema = z.object({
  argv: z.array(z.string()).min(1), cwd: absolutePath,
  // oxlint-disable-next-line anti-slop/no-shape-in-symbol-names -- This is the existing launch-capture wire field.
  env: z.record(z.string(), z.string()), matchedShape: z.enum(['source', 'packaged']),
})

interface SmokeEnvironment extends Record<string, string> {
  HOME: string
  USERPROFILE: string
  HERMES_HOME: string
  HERMES_DESKTOP_USER_DATA_DIR: string
  XDG_CONFIG_HOME: string
  XDG_DATA_HOME: string
  XDG_CACHE_HOME: string
}

interface Launch {
  executablePath: string
  args: string[]
  cwd: string
  env: SmokeEnvironment
}

interface RunningDesktop {
  pid: number
  executable: string
  resources: string
  userData: string
  home: string
}

interface ElectronProcess extends NodeJS.Process {
  resourcesPath: string
}

export function smokeEnvironment(inherited: NodeJS.ProcessEnv, home: string, userData: string): SmokeEnvironment {
  const clean: Record<string, string> = {}
  const keepHermes = new Set(['HERMES_TEST_INSTALL_REF', 'HERMES_TEST_INSTALL_REPO'])
  for (const [key, value] of Object.entries(inherited)) {
    const name = key.toUpperCase()
    if (!value || /(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)/.test(name)
        || /^(?:PYTHON|VIRTUAL_ENV|CONDA|NODE_|NPM_|ELECTRON_|VITE_|UV_|PM_)/.test(name)
        || /^(?:OPENAI|ANTHROPIC|OPENROUTER|OLLAMA|GEMINI|GROQ|XAI)_/.test(name)
        || (name.startsWith('HERMES_') && !keepHermes.has(name))
        || ['INIT_CWD', 'TERMINAL_CWD', 'LD_PRELOAD', 'DYLD_INSERT_LIBRARIES', 'SSH_AUTH_SOCK', 'SSH_ASKPASS', 'GIT_ASKPASS'].includes(name)) {
      continue
    }
    clean[key] = value
  }
  return {
    ...clean, HOME: path.join(home, '.desktop-smoke-home'), USERPROFILE: path.join(home, '.desktop-smoke-home'),
    XDG_CONFIG_HOME: path.join(home, '.desktop-smoke-home', '.config'),
    XDG_DATA_HOME: path.join(home, '.desktop-smoke-home', '.local', 'share'),
    XDG_CACHE_HOME: path.join(home, '.desktop-smoke-home', '.cache'),
    HERMES_HOME: home, HERMES_DESKTOP_USER_DATA_DIR: userData,
    APPDATA: path.join(home, '.desktop-smoke-home', 'AppData', 'Roaming'),
    LOCALAPPDATA: path.join(home, '.desktop-smoke-home', 'AppData', 'Local'),
  }
}

/** The bundled app resolves its Hermes home the same way electron/data-paths.ts does:
 * an explicit HERMES_HOME wins, but a bundle-env clear (HERMES_HOME=null) empties it before
 * this runs, and the HERMES_DESKTOP_USER_DATA_DIR branch then resolves <userData>/hermes-home.
 * Seed both so the driver works against baked-clear bundles and plain source launches. */
export function candidateSmokeHermesHomes(home: string, userData: string): string[] {
  const fallback = path.join(userData, 'hermes-home')
  return home === fallback ? [home] : [home, fallback]
}

export function resolveSmokeLaunch(options: SmokeOptions): Launch {
  const spec = options['launch-spec'] ? launchSpecSchema.parse(JSON.parse(fs.readFileSync(options['launch-spec'], 'utf8').replace(/^\uFEFF/, ''))) : null
  const env = smokeEnvironment({ ...process.env, ...spec?.env }, options.home, options['user-data'])
  let args: string[] = []
  let cwd = env.HOME!
  if (spec) {
    if (options.origin !== 'source') {
      throw new Error('A captured source launch specification cannot select a bundled backend')
    }
    if (!within(options.root, spec.cwd)) {
      throw new Error('Source launch cwd is outside the expected installation')
    }
    cwd = spec.cwd
    // oxlint-disable-next-line anti-slop/no-shape-in-symbol-names -- This is the existing launch-capture wire field.
    if (spec.matchedShape === 'source') {
      const index = spec.argv.indexOf('electron')
      if (index < 0 || !spec.argv.includes('.')) {
        throw new Error('Unsupported captured source launch shape')
      }
      args = [spec.cwd, ...spec.argv.slice(index + 1).filter((arg: string): boolean => arg !== '.')]
    } else {
      if (fs.realpathSync(spec.argv[0]) !== fs.realpathSync(options.exe)) {
        throw new Error('Launch spec executable differs from --exe')
      }
      args = spec.argv.slice(1)
    }
    const editableRoot = spec.env.HERMES_PYTHON_SRC_ROOT
    if (editableRoot) {
      if (!path.isAbsolute(editableRoot) || fs.realpathSync(editableRoot) !== fs.realpathSync(options.root)) {
        throw new Error('Captured HERMES_PYTHON_SRC_ROOT differs from the expected source installation')
      }
      env.HERMES_PYTHON_SRC_ROOT = editableRoot
    }
    // Retain the product-selected source interpreter, not the driver's activated Python.
    for (const key of ['HERMES_DESKTOP_PYTHON', 'HERMES_DESKTOP_HERMES', 'HERMES_DESKTOP_HERMES_ROOT']) {
      const value = spec.env[key]
      if (value) {
        if (!path.isAbsolute(value) || (key !== 'HERMES_DESKTOP_PYTHON' && !within(options.root, value))) {
          throw new Error(`Captured ${key} escapes the expected source installation`)
        }
        fs.accessSync(value)
        env[key] = value
      }
    }
  }
  return { executablePath: options.exe, args, cwd, env }
}

/** Select local through the persisted connection contract before any baked remote default can dial. */
function selectLocal(userData: string): void {
  const v1Path = path.join(userData, 'connection.json')
  const document = z.object({}).passthrough()
  const v1 = fs.existsSync(v1Path) ? document.parse(JSON.parse(fs.readFileSync(v1Path, 'utf8'))) : {}
  fs.writeFileSync(v1Path, JSON.stringify({ ...v1, mode: 'local' }), { mode: 0o600 })
  const v2Path = path.join(userData, 'connections.json')
  if (fs.existsSync(v2Path)) {
    const v2 = document.parse(JSON.parse(fs.readFileSync(v2Path, 'utf8')))
    fs.writeFileSync(v2Path, JSON.stringify({ ...v2, primary: 'local', launchMode: 'primary', lastUsed: 'local' }), { mode: 0o600 })
  }
}

interface BackendWindow extends Window {
  hermesDesktop?: {
    getConnection: () => Promise<{ baseUrl: string; mode?: string; logs: string[] }>
  }
}

async function backendConnection(page: Page): Promise<{ baseUrl: string; mode?: string; logs: string[] }> {
  return z.object({ baseUrl: z.string(), mode: z.string().optional(), logs: z.array(z.string()) }).parse(
    await page.evaluate(async (): Promise<{ baseUrl: string; mode?: string; logs: string[] }> => {
      // SAFETY: the desktop preload owns this bridge; its result is validated at the boundary.
      const bridge = (window as BackendWindow).hermesDesktop
      if (!bridge) { throw new Error('Desktop connection bridge is unavailable') }
      const { baseUrl, mode, logs } = await bridge.getConnection()
      return { baseUrl, mode, logs }
    }),
  )
}

function redact(text: string): string {
  return text.replace(/([?&](?:token|ticket|key)=)[^&\s"']+/gi, '$1[redacted]')
    .replace(/((?:authorization|api[_ -]?key|token|secret|password)\s*[:=]\s*)(?:Bearer\s+)?[^\s,"']+/gi, '$1[redacted]')
}

function captureBackendLogs(homes: readonly string[], outDir: string, phase: string): void {
  for (const home of homes) {
    for (const name of ['desktop.log', 'errors.log', 'agent.log']) {
      const filename = path.join(home, 'logs', name)
      if (fs.existsSync(filename)) {
        const suffix = homes.length > 1 && home !== homes[0] ? `-${path.basename(path.dirname(home))}` : ''
        fs.writeFileSync(path.join(outDir, `desktop-${phase}-${name}${suffix}`), redact(fs.readFileSync(filename, 'utf8')))
      }
    }
  }
}

async function gracefulClose(app: ElectronApplication): Promise<void> {
  let timer: ReturnType<typeof setTimeout> | undefined
  try {
    await Promise.race([app.close(), new Promise<never>((_resolve, reject): void => {
      timer = setTimeout((): void => reject(new Error('Desktop did not close gracefully')), 20_000)
    })])
  } finally {
    clearTimeout(timer)
  }
}

async function verifyRunningDesktop(app: ElectronApplication, options: SmokeOptions, launch: Launch): Promise<RunningDesktop> {
  const running = await app.evaluate(({ app: electronApp }): RunningDesktop => {
    // SAFETY: this callback runs in Electron main, whose process includes resourcesPath.
    const runtime = process as ElectronProcess
    return { pid: runtime.pid, executable: runtime.execPath, resources: runtime.resourcesPath, userData: electronApp.getPath('userData'), home: electronApp.getPath('home') }
  })
  // Chromium resolves DIR_HOME from the HOME env only on Linux; macOS
  // (NSHomeDirectory) and Windows (CSIDL_PROFILE) ignore it, so the home
  // equality is a contract only there. On those platforms the isolation proof
  // is the userData pin plus the seeded Hermes home the backend booted from.
  const homeHonored = process.platform === 'linux'
    ? fs.realpathSync(running.home) === fs.realpathSync(launch.env.HOME!)
    : true
  if (fs.realpathSync(running.userData) !== fs.realpathSync(options['user-data']) || !homeHonored) {
    throw new Error('Desktop did not honor the isolated home and userData directories')
  }
  if (fs.realpathSync(running.executable) !== fs.realpathSync(options.exe)) {
    throw new Error('Running Electron executable differs from --exe')
  }
  if (options.origin === 'bundled' && fs.realpathSync(path.join(running.resources, 'agent-payload')) !== fs.realpathSync(options.root)) {
    throw new Error('Running Electron resources differ from the expected payload root')
  }
  return running
}

export async function runInstalledDesktopSmoke(options: SmokeOptions): Promise<void> {
  const out = options.out
  fs.mkdirSync(out, { recursive: true })
  const receiptPath = path.join(out, `desktop-chat-${options.phase}.json`)
  let mock: MockServer | undefined
  let app: ElectronApplication | undefined
  let page: Page | undefined
  const consoleLines: string[] = []
  try {
    fs.accessSync(options.exe, fs.constants.X_OK)
    fs.accessSync(options.root)
    fs.mkdirSync(options.home, { recursive: true })
    fs.mkdirSync(options['user-data'], { recursive: true })
    const launch = resolveSmokeLaunch(options)
    fs.mkdirSync(launch.env.HOME!, { recursive: true })
    // Electron resolves shell folders before app 'ready': Windows SHGetFolderPath
    // fails (and applyDesktopIdentity crashes the process) when the roaming/local
    // AppData dirs named by the sandboxed USERPROFILE/APPDATA do not exist. The
    // XDG dirs serve the same role for Linux/Chromium.
    for (const dir of [launch.env.APPDATA, launch.env.LOCALAPPDATA, launch.env.XDG_CONFIG_HOME,
      launch.env.XDG_DATA_HOME, launch.env.XDG_CACHE_HOME]) {
      fs.mkdirSync(dir, { recursive: true })
    }
    selectLocal(options['user-data'])
    if (!options['mock-url']) { mock = await startMockServer() }
    const mockUrl = validateMockUrl(options['mock-url'] ?? mock!.url)
    // A bundle-env HERMES_HOME clear (see candidateSmokeHermesHomes) can make the
    // app resolve a different home than --home, so every candidate gets the mock
    // provider config and .env.
    for (const home of candidateSmokeHermesHomes(options.home, options['user-data'])) {
      writeMockProviderConfig(home, mockUrl)
      writeEnvFile(home)
    }
    app = await _electron.launch({ ...launch, timeout: 120_000 })
    app.process().stdout?.on('data', (chunk: Buffer): void => { consoleLines.push(redact(chunk.toString())) })
    app.process().stderr?.on('data', (chunk: Buffer): void => { consoleLines.push(redact(chunk.toString())) })

    page = await pickAppWindow(app, (message: string): void => { consoleLines.push(redact(message)) })
    page.on('pageerror', (error: Error): void => { consoleLines.push(redact(error.message)) })
    await prepareWindowForInput(app, page)
    await waitForChatReady(page)
    const running = await verifyRunningDesktop(app, options, launch)
    const connection = await backendConnection(page)
    const base = new URL(connection.baseUrl)
    if (connection.mode !== 'local' || !['127.0.0.1', 'localhost', '[::1]'].includes(base.hostname)) {
      throw new Error('Required local backend was replaced by a remote connection')
    }
    const identity = await readChatIdentity(page)
    if (options.origin === 'source' && fs.realpathSync(identity.hermesRoot) !== fs.realpathSync(options.root)) {
      throw new Error('Desktop resolved a different source installation')
    }
    const backend = localBackendProcess(Number(base.port), running.pid)
    assertBackendOrigin(backend, options.root, options.origin)
    const provenanceCommit = readInstallationCommit(options.root, options.origin)
    if (provenanceCommit !== options['expect-commit']) { throw new Error('Installed commit differs from --expect-commit') }
    fs.writeFileSync(path.join(out, `desktop-backend-${options.phase}.log`), connection.logs.map(redact).join('\n'))
    // Trace only chat actions. Connection probes can return credential-bearing logs.
    await app.context().tracing.start({ screenshots: true, snapshots: false, sources: false })
    const chat = await runDesktopChatSmoke(page, { mockUrl, phase: options.phase, outDir: out, expectCommit: options['expect-commit'], provenanceCommit })
    fs.writeFileSync(receiptPath, `${JSON.stringify({ ...chat, status: 'closing' }, null, 2)}\n`)
    const result = { ...chat, origin: options.origin, root: options.root, executable: options.exe, running,
      localModeConfigured: true, backend: { ...backend, command: redact(backend.command) }, launchKind: options.phase === 'new' ? 'post-update-launch' : 'installed-launch' }
    await app.context().tracing.stop({ path: path.join(out, `desktop-chat-${options.phase}.zip`) })
    await gracefulClose(app)
    app = undefined
    fs.writeFileSync(receiptPath, `${JSON.stringify(result, null, 2)}\n`)
  } catch (error) {
    if (page) { await page.screenshot({ path: path.join(out, `desktop-chat-${options.phase}-failed.png`) }).catch((): void => {}) }
    if (app) { await app.context().tracing.stop({ path: path.join(out, `desktop-chat-${options.phase}-failed.zip`) }).catch((): void => {}) }
    fs.writeFileSync(receiptPath, `${JSON.stringify({ status: 'failed', phase: options.phase, origin: options.origin, root: options.root,
      executable: options.exe, expectedCommit: options['expect-commit'], error: redact(String(error)) }, null, 2)}\n`)
    throw error
  } finally {
    fs.writeFileSync(path.join(out, `desktop-app-${options.phase}.log`), consoleLines.join('\n'))
    try {
      captureBackendLogs(candidateSmokeHermesHomes(options.home, options['user-data']), out, options.phase)
    } finally {
      if (app) { await gracefulClose(app).catch((error: Error): void => { console.error(redact(error.message)) }) }
      if (mock) { await mock.close() }
    }
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const { values } = parseArgs({ options: {
    exe: { type: 'string' }, root: { type: 'string' }, origin: { type: 'string' }, home: { type: 'string' },
    'user-data': { type: 'string' }, out: { type: 'string' }, phase: { type: 'string' },
    'expect-commit': { type: 'string' }, 'launch-spec': { type: 'string' }, 'mock-url': { type: 'string' },
  } })
  try {
    await runInstalledDesktopSmoke(optionsSchema.parse(values))
  } catch (error) {
    console.error(redact(String(error)))
    process.exitCode = 1
  }
}