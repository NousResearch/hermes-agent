/**
 * Lane-private harness for the core Desktop suite: an isolated sandbox with a
 * fake HOME, a launch helper, a per-socket WebSocket recorder, a /proc-based
 * process census for the backend, and the transcript oracle.
 *
 * Synchronisation rule for every helper here: wait on an observable fact
 * (a frame, a DOM state, a persisted row, a pid) with a deadline — never a
 * fixed sleep.
 */

import * as fs from 'node:fs'
import * as net from 'node:net'
import * as os from 'node:os'
import * as path from 'node:path'
import { DatabaseSync } from 'node:sqlite'

import { _electron, type ElectronApplication, expect, type Page } from '@playwright/test'

import { resolveElectronBinary } from '../electron-binary'

export const DESKTOP_ROOT = path.resolve(import.meta.dirname, '..', '..')
export const REPO_ROOT = path.resolve(DESKTOP_ROOT, '..', '..')

// ─── Sandbox ────────────────────────────────────────────────────────────

export interface CoreSandbox {
  root: string
  /** Prepended to PATH: external-platform fakes (see createCoreSandbox). */
  bin: string
  home: string
  hermesHome: string
  userDataDir: string
  cleanup: () => void
}

/**
 * HOME is faked too, not just HERMES_HOME: profile roots are anchored to
 * `Path.home()/.hermes`, so a sandbox HERMES_HOME under the real ~/.hermes
 * would read and write the real install's profiles/.
 */
export function createCoreSandbox(label: string): CoreSandbox {
  const parent = process.env.HERMES_E2E_CORE_ROOT || os.tmpdir()
  fs.mkdirSync(parent, { recursive: true })
  const root = fs.mkdtempSync(path.join(parent, `core-${label}-`))
  const home = path.join(root, 'home')
  const hermesHome = path.join(home, '.hermes')
  const userDataDir = path.join(root, 'user-data')
  fs.mkdirSync(hermesHome, { recursive: true })
  fs.mkdirSync(userDataDir, { recursive: true })
  fs.writeFileSync(
    path.join(userDataDir, 'window-state.json'),
    JSON.stringify({ x: 0, y: 0, width: 1280, height: 860, isMaximized: false })
  )
  fs.writeFileSync(path.join(userDataDir, 'zoom-state.json'), JSON.stringify({ zoomLevel: 0 }))
  // External platform fake: a logged-out `gh`. The backend probes `gh auth
  // token` for GitHub credentials; with a sandbox HOME a real gh can block on
  // the desktop keyring for ~60 s, and a probe in flight at quit outlives the
  // backend (reported as a finding) — which would make the orphan census
  // depend on the runner's keyring rather than on Hermes.
  const bin = path.join(root, 'bin')
  fs.mkdirSync(bin, { recursive: true })
  fs.writeFileSync(path.join(bin, 'gh'), '#!/bin/sh\necho "no oauth token found for github.com" >&2\nexit 1\n', {
    mode: 0o755
  })

  return {
    root,
    bin,
    home,
    hermesHome,
    userDataDir,
    cleanup: () => {
      if (!process.env.HERMES_E2E_CORE_KEEP) {
        fs.rmSync(root, { recursive: true, force: true })
      }
    }
  }
}

/**
 * Sandbox config: only the scripted provider. The external tirith scanner is
 * off: with none on PATH the backend downloads it from GitHub on the first
 * terminal command (network in a required lane), and with one on PATH it
 * fetched a 12 MB threat DB that was still being written after quit. The
 * approval prompts under test come from Hermes's own detector.
 */
export function providerConfigYaml(providerUrl: string, extra = '', approvals: 'manual' | 'off' = 'off'): string {
  return `model:
  default: mock-model
  provider: mock
providers:
  mock:
    api: ${providerUrl}/v1
    name: Mock
    api_mode: chat_completions
    key_env: MOCK_API_KEY
    models:
      mock-model: {}
    context_length: 64000
auxiliary:
  title_generation:
    enabled: false
security:
  tirith_enabled: false
approvals:
  mode: "${approvals}"
${extra}`
}

export function writeProviderHome(
  dir: string,
  providerUrl: string,
  extra = '',
  approvals: 'manual' | 'off' = 'off'
): void {
  fs.mkdirSync(dir, { recursive: true })
  fs.writeFileSync(path.join(dir, 'config.yaml'), providerConfigYaml(providerUrl, extra, approvals))
  fs.writeFileSync(path.join(dir, '.env'), 'MOCK_API_KEY=core-e2e-key\n')
}

const CREDENTIAL_RE = /(_API_KEY|_TOKEN|_SECRET|_PASSWORD|_CREDENTIALS|_ACCESS_KEY|_PRIVATE_KEY|_BASE_URL)$/

/**
 * The runner's own env minus credentials and every HERMES_* knob: an agent
 * shell exports HERMES_YOLO_MODE / _HERMES_GATEWAY, which the spawned backend
 * would inherit (auto-approving every command, changing the run under test).
 */
export function coreAppEnv(sandbox: CoreSandbox, extra: Record<string, string> = {}): Record<string, string> {
  const env: Record<string, string> = {}

  for (const [key, value] of Object.entries(process.env)) {
    if (!value || CREDENTIAL_RE.test(key) || /^_?HERMES_/.test(key) || key === 'VIRTUAL_ENV') {
      continue
    }

    env[key] = value
  }

  return {
    ...env,
    PATH: `${sandbox.bin}${path.delimiter}${env.PATH ?? ''}`,
    HOME: sandbox.home,
    HERMES_HOME: sandbox.hermesHome,
    HERMES_DESKTOP_USER_DATA_DIR: sandbox.userDataDir,
    HERMES_DESKTOP_IGNORE_EXISTING: '1',
    HERMES_DESKTOP_HERMES_ROOT: REPO_ROOT,
    // setup-pm exports an install-scoped interpreter, not a checkout .venv.
    // The Desktop override must be explicit because the sandbox strips HERMES_*.
    ...(process.env.HERMES_E2E_PYTHON ? { HERMES_DESKTOP_PYTHON: process.env.HERMES_E2E_PYTHON } : {}),
    HERMES_DESKTOP_APP_NAME: `HermesCoreE2E-${path.basename(sandbox.root)}`,
    HERMES_DESKTOP_SKIP_QUIT_CONFIRM: '1',
    HERMES_DESKTOP_CDP_PORT: 'off',
    // A partial-clone (blob:none) dev checkout turns some backend git read into
    // a lazy `git fetch origin` over the network, which outlived quit by >60 s
    // (reported as a finding). CI checkouts are not partial; keep dev runs
    // offline and deterministic the same way.
    GIT_NO_LAZY_FETCH: '1',
    ...extra
  }
}

export async function launchCoreApp(env: Record<string, string>): Promise<{ app: ElectronApplication; page: Page }> {
  if (!fs.existsSync(path.join(DESKTOP_ROOT, 'dist', 'electron-main.mjs'))) {
    throw new Error("Desktop dist not built: run 'npm run build' in apps/desktop first")
  }

  const app = await _electron.launch({
    executablePath: resolveElectronBinary([DESKTOP_ROOT, REPO_ROOT]),
    args: [DESKTOP_ROOT, '--disable-gpu', '--no-sandbox'],
    env,
    cwd: DESKTOP_ROOT
  })

  // Keep the main process's stdout/stderr (backend supervisor lines included)
  // so a boot that never becomes interactive fails with its own story.
  const lines: string[] = []

  const collect = (chunk: Buffer) => {
    lines.push(...chunk.toString('utf8').split('\n').filter(Boolean))
    lines.splice(0, Math.max(0, lines.length - 200))
  }

  app.process().stdout?.on('data', collect)
  app.process().stderr?.on('data', collect)
  APP_LOGS.set(app, lines)

  const page = await app.firstWindow()

  return { app, page }
}

const APP_LOGS = new WeakMap<ElectronApplication, string[]>()

/** Last main-process output lines of `app` (for failure messages). */
export function appLogTail(app: ElectronApplication, n = 60): string {
  return (APP_LOGS.get(app) ?? []).slice(-n).join('\n')
}

// ─── Process census ─────────────────────────────────────────────────────

export interface ProcInfo {
  pid: number
  ppid: number
  cmdline: string
}

function readProc(pid: number): null | { environ: string; cmdline: string; ppid: number } {
  try {
    const environ = fs.readFileSync(`/proc/${pid}/environ`, 'utf8')
    const cmdline = fs.readFileSync(`/proc/${pid}/cmdline`, 'utf8').split('\0').join(' ').trim()
    const stat = fs.readFileSync(`/proc/${pid}/stat`, 'utf8')
    // Field 4 (ppid) follows the parenthesised comm, which may contain spaces.
    const ppid = Number(stat.slice(stat.lastIndexOf(')') + 2).split(' ')[1])

    return { environ, cmdline, ppid }
  } catch {
    return null
  }
}

/** Every live process whose environment carries this sandbox's HERMES_HOME (orphans included). */
export function sandboxProcesses(sandbox: CoreSandbox): ProcInfo[] {
  const needle = `HERMES_HOME=${sandbox.hermesHome}\0`
  const out: ProcInfo[] = []

  for (const entry of fs.readdirSync('/proc')) {
    const pid = Number(entry)

    if (!Number.isInteger(pid) || pid === process.pid) {
      continue
    }

    const info = readProc(pid)

    if (!info || !(info.environ + '\0').includes(needle)) {
      continue
    }

    // Zombies have an empty cmdline and are already dead for our purposes.
    if (!info.cmdline) {
      continue
    }

    out.push({ pid, ppid: info.ppid, cmdline: info.cmdline })
  }

  return out
}

/**
 * The `hermes serve` backend(s) spawned for this sandbox.
 *
 * A child of the backend still shows the backend's argv and environ between
 * fork and exec, and the backend forks ~40 probes per boot (git, ps,
 * ldconfig/gcc, pip): a 100 ms sampler catches one in that window every few
 * boots. Such a child is not a second backend, so a serve process whose parent
 * is itself a serve process is excluded. A real second spawn has the
 * supervisor (Electron main) as its parent, and a pre-exec child orphaned by a
 * dead backend is reparented away and still counted.
 */
export function backendProcesses(sandbox: CoreSandbox): ProcInfo[] {
  const serve = sandboxProcesses(sandbox).filter(
    proc => / serve( |$)/.test(proc.cmdline) && !/electron/i.test(proc.cmdline.split(' ')[0])
  )

  const pids = new Set(serve.map(proc => proc.pid))

  return serve.filter(proc => !pids.has(proc.ppid))
}

// ─── WebSocket recorder ─────────────────────────────────────────────────

export interface GatewayEventFrame {
  socket: number
  type: string
  sessionId: string
  seq: null | number
  payload: any
}

export interface WsRecorder {
  sockets: { id: number; url: string; closed: boolean }[]
  events: GatewayEventFrame[]
  sent: { socket: number; method: string; params: any }[]
}

export function recordWebSockets(page: Page): WsRecorder {
  const rec: WsRecorder = { sockets: [], events: [], sent: [] }

  page.on('websocket', ws => {
    if (!ws.url().includes('/api/ws')) {
      return
    }

    const id = rec.sockets.length
    const entry = { id, url: ws.url(), closed: false }
    rec.sockets.push(entry)
    ws.on('close', () => {
      entry.closed = true
    })
    ws.on('framereceived', frame => {
      try {
        const msg = JSON.parse(String(frame.payload))

        if (msg?.method === 'event' && msg.params) {
          rec.events.push({
            socket: id,
            type: String(msg.params.type ?? ''),
            sessionId: String(msg.params.session_id ?? ''),
            seq: typeof msg.params.seq === 'number' ? msg.params.seq : null,
            payload: msg.params.payload
          })
        }
      } catch {
        /* binary / non-JSON frames carry no gateway event */
      }
    })
    ws.on('framesent', frame => {
      try {
        const msg = JSON.parse(String(frame.payload))

        if (typeof msg?.method === 'string') {
          rec.sent.push({ socket: id, method: msg.method, params: msg.params })
        }
      } catch {
        /* ignore */
      }
    })
  })

  return rec
}

// ─── Network fault injection ────────────────────────────────────────────

export interface TcpProxy {
  port: number
  /** Live client connections through the proxy. */
  connections: () => number
  /** Reset every live connection (the renderer sees an abnormal close, as on a network drop). */
  dropAll: () => number
  close: () => Promise<void>
}

/** A loopback TCP proxy in the test process; the renderer's primary socket is routed through it. */
export function startTcpProxy(targetPort: number): Promise<TcpProxy> {
  const live = new Set<net.Socket>()

  const server = net.createServer(client => {
    const upstream = net.connect(targetPort, '127.0.0.1')
    live.add(client)

    const forget = () => {
      live.delete(client)
      client.destroy()
      upstream.destroy()
    }

    client.on('error', forget)
    upstream.on('error', forget)
    client.on('close', forget)
    upstream.on('close', forget)
    client.pipe(upstream)
    upstream.pipe(client)
  })

  return new Promise((resolve, reject) => {
    server.on('error', reject)
    server.listen(0, '127.0.0.1', () => {
      resolve({
        port: (server.address() as net.AddressInfo).port,
        connections: () => live.size,
        dropAll: () => {
          const count = live.size

          for (const socket of [...live]) {
            socket.resetAndDestroy()
          }

          return count
        },
        close: () =>
          new Promise<void>(done => {
            for (const socket of [...live]) {
              socket.destroy()
            }

            server.close(() => done())
          })
      })
    })
  })
}

/**
 * Rewrite the PRIMARY gateway WebSocket URL main hands the renderer (initial
 * connection and every reconnect mint) so it dials `proxyPort`. REST stays on
 * main's IPC bridge. Takes effect on the renderer's next dial (reload/reconnect).
 */
export async function routePrimaryWebSocket(app: ElectronApplication, backendPort: number, proxyPort: number) {
  await app.evaluate(
    ({ ipcMain }, { backendPort, proxyPort }) => {
      const handlers = (ipcMain as any)._invokeHandlers as Map<string, (...args: any[]) => Promise<any>>
      const from = `ws://127.0.0.1:${backendPort}/`
      const to = `ws://127.0.0.1:${proxyPort}/`

      const rewrite = (value: any): any => {
        if (typeof value === 'string') {
          return value.startsWith(from) ? to + value.slice(from.length) : value
        }

        if (Array.isArray(value)) {
          return value.map(rewrite)
        }

        if (value && typeof value === 'object') {
          return Object.fromEntries(Object.entries(value).map(([key, inner]) => [key, rewrite(inner)]))
        }

        return value
      }

      for (const channel of ['hermes:connection', 'hermes:gateway:ws-url']) {
        const original = handlers.get(channel)

        if (!original) {
          throw new Error(`no ipc handler ${channel}`)
        }

        ipcMain.removeHandler(channel)
        ipcMain.handle(channel, async (event: unknown, profile?: unknown, ...rest: unknown[]) => {
          const result = await original(event, profile, ...rest)
          const primary = profile === undefined || profile === null || profile === '' || profile === 'default'

          return primary ? rewrite(result) : result
        })
      }
    },
    { backendPort, proxyPort }
  )
}

/**
 * Make main stop tagging `profile`'s route as served by the shared primary
 * backend (while still pointing at that same backend). The renderer then dials
 * a SECOND WebSocket to the same process for that profile's session calls —
 * the #120005 topology — while calls made earlier keep the primary joined.
 */
export async function splitProfileRoute(app: ElectronApplication, profile: string): Promise<void> {
  await app.evaluate(({ ipcMain }, profile) => {
    const handlers = (ipcMain as any)._invokeHandlers as Map<string, (...args: any[]) => Promise<any>>
    const original = handlers.get('hermes:connection:for')

    if (!original) {
      throw new Error('no ipc handler hermes:connection:for')
    }

    ipcMain.removeHandler('hermes:connection:for')
    ipcMain.handle('hermes:connection:for', async (event: unknown, payload: any) => {
      const result = await original(event, payload)

      if (payload?.profile === profile && result && typeof result === 'object') {
        const { sharedPrimary: _shared, ...rest } = result

        return rest
      }

      return result
    })
  }, profile)
}

// ─── Renderer helpers ───────────────────────────────────────────────────

export function composer(page: Page) {
  return page.locator('[data-slot="composer-root"] [contenteditable="true"]').filter({ visible: true }).first()
}

/** Composer mounted, no full-viewport overlay above it, window visible. */
export async function waitForInteractive(app: ElectronApplication, page: Page, timeout = 180_000): Promise<void> {
  try {
    await expect(composer(page)).toBeVisible({ timeout })
    await page.waitForFunction(
      () => {
        const el = document.elementFromPoint(window.innerWidth / 2, window.innerHeight / 2)
        let node: Element | null = el

        if (!el) {
          return false
        }

        while (node) {
          const cs = window.getComputedStyle(node)

          if (cs.position === 'fixed') {
            const r = node.getBoundingClientRect()

            if (r.left <= 0 && r.top <= 0 && r.right >= window.innerWidth && r.bottom >= window.innerHeight) {
              return false
            }
          }

          node = node.parentElement
        }

        return true
      },
      undefined,
      { timeout, polling: 250 }
    )
  } catch (error) {
    const blocker = await page
      .evaluate(() => {
        let node: Element | null = document.elementFromPoint(window.innerWidth / 2, window.innerHeight / 2)
        const chain: string[] = []

        while (node && chain.length < 12) {
          const slot = node.getAttribute('data-slot') ?? ''
          const role = node.getAttribute('role') ?? ''
          chain.push(
            `${node.tagName.toLowerCase()}${slot ? `[data-slot=${slot}]` : ''}${role ? `[role=${role}]` : ''}${window.getComputedStyle(node).position === 'fixed' ? '{fixed}' : ''}`
          )
          node = node.parentElement
        }

        return { chain, route: location.hash, text: document.body.innerText.replace(/\s+/g, ' ').slice(0, 800) }
      })
      .catch(e => ({ chain: [], route: '', text: `evaluate failed: ${String(e)}` }))

    throw new Error(
      `app never became interactive: ${(error as Error).message.split('\n')[0]}\n` +
        `route: ${blocker.route}\ncenter element chain: ${blocker.chain.join(' < ')}\nbody text: ${blocker.text}\n` +
        `main-process log tail:\n${appLogTail(app)}`
    )
  }

  await expect
    .poll(
      () =>
        app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0]?.isVisible() ?? false).catch(() => false),
      { timeout, intervals: [250] }
    )
    .toBe(true)
}

/**
 * Enter submits (and steers a live turn); Control+Enter queues a follow-up
 * behind it. While the gateway is reconnecting the composer keeps the draft
 * and ignores Enter (by design), so the press is repeated until the draft is
 * accepted — but only while the draft is still in the composer AND no
 * prompt.submit carrying it went out on any socket, so a retry can never
 * double-submit.
 */
export async function send(
  page: Page,
  text: string,
  key: 'Control+Enter' | 'Enter' = 'Enter',
  ws?: WsRecorder
): Promise<void> {
  const box = composer(page)
  const probe = text.slice(0, 12)
  await box.click()
  await box.fill(text)
  await expect(box).toContainText(probe)
  const submitted = () => (ws?.sent ?? []).some(frame => JSON.stringify(frame.params ?? '').includes(probe))

  await expect
    .poll(
      async () => {
        const draft = (await box.textContent().catch(() => '')) ?? ''

        if (!draft.includes(probe)) {
          return 'accepted'
        }

        if (!submitted()) {
          await box.press(key)
        }

        return 'pending'
      },
      { timeout: 120_000, intervals: [1_000, 2_000, 4_000], message: `composer accepted ${probe}` }
    )
    .toBe('accepted')
}

export async function currentSessionId(page: Page): Promise<string> {
  return page.evaluate(() => decodeURIComponent(location.hash.replace(/^#\/?/, '').split('?')[0] ?? ''))
}

export interface PersistedMessage {
  role: string
  content: string
}

/**
 * The stored session id holding a user message with `marker`, read straight
 * from the profile's state.db (read-only) — independent of any renderer or
 * REST projection.
 */
export function storedSessionForMarker(sandbox: CoreSandbox, profile: string, marker: string): null | string {
  const dbPath =
    profile === 'default'
      ? path.join(sandbox.hermesHome, 'state.db')
      : path.join(sandbox.hermesHome, 'profiles', profile, 'state.db')

  if (!fs.existsSync(dbPath)) {
    return null
  }

  const db = new DatabaseSync(dbPath, { readOnly: true })

  try {
    const row = db
      .prepare("SELECT session_id FROM messages WHERE role = 'user' AND content LIKE ? ORDER BY id LIMIT 1")
      .get(`%${marker}%`) as undefined | { session_id: string }

    return row?.session_id ?? null
  } catch {
    // Mid-WAL-checkpoint reads can fail transiently; the caller polls.
    return null
  } finally {
    db.close()
  }
}

/** The backend's persisted display transcript (REST, the same read the renderer hydrates from). */
export async function persistedTranscript(
  page: Page,
  sessionId: string,
  profile?: string
): Promise<PersistedMessage[]> {
  const query = profile ? `&profile=${encodeURIComponent(profile)}` : ''

  const result = await page.evaluate(
    async ({ sessionId, query }) =>
      (window as any).hermesDesktop.api({ path: `/api/sessions/${sessionId}/messages?order=oldest&limit=500${query}` }),
    { sessionId, query }
  )

  return (result?.messages ?? []).map((m: any) => ({
    role: String(m.role ?? ''),
    content: typeof m.content === 'string' ? m.content : JSON.stringify(m.content ?? '')
  }))
}

export interface RenderedMessage {
  role: 'assistant' | 'user'
  text: string
}

/** Every user/assistant bubble in the visible thread, in DOM order. */
export async function renderedTranscript(page: Page): Promise<{ bubbles: RenderedMessage[]; fullText: string }> {
  return page.evaluate(() => {
    const viewport = document.querySelector('[data-slot="aui_thread-viewport"]')

    if (!viewport) {
      return { bubbles: [], fullText: '' }
    }

    const bubbles = [
      ...viewport.querySelectorAll('[data-slot="aui_user-message-root"], [data-slot="aui_assistant-message-root"]')
    ].map(el => ({
      role: (el.getAttribute('data-slot') === 'aui_user-message-root' ? 'user' : 'assistant') as 'assistant' | 'user',
      text: (el as HTMLElement).innerText.replace(/\s+/g, ' ').trim()
    }))

    return { bubbles, fullText: (viewport as HTMLElement).innerText.replace(/\s+/g, ' ') }
  })
}
