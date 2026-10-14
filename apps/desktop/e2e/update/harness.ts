/**
 * Harness for the Desktop install/update suite: a REAL local install (made by
 * scripts/install.sh + `hermes desktop --build-only` in the upgrade suite's
 * sandbox, see seed.py), a local bare origin standing in for GitHub, the
 * packaged app that install built, and the core suite's scripted provider.
 *
 * The one fake per external edge: the git server (a local bare repo behind a
 * url.insteadOf rewrite) and the LLM provider (e2e/core/provider.ts). Nothing
 * inside Hermes is mocked.
 *
 * Synchronisation rule (same as e2e/core): wait on an observable fact — a log
 * line, a pid, a file, a DOM state — with a deadline, never a fixed sleep.
 */

import { spawnSync } from 'node:child_process'
import * as fs from 'node:fs'
import * as os from 'node:os'
import * as path from 'node:path'

import { _electron, type ElectronApplication, expect, type Page } from '@playwright/test'

import { type CoreSandbox, type ProcInfo, providerConfigYaml, sandboxProcesses } from '../core/harness'
import { type ScriptedProvider, startScriptedProvider } from '../core/provider'

import { gatedOn, KNOWN } from './gates'
import { type GithubEdge, startGithubEdge } from './github-edge'

export const DESKTOP_ROOT = path.resolve(import.meta.dirname, '..', '..')
export const REPO_ROOT = path.resolve(DESKTOP_ROOT, '..', '..')
const SEED = path.join(import.meta.dirname, 'seed.py')

/** Where the seeded install lives. Short on purpose: the install bakes it into AF_UNIX socket paths. */
export const UPDATE_ROOT = process.env.HERMES_E2E_UPDATE_ROOT || path.join(os.tmpdir(), 'hdu-e2e')

export interface InstallFacts {
  sandboxRoot: string
  home: string
  hermesHome: string
  checkout: string
  hermes: string
  origin: string
  env: Record<string, string>
  headSha: string
}

function python(): string {
  return process.env.HERMES_E2E_PYTHON || 'python3'
}

function seed(args: string[], timeoutMs = 60_000): string {
  const cp = spawnSync(python(), [SEED, ...args], {
    cwd: REPO_ROOT,
    encoding: 'utf8',
    timeout: timeoutMs,
    maxBuffer: 64 * 1024 * 1024
  })

  if (cp.status !== 0) {
    throw new Error(`seed.py ${args[0]} failed (status ${cp.status}, signal ${cp.signal}):\n${cp.stderr?.slice(-6000)}`)
  }

  return cp.stdout.trim()
}

/** Build the install once per run (Playwright globalSetup). */
export function seedInstall(): InstallFacts {
  return JSON.parse(
    seed(['install', UPDATE_ROOT], 40 * 60_000)
      .split('\n')
      .pop()!
  )
}

/** Put the pristine install + origin back (every spec starts from the same healthy install). */
export function restoreInstall(): InstallFacts {
  killInstallProcesses(readFacts())
  const facts = JSON.parse(seed(['restore', UPDATE_ROOT], 10 * 60_000)) as InstallFacts
  fs.rmSync(userDataDir(facts), { recursive: true, force: true })

  return facts
}

export function readFacts(): InstallFacts {
  return JSON.parse(fs.readFileSync(path.join(UPDATE_ROOT, 'install.json'), 'utf8'))
}

/** One new upstream commit on origin/main (the release the user updates to); returns its sha. */
export function publishUpstream(message: string, relPath: string, content: string): string {
  const file = path.join(UPDATE_ROOT, 'publish-content')
  fs.writeFileSync(file, content)

  return seed(['publish', UPDATE_ROOT, message, relPath, file]).split('\n').pop()!
}

export function git(cwd: string, ...args: string[]): string {
  const cp = spawnSync('git', args, { cwd, encoding: 'utf8', env: { ...process.env, GIT_NO_LAZY_FETCH: '1' } })

  if (cp.status !== 0) {
    throw new Error(`git ${args.join(' ')} failed in ${cwd}: ${cp.stderr}`)
  }

  return cp.stdout.trim()
}

// ─── One spec's world ──────────────────────────────────────────────────

export interface InstallSession {
  facts: InstallFacts
  provider: ScriptedProvider
  edge: GithubEdge
  /** The environment the user's desktop session starts the app with. */
  env: Record<string, string>
  close: () => Promise<void>
}

/**
 * The pristine install, the scripted provider configured as the user's
 * endpoint, and the GitHub edge the install's update check goes through.
 */
export async function startInstallSession(): Promise<InstallSession> {
  const facts = restoreInstall()
  const provider = await startScriptedProvider()
  configureProvider(facts, provider.url)
  const edge = await startGithubEdge(facts.sandboxRoot, facts.origin, facts.env.SSL_CERT_FILE)
  // The install's own shell env (a user who works behind the proxy) sees the same edge.
  const env = appEnv(facts, edge.env)

  return {
    facts,
    provider,
    edge,
    env,
    close: async () => {
      killInstallProcesses(facts)
      await provider.close()
      await edge.close()
    }
  }
}

// ─── The installed app ─────────────────────────────────────────────────

export function releaseDir(facts: InstallFacts): string {
  return path.join(facts.checkout, 'apps', 'desktop', 'release', 'linux-unpacked')
}

/** The packaged executable `hermes desktop` built into the install (electron-builder names it after productName). */
export function packagedExe(facts: InstallFacts): string {
  const dir = releaseDir(facts)
  const name = fs.readdirSync(dir).find(entry => /^hermes$/i.test(entry))

  if (!name) {
    throw new Error(`no packaged Hermes executable in ${dir}: ${fs.readdirSync(dir).join(', ')}`)
  }

  return path.join(dir, name)
}

export function userDataDir(facts: InstallFacts): string {
  return path.join(facts.sandboxRoot, 'user-data')
}

/** Display vars the runner provides (xvfb in CI, the live Wayland session locally). */
function displayEnv(): Record<string, string> {
  const out: Record<string, string> = {}

  for (const key of ['DISPLAY', 'XAUTHORITY', 'XDG_SESSION_TYPE', 'ELECTRON_OZONE_PLATFORM_HINT']) {
    if (process.env[key]) {
      out[key] = process.env[key]!
    }
  }

  // The install's XDG_RUNTIME_DIR is sandbox-private, so a Wayland socket name is made absolute.
  const wayland = process.env.WAYLAND_DISPLAY

  if (wayland) {
    out.WAYLAND_DISPLAY = path.isAbsolute(wayland)
      ? wayland
      : path.join(process.env.XDG_RUNTIME_DIR || `/run/user/${process.getuid?.()}`, wayland)
  }

  return out
}

/**
 * The environment a user's desktop session hands the app: the install's HOME /
 * HERMES_HOME / PATH (with the git URL rewrite that points "GitHub" at the
 * local origin) plus the display. No HERMES_DESKTOP_HERMES_ROOT: the app must
 * find the install the way it does for a real user.
 */
export function appEnv(facts: InstallFacts, extra: Record<string, string> = {}): Record<string, string> {
  fs.mkdirSync(userDataDir(facts), { recursive: true })
  const windowState = path.join(userDataDir(facts), 'window-state.json')

  if (!fs.existsSync(windowState)) {
    fs.writeFileSync(windowState, JSON.stringify({ x: 0, y: 0, width: 1280, height: 860, isMaximized: false }))
  }

  return {
    ...facts.env,
    ...displayEnv(),
    HERMES_DESKTOP_USER_DATA_DIR: userDataDir(facts),
    HERMES_DESKTOP_SKIP_QUIT_CONFIRM: '1',
    HERMES_DESKTOP_CDP_PORT: 'off',
    ...extra
  }
}

/** A user who configured a custom OpenAI-compatible endpoint (the scripted provider). */
export function configureProvider(facts: InstallFacts, providerUrl: string): void {
  const config = path.join(facts.hermesHome, 'config.yaml')
  // Keep the seed's top-level `update:` block (the install's `--set-channel main` record): the
  // specs publish commits on main, never a release, so the stable default would offer nothing.
  const update = /^update:\n(?:[ \t].*\n?)*/m.exec(readText(config))?.[0] ?? ''
  fs.writeFileSync(config, providerConfigYaml(providerUrl, update.endsWith('\n') || !update ? update : `${update}\n`))
  fs.writeFileSync(path.join(facts.hermesHome, '.env'), 'MOCK_API_KEY=update-e2e-key\n')
}

export interface LaunchedApp {
  app: ElectronApplication
  page: Page
  logs: string[]
  logTail: () => string
}

export async function launchInstalledApp(facts: InstallFacts, env: Record<string, string>): Promise<LaunchedApp> {
  const logs: string[] = []

  const app = await _electron.launch({
    executablePath: packagedExe(facts),
    args: ['--disable-gpu', '--no-sandbox'],
    env,
    cwd: facts.home
  })

  const collect = (chunk: Buffer) => {
    logs.push(...chunk.toString('utf8').split('\n').filter(Boolean))
    logs.splice(0, Math.max(0, logs.length - 300))
  }

  app.process().stdout?.on('data', collect)
  app.process().stderr?.on('data', collect)
  const logTail = () => logs.slice(-80).join('\n')

  const page = await app.firstWindow().catch(async error => {
    await app.close().catch(() => undefined)

    throw new Error(`installed app never opened a window: ${(error as Error).message.split('\n')[0]}\n${logTail()}`)
  })

  return { app, page, logs, logTail }
}

// ─── Observation ───────────────────────────────────────────────────────

export function installProcesses(facts: InstallFacts): ProcInfo[] {
  return sandboxProcesses({ hermesHome: facts.hermesHome } as CoreSandbox)
}

function exeOf(pid: number): string {
  try {
    return fs.readlinkSync(`/proc/${pid}/exe`)
  } catch {
    return ''
  }
}

/** Live Desktop main processes of this install (the packaged exe, not its zygote/renderer/gpu children). */
export function desktopMainProcesses(facts: InstallFacts): ProcInfo[] {
  const exe = fs.realpathSync(path.dirname(packagedExe(facts)))

  return installProcesses(facts).filter(
    proc => exeOf(proc.pid).startsWith(exe + path.sep) && !/--type=/.test(proc.cmdline)
  )
}

/** The `hermes serve` backend(s) of this install (children of the backend excluded). */
export function backendServeProcesses(facts: InstallFacts): ProcInfo[] {
  const serve = installProcesses(facts).filter(
    proc =>
      / serve( |$)/.test(proc.cmdline) && !/--type=/.test(proc.cmdline) && !exeOf(proc.pid).includes('linux-unpacked')
  )

  const pids = new Set(serve.map(proc => proc.pid))

  return serve.filter(proc => !pids.has(proc.ppid))
}

export function readText(file: string): string {
  try {
    return fs.readFileSync(file, 'utf8')
  } catch {
    return ''
  }
}

export function desktopLog(facts: InstallFacts): string {
  return readText(path.join(facts.hermesHome, 'logs', 'desktop.log'))
}

export function handoffLog(facts: InstallFacts): string {
  return readText(path.join(facts.hermesHome, 'logs', 'desktop-update-handoff.log'))
}

function tail(text: string, n: number): string {
  return text.split('\n').slice(-n).join('\n')
}

/** Everything a red cell needs to explain itself. */
export function diagnostics(facts: InstallFacts, extra = ''): string {
  const logsDir = path.join(facts.hermesHome, 'logs')
  const parts: string[] = [extra]

  try {
    parts.push(`checkout HEAD: ${git(facts.checkout, 'rev-parse', 'HEAD')}`)
    parts.push(`git status:\n${git(facts.checkout, 'status', '--short')}`)
  } catch (error) {
    parts.push(String(error))
  }

  for (const name of fs.existsSync(logsDir) ? fs.readdirSync(logsDir) : []) {
    if (/\.log$/.test(name)) {
      parts.push(`── ${name} (tail) ──\n${tail(readText(path.join(logsDir, name)), 40)}`)
    }
  }

  const receipts = path.join(facts.hermesHome, 'update-receipts')

  if (fs.existsSync(receipts)) {
    for (const name of fs.readdirSync(receipts).slice(-3)) {
      parts.push(`── receipt ${name} ──\n${tail(readText(path.join(receipts, name)), 30)}`)
    }
  }

  parts.push(
    `── processes of this install ──\n${installProcesses(facts)
      .map(proc => `${proc.pid} (ppid ${proc.ppid}) ${proc.cmdline.slice(0, 200)}`)
      .join('\n')}`
  )

  return parts.filter(Boolean).join('\n')
}

/** SIGKILL every process that carries this install's HERMES_HOME (end of every spec). */
export function killInstallProcesses(facts: InstallFacts | null): void {
  if (!facts) {
    return
  }

  for (const proc of installProcesses(facts)) {
    try {
      process.kill(proc.pid, 'SIGKILL')
    } catch {
      /* already gone */
    }
  }
}

export async function waitFor<T>(
  what: string,
  probe: () => T | Promise<T>,
  { timeout, interval = 500, explain }: { timeout: number; interval?: number; explain?: () => string }
): Promise<NonNullable<T>> {
  const deadline = Date.now() + timeout

  for (;;) {
    const value = await probe()

    if (value) {
      return value as NonNullable<T>
    }

    if (Date.now() > deadline) {
      throw new Error(`timed out after ${timeout} ms waiting for ${what}\n${explain?.() ?? ''}`)
    }

    await new Promise(resolve => setTimeout(resolve, interval))
  }
}

// ─── UI steps ──────────────────────────────────────────────────────────

/** Copy of the first-run chooser / bootstrap installer overlay (the screens a healthy install must never show). */
export const FIRST_RUN_SCREENS = [
  'Set up Hermes Desktop',
  'Hermes needs a one-time install',
  'Setting up Hermes Agent',
  'Install Hermes locally',
  'Use Hermes on this computer'
]

/**
 * Record every first-run/bootstrap screen that is ever painted, even for a
 * frame (the #123888 chooser flashed before the backend won on some boots).
 */
export async function installFirstRunSampler(page: Page): Promise<void> {
  await page.evaluate(screens => {
    const w = window as unknown as { __firstRunSeen?: string[] }
    w.__firstRunSeen = []

    const scan = () => {
      const text = document.body?.innerText ?? ''

      for (const screen of screens) {
        if (text.includes(screen) && !w.__firstRunSeen!.includes(screen)) {
          w.__firstRunSeen!.push(screen)
        }
      }
    }

    scan()
    new MutationObserver(scan).observe(document.documentElement, {
      childList: true,
      subtree: true,
      characterData: true
    })
  }, FIRST_RUN_SCREENS)
}

export async function firstRunScreensSeen(page: Page): Promise<string[]> {
  return page.evaluate(() => (window as unknown as { __firstRunSeen?: string[] }).__firstRunSeen ?? [])
}

export async function openAbout(page: Page): Promise<void> {
  await page.evaluate(() => {
    location.hash = '#/settings?tab=about'
  })
  await expect(page.getByRole('button', { name: /check now|update now/i }).first()).toBeVisible({ timeout: 60_000 })
}

/** Force a check against the (local) origin and wait until the About panel offers "Update now". */
export async function waitForUpdateOffer(page: Page, targetSha: string): Promise<void> {
  await expect
    .poll(
      async () => {
        const status = await checkForUpdates(page).catch(e => ({ e: String(e) }))

        return JSON.stringify(status)
      },
      { timeout: 120_000, intervals: [2_000], message: `Desktop update check offers ${targetSha}` }
    )
    .toContain(targetSha)
  await expect(page.getByRole('button', { name: /^update now$/i }).first()).toBeVisible({ timeout: 60_000 })
}

/** The `[updates]` lines the app logged after `offset` (what it told the user about this update). */
export function updateLogLines(facts: InstallFacts, offset = 0): string {
  return desktopLog(facts)
    .slice(offset)
    .split('\n')
    .filter(line => line.includes('[updates]'))
    .join('\n')
}

/**
 * Click the About panel's "Update now" and wait until the app either quits for
 * the hand-off or logs why it did not. The final assertion is gated on the
 * open PM-install pre-flight bug (#122991) by its exact message.
 */
export async function clickUpdateNowAndExpectHandoff(launched: LaunchedApp, facts: InstallFacts): Promise<void> {
  const offset = desktopLog(facts).length
  let quit = false
  launched.app.process().once('exit', () => {
    quit = true
  })
  await launched.page
    .getByRole('button', { name: /^update now$/i })
    .first()
    .click()
  await waitFor(
    'the app to quit for the update hand-off, or to log why it did not',
    () => quit || /\[updates\] .*(fail|cancel|refus|could not|already running)/i.test(updateLogLines(facts, offset)),
    { timeout: 120_000, explain: () => diagnostics(facts, launched.logTail()) }
  )
  await gatedOn(KNOWN.preflightPython, () => {
    expect(
      quit,
      `"Update now" hands off to the updater and the app quits for it; the app logged:\n${updateLogLines(facts, offset)}`
    ).toBe(true)
  })
}

export function isAlive(pid: number): boolean {
  try {
    process.kill(pid, 0)

    return !/^\S+ \(.*\) Z /.test(readText(`/proc/${pid}/stat`))
  } catch {
    return false
  }
}

/** The fields of the Desktop update status these specs read. */
export interface UpdateStatus {
  currentSha?: string
  targetSha?: string
  updateAvailable?: boolean
  behind?: null | number
}

/** A forced update check through the app's own preload bridge (what the About panel calls). */
export async function checkForUpdates(page: Page): Promise<null | UpdateStatus> {
  return page.evaluate(() => {
    const bridge = (
      window as unknown as { hermesDesktop: { updates: { check(o: { force: boolean }): Promise<unknown> } } }
    ).hermesDesktop

    return bridge.updates.check({ force: true }) as Promise<null | UpdateStatus>
  })
}

/** "Is this install current on `sha`?" as the app reports it. */
export async function currentAs(page: Page): Promise<{ currentSha?: string; updateAvailable: boolean }> {
  const status = await checkForUpdates(page)

  return {
    currentSha: status?.currentSha,
    updateAvailable: Boolean(status?.updateAvailable ?? (status?.behind ?? 0) > 0)
  }
}

/** Close an app whose process may already be gone (Playwright's handle throws synchronously then). */
export async function closeQuietly(app: ElectronApplication): Promise<void> {
  try {
    await app.close()
  } catch {
    // already exited
  }
}
