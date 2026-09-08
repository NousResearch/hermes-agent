import { type ChildProcess, execFileSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

import type { TestInfo } from '@playwright/test'

import {
  buildAppEnv,
  createSandbox,
  launchDesktop,
  type MockBackendFixture,
  type Sandbox,
  waitForAppReady,
  writeEnvFile,
  writeMockProviderConfig
} from './fixtures'
import { startMockServer } from './mock-server'
import { cleanupAfterOwnedElectron, finishOwnedElectronShutdown } from './owned-electron-cleanup'
import { collectErrorBanners, expect, type Page, test } from './test'

interface UpdateStatus {
  supported: boolean
  hermesRoot: string
  currentSha: string
  targetSha: string
  behind: number
  updateAvailable?: boolean
  error?: string
}

interface DesktopBridge {
  api: <T>(request: { path: string; method?: string; body?: unknown }) => Promise<T>
  updates: { check: () => Promise<UpdateStatus> }
}

interface MemoryStatus {
  builtin_paths: { memory: string; user: string }
  builtin_files: { memory: number; user: number }
}

interface Skill {
  name: string
  provenance: 'agent' | 'bundled' | 'hub'
}

type OwnedFixture = MockBackendFixture & { child: ChildProcess }

interface ContextMenuDiagnostics {
  observerStartedAt: number
  events: { at: number; trusted: boolean; button: number }[]
}

type ObservedWindow = Window & { __hermesE2EContextMenus?: ContextMenuDiagnostics }

const FIXTURE_SKILLS = ['acceptance-learned', 'acceptance-hub'] as const

// This is an independent Git checkout, never a linked worktree or a clone of
// the developer's refs. The native checker currently requires a .git directory.
// Deliberately omit hermes_cli/main.py: backend resolution must reject this
// update-only fixture and fall back to the real dev SOURCE_REPO_ROOT.
function seedUpdateRepository(sandbox: Sandbox): { root: string; sha: string } {
  const root = path.join(sandbox.root, 'update-repository')
  const origin = path.join(sandbox.root, 'update-origin.git')

  const git = (...args: string[]) => execFileSync('git', args, {
    cwd: sandbox.root,
    encoding: 'utf8',
    windowsHide: true,
    env: { ...process.env, GIT_CONFIG_NOSYSTEM: '1', GIT_CONFIG_GLOBAL: path.join(sandbox.root, 'no-global-config') }
  }).trim()

  git('init', '--bare', '--initial-branch=main', origin)
  git('init', '--initial-branch=main', root)
  git('-C', root, '-c', 'user.name=Acceptance Fixture', '-c', 'user.email=acceptance@example.invalid',
    '-c', 'commit.gpgSign=false', 'commit', '--allow-empty', '-m', 'Synthetic update acceptance fixture')
  git('-C', root, 'remote', 'add', 'origin', origin)
  git('-C', root, 'push', '--set-upstream', 'origin', 'main')
  fs.writeFileSync(path.join(sandbox.userDataDir, 'updates.json'), JSON.stringify({ branch: 'main' }))

  return { root, sha: git('-C', root, 'rev-parse', 'HEAD') }
}

function seedSettingsFiles(sandbox: Sandbox): void {
  const memories = path.join(sandbox.hermesHome, 'memories')
  fs.mkdirSync(memories, { recursive: true })
  fs.writeFileSync(path.join(memories, 'MEMORY.md'), '# Synthetic memory\nAcceptance fixture only.\n')
  fs.writeFileSync(path.join(memories, 'USER.md'), '# Synthetic user\nNo real user data.\n')

  const logs = path.join(sandbox.hermesHome, 'logs')
  fs.mkdirSync(logs, { recursive: true })

  for (const file of ['agent', 'errors', 'gateway']) {
    fs.writeFileSync(path.join(logs, `${file}.log`), [
      `2026-01-01 00:00:00 INFO acceptance: ${file}-info-sentinel`,
      `2026-01-01 00:00:01 WARNING acceptance: ${file}-warning-sentinel`,
      `2026-01-01 00:00:02 ERROR acceptance: ${file}-error-sentinel`,
      ''
    ].join('\n'))
  }

  const skills = path.join(sandbox.hermesHome, 'skills')

  for (const name of FIXTURE_SKILLS) {
    const dir = path.join(skills, name)
    fs.mkdirSync(dir, { recursive: true })
    fs.writeFileSync(path.join(dir, 'SKILL.md'),
      `---\nname: ${name}\ndescription: Synthetic acceptance skill.\n---\n# ${name}\nSynthetic fixture only.\n`)
  }

  fs.mkdirSync(path.join(skills, '.hub'), { recursive: true })
  fs.writeFileSync(path.join(skills, '.hub', 'lock.json'), JSON.stringify({
    installed: { 'acceptance-hub': { install_path: 'acceptance-hub' } }
  }))
}

function findBundledSkill(sandbox: Sandbox, listedSkills: Skill[]): { name: string; sourcePath: string } {
  const names = new Set(fs.readFileSync(path.join(sandbox.hermesHome, 'skills', '.bundled_manifest'), 'utf8')
    .split('\n').map(line => line.split(':')[0]!.trim()).filter(Boolean))

  const listedNames = new Set(listedSkills.map(skill => skill.name))
  const bundledRoot = path.resolve(import.meta.dirname, '../../../skills')

  // Startup owns this manifest. Find a real repository SKILL.md using only
  // directory/file metadata; never forge a bundled identity or read user data.
  const find = (directory: string): { name: string; sourcePath: string } | undefined => {
    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
      if (!entry.isDirectory() || entry.isSymbolicLink()) {
        continue
      }

      const skillDir = path.join(directory, entry.name)
      const sourcePath = path.join(skillDir, 'SKILL.md')

      if (names.has(entry.name) && listedNames.has(entry.name) && fs.existsSync(sourcePath) && fs.lstatSync(sourcePath).isFile()) {
        return { name: entry.name, sourcePath }
      }

      const nested = find(skillDir)

      if (nested) {
        return nested
      }
    }

    return undefined
  }

  const bundled = find(bundledRoot)
  expect(bundled, 'Startup manifest must identify a listed bundled skill backed by a repository SKILL.md').toBeDefined()

  return bundled!
}

async function backend<T>(page: Page, endpoint: string): Promise<T> {
  return page.evaluate(async apiPath => {
    const desktop = (window as unknown as { hermesDesktop: DesktopBridge }).hermesDesktop

    return desktop.api<T>({ path: apiPath })
  }, endpoint)
}

async function navigate(page: Page, route: string): Promise<void> {
  // Neutral setup may dismiss one pre-existing menu before testing starts.
  // A menu appearing during a feature or navigation is never auto-dismissed.
  const menus = page.getByRole('menu').filter({ visible: true })
  await expect(menus, 'Feature navigation must start without a modal menu').toHaveCount(0)
  await page.evaluate(next => { window.location.hash = next }, route)
  await expect(menus, 'Feature navigation must not open a modal menu').toHaveCount(0)
}

async function captureMenuCheckpoint(page: Page, testInfo: TestInfo, label: string): Promise<void> {
  await testInfo.attach(`${label}-menu-state`, {
    body: JSON.stringify({
      visibleMenus: await page.getByRole('menu').filter({ visible: true }).count(),
      // Diagnostic only: acceptance always uses accessible roles normally.
      allMenus: await page.getByRole('menu', { includeHidden: true }).count(),
      observer: await page.evaluate(() => (window as ObservedWindow).__hermesE2EContextMenus ?? null),
      earlierEventOrigin: 'unknown; observer cannot recover events before installation'
    }), contentType: 'application/json'
  })
}

async function selectLogControl(page: Page, group: string, name: string): Promise<void> {
  await page.getByRole('group', { name: group, exact: true })
    .getByRole('button', { name, exact: true }).filter({ visible: true }).click()
}

function redactDiagnostics(text: string): string {
  return text
    .replace(/-----BEGIN [^-]*PRIVATE KEY-----[\s\S]*?(?:-----END [^-]*PRIVATE KEY-----|$)/g, '[redacted private key]')
    .split('\n')
    .map(line => /authorization|cookie|password|credential|secret|token|ticket|api[_ -]?key|bearer|https?:\/\/[^\s/]+@/i.test(line)
      ? '[redacted sensitive diagnostic line]'
      : line.replace(/\b[A-Za-z0-9_+/=-]{32,}\b/g, '[redacted opaque value]'))
    .join('\n')
}

function describeFailure(error: unknown): string {
  if (error instanceof AggregateError) {
    return `${redactDiagnostics(error.message)}\n${error.errors.map(describeFailure).join('\n\n')}`
  }

  return redactDiagnostics(error instanceof Error ? (error.stack ?? error.message) : String(error))
}

async function captureSandboxDiagnostics(sandbox: Sandbox, testInfo: TestInfo, label: string): Promise<void> {
  const failures: unknown[] = []
  const hermesHome = path.resolve(sandbox.hermesHome)

  // main.ts copies backend stdout/stderr into desktop.log. Keep that tail,
  // including tracebacks, not just [boot] lines. Never scan the home or read
  // config, .env, auth files, session stores, or another profile's directory.
  for (const name of ['desktop.log', 'gui.log', 'errors.log', 'agent.log']) {
    try {
      const file = path.join(hermesHome, 'logs', name)

      if (!fs.existsSync(file)) {
        continue
      }

      if (fs.realpathSync(hermesHome) !== hermesHome || fs.realpathSync(file) !== file || !fs.lstatSync(file).isFile()) {
        throw new Error(`Refusing redirected or non-regular synthetic diagnostic file: ${name}`)
      }

      const fd = fs.openSync(file, 'r')
      let tail: string

      try {
        const size = fs.fstatSync(fd).size
        const start = Math.max(0, size - 128 * 1024)
        const buffer = Buffer.alloc(size - start)
        const bytesRead = fs.readSync(fd, buffer, 0, buffer.length, start)
        const text = buffer.subarray(0, bytesRead).toString('utf8')
        const firstNewline = text.indexOf('\n')

        // Do not retain a partial first line: it could lose a credential label.
        tail = start > 0 ? (firstNewline < 0 ? '' : text.slice(firstNewline + 1)) : text
      } finally {
        fs.closeSync(fd)
      }

      await testInfo.attach(`${label}-redacted-${name}`, {
        body: redactDiagnostics(tail), contentType: 'text/plain'
      })
    } catch (error) {
      failures.push(error)
    }
  }

  if (failures.length > 0) {
    throw new AggregateError(failures, `${label} synthetic diagnostics capture failed`)
  }
}

async function closeOwnedElectron(
  app: MockBackendFixture['app'],
  child: ChildProcess
): Promise<void> {
  await finishOwnedElectronShutdown({
    close: async () => {
      let closeTimer: ReturnType<typeof setTimeout> | undefined

      try {
        await Promise.race([
          app.close(),
          new Promise<never>((_, reject) => {
            closeTimer = setTimeout(() => reject(new Error('Test Electron did not close within 30 seconds')), 30_000)
          })
        ])
      } finally {
        clearTimeout(closeTimer)
      }
    },
    killIfRunning: () => {
      // Only this launch's retained child handle is eligible. No process scans.
      if (child.exitCode === null && child.signalCode === null && !child.kill()) {
        throw new Error('Could not signal the owned test Electron process')
      }
    },
    waitForExit: async () => {
      await expect.poll(() => child.exitCode !== null || child.signalCode !== null, {
        timeout: 5_000, message: 'The owned test Electron process must exit after the close attempt'
      }).toBe(true)
    }
  })
}

async function closeLifecycle(fixture: OwnedFixture, testInfo: TestInfo, label: string): Promise<void> {
  const failures: unknown[] = []

  try {
    await captureMenuCheckpoint(fixture.page, testInfo, `${label}-before-close`).catch(error => failures.push(error))
    // Collect BEFORE closing/reopening; installing the next page's guard clears
    // its shared buffer. Explicit assertions preserve each lifecycle's errors.
    const errors = await collectErrorBanners(fixture.page)
    await testInfo.attach(`${label}-error-banners`, {
      body: JSON.stringify(errors.map(redactDiagnostics)), contentType: 'application/json'
    }).catch(error => failures.push(error))

    await fixture.page.screenshot({ path: testInfo.outputPath(`${label}.png`) }).catch(error => failures.push(error))

    const tracePath = testInfo.outputPath(`${label}-trace.zip`)
    await fixture.app.context().tracing.stopChunk({ path: tracePath }).catch(error => failures.push(error))

    if (fs.existsSync(tracePath)) {
      await testInfo.attach(`${label}-trace`, { path: tracePath, contentType: 'application/zip' }).catch(error => failures.push(error))
    }

    expect(errors, `${label} must not show error banners`).toEqual([])
  } catch (error) {
    failures.push(error)
  } finally {
    await closeOwnedElectron(fixture.app, fixture.child).catch(error => failures.push(error))
    // Capture after the close attempt. The sandbox remains available because
    // Electron closure cannot prove the separately spawned backend has exited.
    await captureSandboxDiagnostics(fixture.sandbox, testInfo, label).catch(error => failures.push(error))
  }

  if (failures.length > 0) {
    throw new AggregateError(failures, `${label} lifecycle verification or cleanup failed`)
  }
}

// Electron owns its page; requesting Playwright's browser page would launch a second surface.
// eslint-disable-next-line no-empty-pattern
test('settings retain backend truth across reopen, log filters, memory paths and skill provenance', async ({}, testInfo) => {
  test.setTimeout(300_000)
  const sandbox = createSandbox('settings-recovery')
  let mock: Awaited<ReturnType<typeof startMockServer>> | undefined
  let fixture: OwnedFixture | undefined
  let ownedApp: MockBackendFixture['app'] | undefined
  let ownedChild: ChildProcess | undefined
  let launchAttempted = false
  let lifecycle = 'english'
  const failures: unknown[] = []

  try {
    mock = await startMockServer()
    writeMockProviderConfig(sandbox.hermesHome, mock.url, '  language: en',
      'desktop:\n  automatic_update_checks: false\ncurator:\n  enabled: false')
    writeEnvFile(sandbox.hermesHome)
    seedSettingsFiles(sandbox)

    for (const relativePath of ['memories/MEMORY.md', 'memories/USER.md', 'logs/agent.log', 'logs/errors.log', 'logs/gateway.log']) {
      await testInfo.attach(`synthetic-${relativePath.replace('/', '-')}`, {
        body: fs.readFileSync(path.join(sandbox.hermesHome, relativePath)), contentType: 'text/plain'
      })
    }

    const updateRepo = seedUpdateRepository(sandbox)

    const env = buildAppEnv(sandbox, {
      HERMES_DESKTOP_HERMES_ROOT: updateRepo.root,
      HERMES_DESKTOP_APP_NAME: `HermesE2E-SettingsRecovery-${path.basename(sandbox.root)}`
    })

    const open = async (): Promise<OwnedFixture> => {
      // Retain evidence after any launch attempt; backend exit is not proven.
      launchAttempted = true

      const launched = await launchDesktop(env, app => {
        ownedApp = app
        // Playwright disposes the application's dispatcher during close().
        // Its process() accessor is no longer usable afterward; retain the
        // actual child handle now for both lifecycle and final cleanup.
        ownedChild = app.process()
      })

      const next = { ...launched, child: ownedChild!, mock: mock!, mockUrl: mock!.url, sandbox, cleanup: async () => {} }

      // Assign before waiting so boot failures still close the app and mock.
      fixture = next
      await next.page.evaluate(() => {
        const diagnostics: ContextMenuDiagnostics = { observerStartedAt: Date.now(), events: [] }
        const observedWindow = window as ObservedWindow

        observedWindow.__hermesE2EContextMenus = diagnostics
        window.addEventListener('contextmenu', event => {
          // isTrusted is technical metadata, not proof of human input.
          diagnostics.events.push({ at: Date.now(), trusted: event.isTrusted, button: event.button })

          if (diagnostics.events.length > 20) {
            diagnostics.events.shift()
          }
        }, true)
      })
      await waitForAppReady(next, 120_000)
      expect(await next.app.evaluate(({ app }) => app.getPath('userData'))).toBe(sandbox.userDataDir)
      const nativeTitle = await next.app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0]?.getTitle())
      expect(nativeTitle).toMatch(/^\[E2E\] /)
      await testInfo.attach(`${lifecycle}-window-title`, {
        body: JSON.stringify({ nativeTitle, documentTitle: await next.page.title() }), contentType: 'application/json'
      })

      await test.step(`${lifecycle}: establish neutral UI before feature navigation`, async () => {
        const menus = next.page.getByRole('menu').filter({ visible: true })
        const neutralFailures: unknown[] = []

        try {
          await captureMenuCheckpoint(next.page, testInfo, `${lifecycle}-before-neutral-setup`)

          if (await menus.count() > 0) {
            await next.page.keyboard.press('Escape')
          }

          await expect(menus, 'One setup Escape must close any pre-existing menu').toHaveCount(0)
        } catch (error) {
          neutralFailures.push(error)
        } finally {
          await captureMenuCheckpoint(next.page, testInfo, `${lifecycle}-after-neutral-setup`)
            .catch(error => neutralFailures.push(error))
        }

        if (neutralFailures.length > 0) {
          throw new AggregateError(neutralFailures, 'Neutral setup or its diagnostic capture failed')
        }
      })

      return next
    }

    let current = await open()
    let page = current.page

    await test.step('automatic checks start off and native Check now reaches a truthful terminal state', async () => {
      await navigate(page, '/settings?tab=about')
      const automatic = page.getByRole('switch', { name: 'Automatic update checks', exact: true })
      await expect(automatic).not.toBeChecked()

      const check = page.getByRole('button', { name: 'Check now', exact: true })

      await expect(check).toBeEnabled()
      await check.click()
      await expect(page.getByText("You're on the latest version.", { exact: true })).toBeVisible({ timeout: 30_000 })
      await expect(check).toBeEnabled()
      await expect(page.getByRole('button', { name: 'Checking…', exact: true })).toHaveCount(0)

      const native = await page.evaluate(() =>
        (window as unknown as { hermesDesktop: DesktopBridge }).hermesDesktop.updates.check())

      expect(native).toMatchObject({
        supported: true, hermesRoot: updateRepo.root, currentSha: updateRepo.sha, targetSha: updateRepo.sha, behind: 0
      })
      expect(native.error).toBeUndefined()

      if (native.updateAvailable !== undefined) {
        expect(native.updateAvailable).toBe(false)
      }

      await testInfo.attach('native-local-git-check', { body: JSON.stringify(native), contentType: 'application/json' })
      await automatic.click()
      await expect(automatic).toBeChecked()
      await expect.poll(async () => (await backend<{ desktop: { automatic_update_checks: boolean } }>(page, '/api/config'))
        .desktop.automatic_update_checks).toBe(true)
    })

    await test.step('file and severity stay independent, including All levels and search empty state', async () => {
      await navigate(page, '/command-center?section=system')
      const loadStart = Date.now()
      const loadFailures: unknown[] = []

      // System loads status and logs together. Observe its real content rather
      // than prewarming either API; the first status import can exceed 5s.
      await expect(page.getByText(/agent-warning-sentinel/)).toBeVisible({ timeout: 30_000 })
        .catch(error => loadFailures.push(error))
      const visibleElapsedMs = Date.now() - loadStart
      const uiVisible = loadFailures.length === 0
      await testInfo.attach('first-system-log-timing', {
        body: JSON.stringify({ visibleElapsedMs, uiVisible }), contentType: 'application/json'
      }).catch(error => loadFailures.push(error))
      const apiStart = Date.now()

      try {
        const logResponse = await backend<{ file: string; lines: string[] }>(page, '/api/logs?file=agent&level=WARNING&lines=100')
        await testInfo.attach('first-system-log-readback', {
          body: JSON.stringify({
            visibleElapsedMs, uiVisible, apiElapsedMs: Date.now() - apiStart,
            file: logResponse.file, lines: logResponse.lines.map(redactDiagnostics)
          }), contentType: 'application/json'
        })
      } catch (error) {
        loadFailures.push(error)
      }

      if (loadFailures.length > 0) {
        throw new AggregateError(loadFailures, 'First System log visibility/readback failed')
      }

      await expect(page.getByText(/agent-info-sentinel/)).toHaveCount(0)
      await selectLogControl(page, 'Log file', 'errors.log')
      await expect(page.getByText(/errors-warning-sentinel/)).toBeVisible()
      await expect(page.getByText(/agent-warning-sentinel/)).toHaveCount(0)
      await selectLogControl(page, 'Level', 'All levels')
      await expect(page.getByText(/errors-info-sentinel/)).toBeVisible()
      await expect(page.getByRole('group', { name: 'Log file', exact: true })
        .getByRole('button', { name: 'errors.log', exact: true }).filter({ visible: true })).toHaveAttribute('data-active', 'true')
      await selectLogControl(page, 'Level', 'error')
      await expect(page.getByText(/errors-error-sentinel/)).toBeVisible()
      await expect(page.getByText(/errors-warning-sentinel/)).toHaveCount(0)
      await selectLogControl(page, 'Log file', 'gateway.log')
      await expect(page.getByText(/gateway-error-sentinel/)).toBeVisible()
      await expect(page.getByText(/gateway-warning-sentinel/)).toHaveCount(0)
      await selectLogControl(page, 'Level', 'All levels')
      await expect(page.getByText(/gateway-info-sentinel/)).toBeVisible()
      await expect(page.getByText('Showing up to 100 recent lines from the selected file and level.', { exact: true })).toBeVisible()
      const search = page.getByPlaceholder('Filter log lines...')
      await search.fill('no-such-acceptance-entry')
      await expect(page.getByText('No log lines match the search.', { exact: true })).toBeVisible()
      await search.fill('')
      await expect(page.getByText(/gateway-info-sentinel/)).toBeVisible()
      await page.screenshot({ path: testInfo.outputPath('english-log-filters.png') })
    })

    await test.step('an empty selected log has its own empty state, independent of search', async () => {
      // No gateway is started in this sandbox, so this synthetic fixture file
      // has no writer. Re-selecting it exercises the real backend empty tail.
      fs.writeFileSync(path.join(sandbox.hermesHome, 'logs', 'gateway.log'), '')
      await selectLogControl(page, 'Log file', 'errors.log')
      await expect(page.getByText(/errors-info-sentinel/)).toBeVisible()
      await selectLogControl(page, 'Log file', 'gateway.log')
      await expect(page.getByText('No logs loaded yet.', { exact: true })).toBeVisible()
      await expect(page.getByText('No log lines match the search.', { exact: true })).toHaveCount(0)
      expect(await backend(page, '/api/logs?file=gateway&lines=100')).toEqual({ file: 'gateway', lines: [] })
    })

    await test.step('memory metadata points at active synthetic files and Open file is enabled', async () => {
      await navigate(page, '/command-center?section=maintenance')

      const memory = await backend<MemoryStatus>(page, '/api/memory')
      expect(memory.builtin_paths).toEqual({
        memory: path.join(sandbox.hermesHome, 'memories', 'MEMORY.md'),
        user: path.join(sandbox.hermesHome, 'memories', 'USER.md')
      })

      for (const key of ['memory', 'user'] as const) {
        expect(memory.builtin_files[key]).toBe(fs.statSync(memory.builtin_paths[key]).size)
        expect(memory.builtin_files[key]).toBeGreaterThan(0)
      }

      const openFiles = page.getByRole('button', { name: 'Open file', exact: true })
      // Maintenance hydrates memory metadata after the panel shell is painted.
      // Keep the assertion tied to the real accessible controls without racing
      // that asynchronous load.
      await expect(openFiles).toHaveCount(2, { timeout: 30_000 })
      await expect(openFiles.nth(0)).toBeEnabled({ timeout: 30_000 })
      await expect(openFiles.nth(1)).toBeEnabled({ timeout: 30_000 })
      await testInfo.attach('synthetic-memory-paths', { body: JSON.stringify(memory.builtin_paths), contentType: 'application/json' })
    })

    await test.step('only agent provenance is labelled learned', async () => {
      const skills = await backend<Skill[]>(page, '/api/skills')
      const bundled = findBundledSkill(sandbox, skills)

      const cases = [
        { name: FIXTURE_SKILLS[0], provenance: 'agent', label: 'learned' },
        { name: bundled.name, provenance: 'bundled', label: 'built-in' },
        { name: FIXTURE_SKILLS[1], provenance: 'hub', label: 'hub' }
      ] as const

      await testInfo.attach('bundled-skill-source-metadata', {
        body: JSON.stringify(bundled), contentType: 'application/json'
      })

      for (const { name, provenance } of cases) {
        expect(skills.find(skill => skill.name === name)?.provenance).toBe(provenance)
      }

      await navigate(page, '/skills?tab=skills')

      const counts = { agent: 0, bundled: 0, hub: 0 }

      for (const skill of skills) {
        counts[skill.provenance] += 1
      }

      // The skills route can render its shell before the capabilities query has
      // hydrated.  Wait for the computed summary rather than racing the first
      // paint; the exact text still verifies the backend provenance counts.
      const summaryPattern = new RegExp(
        `${counts.agent}\\s+learned\\s+·\\s+${counts.bundled}\\s+built-in\\s+·\\s+${counts.hub}\\s+hub`
      )

      await expect(page.getByText(summaryPattern).filter({ visible: true }).first()).toBeVisible({ timeout: 30_000 })

      for (const { name, provenance, label } of cases) {
        const row = page.getByRole('button').filter({ has: page.getByText(name, { exact: true }) }).first()
        await expect(row).toBeVisible({ timeout: 30_000 })
        await expect(row).toContainText(label, { timeout: 30_000 })

        if (provenance !== 'agent') {
          await expect(row).not.toContainText('learned')
        }
      }

      await page.screenshot({ path: testInfo.outputPath('skill-provenance.png') })
    })

    await page.evaluate(async () => {
      const desktop = (window as unknown as { hermesDesktop: DesktopBridge }).hermesDesktop
      const config = await desktop.api<{ display?: Record<string, unknown> }>({ path: '/api/config' })
      await desktop.api({ path: '/api/config', method: 'PUT', body: {
        config: { ...config, display: { ...config.display, language: 'ar' } }
      } })
    })
    fixture = undefined
    await closeLifecycle(current, testInfo, lifecycle)
    lifecycle = 'arabic-reopened'
    current = await open()
    page = current.page

    await test.step('same sandbox reopens with persisted checks and Arabic presentation', async () => {
      await expect(page.locator('html')).toHaveAttribute('lang', 'ar')
      await expect(page.locator('html')).toHaveAttribute('dir', 'rtl')
      await navigate(page, '/settings?tab=about')
      await expect(page.getByRole('switch', { name: 'التحقق التلقائي من التحديثات', exact: true })).toBeChecked()
      expect((await backend<{ desktop: { automatic_update_checks: boolean } }>(page, '/api/config'))
        .desktop.automatic_update_checks).toBe(true)
      await expect(page.getByRole('button', { name: 'التحقق الآن', exact: true })).toBeEnabled()
      await page.screenshot({ path: testInfo.outputPath('arabic-about-persisted.png') })
      await navigate(page, '/command-center?section=system')
      await selectLogControl(page, 'ملف السجل', 'errors.log')
      await selectLogControl(page, 'مستوى السجل', 'كل المستويات')
      await expect(page.getByText(/errors-info-sentinel/)).toBeVisible()
      await expect(page.getByText('يُعرض آخر 100 سطر كحد أقصى من الملف والمستوى المحددين.', { exact: true })).toBeVisible()
      await page.getByPlaceholder('البحث في سطور السجل...').fill('no-such-acceptance-entry')
      await expect(page.getByText('لا توجد سطور تطابق البحث.', { exact: true })).toBeVisible()
    })
  } catch (error) {
    failures.push(error)
  } finally {
    try {
      if (fixture) {
        await closeLifecycle(fixture, testInfo, lifecycle).catch(error => failures.push(error))
      } else if (failures.length > 0) {
        if (ownedApp && ownedChild?.exitCode === null && ownedChild.signalCode === null) {
          await closeOwnedElectron(ownedApp, ownedChild).catch(error => failures.push(error))
        }

        await captureSandboxDiagnostics(sandbox, testInfo, 'setup-failure').catch(error => failures.push(error))
      }
    } finally {
      try {
        await mock?.close().catch(error => failures.push(error))
      } finally {
        try {
          const cleaned = cleanupAfterOwnedElectron({ launchAttempted }, sandbox.cleanup)

          if (!cleaned) {
            await testInfo.attach('retained-synthetic-sandbox', {
              body: JSON.stringify({ root: sandbox.root, reason: 'Owned backend exit is unproven after launch' }),
              contentType: 'application/json'
            })
          }
        } catch (error) {
          failures.push(error)
        }
      }
    }
  }

  if (failures.length > 0) {
    // Some reporters print only AggregateError.message, not its .errors.
    // Keep the original failure first and include the redacted cleanup chain.
    throw new AggregateError(failures,
      `Settings acceptance failed; original and cleanup errors:\n${failures.map(describeFailure).join('\n\n')}`)
  }
})
