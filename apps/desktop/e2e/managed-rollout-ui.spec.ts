/**
 * Browser-rendered rehearsal of the production managed-rollout section.
 *
 * The companion Vite fixture mounts the real React component, CSS, theme, and
 * Arabic locale with deterministic read-only fleet data. It does not exercise
 * Electron main, preload IPC, native adapters, or SSH. Those acceptance gates
 * remain open until a disposable Electron backend can start.
 */

import * as fs from 'node:fs/promises'
import * as os from 'node:os'
import * as path from 'node:path'

import { createServer, type ViteDevServer } from 'vite'

import { expect, type Page, test } from './test'

const DESKTOP_ROOT = path.resolve(import.meta.dirname, '..')
const FLEET_SIZE = 160
const ROLLOUT_TITLE = 'عمليات النشر المُدارة'

let server: ViteDevServer
let scratch: string
let url: string

function fleetRows() {
  return Array.from({ length: FLEET_SIZE }, (_, index) => {
    const connectionId = '00000000-0000-4000-8000-' + (index + 1).toString(16).padStart(12, '0')
    const installId = (index + 1).toString(16).padStart(32, '0')

    return {
      installId,
      connectionId,
      aliasConnectionIds: index === 0
        ? Array.from({ length: 12 }, (_, alias) => '00000000-0000-4000-8000-' + (alias + 200).toString(16).padStart(12, '0'))
        : [],
      codeRoot: '/disposable/hermes/' + index,
      repositoryId: 'github.com/NousResearch/hermes-agent',
      headSha: 'a'.repeat(40),
      requiredScopeIds: ['default'],
      source: {
        connectionId,
        verifiedHostKeyFingerprint: index < 2 ? 'SHA256:shared-fixture-machine' : 'SHA256:fixture-machine-' + index
      }
    }
  })
}

function managedRollouts(page: Page) {
  return page.getByRole('region', { name: ROLLOUT_TITLE })
}

function fleet(page: Page) {
  return managedRollouts(page).getByRole('region', { name: 'Managed rollout fleet' })
}

async function capture(page: Page, name: string): Promise<void> {
  await managedRollouts(page).getByRole('heading', { name: ROLLOUT_TITLE }).scrollIntoViewIfNeeded()
  const file = test.info().outputPath(name + '.png')

  await page.screenshot({ path: file, animations: 'disabled', caret: 'hide' })
  await test.info().attach(name, { path: file, contentType: 'image/png' })
}

test.beforeAll(async () => {
  scratch = await fs.mkdtemp(path.join(os.tmpdir(), 'hermes-managed-rollout-ui-'))
  Object.assign(globalThis, { __dirname: DESKTOP_ROOT })
  server = await createServer({
    root: DESKTOP_ROOT,
    configFile: path.join(DESKTOP_ROOT, 'vite.config.ts'),
    configLoader: 'runner',
    cacheDir: path.join(scratch, 'node_modules/.vite'),
    server: { host: '127.0.0.1', port: 0, strictPort: false },
    optimizeDeps: { entries: ['scripts/fixtures/managed-rollout-ui.html'] }
  })
  await server.listen()
  url = server.resolvedUrls!.local[0] + 'scripts/fixtures/managed-rollout-ui.html'
})

test.afterAll(async () => {
  await server?.close()
  const resolved = path.resolve(scratch ?? '')
  const tempRoot = path.resolve(os.tmpdir()) + path.sep

  if (resolved.startsWith(tempRoot) && path.basename(resolved).startsWith('hermes-managed-rollout-ui-')) {
    await fs.rm(resolved, { recursive: true, force: true })
  }
})

test.beforeEach(async ({ page }) => {
  await page.addInitScript(rows => {
    ;(window as unknown as { __managedRolloutRows: unknown[] }).__managedRolloutRows = rows
  }, fleetRows())
  await page.goto(url)
  await expect(managedRollouts(page)).toBeVisible({ timeout: 45_000 })
  await expect(fleet(page).locator('button[aria-pressed]')).toHaveCount(FLEET_SIZE)
})

test('keyboard selection retains focus and cannot authorize preparation', async ({ page }) => {
  const buttons = fleet(page).locator('button[aria-pressed]')
  const first = buttons.first()
  const second = buttons.nth(1)

  await first.focus()
  await page.keyboard.press('Enter')
  await expect(first).toHaveAttribute('aria-pressed', 'true')
  await expect(first).toBeFocused()
  await page.keyboard.press('Tab')
  await expect(second).toBeFocused()
  await page.keyboard.press('Space')
  await expect(second).toHaveAttribute('aria-pressed', 'true')
  await expect(managedRollouts(page).getByRole('button', { name: 'Prepare selected targets' })).toBeDisabled()
  await expect(managedRollouts(page).getByRole('button', { name: 'Start rollout' })).toHaveCount(0)
  expect(await page.evaluate(() => (window as unknown as {
    __managedRolloutCalls: { preparation: number; start: number; command: number }
  }).__managedRolloutCalls)).toEqual({ preparation: 0, start: 0, command: 0 })
})

test('light and dark layouts stay reachable at wide and narrow viewports', async ({ page }) => {
  for (const scheme of ['light', 'dark'] as const) {
    await page.emulateMedia({ colorScheme: scheme })
    await expect(page.locator('html')).toHaveAttribute('data-hermes-mode', scheme)

    for (const width of [1220, 760]) {
      await page.setViewportSize({ width, height: 800 })
      await expect(managedRollouts(page)).toBeVisible()
      await expect(fleet(page).locator('button[aria-pressed]').first()).toBeVisible()

      const layout = await page.evaluate(() => ({
        viewport: window.innerWidth,
        overflow: document.documentElement.scrollWidth - window.innerWidth
      }))

      expect(layout.viewport).toBe(width)
      expect(layout.overflow).toBeLessThanOrEqual(2)
      await capture(page, 'managed-rollout-' + scheme + '-' + width)
    }
  }
})

test('Arabic RTL, long identity text, and reduced motion remain usable', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' })
  await expect(page.locator('html')).toHaveAttribute('lang', 'ar')
  await expect(page.locator('html')).toHaveAttribute('dir', 'rtl')

  const firstRow = fleet(page).locator('div.border-b').first()

  const observed = await managedRollouts(page).evaluate(section => ({
    title: section.querySelector('h2')?.textContent?.trim() ?? null,
    inventoryExplanation: section.querySelector('section[aria-label="Observed managed SSH installations"] > p')?.textContent?.trim() ?? null,
    inventoryRevision: section.querySelector('section[aria-label="Observed managed SSH installations"] > p:nth-of-type(2)')?.textContent?.trim() ?? null,
    firstSelect: section.querySelector('section[aria-label="Managed rollout fleet"] button[aria-pressed]')?.textContent?.trim() ?? null,
    sharedMachineWarning: section.querySelector('section[aria-label="Managed rollout fleet"] .text-amber-600')?.textContent?.trim() ?? null
  }))

  expect(observed.inventoryExplanation).toBe('تعكس قائمة الأجهزة الحالة المرصودة فقط. لا يعني تحديد هدف وحده أنه مؤهل أو أن تحديثه مُصرَّح به.')
  expect(observed.inventoryRevision).toBe('مراجعة القائمة: ui-rehearsal-1؛ وقت الالتقاط وفق الساعة الرتيبة: 100.')
  expect(observed.firstSelect).toBe('تحديد')
  expect(observed.sharedMachineWarning).toBe('آلة مشتركة؛ راجع الملكية قبل الإعداد.')
  await expect(firstRow.locator('p.text-amber-600')).toBeVisible()
  expect((await firstRow.textContent())?.length ?? 0).toBeGreaterThan(200)
  await expect(firstRow.locator('button[aria-pressed]')).toBeVisible()

  const durations = await firstRow.locator('button[aria-pressed]').evaluate(button =>
    getComputedStyle(button).transitionDuration.split(',').map(duration => Number.parseFloat(duration))
  )

  expect(durations.length).toBeGreaterThan(0)
  expect(durations.every(seconds => Number.isFinite(seconds) && seconds <= 0.001)).toBe(true)

  const observedFile = test.info().outputPath('managed-rollout-arabic-dom-copy.json')

  await fs.writeFile(observedFile, JSON.stringify({ locale: 'ar', direction: 'rtl', observed }, null, 2))
  await test.info().attach('managed-rollout-arabic-dom-copy', { path: observedFile, contentType: 'application/json' })
  await capture(page, 'managed-rollout-arabic-rtl-long-reduced-motion')
})

test('160 rendered rows respond to selection with measured interaction time', async ({ page }) => {
  const buttons = fleet(page).locator('button[aria-pressed]')
  const timings: number[] = []

  for (const index of [79, 159]) {
    const button = buttons.nth(index)

    await button.scrollIntoViewIfNeeded()
    const start = await page.evaluate(() => performance.now())

    await button.click()
    await expect(button).toHaveAttribute('aria-pressed', 'true')
    timings.push(await page.evaluate(() => performance.now()) - start)
  }

  expect(timings.every(elapsed => elapsed < 5000)).toBe(true)

  const timingsFile = test.info().outputPath('managed-rollout-interaction-ms.json')

  await fs.writeFile(timingsFile, JSON.stringify({ renderedRows: FLEET_SIZE, selectedIndices: [79, 159], timings }, null, 2))
  await test.info().attach('managed-rollout-interaction-ms', { path: timingsFile, contentType: 'application/json' })
  await expect(managedRollouts(page).getByRole('button', { name: 'Start rollout' })).toHaveCount(0)
})
