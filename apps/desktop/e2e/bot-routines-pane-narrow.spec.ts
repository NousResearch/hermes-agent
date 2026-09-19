import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

import { startMockServer } from '../../../tests-js/scripts/mock-server'

import { writeEnvFile, writeMockProviderConfig } from '../../../tests-js/scripts/mock-provider-config'
import {
  buildAppEnv,
  createSandbox,
  launchDesktop,
  type MockBackendFixture,
  waitForAppReady
} from './fixtures'
import { RealSessionBuilder } from './real-session-builder'
import { expect, test } from './test'

// The Scheduled jobs (Routines) pane docks at 250px beside a bot chat. A job
// row is a grid item, and grid items default to `min-width: auto`, so a long
// nowrap title pinned the row wider than the pane: the enable/disable Switch
// and the delete control were clipped off the right edge and the next-run
// label was cut mid-word (#91623, #89534). Closing the pane with its ✕ also
// remembered the dismissal forever — re-entering Bot Mode never brought the
// pane back (#102224).

type Page = MockBackendFixture['page']

const LONG_TITLE = '[bot:alpha] Weekly research digest with a deliberately long routine title that overflows'

let fixture: MockBackendFixture | null = null
let alphaHome = ''

async function openBots(page: Page): Promise<void> {
  const tab = page
    .getByRole('button', { name: 'Bots', exact: true })
    .or(page.getByRole('tab', { name: 'Bots', exact: true }))
    .first()

  await tab.click()
  await expect(page.getByRole('button', { name: 'New bot or group chat' })).toBeVisible()
}

async function openSessions(page: Page): Promise<void> {
  const tab = page
    .getByRole('button', { name: /^sessions$/i })
    .or(page.getByRole('tab', { name: /^sessions$/i }))
    .first()

  await tab.click()
}

async function openUntil(action: () => Promise<void>, expected: () => Promise<void>, attempts = 3): Promise<void> {
  for (let attempt = 1; ; attempt += 1) {
    await action()

    try {
      await expected()

      return
    } catch (error) {
      if (attempt >= attempts) {
        throw error
      }
    }
  }
}

async function seedBot(hermesHome: string, mockUrl: string, name: string): Promise<string> {
  const dir = path.join(hermesHome, 'profiles', name)
  fs.mkdirSync(dir, { recursive: true })
  writeMockProviderConfig(dir, mockUrl)
  writeEnvFile(dir)

  const builder = await RealSessionBuilder.start(dir)

  try {
    await builder.createSession({ title: 'Bot Chat', turns: [`Hello ${name}`] })
  } finally {
    await builder.close()
  }

  return dir
}

/** A real cron job in the profile's own cron store, written by the real store code. */
function seedRoutine(profileHome: string, title: string): void {
  const python = path.resolve(process.cwd(), '../../venv/bin/python')
  const repoRoot = path.resolve(process.cwd(), '../..')

  const script = [
    'from cron.jobs import create_job',
    `create_job(name=${JSON.stringify(title)}, schedule='every 4 days', prompt='say hello')`
  ].join('\n')

  execFileSync(python, ['-c', script], {
    cwd: repoRoot,
    env: { ...process.env, HERMES_HOME: profileHome },
    stdio: 'pipe'
  })
}

function readJobs(profileHome: string): Array<{ enabled?: boolean; name?: string; state?: string }> {
  const raw = JSON.parse(fs.readFileSync(path.join(profileHome, 'cron', 'jobs.json'), 'utf8'))

  return Array.isArray(raw) ? raw : (raw.jobs ?? [])
}

async function openAlphaChat(page: Page): Promise<void> {
  await openBots(page)

  const alphaRow = page
    .getByRole('button', { name: /^alpha\b/i })
    .filter({ visible: true })
    .first()

  await expect(alphaRow).toBeVisible({ timeout: 30_000 })
  await openUntil(
    () => alphaRow.click(),
    () =>
      expect(page.getByText('Hello alpha', { exact: true }).filter({ visible: true }).first()).toBeVisible({
        timeout: 45_000
      })
  )
}

function routinesTab(page: Page) {
  return page.locator('[data-tree-tab="hermes-bots:routines"]').filter({ visible: true }).first()
}

/** Expand the collapsed right-edge Scheduled jobs tab (a no-op when the pane is already open); resolve the row. */
async function expandRoutines(page: Page) {
  const row = page
    .getByRole('button', { name: /Weekly research digest/ })
    .filter({ visible: true })
    .first()

  if (!(await row.isVisible())) {
    const tab = routinesTab(page)
    await expect(tab).toBeVisible({ timeout: 30_000 })
    await tab.click()
  }

  await expect(row).toBeVisible({ timeout: 30_000 })
  await page.waitForTimeout(2_500)
  await expect(row).toBeVisible()

  return row
}

test.beforeAll(async () => {
  const mock = await startMockServer()
  const sandbox = createSandbox('bots-routines')
  writeMockProviderConfig(sandbox.hermesHome, mock.url)
  writeEnvFile(sandbox.hermesHome)
  alphaHome = await seedBot(sandbox.hermesHome, mock.url, 'alpha')
  seedRoutine(alphaHome, LONG_TITLE)

  const { app, page } = await launchDesktop(buildAppEnv(sandbox))

  fixture = {
    app,
    page,
    mock,
    mockUrl: mock.url,
    sandbox,
    cleanup: async () => {
      await app.close().catch(() => undefined)
      await mock.close()
      sandbox.cleanup()
    }
  }
  await waitForAppReady(fixture, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('a long routine title never pushes the Switch, delete control or next-run label out of the 250px pane', async ({}, testInfo) => {
  test.setTimeout(300_000)
  const page = fixture!.page

  await openAlphaChat(page)
  const row = await expandRoutines(page)
  await page.screenshot({ path: testInfo.outputPath('routines-pane.png') })

  // Geometry against the pane's own scroll container, not body text: the row
  // is a grid item, so its intrinsic width is what pins the controls off-edge.
  const geometry = await row.evaluate(button => {
    const card = button.parentElement!.parentElement as HTMLElement
    const scroller = card.closest<HTMLElement>('.overflow-y-auto')!
    const paneRect = scroller.getBoundingClientRect()

    const rect = (el: Element | null) => {
      const r = el!.getBoundingClientRect()

      return { left: r.left, right: r.right, width: r.width }
    }

    const nextRun = [...card.querySelectorAll<HTMLElement>('span')].find(s => /^next/i.test(s.textContent ?? ''))!
    const title = button.querySelector<HTMLElement>('span.truncate')!

    return {
      card: rect(card),
      pane: rect(scroller),
      switch: rect(card.querySelector('[role="switch"]')),
      trash: rect(card.querySelector('button[aria-label]')),
      nextRun: { ...rect(nextRun), clipped: nextRun.scrollWidth > nextRun.clientWidth + 1, text: nextRun.textContent },
      titleEllipsized: title.scrollWidth > title.clientWidth,
      paneRight: paneRect.right
    }
  })


  expect.soft(geometry.card.right).toBeLessThanOrEqual(geometry.pane.right + 1)
  expect.soft(geometry.switch.right).toBeLessThanOrEqual(geometry.pane.right + 1)
  expect.soft(geometry.trash.right).toBeLessThanOrEqual(geometry.pane.right + 1)
  expect.soft(geometry.nextRun.right).toBeLessThanOrEqual(geometry.pane.right + 1)
  expect.soft(geometry.nextRun.clipped).toBe(false)
  expect.soft(geometry.nextRun.text).toMatch(/next: in \d+ days?/i)
  // The title is the one thing that MAY be cut, and it must be cut with an ellipsis.
  expect.soft(geometry.titleEllipsized).toBe(true)

  await test.step("clicking the row's Switch toggles the job in place — the pane stays, jobs.json flips (#95031)", async () => {
    const toggle = row.locator('xpath=..').getByRole('switch')
    await expect(toggle).toBeVisible()
    expect(readJobs(alphaHome)[0]?.enabled).not.toBe(false)

    await toggle.click()

    await expect.poll(() => readJobs(alphaHome)[0]?.enabled, { timeout: 30_000 }).toBe(false)
    await expect(row).toBeVisible()
    await expect(page.getByText(/Open this bot's continuous chat/i)).toHaveCount(0)
    await page.screenshot({ path: testInfo.outputPath('routines-pane-toggled.png') })
  })

  await test.step('closing Scheduled jobs with ✕ is recoverable — leaving and re-entering Bot Mode brings the pane back (#102224)', async () => {
    const tab = page.locator('[data-tree-tab="hermes-bots:routines"]').first()
    await expect(tab).toBeVisible({ timeout: 30_000 })
    await tab.hover()
    const closer = tab.getByRole('button', { name: /^close$/i }).first()
    await closer.click({ force: true })
    await page.waitForTimeout(1_500)
    await expect(page.getByRole('button', { name: /Weekly research digest/ })).toHaveCount(0, { timeout: 15_000 })
    await page.screenshot({ path: testInfo.outputPath('routines-closed.png') })

    await openSessions(page)
    await page.waitForTimeout(1_000)
    await openAlphaChat(page)
    await page.waitForTimeout(2_000)
    await page.screenshot({ path: testInfo.outputPath('routines-restored.png') })

    await expect(page.locator('[data-tree-tab="hermes-bots:routines"]').first()).toBeVisible({ timeout: 30_000 })
  })
})
