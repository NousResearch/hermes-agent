/**
 * Electron regression for stale virtual-row measurements when the session
 * sidebar switches from compact rows to Inbox cards.
 */

import * as path from 'node:path'
import { DatabaseSync } from 'node:sqlite'

import type { Page } from '@playwright/test'

import {
  buildAppEnv,
  createSandbox,
  launchDesktop,
  type MockBackendFixture,
  waitForAppReady,
  writeEnvFile,
  writeMockProviderConfig
} from './fixtures'
import { startMockServer } from './mock-server'
import { RealSessionBuilder } from './real-session-builder'
import { expect, test } from './test'

const SESSION_COUNT = 28
const SESSION_PREFIX = 'E2E Inbox measurement row'
const OVERLAP_TOLERANCE_PX = 0.5
const RESIZE_GATE = '__hermesE2EDropResizeObservations'

interface SessionRowRect {
  bottom: number
  height: number
  index: number
  session: boolean
  text: string
  top: number
}

interface LayoutSample {
  maxOverlap: number
  overlaps: Array<{ current: number; next: number; pixels: number }>
  rows: SessionRowRect[]
}

async function setupSeededSidebar(): Promise<MockBackendFixture> {
  const mock = await startMockServer()
  const sandbox = createSandbox('inbox-card-measurement')

  try {
    writeMockProviderConfig(sandbox.hermesHome, mock.url)
    writeEnvFile(sandbox.hermesHome)

    const builder = await RealSessionBuilder.start(sandbox.hermesHome)
    const sessions: string[] = []

    try {
      for (let index = 0; index < SESSION_COUNT; index += 1) {
        const number = String(index + 1).padStart(2, '0')

        const session = await builder.createSession({
          title: `${SESSION_PREFIX} ${number}`,
          turns: [`${SESSION_PREFIX} ${number} with enough preview text to render a complete Inbox card.`]
        })

        sessions.push(session.sessionId)
      }
    } finally {
      await builder.close()
    }

    const nowSeconds = Date.now() / 1000
    const stateDb = path.join(sandbox.hermesHome, 'state.db')

    const database = new DatabaseSync(stateDb)

    try {
      const updateSession = database.prepare('UPDATE sessions SET started_at = ?, last_activity_at = ? WHERE id = ?')
      const updateMessages = database.prepare('UPDATE messages SET timestamp = ? WHERE session_id = ?')

      sessions.forEach((sessionId, index) => {
        const ageSeconds = index < 2 ? index * 60 : 32 * 24 * 60 * 60 + (index - 2) * 60
        const timestamp = nowSeconds - ageSeconds

        updateSession.run(timestamp, timestamp, sessionId)
        updateMessages.run(timestamp, sessionId)
      })
    } finally {
      database.close()
    }

    const { app, page } = await launchDesktop(buildAppEnv(sandbox))
    await page.addInitScript(gateName => {
      const NativeResizeObserver = window.ResizeObserver

      window.ResizeObserver = class extends NativeResizeObserver {
        constructor(callback: ResizeObserverCallback) {
          super((entries, observer) => {
            if (!(window as unknown as Record<string, boolean>)[gateName]) {
              callback(entries, observer)
            }
          })
        }
      }
    }, RESIZE_GATE)
    await page.reload()

    return {
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
  } catch (error) {
    await mock.close()
    sandbox.cleanup()
    throw error
  }
}

async function readVisibleSessionLayout(page: Page): Promise<LayoutSample> {
  const firstSession = page.locator('[data-slot="sidebar"] button').filter({ hasText: SESSION_PREFIX }).first()

  await expect(firstSession).toBeAttached()

  return firstSession.evaluate(
    (button, { prefix, tolerance }) => {
      const item = button.closest('[data-index]')
      const spacer = item?.parentElement

      if (!spacer) {
        throw new Error('Could not resolve the virtual session-list spacer')
      }

      const rows = Array.from(spacer.children)
        .filter(
          (candidate): candidate is HTMLElement =>
            candidate instanceof HTMLElement && candidate.hasAttribute('data-index')
        )
        .map(candidate => {
          const rect = candidate.getBoundingClientRect()
          const text = (candidate.textContent ?? '').trim().replace(/\s+/g, ' ').slice(0, 100)

          return {
            bottom: rect.bottom,
            height: rect.height,
            index: Number(candidate.dataset.index),
            session: text.includes(prefix),
            text,
            top: rect.top
          }
        })
        .sort((left, right) => left.top - right.top)

      const overlaps: LayoutSample['overlaps'] = []

      for (let index = 0; index < rows.length - 1; index += 1) {
        const current = rows[index]
        const next = rows[index + 1]
        const pixels = current.bottom - next.top

        if (pixels > tolerance) {
          overlaps.push({ current: current.index, next: next.index, pixels })
        }
      }

      return {
        maxOverlap: Math.max(0, ...overlaps.map(overlap => overlap.pixels)),
        overlaps,
        rows
      }
    },
    { prefix: SESSION_PREFIX, tolerance: OVERLAP_TOLERANCE_PX }
  )
}

async function enableInboxStyle(page: Page): Promise<void> {
  await page.getByRole('button', { name: 'Filters' }).click()
  const option = page.getByRole('menuitemcheckbox', { name: 'Inbox style' })
  await expect(option).toHaveAttribute('aria-checked', 'false')
  await option.click()
  await page.keyboard.press('Escape')
}

async function disableInboxStyle(page: Page): Promise<void> {
  await page.getByRole('button', { name: 'Filters' }).click()
  const option = page.getByRole('menuitemcheckbox', { name: 'Inbox style' })
  await expect(option).toHaveAttribute('aria-checked', 'true')
  await option.click()
  await page.keyboard.press('Escape')
}

async function scrollSessionListToEnd(page: Page): Promise<void> {
  const session = page.locator('[data-slot="sidebar"] button').filter({ hasText: SESSION_PREFIX }).first()

  await session.evaluate(button => {
    const scroller = button.closest('[data-index]')?.parentElement?.parentElement

    if (!(scroller instanceof HTMLElement)) {
      throw new Error('Could not resolve the virtual session-list scroller')
    }

    scroller.scrollTop = scroller.scrollHeight
  })
}

test('Inbox cards keep virtualized sessions and date dividers separated', async () => {
  test.setTimeout(240_000)
  const fixture = await setupSeededSidebar()

  try {
    const { page } = fixture
    await waitForAppReady(fixture, 120_000)

    const compact = await readVisibleSessionLayout(page)
    expect(compact.rows.length, 'the fixture must render enough rows to compare adjacent geometry').toBeGreaterThan(2)
    expect(compact.overlaps).toEqual([])

    await page.evaluate(gateName => {
      ;(window as unknown as Record<string, boolean>)[gateName] = true
    }, RESIZE_GATE)
    await enableInboxStyle(page)
    await expect
      .poll(async () => (await readVisibleSessionLayout(page)).rows[0]?.height ?? 0, {
        message: 'Inbox rows should paint at card height'
      })
      .toBeGreaterThan(50)
    await page.waitForTimeout(250)

    const inbox = await readVisibleSessionLayout(page)
    expect(
      inbox.rows.some(row => !row.session),
      'the fixture must include an interleaved date divider'
    ).toBe(true)
    expect(inbox.overlaps, `Inbox rows overlap by up to ${inbox.maxOverlap.toFixed(2)}px`).toEqual([])

    await scrollSessionListToEnd(page)
    await expect(
      page.locator('[data-slot="sidebar"] [data-slot="row-button"]').filter({ hasText: `${SESSION_PREFIX} 28` })
    ).toBeVisible()

    const bottom = await readVisibleSessionLayout(page)
    expect(bottom.overlaps, `scrolled Inbox rows overlap by up to ${bottom.maxOverlap.toFixed(2)}px`).toEqual([])

    await disableInboxStyle(page)
    await expect
      .poll(async () => (await readVisibleSessionLayout(page)).rows.find(row => row.session)?.height ?? Infinity, {
        message: 'disabling Inbox style should restore compact row geometry'
      })
      .toBeLessThan(40)

    const restored = await readVisibleSessionLayout(page)
    expect(restored.overlaps, `restored compact rows overlap by up to ${restored.maxOverlap.toFixed(2)}px`).toEqual([])
  } finally {
    await fixture.cleanup()
  }
})
