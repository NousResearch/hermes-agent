/** Real SQLite -> serve -> Electron history navigation; the mock only replaces inference. */
import { execFileSync } from 'node:child_process'
import * as path from 'node:path'

import { writeEnvFile, writeMockProviderConfig } from '../../../tests-js/scripts/mock-provider-config'
import { startMockServer } from '../../../tests-js/scripts/mock-server'

import { buildAppEnv, createSandbox, launchDesktop, waitForAppReady } from './fixtures'
import { expect, test } from './test'

const title = 'E2E bounded history navigation'

test('exact rail targets, sequential newer pages and explicit live-tail return', async ({}, testInfo) => {
  test.setTimeout(180_000)
  const python = process.env.HERMES_DESKTOP_PYTHON
  if (!python) throw new Error('Set HERMES_DESKTOP_PYTHON to an isolated dependency-complete test interpreter')
  const sandbox = createSandbox('history-navigation')
  const mock = await startMockServer()
  writeMockProviderConfig(sandbox.hermesHome, mock.url)
  writeEnvFile(sandbox.hermesHome, 'e2e-mock-key', mock.url)
  let app: Awaited<ReturnType<typeof launchDesktop>>['app'] | undefined
  try {
    const root = path.resolve(import.meta.dirname, '../../..')
    execFileSync(
      python,
      [
        '-c',
        `
from pathlib import Path
from hermes_state import SessionDB
import sys
with SessionDB(db_path=Path(sys.argv[1]) / 'state.db') as db:
    sid = 'e2e-history-navigation'
    db.create_session(session_id=sid, source='desktop')
    db.set_session_title(sid, sys.argv[2])
    rows = []
    for turn in range(40):
        rows.append({'role': 'user', 'content': f'NAV prompt {turn:02}', 'timestamp': turn + 1})
        rows.append({'role': 'assistant', 'content': ('### Navigation answer\\n\\n' + 'A durable paragraph for measuring the selected turn. ' * 30), 'timestamp': turn + 1})
        if turn == 20:
            for tool in range(250):
                rows.append({'role': 'assistant', 'content': f'Navigation tool step {tool}', 'timestamp': 21,
                    'tool_calls': [{'id': f'nav-{tool}', 'type': 'function', 'function': {'name': 'terminal', 'arguments': '{}'}}]})
                rows.append({'role': 'tool', 'tool_name': 'terminal', 'tool_call_id': f'nav-{tool}', 'content': f'Navigation result {tool}', 'timestamp': 21})
    db.append_messages_batch(sid, rows)
    db.end_session(sid, 'completed')
`,
        sandbox.hermesHome,
        title
      ],
      { cwd: root, env: buildAppEnv(sandbox) }
    )
    const launched = await launchDesktop(buildAppEnv(sandbox))
    app = launched.app
    const page = launched.page
    await waitForAppReady({ app, page, mock, mockUrl: mock.url, sandbox, cleanup: async () => {} }, 120_000)
    await page.locator('[data-slot="sidebar"] button').filter({ hasText: title }).first().click()
    const viewport = page.locator('[data-slot="aui_thread-viewport"]').first()
    await expect(viewport).toContainText('NAV prompt 39')
    const rail = page.locator('[data-slot="thread-timeline-ticks"]').first()
    await expect(rail).toBeVisible()
    await rail.evaluate(element => {
      element.scrollTop = 0
    })
    await rail.getByRole('button', { name: 'NAV prompt 00', exact: true }).click()
    const target = viewport.locator('[data-message-id="history-row-1"]')
    await expect(target).toBeVisible()
    await expect
      .poll(() =>
        target.evaluate(element => {
          const viewport = element.closest('[data-slot="aui_thread-viewport"]')!
          const group = element.closest('[data-slot="aui_message-group"]')!
          return Math.abs(group.getBoundingClientRect().top - viewport.getBoundingClientRect().top - 8)
        })
      )
      .toBeLessThanOrEqual(2)
    await expect(rail.getByRole('button', { name: 'NAV prompt 00', exact: true })).toHaveAttribute(
      'aria-current',
      'location'
    )
    await page.screenshot({ path: testInfo.outputPath('exact-history-target.png') })
    const later = viewport.getByRole('button', { name: 'Show later messages' })
    for (let pageNumber = 0; pageNumber < 4; pageNumber += 1) {
      const before = await viewport.textContent()
      await later.click()
      await expect.poll(() => viewport.textContent()).not.toBe(before)
      expect(await viewport.locator('[data-message-id]').count()).toBeLessThanOrEqual(360)
    }
    await expect(viewport).toContainText('NAV prompt 39')
    await expect(later).toHaveCount(0)
    await expect(viewport.getByRole('button', { name: 'Show earlier messages' })).toBeVisible()
    // Return through adjacent pages, not by jumping to another prompt or live.
    for (let pageNumber = 0; pageNumber < 2; pageNumber += 1) {
      const before = await viewport.textContent()
      await viewport.getByRole('button', { name: 'Show earlier messages' }).click()
      await expect.poll(() => viewport.textContent()).not.toBe(before)
      expect(await viewport.locator('[data-message-id]').count()).toBeLessThanOrEqual(360)
    }
    await expect(viewport).toContainText('NAV prompt 00')
    await expect(viewport.getByRole('button', { name: 'Show earlier messages' })).toHaveCount(0)
    await expect(later).toBeVisible()
    await page.getByRole('button', { name: 'Jump to latest', exact: true }).click()
    await expect(viewport.locator('[data-message-id^="history-"]')).toHaveCount(0)
    await expect(viewport).toContainText('NAV prompt 39')
    await page.screenshot({ path: testInfo.outputPath('returned-live-tail.png') })
  } finally {
    await app?.close().catch(() => undefined)
    await mock.close()
    sandbox.cleanup()
  }
})
