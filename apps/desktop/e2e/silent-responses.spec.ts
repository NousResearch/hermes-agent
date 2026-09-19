import { spawnSync } from 'node:child_process'
import * as path from 'node:path'

import { setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test } from './test'

const TRIGGER = 'E2E_SILENT_RESPONSE'

// Exercise the real agent/gateway/persistence/renderer chain. Only the provider
// is scripted; do not seed history or replace window.hermesClient.
test('silent replies stay invisible live and after resume without deleting history', async ({}, testInfo) => {
  const fixture = await setupMockBackend({
    modelContextLength: 128_000,
    mockServer: { textReply: '[[SILENT]]', holdFirstStreamForPrompt: TRIGGER },
  })

  try {
    const { page, mock } = fixture
    await waitForAppReady(fixture, 120_000)
    const transcript = page.locator('[data-slot="aui_thread-viewport"]')
    const composer = page.locator('[contenteditable="true"]').first()
    await composer.fill(TRIGGER)
    await page.keyboard.press('Enter')
    await Promise.race([
      mock.waitForHeldStream(),
      page.getByRole('alert').waitFor().then(async () => {
        throw new Error(await page.getByRole('alert').allTextContents().then(texts => texts.join('\n')))
      }),
    ])
    await expect(transcript).toContainText(TRIGGER)
    await expect(transcript).not.toContainText('[[SILENT]]')
    mock.releaseHeldStream()

    // Persistence is the completion barrier: a negative DOM assertion alone
    // could pass before a single provider token has reached the client.
    const storedReplies = () => {
      const result = spawnSync('python', ['-c',
        'import sqlite3,sys; c=sqlite3.connect("file:"+sys.argv[1]+"?mode=ro",uri=True); print(c.execute("SELECT count(*) FROM messages WHERE role=? AND trim(content)=?",("assistant","[[SILENT]]")).fetchone()[0])',
        path.join(fixture.sandbox.hermesHome, 'state.db'),
      ], { encoding: 'utf8' })
      return result.status === 0 ? Number(result.stdout.trim()) : 0
    }
    await expect.poll(storedReplies, { timeout: 60_000 }).toBeGreaterThanOrEqual(1)
    await expect(page.getByRole('button', { name: 'Stop', exact: true })).not.toBeVisible()
    await expect(transcript).not.toContainText('[[SILENT]]')
    await expect(transcript.locator('[data-slot="aui_assistant-message-root"]')).toHaveCount(0)
    await page.screenshot({ path: testInfo.outputPath('silent-live.png') })

    await page.reload()
    await waitForAppReady(fixture, 120_000)

    await expect(transcript).toContainText(TRIGGER)
    await expect(transcript).not.toContainText('[[SILENT]]')
    await expect(transcript.locator('[data-slot="aui_assistant-message-root"]')).toHaveCount(0)
    expect(storedReplies()).toBeGreaterThanOrEqual(1)
    await page.screenshot({ path: testInfo.outputPath('silent-resume.png') })
  } finally {
    fixture.mock.releaseHeldStream()
    await fixture.cleanup()
  }
})
