/**
 * E2E `/save` regression — verifies the desktop slash command persists the
 * transcript privately under the Hermes profile home.
 *
 * Covers the filesystem contract in tui_gateway/methods_session.py's
 * `session.save` handler: `secure_artifact_dir()` on `~/.hermes/sessions/saved/`
 * (0700) and `artifact_file_mode()` on the written JSON (0600), plus the
 * saved transcript containing both the user's message and the mock reply.
 *
 * Prerequisite: `npm run build` must have been run so dist/ exists.
 */

import * as fs from 'node:fs'
import * as path from 'node:path'

import { expect, test } from './test'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { MOCK_REPLY } from './mock-server'

let fixture: MockBackendFixture | null = null

test.beforeAll(async () => {
  fixture = await setupMockBackend()
  await waitForAppReady(fixture!, 120_000)
})
test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test.describe('/save command', () => {
  test('persists the transcript privately with 0700 dir / 0600 file', async ({}, testInfo) => {
    const page = fixture!.page

    const composer = page.locator('[contenteditable="true"]').first()
    await composer.waitFor({ state: 'visible', timeout: 10_000 })

    await composer.click()
    await composer.type('Remember this for later', { delay: 20 })
    await page.keyboard.press('Enter')

    // Wait for the mock reply so the transcript has both a user and an
    // assistant message before we save it.
    await page.waitForFunction(
      (reply: string) => (document.body.textContent ?? '').includes(reply),
      MOCK_REPLY,
      { timeout: 60_000 },
    )

    // A streamed token can render before message.complete commits the turn to
    // session history. The voice primary only exists with an empty composer
    // when `busy === false`, so it is the UI's authoritative settled edge.
    await page.getByRole('button', { name: 'Start voice conversation' }).waitFor({
      state: 'visible',
      timeout: 30_000,
    })

    const savedDir = path.join(fixture!.sandbox.hermesHome, 'sessions', 'saved')
    fs.mkdirSync(savedDir, { recursive: true })
    if (process.platform !== 'win32') fs.chmodSync(savedDir, 0o755)
    await composer.click()
    await composer.type('/save', { delay: 20 })
    // Enter may accept the slash-completion row without submitting it.
    await page.getByRole('button', { name: 'Send', exact: true }).click()

    // The sandbox starts empty, so the created JSON is unambiguous and this
    // remains correct when HERMES_HOME contains spaces.
    await expect
      .poll(
        () => (fs.existsSync(savedDir) ? fs.readdirSync(savedDir).filter(name => name.endsWith('.json')).length : 0),
        { timeout: 30_000 },
      )
      .toBe(1)
    const savedPath = path.join(savedDir, fs.readdirSync(savedDir).find(name => name.endsWith('.json'))!)

    if (process.platform !== 'win32') {
      expect(fs.statSync(savedDir).mode & 0o777).toBe(0o700)
      expect(fs.statSync(savedPath).mode & 0o777).toBe(0o600)
    }

    const saved = JSON.parse(fs.readFileSync(savedPath, 'utf8')) as {
      messages: Array<{ role: string; content: unknown }>
    }

    const roles = saved.messages.map(m => m.role)
    expect(roles).toContain('user')
    expect(roles).toContain('assistant')

    const flatten = (content: unknown): string =>
      typeof content === 'string' ? content : JSON.stringify(content)

    expect(saved.messages.some(m => m.role === 'user' && flatten(m.content).includes('Remember this for later'))).toBe(true)
    expect(saved.messages.some(m => m.role === 'assistant' && flatten(m.content).includes(MOCK_REPLY))).toBe(true)
    await testInfo.attach('saved-transcript', {
      body: fs.readFileSync(savedPath),
      contentType: 'application/json',
    })
    await page.screenshot({ path: testInfo.outputPath('save-confirmed.png') })
  })
})
