/**
 * E2E regression: desktop resume must hide agent-only transcript rows.
 *
 * Compaction handoffs are active user rows because the model needs them for
 * context continuity. They are not authored chat content, so the desktop
 * transcript must never display them after a real compressor-generated resume.
 */

import * as fs from 'node:fs'
import * as path from 'node:path'

import { expect, test } from './test'

import {
  type MockBackendFixture,
  buildAppEnv,
  createSandbox,
  launchDesktop,
  waitForAppReady,
  writeEnvFile,
  writeMockProviderConfig,
} from './fixtures'
import {
  MOCK_REPLY,
  startMockServer,
  VERIFICATION_STOP_TEXT,
  VERIFICATION_STOP_TRIGGER,
} from './mock-server'
import { RealSessionBuilder } from './real-session-builder'

const SESSION_TITLE = 'E2E Hidden History Messages'
const VISIBLE_USER_TEXT = 'E2E_VISIBLE_USER_HISTORY'
const VISIBLE_POST_COMPACTION_TEXT = 'E2E_VISIBLE_POST_COMPACTION_HISTORY'
const COMPACTION_TRIGGER_PADDING = ' force real context compression'.repeat(600)

async function setupSeededMockBackend(): Promise<MockBackendFixture> {
  const mock = await startMockServer()
  const sandbox = createSandbox('hidden-history')
  writeMockProviderConfig(sandbox.hermesHome, mock.url)
  fs.appendFileSync(
    path.join(sandbox.hermesHome, 'config.yaml'),
    '\ncompression:\n  threshold_tokens: 1\n',
    'utf8',
  )
  writeEnvFile(sandbox.hermesHome)
  const builder = await RealSessionBuilder.start(sandbox.hermesHome)
  try {
    await builder.createSession({
      title: SESSION_TITLE,
      turns: [
        `${VISIBLE_USER_TEXT}${COMPACTION_TRIGGER_PADDING}`,
        VISIBLE_POST_COMPACTION_TEXT,
      ],
    })
  } finally {
    await builder.close()
  }

  const { app, page } = await launchDesktop(buildAppEnv(sandbox))

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
    },
  }
}

test('resume hides real context-compaction handoffs', async ({}, testInfo) => {
  const fixture = await setupSeededMockBackend()

  try {
    const { page } = fixture
    await waitForAppReady(fixture, 120_000)

    const sessionRow = page
      .locator('[data-slot="sidebar"] button')
      .filter({ hasText: SESSION_TITLE })
      .first()
    await sessionRow.click()

    const transcript = page.locator('[data-slot="aui_thread-viewport"]')
    await expect(transcript).toContainText(VISIBLE_USER_TEXT)
    await expect(transcript).toContainText(VISIBLE_POST_COMPACTION_TEXT)
    await expect(transcript).toContainText(MOCK_REPLY)
    await expect(transcript).not.toContainText('[CONTEXT COMPACTION — REFERENCE ONLY]')
    await page.screenshot({ path: testInfo.outputPath('hidden-history-resume.png') })
  } finally {
    await fixture.cleanup()
  }
})

test('live verify-on-stop continuations stay out of the transcript', async ({}, testInfo) => {
  const sandbox = createSandbox('live-verification-nudge')
  const projectRoot = path.join(sandbox.root, 'project')
  const changedFile = path.join(projectRoot, 'e2e-verification-target.py')
  fs.mkdirSync(projectRoot)
  fs.writeFileSync(
    path.join(projectRoot, 'pyproject.toml'),
    '[project]\nname = "e2e-verification-project"\nversion = "0.0.0"\n',
    'utf8',
  )

  const mock = await startMockServer({ verificationWritePath: changedFile })
  writeMockProviderConfig(sandbox.hermesHome, mock.url)
  fs.appendFileSync(path.join(sandbox.hermesHome, 'config.yaml'), '\nagent:\n  verify_on_stop: true\n', 'utf8')
  writeEnvFile(sandbox.hermesHome)
  const { app, page } = await launchDesktop(buildAppEnv(sandbox))
  const fixture: MockBackendFixture = {
    app,
    page,
    mock,
    mockUrl: mock.url,
    sandbox,
    cleanup: async () => {
      await app.close().catch(() => undefined)
      await mock.close()
      sandbox.cleanup()
    },
  }

  try {
    await waitForAppReady(fixture, 120_000)
    const composer = page.locator('[contenteditable="true"]').first()
    await composer.click()
    await composer.type(VERIFICATION_STOP_TRIGGER)
    await page.keyboard.press('Enter')

    const transcript = page.locator('[data-slot="aui_thread-viewport"]')
    await expect(transcript).toContainText(VERIFICATION_STOP_TEXT, { timeout: 60_000 })
    await expect.poll(
      () => mock.receivedPrompts.some(prompt => prompt.includes('[System: You edited code in this turn')),
      { timeout: 30_000 },
    ).toBe(true)
    expect(fs.existsSync(changedFile), 'The scripted write_file call should edit only the sandbox project').toBe(true)
    await expect(transcript).not.toContainText('[System: You edited code in this turn')
    await page.screenshot({ path: testInfo.outputPath('live-verification-nudge.png') })
  } finally {
    await fixture.cleanup()
  }
})
