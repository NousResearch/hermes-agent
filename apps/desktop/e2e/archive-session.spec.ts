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

const PROMPT = 'Inbox archive visual proof'
let fixture: MockBackendFixture | null = null
let seededSessionId = ''

async function setupSeededMockBackend(): Promise<MockBackendFixture> {
  const mock = await startMockServer()
  const sandbox = createSandbox('archive-visual')

  writeMockProviderConfig(sandbox.hermesHome, mock.url)
  writeEnvFile(sandbox.hermesHome)

  const builder = await RealSessionBuilder.start(sandbox.hermesHome)

  try {
    seededSessionId = (await builder.createSession({ title: PROMPT, turns: [PROMPT] })).sessionId
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
    }
  }
}

test.beforeAll(async () => {
  fixture = await setupSeededMockBackend()
  await waitForAppReady(fixture, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

// This E2E launches Electron through its custom fixture rather than Playwright's browser fixture.
// eslint-disable-next-line no-empty-pattern
test('archived card stays absent from Inbox-style sidebar', async ({}, testInfo) => {
  const { page } = fixture!
  const initialRoute = await page.evaluate(() => window.location.hash)

  const actions = page.locator('[data-row-actions] button[aria-label="Session actions"]:visible')
  await expect(actions).toHaveCount(1, { timeout: 30_000 })
  await page.getByRole('button', { name: 'Filters' }).click()
  await page.getByText('Inbox style', { exact: true }).click()
  await page.keyboard.press('Escape')

  await expect(actions).toHaveCount(1)
  await actions.click()
  await page.getByText('Archive', { exact: true }).click()

  await expect(actions).toHaveCount(0, { timeout: 30_000 })

  // Wait on the authoritative write, not a grace-period sleep. This raw bridge
  // read bypasses the renderer cache and proves the archive RPC reached state.db.
  await expect
    .poll(
      () =>
        page.evaluate(async (sessionId: string) => {
          const desktop = window as unknown as {
            hermesDesktop: { api: <T>(request: { path: string }) => Promise<T> }
          }

          const archived = await desktop.hermesDesktop.api<{ archived?: boolean | number }>({
            path: `/api/sessions/${encodeURIComponent(sessionId)}`
          })

          // The by-id backend row is a raw SQLite projection, so the flag may
          // arrive as integer 1 rather than the list endpoint's normalized true.
          return Boolean(archived.archived)
        }, seededSessionId),
      { timeout: 30_000 }
    )
    .toBe(true)

  // A leaked menu event navigates synchronously through the row's resume path.
  await expect.poll(() => page.evaluate(() => window.location.hash)).toBe(initialRoute)
  await expect(actions).toHaveCount(0)
  await page.screenshot({ path: testInfo.outputPath('inbox-after-archive.png') })
})
