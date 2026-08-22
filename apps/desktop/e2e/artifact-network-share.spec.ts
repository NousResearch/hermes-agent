import {
  buildAppEnv,
  createSandbox,
  launchDesktop,
  type Sandbox,
  waitForAppReady,
  writeEnvFile,
  writeMockProviderConfig
} from './fixtures'
import { type MockServer, startMockServer } from './mock-server'
import { RealSessionBuilder } from './real-session-builder'
import { type ElectronApplication, expect, type Page, test } from './test'

const NETWORK_SHARE_PATH = String.raw`\\fileserver\share\generated.png`
const NETWORK_SHARE_REPLY = `Generated preview follows.\nMEDIA: ${NETWORK_SHARE_PATH}`
const SESSION_TITLE = 'E2E remote network-share artifact'

interface NetworkShareFixture {
  app: ElectronApplication
  mock: MockServer
  page: Page
  sandbox: Sandbox
  cleanup: () => Promise<void>
}

async function setupNetworkShareDesktop(): Promise<NetworkShareFixture> {
  const mock = await startMockServer({ reply: NETWORK_SHARE_REPLY })
  const sandbox = createSandbox('artifact-network-share')
  writeMockProviderConfig(sandbox.hermesHome, mock.url)
  writeEnvFile(sandbox.hermesHome)

  const builder = await RealSessionBuilder.start(sandbox.hermesHome)

  try {
    await builder.createSession({
      title: SESSION_TITLE,
      turns: ['Create the E2E artifact record.']
    })
  } finally {
    await builder.close()
  }

  const { app, page } = await launchDesktop(buildAppEnv(sandbox))

  return {
    app,
    mock,
    page,
    sandbox,
    cleanup: async () => {
      await app.close().catch(() => undefined)
      await mock.close()
      sandbox.cleanup()
    }
  }
}

async function interceptPreviewReads(app: ElectronApplication): Promise<void> {
  await app.evaluate(({ ipcMain }) => {
    const state = globalThis as typeof globalThis & { __artifactNetworkSharePreviewReads?: number }

    state.__artifactNetworkSharePreviewReads = 0
    ipcMain.removeHandler('hermes:readFileDataUrl')
    ipcMain.handle('hermes:readFileDataUrl', async () => {
      state.__artifactNetworkSharePreviewReads = (state.__artifactNetworkSharePreviewReads ?? 0) + 1

      return 'data:image/png;base64,QU5Z'
    })
  })
}

async function previewReadCount(app: ElectronApplication): Promise<number> {
  return app.evaluate(() => {
    const state = globalThis as typeof globalThis & { __artifactNetworkSharePreviewReads?: number }

    return state.__artifactNetworkSharePreviewReads ?? 0
  })
}

async function openArtifacts(page: Page): Promise<void> {
  await page.evaluate(() => {
    window.location.hash = '#/artifacts'
  })
  await page.waitForFunction(
    path => window.location.hash === '#/artifacts' && (document.body.textContent ?? '').includes(path),
    NETWORK_SHARE_PATH,
    { timeout: 30_000 }
  )
}

test.describe('artifact network-share previews', () => {
  let fixture: NetworkShareFixture | null = null

  test.afterEach(async () => {
    await fixture?.cleanup()
    fixture = null
  })

  test('does not read a persisted remote-share image merely by opening Artifacts', async () => {
    test.slow()
    fixture = await setupNetworkShareDesktop()
    await waitForAppReady(fixture, 120_000)
    await interceptPreviewReads(fixture.app)
    await openArtifacts(fixture.page)

    const card = fixture.page.locator('article').filter({ hasText: NETWORK_SHARE_PATH })
    await expect(card).toBeVisible()
    await expect(card.locator('.cursor-default')).toHaveCount(1)

    expect(await previewReadCount(fixture.app)).toBe(0)
    await expect(card.locator('[slot="artifact-media"]')).toHaveCount(0)
  })
})
