/**
 * Browser-level coverage for the lunar-city visualizer.
 *
 * The test deliberately uses the real desktop shell and mock backend so the
 * screenshot and interaction checks exercise the route as a user sees it.
 */
import * as fs from 'node:fs'
import * as path from 'node:path'

import {
  buildAppEnv,
  findElectron,
  type MockBackendFixture,
  setupMockBackend,
  waitForAppReady,
  writeEnvFile,
  writeMockProviderConfig
} from './fixtures'
import {
  createLunarCityPopulationFixture,
  type StandardGatewaySource,
  startPopulationGateways
} from './lunar-city-fixtures'
import { startMockServer } from '../../../tests-js/scripts/mock-server'
import { _electron, expect, installErrorBannerGuard, test } from './test'

const DESKTOP_ROOT = path.resolve(import.meta.dirname, '..')

let fixture: MockBackendFixture | null = null

test.beforeAll(async () => {
  fixture = await setupMockBackend()
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('renders the live lunar city canvas and its snapshot-backed accessible controls', async ({
  page: _page
}, testInfo) => {
  const page = fixture!.page
  const lunarRequests: string[] = []

  const recordLunarRequest = (request: { url(): string }) => {
    if (request.url().includes('/lunar-city/')) {
      lunarRequests.push(request.url())
    }
  }

  await waitForAppReady(fixture!, 120_000)
  page.on('request', recordLunarRequest)
  await page.evaluate(() => {
    window.location.hash = '#/lunar-city'
  })

  await expect(page.getByRole('heading', { name: 'Lunar City' })).toBeVisible()
  const canvas = page.getByLabel('Interactive 3D Lunar City')
  await expect(canvas).toBeVisible()
  await expect(canvas).toHaveAttribute('data-world-status', 'ready', { timeout: 30_000 })
  expect(lunarRequests.some(url => /\/v2\/models\/terrain\.glb(?:[?#]|$)/.test(url))).toBe(true)
  expect(lunarRequests.some(url => /moon-settlement-approved\.jpg/i.test(url))).toBe(false)
  await expect(page.getByRole('complementary', { name: 'Lunar City accessible operations' })).toBeVisible()
  await expect(page.getByRole('combobox', { name: /3D quality/i })).toBeVisible()
  await expect(page.getByRole('region', { name: 'Source health' })).toBeVisible()
  await expect(page.getByRole('button', { name: /Open (Library|Research Lab)/i })).toHaveCount(0)
  await expect(page.getByRole('button', { name: /Inspect .* worker/i })).toHaveCount(0)
  await expect(page.getByText('MISSIONS')).toHaveCount(0)

  const viewport = page.getByTestId('lunar-city-viewport')
  await expect(page.getByRole('button', { name: 'Zoom In' })).toBeVisible()
  await page.getByRole('button', { name: 'Zoom In' }).click()
  await expect(viewport).not.toHaveAttribute('data-zoom')
  await page.getByRole('button', { name: 'Return to City' }).click()
  await page.screenshot({ path: testInfo.outputPath('lunar-city-overview.png'), fullPage: true })
  page.off('request', recordLunarRequest)
})

async function profileSessionCount(source: StandardGatewaySource, profile: string): Promise<number> {
  const response = await fetch(`${source.url}/api/profiles/sessions?limit=500&profile=all`, {
    headers: { 'X-Hermes-Session-Token': source.token }
  })

  const payload = (await response.json()) as { sessions: Array<{ profile: string }> }

  return payload.sessions.filter(session => session.profile === profile).length
}

function expectedLeaderModelId(owner: { connectionId: string; profile: string }): string {
  const models = ['owl', 'fox', 'badger', 'otter', 'bird', 'stag'] as const
  const ownerKey = `${encodeURIComponent(owner.connectionId)}::${encodeURIComponent(owner.profile)}`
  let hash = 0

  for (const character of ownerKey) {
    hash = (hash * 31 + character.codePointAt(0)!) >>> 0
  }

  return models[hash % models.length]!
}

function seedDuplicateProfile(home: string, mockUrl: string): void {
  const profileDir = path.join(home, 'profiles', 'duplicate')
  fs.mkdirSync(profileDir, { recursive: true })
  writeMockProviderConfig(profileDir, mockUrl)
  writeEnvFile(profileDir)
  fs.writeFileSync(path.join(profileDir, 'SOUL.md'), '# Duplicate fixture leader\n', 'utf8')
  fs.writeFileSync(
    path.join(profileDir, 'profile.yaml'),
    'display_name: Duplicate fixture leader\ndescription: Exact-owner E2E fixture\n',
    'utf8'
  )
}

function writeCollisionRegistry(
  userDataDir: string,
  source0: StandardGatewaySource,
  source2: StandardGatewaySource
): void {
  fs.writeFileSync(
    path.join(userDataDir, 'connections.json'),
    `${JSON.stringify(
      {
        version: 2,
        primary: 'local',
        launchMode: 'primary',
        lastUsed: 'local',
        connections: [
          { id: 'local', kind: 'local', label: 'This device' },
          {
            id: 'source-0',
            kind: 'remote',
            label: 'Source zero',
            authMode: 'token',
            token: { encoding: 'plain', value: source0.token },
            url: source0.url
          },
          {
            id: 'source-2',
            kind: 'remote',
            label: 'Source two',
            authMode: 'token',
            token: { encoding: 'plain', value: source2.token },
            url: source2.url
          }
        ]
      },
      null,
      2
    )}\n`,
    { encoding: 'utf8', mode: 0o600 }
  )
}

async function cleanupPopulationFixture(cleanup: () => void): Promise<void> {
  let lastError: unknown

  for (let attempt = 0; attempt < 5; attempt += 1) {
    try {
      cleanup()

      return
    } catch (error) {
      lastError = error
      await new Promise(resolve => setTimeout(resolve, 100 * (attempt + 1)))
    }
  }

  throw lastError
}

test('a physical-model accessibility pick disambiguates exact owners, focuses the camera, and persists dialogue', async () => {
  test.slow()
  test.setTimeout(240_000)
  const population = createLunarCityPopulationFixture(25)
  const mock = await startMockServer()
  let gateways: Awaited<ReturnType<typeof startPopulationGateways>> | undefined
  let app: Awaited<ReturnType<typeof _electron.launch>> | undefined

  try {
    for (const home of Object.values(population.sourceHomes)) {
      writeMockProviderConfig(home, mock.url)
      writeEnvFile(home)
    }

    seedDuplicateProfile(population.sourceHomes['remote-lab']!, mock.url)
    seedDuplicateProfile(population.sourceHomes['remote-archive']!, mock.url)

    gateways = await startPopulationGateways(population)
    const source0 = gateways.sources.find(source => source.connectionId === 'remote-lab')!
    const source2 = gateways.sources.find(source => source.connectionId === 'remote-archive')!
    writeCollisionRegistry(population.userDataDir, source0, source2)

    const env = buildAppEnv({
      cleanup: population.cleanup,
      hermesHome: population.hermesHome,
      root: population.root,
      userDataDir: population.userDataDir
    })

    app = await _electron.launch({
      args: [DESKTOP_ROOT, '--no-sandbox'],
      cwd: DESKTOP_ROOT,
      env,
      executablePath: findElectron()
    })
    const page = await app.firstWindow()
    installErrorBannerGuard(page)
    await waitForAppReady({ app, page } as MockBackendFixture, 120_000)
    await page.evaluate(() => {
      window.location.hash = '#/lunar-city'
    })

    const canvas = page.getByLabel('Interactive 3D Lunar City')
    await expect(canvas).toHaveAttribute('data-world-status', 'ready', { timeout: 60_000 })
    const owner = { connectionId: 'source-2', profile: 'duplicate' }
    const otherOwner = { connectionId: 'source-0', profile: 'duplicate' }
    const modelId = expectedLeaderModelId(owner)
    expect(expectedLeaderModelId(otherOwner)).toBe(modelId)

    const remoteBefore = await profileSessionCount(source2, owner.profile)
    const otherBefore = await profileSessionCount(source0, otherOwner.profile)
    const cameraStatus = page.getByRole('status', { name: 'Camera position' })
    const cameraBefore = await cameraStatus.textContent()

    await page
      .getByRole('button', { name: new RegExp(`^Select ${modelId} leader model with \\d+ exact profiles$`, 'u') })
      .press('Enter')
    const chooser = page.getByRole('region', { name: `Choose exact ${modelId} profile` })
    await expect(chooser.getByRole('button', { name: 'Talk to duplicate leader on source-0' })).toBeVisible()
    await chooser.getByRole('button', { name: 'Talk to duplicate leader on source-2' }).press('Enter')

    const dialogue = page.getByRole('dialog', { name: 'duplicate leader conversation' })
    await expect(dialogue).toBeVisible({ timeout: 60_000 })
    await expect(dialogue).toContainText('source-2 / duplicate')
    await expect(cameraStatus).toContainText('duplicate leader')
    expect(await cameraStatus.textContent()).not.toBe(cameraBefore)

    const message = `remote-lab persistent leader turn ${Date.now()}`
    await page.getByRole('textbox', { name: 'Message duplicate leader' }).fill(message)
    await page.getByRole('textbox', { name: 'Message duplicate leader' }).press('Enter')
    await expect.poll(() => mock.receivedPrompts).toContain(message)
    await expect.poll(() => profileSessionCount(source2, owner.profile)).toBe(remoteBefore + 1)
    await expect.poll(() => profileSessionCount(source0, otherOwner.profile)).toBe(otherBefore)

    await page.getByRole('button', { name: 'Close conversation' }).press('Enter')
    await page.getByRole('button', { name: 'Talk to duplicate leader on source-2' }).press('Enter')
    await expect(dialogue).toContainText('source-2 / duplicate')
    const resumedMessage = `source-2 resumed persistent leader turn ${Date.now()}`
    await page.getByRole('textbox', { name: 'Message duplicate leader' }).fill(resumedMessage)
    await page.getByRole('textbox', { name: 'Message duplicate leader' }).press('Enter')
    await expect.poll(() => mock.receivedPrompts).toContain(resumedMessage)
    await expect.poll(() => profileSessionCount(source2, owner.profile)).toBe(remoteBefore + 1)
    await expect.poll(() => profileSessionCount(source0, otherOwner.profile)).toBe(otherBefore)
  } finally {
    await app?.close().catch(() => undefined)
    await gateways?.close()
    await mock.close()
    await cleanupPopulationFixture(population.cleanup)
  }
})
