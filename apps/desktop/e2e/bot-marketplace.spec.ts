import fs from 'node:fs'
import http, { type Server } from 'node:http'
import path from 'node:path'

import { type ElectronApplication, type Page, type TestInfo } from '@playwright/test'

import { startMockServer } from '../../../tests-js/scripts/mock-server'

import {
  buildAppEnv,
  createSandbox,
  launchDesktop,
  type MockBackendFixture,
  waitForAppReady,
  writeEnvFile,
  writeMockProviderConfig
} from './fixtures'
import { expect, test } from './test'

// Local cross-product acceptance only: build website and Desktop first, then run
// BOT_MARKETPLACE_ACCEPTANCE=1 npx playwright test e2e/bot-marketplace.spec.ts
// The general Desktop CI suite must not acquire a website build dependency.
test.skip(process.env.BOT_MARKETPLACE_ACCEPTANCE !== '1', 'Opt-in local site/Desktop acceptance')

const REPO_ROOT = path.resolve(import.meta.dirname, '..', '..', '..')
const WEBSITE_BUILD = path.join(REPO_ROOT, 'website', 'build')

const PICKER_ORIGIN = 'https://hermes-agent.nousresearch.com'
const STARTER_PROMPT = 'Load the research-analyst workflow skill and perform its first useful task using only its bundled synthetic sample.'

interface WebsiteServer {
  baseUrl: string
  requests: string[]
  close: () => Promise<void>
}

interface AcceptanceFixture extends MockBackendFixture {
  mappedWebsiteRequests: string[]
}

function contentType(file: string): string {
  if (file.endsWith('.html')) {return 'text/html; charset=utf-8'}

  if (file.endsWith('.js')) {return 'text/javascript; charset=utf-8'}

  if (file.endsWith('.css')) {return 'text/css; charset=utf-8'}

  if (file.endsWith('.json')) {return 'application/json; charset=utf-8'}

  if (file.endsWith('.svg')) {return 'image/svg+xml'}

  if (file.endsWith('.png')) {return 'image/png'}

  if (file.endsWith('.woff2')) {return 'font/woff2'}

  return 'application/octet-stream'
}

function startBuiltWebsite(): Promise<WebsiteServer> {
  if (!fs.existsSync(path.join(WEBSITE_BUILD, 'bots', 'index.html'))) {
    throw new Error(`Locally built Bot Marketplace is missing: ${WEBSITE_BUILD}. Run the website build first.`)
  }

  return new Promise((resolve, reject) => {
    const requests: string[] = []

    const server: Server = http.createServer((request, response) => {
      requests.push(request.url || '/')
      const requestUrl = new URL(request.url || '/', 'http://127.0.0.1')
      let pathname = decodeURIComponent(requestUrl.pathname)

      if (pathname === '/docs') {pathname = '/'}
      else if (pathname.startsWith('/docs/')) {pathname = pathname.slice('/docs'.length)}

      const relative = pathname.replace(/^\/+/, '')
      const candidate = path.resolve(WEBSITE_BUILD, relative)

      if (candidate !== WEBSITE_BUILD && !candidate.startsWith(`${WEBSITE_BUILD}${path.sep}`)) {
        response.writeHead(403).end('forbidden')

        return
      }

      let file = candidate

      try {
        if (fs.statSync(file).isDirectory()) {file = path.join(file, 'index.html')}
      } catch {
        if (!path.extname(file)) {file = path.join(file, 'index.html')}
      }

      if (!fs.existsSync(file) || !fs.statSync(file).isFile()) {
        response.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' }).end(`missing ${pathname}`)

        return
      }

      response.writeHead(200, {
        'cache-control': 'no-store',
        'content-type': contentType(file)
      })
      fs.createReadStream(file).pipe(response)
    })

    server.once('error', reject)
    server.listen(0, '127.0.0.1', () => {
      const address = server.address()

      if (!address || typeof address === 'string') {
        reject(new Error('Built website server did not expose a TCP port'))

        return
      }

      resolve({
        baseUrl: `http://127.0.0.1:${address.port}`,
        requests,
        close: () => new Promise<void>((done, fail) => server.close(error => error ? fail(error) : done()))
      })
    })
  })
}

async function mapProductionPickerToLocalBuild(app: ElectronApplication, website: WebsiteServer): Promise<void> {
  await app.evaluate(async ({ net, session }, input) => {
    await new Promise<void>((resolve, reject) => {
      session.defaultSession.protocol.interceptBufferProtocol(
        'https',
        (request: { url: string }, callback: (response: {
          data: Buffer
          headers?: Record<string, string[]>
          mimeType?: string
          statusCode: number
        }) => void) => {
        const requested = new URL(request.url)

        if (requested.origin !== input.origin || !requested.pathname.startsWith('/docs/')) {
          callback({ statusCode: 404, data: Buffer.from('not mapped by acceptance harness') })

          return
        }

        const local = new URL(`${requested.pathname}${requested.search}`, input.baseUrl)
        void net.fetch(local.toString()).then(async (response: Response) => {
          const headers: Record<string, string[]> = {}
          response.headers.forEach((value: string, key: string) => { headers[key] = [value] })
          callback({
            data: Buffer.from(await response.arrayBuffer()),
            headers,
            mimeType: response.headers.get('content-type')?.split(';')[0],
            statusCode: response.status
          })
        }).catch((error: unknown) => {
          callback({ statusCode: 502, data: Buffer.from(`local build mapping failed: ${String(error)}`) })
        })
      }, (error: Error | null) => error ? reject(error) : resolve())
    })
  }, { baseUrl: website.baseUrl, origin: PICKER_ORIGIN })
}

async function setupAcceptance(website: WebsiteServer, prefix: string): Promise<AcceptanceFixture> {
  const mock = await startMockServer({
    replyForPrompt: prompt => prompt.includes('bundled synthetic sample')
      ? 'Synthetic demonstration complete. What real input would you like to use next?'
      : 'Hello from the isolated mock inference server.'
  })

  const sandbox = createSandbox(prefix)
  const isolatedHome = path.join(sandbox.root, 'os-home')
  const xdgConfig = path.join(sandbox.root, 'xdg-config')
  const xdgCache = path.join(sandbox.root, 'xdg-cache')
  const xdgData = path.join(sandbox.root, 'xdg-data')
  const xdgState = path.join(sandbox.root, 'xdg-state')
  const xdgRuntime = path.join(sandbox.root, 'xdg-runtime')

  for (const directory of [isolatedHome, xdgConfig, xdgCache, xdgData, xdgState, xdgRuntime]) {
    fs.mkdirSync(directory, { recursive: true })
  }

  fs.chmodSync(xdgRuntime, 0o700)

  writeMockProviderConfig(
    sandbox.hermesHome,
    mock.url,
    undefined,
    'approvals:\n  mode: manual\ntimezone: Etc/UTC'
  )
  writeEnvFile(sandbox.hermesHome)

  const env = buildAppEnv(sandbox, {
    HOME: isolatedHome,
    USERPROFILE: isolatedHome,
    XDG_CONFIG_HOME: xdgConfig,
    XDG_CACHE_HOME: xdgCache,
    XDG_DATA_HOME: xdgData,
    XDG_STATE_HOME: xdgState,
    XDG_RUNTIME_DIR: xdgRuntime
  })

  const { app, page } = await launchDesktop(env)
  const mappedWebsiteRequests = website.requests
  await mapProductionPickerToLocalBuild(app, website)

  const fixture: AcceptanceFixture = {
    app,
    page,
    mock,
    mockUrl: mock.url,
    sandbox,
    mappedWebsiteRequests,
    cleanup: async () => {
      await app.close().catch(() => undefined)
      await mock.close()
      sandbox.cleanup()
    }
  }

  await waitForAppReady(fixture, 120_000)

  return fixture
}

async function openMarketplace(page: Page): Promise<void> {
  await page.evaluate(() => { window.location.hash = '#/capabilities?tab=bots' })
  await expect(page.getByRole('region', { name: 'Native catalog' })).toBeVisible({ timeout: 60_000 })
}

async function pickResearchAnalystFromBuiltSite(fixture: AcceptanceFixture): Promise<void> {
  const { page } = fixture
  await openMarketplace(page)

  // Native backend-backed fallback must work before the optional site picker.
  await expect(page.getByRole('button', { name: 'Add Bot Research Desk' })).toBeVisible({ timeout: 60_000 })

  const picker = page.frameLocator('iframe[title="Bot Marketplace"]')
  await expect(picker.getByRole('heading', { name: 'Bot Marketplace', exact: true })).toBeVisible({ timeout: 60_000 })
  await picker.getByRole('textbox', { name: 'Search bots' }).fill('Research Desk')
  const card = picker.locator('article').filter({ hasText: 'Research Desk' }).first()
  await expect(card).toContainText('Get a sourced answer to a specific question')
  await card.getByRole('button', { name: 'Add Bot' }).click()

  const dialog = page.getByRole('dialog')
  await expect(dialog.getByRole('heading', { name: 'Research Desk' })).toBeVisible()
  await expect(dialog).toContainText('Produce accurate, dated research briefs grounded in primary and reputable sources.')
  await expect(dialog).toContainText('Source profile: default')
}

async function installReviewedBot(
  fixture: AcceptanceFixture,
  profile: string,
  testInfo: TestInfo
): Promise<string> {
  const { page, sandbox } = fixture
  await pickResearchAnalystFromBuiltSite(fixture)
  const dialog = page.getByRole('dialog')
  await dialog.getByRole('textbox', { name: 'Bot name' }).fill(profile)
  await captureEvidence(page, testInfo, `${profile}-review`)
  await dialog.getByRole('button', { name: 'Add Bot', exact: true }).click()
  await expect(dialog).toBeHidden({ timeout: 120_000 })

  const profileDir = path.join(sandbox.hermesHome, 'profiles', profile)
  await expect.poll(() => fs.existsSync(path.join(profileDir, 'profile.yaml')), { timeout: 60_000 }).toBe(true)
  expect(fs.readdirSync(path.join(sandbox.hermesHome, 'profiles')).sort()).toEqual([profile])

  return profileDir
}

function readText(file: string): string {
  return fs.readFileSync(file, 'utf8')
}

async function captureEvidence(page: Page, testInfo: TestInfo, name: string): Promise<void> {
  const screenshot = testInfo.outputPath(`${name}.png`)

  await page.screenshot({ path: screenshot, fullPage: true })
  await testInfo.attach(name, { path: screenshot, contentType: 'image/png' })
}

async function selectInstalledBot(page: Page, profile: string): Promise<void> {
  const finish = page.getByRole('button', { name: `Finish setup ${profile}` })
  await expect(finish).toBeVisible({ timeout: 60_000 })
  await finish.click()
  await expect(page.getByRole('button', { name: `Run sample task ${profile}` })).toBeVisible({ timeout: 60_000 })
}

let website: WebsiteServer | null = null

test.beforeAll(async () => {
  website = await startBuiltWebsite()
})

test.afterAll(async () => {
  await website?.close()
  website = null
})

test('native fallback + locally built site picker installs and finishes one isolated bot profile', async ({ browserName: _browserName }, testInfo) => {
  test.setTimeout(600_000)
  const fixture = await setupAcceptance(website!, 'bot-marketplace-acceptance')
  const profile = 'acceptance-researcher'

  try {
    const profileDir = await installReviewedBot(fixture, profile, testInfo)
    const profileYaml = readText(path.join(profileDir, 'profile.yaml'))
    const configYaml = readText(path.join(profileDir, 'config.yaml'))
    const blueprintYaml = readText(path.join(profileDir, 'bot-blueprint.yaml'))
    const soul = readText(path.join(profileDir, 'SOUL.md'))

    expect(profileYaml).toMatch(/catalog_name: research-analyst/)
    expect(profileYaml).toMatch(/source_profile: default/)
    expect(profileYaml).toMatch(/setup_state: needs_setup/)
    expect(profileYaml).toMatch(/state: paused/)
    expect(configYaml).toMatch(/provider: mock/)
    expect(configYaml).toMatch(/default: mock-model/)
    expect(blueprintYaml).toMatch(/schema_version: 1/)
    expect(blueprintYaml).toMatch(/name: research-analyst/)
    expect(soul).toContain('Never publish, purchase, subscribe, or contact people without explicit approval.')
    expect(fs.existsSync(path.join(profileDir, 'skills', 'bots', 'research-analyst', 'SKILL.md'))).toBe(true)

    await selectInstalledBot(fixture.page, profile)
    const activate = fixture.page.getByRole('button', { name: 'Activate Weekly topic brief' })
    await expect(activate).toBeDisabled()
    expect(fs.existsSync(path.join(profileDir, 'cron', 'jobs.json'))).toBe(false)

    // Exercise the explicit Finish setup refresh before any model turn.
    await fixture.page.getByRole('button', { name: `Finish setup ${profile}` }).last().click()
    await expect(fixture.page.getByRole('button', { name: `Run sample task ${profile}` })).toBeVisible()
    await captureEvidence(fixture.page, testInfo, `${profile}-finish-setup`)

    await fixture.page.getByRole('button', { name: `Run sample task ${profile}` }).click()
    await expect.poll(
      () => fixture.mock.receivedPrompts.filter(prompt => prompt.includes(STARTER_PROMPT)).length,
      { timeout: 120_000 }
    ).toBe(1)
    await expect(fixture.page.getByText('Synthetic demonstration complete. What real input would you like to use next?')).toBeVisible({ timeout: 180_000 })

    await openMarketplace(fixture.page)
    await expect(fixture.page.getByRole('button', { name: `Open ${profile}` })).toBeVisible({ timeout: 120_000 })
    await fixture.page.getByText(profile, { exact: true }).filter({ visible: true }).first().click()
    await expect(fixture.page.getByRole('button', { name: 'Activate Weekly topic brief' })).toBeVisible()

    const schedule = fixture.page.getByRole('textbox', { name: 'Weekly topic brief Schedule' })
    const timezone = fixture.page.getByRole('textbox', { name: 'Weekly topic brief Timezone' })
    const destination = fixture.page.getByRole('textbox', { name: 'Weekly topic brief Destination' })
    await schedule.fill('0 0 1 1 *')
    await timezone.fill('Etc/UTC')
    await destination.fill('file:isolated-acceptance-never-runs')
    await fixture.page.getByRole('button', { name: 'Activate Weekly topic brief' }).click()
    await expect(fixture.page.getByRole('button', { name: 'Pause Weekly topic brief' })).toBeVisible({ timeout: 60_000 })

    const jobsFile = path.join(profileDir, 'cron', 'jobs.json')
    await expect.poll(() => fs.existsSync(jobsFile), { timeout: 60_000 }).toBe(true)

    const jobsDocument = JSON.parse(readText(jobsFile)) as {
      jobs: Array<{
        enabled: boolean
        schedule: { expr: string; timezone?: string }
      }>
    }

    expect(jobsDocument.jobs).toHaveLength(1)
    expect(jobsDocument.jobs[0]).toMatchObject({
      enabled: true,
      schedule: { expr: '0 0 1 1 *', timezone: 'Etc/UTC' }
    })
    expect(readText(path.join(profileDir, 'profile.yaml'))).toMatch(/setup_state: ready/)
    expect(readText(path.join(profileDir, 'profile.yaml'))).toMatch(/state: active/)

    expect(fixture.mappedWebsiteRequests).toContain('/docs/bots?embed=picker')
    expect(fixture.mappedWebsiteRequests).toContain('/docs/api/bots.json')
    await captureEvidence(fixture.page, testInfo, `${profile}-ready-routine-active`)
  } finally {
    await fixture.cleanup()
  }
})

test('Run sample submits exactly once when the canonical Bot Chat already exists', async ({ browserName: _browserName }, testInfo) => {
  test.setTimeout(480_000)
  const fixture = await setupAcceptance(website!, 'bot-marketplace-existing-chat')
  const profile = 'opened-before-sample'

  try {
    await installReviewedBot(fixture, profile, testInfo)
    await selectInstalledBot(fixture.page, profile)

    // Leave Capabilities and open the installed bot from the real Bot Mode roster.
    await fixture.page.evaluate(() => { window.location.hash = '#/' })
    const botsTab = fixture.page.getByRole('button', { name: 'Bots', exact: true }).or(fixture.page.getByRole('tab', { name: 'Bots', exact: true })).first()
    await botsTab.click()
    const rosterRow = fixture.page.getByRole('button', { name: new RegExp(`Research Desk · @${profile}\\b`) }).filter({ visible: true }).first()
    await expect(rosterRow).toBeVisible({ timeout: 90_000 })
    await rosterRow.click()
    await expect(fixture.page.getByText('Say something to get started.').filter({ visible: true })).toBeVisible({ timeout: 90_000 })

    await openMarketplace(fixture.page)
    await fixture.page.getByRole('button', { name: `Finish setup ${profile}` }).click()
    const runSample = fixture.page.getByRole('button', { name: `Run sample task ${profile}` })
    await expect(runSample).toBeVisible()
    await runSample.click()
    await expect.poll(
      () => fixture.mock.receivedPrompts.filter(prompt => prompt.includes(STARTER_PROMPT)).length,
      { timeout: 120_000 }
    ).toBe(1)
    await expect(fixture.page.getByText('Synthetic demonstration complete. What real input would you like to use next?')).toBeVisible({ timeout: 180_000 })
    expect(fixture.mock.receivedPrompts.filter(prompt => prompt.includes(STARTER_PROMPT))).toHaveLength(1)
    await captureEvidence(fixture.page, testInfo, `${profile}-kickoff-success`)
  } finally {
    await fixture.cleanup()
  }
})
