/**
 * A manual model choice belongs to the profile where it was made. Switching
 * profiles must restore the target pair before a new session is created, or
 * the target backend receives a model owned by a different provider.
 */

import * as fs from 'node:fs'
import * as path from 'node:path'

import {
  buildAppEnv,
  createSandbox,
  launchDesktop,
  type MockBackendFixture,
  waitForAppReady
} from './fixtures'
import { startMockServer } from './mock-server'
import { expect, test } from './test'

const PROMPT = 'Keep this request on the Xiaomi profile.'
const TARGET_PROFILE = 'xiaomi'

function writeProviderProfile(
  home: string,
  provider: 'deepseek' | 'xiaomi',
  model: string,
  mockUrl: string
): void {
  fs.mkdirSync(home, { recursive: true })
  fs.writeFileSync(
    path.join(home, 'config.yaml'),
    `model:
  default: ${model}
  provider: ${provider}
  base_url: ${mockUrl}/v1
  api_mode: chat_completions
auxiliary:
  title_generation:
    enabled: false
approvals:
  mode: "off"
`,
    'utf8'
  )
  fs.writeFileSync(
    path.join(home, '.env'),
    `${provider === 'deepseek' ? 'DEEPSEEK_API_KEY' : 'XIAOMI_API_KEY'}=e2e-mock-key\n`,
    'utf8'
  )
}

test('creates a target-profile session with that profile model and provider', async ({}, testInfo) => {
  test.setTimeout(420_000)

  const deepseek = await startMockServer()
  const xiaomi = await startMockServer()
  const sandbox = createSandbox('provider-model-profile')

  writeProviderProfile(sandbox.hermesHome, 'deepseek', 'deepseek-v4-pro', deepseek.url)
  writeProviderProfile(
    path.join(sandbox.hermesHome, 'profiles', TARGET_PROFILE),
    'xiaomi',
    'mimo-v2.5-pro',
    xiaomi.url
  )

  const { app, page } = await launchDesktop(buildAppEnv(sandbox))
  const consoleErrors: string[] = []
  const pageErrors: string[] = []
  const failedRequests: string[] = []
  const sessionCreates: Array<Record<string, unknown>> = []

  page.on('console', message => {
    if (message.type() === 'error') {
      consoleErrors.push(message.text())
    }
  })
  page.on('pageerror', error => pageErrors.push(error.message))
  page.on('websocket', socket => {
    socket.on('framesent', event => {
      try {
        const message = JSON.parse(String(event.payload)) as {
          method?: string
          params?: Record<string, unknown>
        }

        if (message.method === 'session.create' && message.params) {
          sessionCreates.push(message.params)
        }
      } catch {
        // Binary and non-JSON WebSocket frames are unrelated to gateway RPC.
      }
    })
  })

  try {
    await page.waitForLoadState('domcontentloaded')
    await page.evaluate(
      selections => {
        for (const [profile, selection] of Object.entries(selections)) {
          const value = JSON.stringify(selection)
          const encodedProfile = encodeURIComponent(profile)

          window.localStorage.setItem(`hermes.desktop.composer.selection.v1.profile.${encodedProfile}`, value)
          window.localStorage.setItem(`hermes.desktop.composer.selection.v1.registry.local.${encodedProfile}`, value)
        }
      },
      {
        default: { model: 'deepseek-v4-pro', provider: 'deepseek', source: 'manual' },
        [TARGET_PROFILE]: { model: 'mimo-v2.5-pro', provider: 'xiaomi', source: 'manual' }
      }
    )
    await page.reload()
    await waitForAppReady({ app, page } as MockBackendFixture, 120_000)
    consoleErrors.length = 0
    pageErrors.length = 0
    page.on('requestfailed', request => {
      failedRequests.push(`${request.method()} ${request.url()}: ${request.failure()?.errorText ?? 'unknown error'}`)
    })

    const rail = page.locator('[data-slot="profile-rail"]')
    const sourceModel = page.getByRole('button', {
      name: /Model · deepseek: deepseek-v4-pro/
    })

    await expect(sourceModel).toBeVisible()
    expect(
      await page.evaluate(() =>
        window.localStorage.getItem('hermes.desktop.composer.selection.v1.profile.default')
      )
    ).toContain('"source":"manual"')

    const target = rail.getByRole('button', { name: TARGET_PROFILE, exact: true })
    await target.click()
    await expect(target).toHaveAttribute('aria-pressed', 'true', { timeout: 120_000 })
    await expect(page.getByRole('button', { name: /Model · xiaomi: mimo-v2.5-pro/ })).toBeVisible({
      timeout: 120_000
    })

    const composer = page.locator('[data-slot="composer-root"] [contenteditable="true"]').filter({ visible: true }).first()
    await composer.click()
    await composer.fill(PROMPT)
    await page.keyboard.press('Enter')

    await expect.poll(() => sessionCreates.length, { timeout: 60_000 }).toBe(1)
    expect(sessionCreates[0]).toMatchObject({
      model: 'mimo-v2.5-pro',
      profile: TARGET_PROFILE,
      provider: 'xiaomi',
      source: 'desktop'
    })
    expect(JSON.stringify(sessionCreates[0])).not.toContain('deepseek-v4-pro')
    expect(JSON.stringify(sessionCreates[0])).not.toContain('"provider":"deepseek"')

    await expect.poll(() => xiaomi.receivedPrompts.includes(PROMPT), { timeout: 180_000 }).toBe(true)
    await expect.poll(() => xiaomi.receivedModels.includes('mimo-v2.5-pro'), { timeout: 180_000 }).toBe(true)
    expect(deepseek.receivedPrompts).not.toContain(PROMPT)
    expect(deepseek.receivedModels).not.toContain('mimo-v2.5-pro')

    await expect(page.locator('[data-slot="aui_thread-viewport"]')).toContainText(
      'Hello from the mock inference server!',
      { timeout: 60_000 }
    )
    await expect(page.locator('[role="alert"]')).toHaveCount(0)
    expect(consoleErrors).toEqual([])
    expect(pageErrors).toEqual([])
    expect(failedRequests).toEqual([])

    await page.screenshot({
      path: testInfo.outputPath('xiaomi-profile-model-provider-isolated.png')
    })
  } finally {
    await app.close().catch(() => undefined)
    await Promise.all([deepseek.close(), xiaomi.close()])
    sandbox.cleanup()
  }
})
