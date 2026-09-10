import * as fs from 'node:fs'
import * as path from 'node:path'

import {
  buildAppEnv,
  createSandbox,
  launchDesktop,
  type Sandbox,
  waitForAppReady,
  writeEnvFile,
  writeMockProviderConfig
} from './fixtures'
import { startMockServer } from './mock-server'
import { collectErrorBanners, type ElectronApplication, expect, type Page, test } from './test'

const APPEARANCE_ROUTE = '/settings?tab=config%3Aappearance&setting=appearance.profile-picker'
const PROFILE = 'picker-e2e'

const rail = (page: Page) => page.locator('[data-slot="profile-rail"]')
const dropdown = (page: Page) => rail(page).locator('[data-slot="profile-dropdown"]')
const preference = (page: Page) => page.getByRole('switch', { name: 'Always use dropdown', exact: true })

async function navigate(page: Page, route: string): Promise<void> {
  await page.evaluate(target => {
    window.location.hash = target
  }, route)
  await expect.poll(() => page.evaluate(() => window.location.hash)).toBe(`#${route}`)
}

test('Appearance profile picker updates immediately and persists through reload and restart', async () => {
  test.setTimeout(300_000)
  const testInfo = test.info()
  const sandbox: Sandbox = createSandbox('profile-picker')
  const mock = await startMockServer()
  let app: ElectronApplication | undefined
  let page: Page | undefined
  const logs: string[] = []

  // Isolate OS-home discovery too, not only Hermes config and Chromium storage.
  const env = buildAppEnv(sandbox, {
    HOME: sandbox.root,
    USERPROFILE: sandbox.root,
    XDG_CONFIG_HOME: path.join(sandbox.root, 'config'),
    XDG_CACHE_HOME: path.join(sandbox.root, 'cache'),
    XDG_DATA_HOME: path.join(sandbox.root, 'data'),
    XDG_RUNTIME_DIR: path.join(sandbox.root, 'runtime'),
    DBUS_SESSION_BUS_ADDRESS: '',
    HERMES_PROFILE: '',
    HERMES_DESKTOP_DEV_SERVER: '',
    HERMES_DESKTOP_HERMES: '',
    PYTHONPATH: ''
  })

  fs.mkdirSync(env.XDG_RUNTIME_DIR, { recursive: true, mode: 0o700 })
  writeMockProviderConfig(sandbox.hermesHome, mock.url)
  writeEnvFile(sandbox.hermesHome)

  async function boot(): Promise<Page> {
    const launched = await launchDesktop(env)
    app = launched.app
    page = launched.page
    app.process().stdout?.on('data', (data: Buffer) => logs.push(data.toString()))
    app.process().stderr?.on('data', (data: Buffer) => logs.push(data.toString()))
    page.on('console', message => logs.push(`[renderer:${message.type()}] ${message.text()}\n`))
    page.on('pageerror', error => logs.push(`[pageerror] ${error.stack ?? error.message}\n`))
    await waitForAppReady({ app, page, sandbox, cleanup: async () => {} }, 120_000)

    return page
  }

  async function capture(name: string): Promise<void> {
    const screenshot = testInfo.outputPath(`${name}.png`)
    await page!.screenshot({ path: screenshot })
    await testInfo.attach(name, { path: screenshot, contentType: 'image/png' })
  }

  try {
    page = await boot()
    await expect(rail(page)).toBeVisible()
    await expect(dropdown(page)).toHaveCount(0)
    await expect(rail(page).getByRole('button', { name: 'default', exact: true })).toBeVisible()

    await navigate(page, APPEARANCE_ROUTE)
    await expect(preference(page)).toBeVisible()
    await expect(preference(page)).not.toBeChecked()
    await preference(page).click()
    await expect(preference(page)).toBeChecked()
    // No route change or reload between the click and the sidebar assertion.
    await expect(dropdown(page)).toBeVisible()
    await capture('01-appearance-on')
    await page.keyboard.press('Escape')
    await expect(preference(page)).toBeHidden()

    await dropdown(page).click()
    await expect(page.getByRole('menuitemradio', { name: 'default', exact: true })).toBeChecked()
    await expect(page.getByRole('menuitem', { name: 'Import profile…', exact: true })).toBeEnabled()
    await capture('02-default-and-actions')
    await page.getByRole('menuitem', { name: 'New profile', exact: true }).click()
    const create = page.getByRole('dialog', { name: 'New profile', exact: true })
    await expect(create).toBeVisible()
    await create.locator('#new-profile-name').fill(PROFILE)
    await create.getByRole('button', { name: 'Create profile', exact: true }).click()
    await expect(create).toBeHidden({ timeout: 30_000 })
    await expect(dropdown(page)).toContainText(PROFILE, { timeout: 30_000 })
    expect(fs.existsSync(path.join(sandbox.hermesHome, 'profiles', PROFILE, 'config.yaml'))).toBe(true)

    // Selecting a different profile must not reset this app-global preference.
    await dropdown(page).click()
    await page.getByRole('menuitemradio', { name: 'default', exact: true }).click()
    await expect(dropdown(page)).toContainText('default')
    await rail(page).getByRole('button', { name: 'Manage profiles…', exact: true }).click()
    await expect.poll(() => page!.evaluate(() => window.location.hash)).toBe('#/profiles')
    await expect(page.getByRole('heading', { name: 'Profiles', exact: true })).toBeVisible()
    await page.keyboard.press('Escape')

    await navigate(page, '/')
    expect(await collectErrorBanners(page)).toEqual([])
    await page.reload()
    await waitForAppReady({ app: app!, page, sandbox, cleanup: async () => {} }, 120_000)
    await expect(dropdown(page)).toBeVisible()
    await navigate(page, APPEARANCE_ROUTE)
    await expect(preference(page)).toBeChecked()
    await page.keyboard.press('Escape')
    expect(await collectErrorBanners(page)).toEqual([])

    await app!.close()
    page = await boot()
    await expect(dropdown(page)).toBeVisible()
    await navigate(page, APPEARANCE_ROUTE)
    await expect(preference(page)).toBeChecked()
    await preference(page).scrollIntoViewIfNeeded()
    await capture('03-restart-still-on')

    await preference(page).click()
    await expect(preference(page)).not.toBeChecked()
    await expect(dropdown(page)).toHaveCount(0)
    await expect(rail(page).getByRole('button', { name: PROFILE, exact: true })).toBeVisible()
    await page.keyboard.press('Escape')
    await capture('04-squares-restored')
    expect(await collectErrorBanners(page)).toEqual([])

    await app!.close()
    page = await boot()
    await expect(dropdown(page)).toHaveCount(0)
    await expect(rail(page).getByRole('button', { name: PROFILE, exact: true })).toBeVisible()
    await expect(rail(page).getByRole('button', { name: 'New profile', exact: true })).toBeEnabled()
    await expect(rail(page).getByRole('button', { name: 'Import profile…', exact: true })).toBeEnabled()
    await expect(rail(page).getByRole('button', { name: 'Manage profiles…', exact: true })).toBeEnabled()
    await navigate(page, APPEARANCE_ROUTE)
    await expect(preference(page)).not.toBeChecked()
    await preference(page).scrollIntoViewIfNeeded()
    await capture('05-restart-still-off')
    expect(await collectErrorBanners(page)).toEqual([])
  } catch (error) {
    if (page && !page.isClosed()) {
      await capture('failure')
      logs.push(`\n[DOM]\n${await page.locator('body').innerText()}\n`)
    }

    throw error
  } finally {
    await app?.close().catch(() => undefined)
    await mock.close()
    const logPath = testInfo.outputPath('runtime.log')
    fs.writeFileSync(logPath, logs.join(''), 'utf8')
    await testInfo.attach('runtime', { path: logPath, contentType: 'text/plain' })
    sandbox.cleanup()
  }
})
