import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

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

// The operator settings lock, end to end against a REAL gateway: the pill reads
// `config.lock.status`, the password goes to `config.unlock`, and the gateway verifies it against
// the scrypt hash in the root config.yaml. The proof is on disk — the unlock window is a file the
// gateway writes, so the spec reads it back rather than trusting the UI's own word.

type Page = MockBackendFixture['page']

const PASSWORD = 'e2e-unlock-password'
const LOCKED_KEYS = ['approvals.mode', 'yolo'] as const

let fixture: MockBackendFixture | null = null

/** Same format `hermes_cli.settings_lock.hash_password` writes: scrypt$n$r$p$salt$hash. */
function scryptHash(password: string): string {
  const salt = crypto.randomBytes(16)
  const digest = crypto.scryptSync(password, salt, 32, { N: 16384, maxmem: 64 * 1024 * 1024, p: 1, r: 8 })

  return `scrypt$16384$8$1$${salt.toString('base64')}$${digest.toString('base64')}`
}

// SETTINGS_LOCK_SCREENSHOT_DIR=<dir> saves a full-window capture at each state.
async function capture(page: Page, name: string): Promise<void> {
  const dir = process.env.SETTINGS_LOCK_SCREENSHOT_DIR

  if (!dir) {
    return
  }

  fs.mkdirSync(dir, { recursive: true })
  await page.screenshot({ path: path.join(dir, `${name}.png`) })
}

const statusbar = (page: Page) => page.locator('[data-slot="statusbar"]')
const lockPill = (page: Page) => statusbar(page).getByRole('button', { name: /Locked|Unlocked/ })

function unlockFile(hermesHome: string): string {
  return path.join(hermesHome, '.settings-unlock')
}

test.beforeAll(async () => {
  const mock = await startMockServer()
  const sandbox = createSandbox('settings-lock')

  writeMockProviderConfig(sandbox.hermesHome, mock.url)
  writeEnvFile(sandbox.hermesHome)

  // The lock lives in the ROOT config, appended after the provider config is written (that call
  // owns the file). A password hash, so the unlock path is exercised for real.
  const configPath = path.join(sandbox.hermesHome, 'config.yaml')
  fs.appendFileSync(
    configPath,
    ['settings_lock:', '  enabled: true', '  keys:', ...LOCKED_KEYS.map(key => `    - ${key}`),
     `  password: '${scryptHash(PASSWORD)}'`, ''].join('\n'),
    'utf8'
  )

  const { app, page } = await launchDesktop(buildAppEnv(sandbox))

  fixture = {
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
  await waitForAppReady(fixture, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('the lock pill reports the lock, takes the password, and the gateway opens the window', async () => {
  test.setTimeout(300_000)
  const page = fixture!.page
  const { hermesHome } = fixture!.sandbox

  // 1. Locked: the pill says so, and no unlock window exists on disk.
  await expect(lockPill(page)).toBeVisible({ timeout: 60_000 })
  await expect(lockPill(page)).toContainText('Locked')
  expect(fs.existsSync(unlockFile(hermesHome))).toBe(false)
  await capture(page, '1-locked')

  // 2. The menu names exactly what is locked — including the implicit self-protection, so the
  //    lock cannot be switched off through the very door it guards.
  await lockPill(page).click()
  const menu = page.getByRole('menu').or(page.locator('[data-radix-menu-content]')).first()
  await expect(menu).toBeVisible()

  for (const key of [...LOCKED_KEYS, 'settings_lock.*']) {
    await expect(menu.getByText(key, { exact: true })).toBeVisible()
  }

  const password = menu.getByLabel('Password')
  await expect(password).toBeVisible()
  await capture(page, '2-menu-open')

  // 3. A wrong password is refused by the GATEWAY, not by the UI, and writes no window.
  await password.fill('not-the-password')
  await menu.getByRole('button', { name: /Unlock/ }).click()
  await expect(menu.getByText('Incorrect password.')).toBeVisible({ timeout: 30_000 })
  expect(fs.existsSync(unlockFile(hermesHome))).toBe(false)
  await capture(page, '3-wrong-password')

  // 4. The real password opens a time-boxed window: the gateway writes the file, and the pill
  //    switches to counting it down so "unlocked" is not a state you forget you left on.
  await password.fill(PASSWORD)
  await menu.getByRole('button', { name: /Unlock/ }).click()
  await expect
    .poll(() => fs.existsSync(unlockFile(hermesHome)), { timeout: 30_000 })
    .toBe(true)
  // The open menu marks the rest of the page aria-hidden, so the pill is unqueryable until it
  // closes — close it before reading the pill back.
  await page.keyboard.press('Escape')
  await expect(lockPill(page)).toContainText('Unlocked', { timeout: 30_000 })
  await capture(page, '4-unlocked')

  // 5. Locking again closes the window immediately.
  await lockPill(page).click()
  await page
    .getByRole('menu')
    .or(page.locator('[data-radix-menu-content]'))
    .first()
    .getByRole('button', { name: 'Lock now' })
    .click()
  await page.keyboard.press('Escape')
  await expect(lockPill(page)).toContainText('Locked', { timeout: 30_000 })
  await expect.poll(() => fs.existsSync(unlockFile(hermesHome)), { timeout: 30_000 }).toBe(false)
  await capture(page, '5-relocked')
})
