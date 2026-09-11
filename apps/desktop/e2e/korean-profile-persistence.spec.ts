/**
 * Dante's reported restart case: only the named profile is Korean while the
 * default profile remains English. Exercise the language picker, native IPC,
 * real Python config persistence, profile activation and a full Electron quit.
 * No inference request is sent; the local mock only supplies provider setup.
 */
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import * as path from 'node:path'

import { startMockServer } from '../../../tests-js/scripts/mock-server'

import {
  buildAppEnv,
  createSandbox,
  launchDesktop,
  type MockBackendFixture,
  waitForAppReady,
  writeMockProviderConfig,
} from './fixtures'
import { collectErrorBanners, type ElectronApplication, expect, type Page, test } from './test'

const { load } = createRequire(import.meta.url)('js-yaml') as { load: (text: string) => unknown }

const { prepareWindowForInput } = createRequire(import.meta.url)(
  '../../../tests/install/e2e-assets/window-input.cjs',
) as { prepareWindowForInput: (app: ElectronApplication, page: Page) => Promise<void> }

test.use({ actionTimeout: 30_000, navigationTimeout: 30_000 })

interface SavedConfig {
  display: { language: string; skin: string }
  terminal: { cwd: string }
}

const readConfig = (home: string) => load(readFileSync(path.join(home, 'config.yaml'), 'utf8')) as SavedConfig
const rail = (page: Page) => page.locator('[data-slot="profile-rail"]')

// Both start states matter: a rail selection must be remembered, and a named
// profile already chosen for startup must not load the default profile's UI.
for (const initialProfile of ['default', 'writer'] as const) {
  // eslint-disable-next-line no-empty-pattern -- Electron supplies the page; only Playwright test metadata is needed.
  test(`Korean stays with writer across profile changes and restart (start: ${initialProfile})`, async ({}, testInfo) => {
    test.setTimeout(360_000)
    const mock = await startMockServer()
    const sandbox = createSandbox('korean-profiles')
    const writerHome = path.join(sandbox.hermesHome, 'profiles', 'writer')
    const workspace = path.join(sandbox.root, 'workspace')
    const osHome = path.join(sandbox.root, 'os-home')
    mkdirSync(writerHome, { recursive: true })
    mkdirSync(workspace, { recursive: true })
    mkdirSync(osHome, { recursive: true })

    for (const [home, language] of [
      [sandbox.hermesHome, 'en'],
      [writerHome, initialProfile === 'writer' ? 'ko-KR' : 'en'],
    ]) {
      writeMockProviderConfig(
        home,
        mock.url,
        `  language: ${language}\n  skin: mono`,
        `terminal:\n  cwd: ${JSON.stringify(workspace)}\napprovals:\n  mode: smart`,
      )
    }

    // This is the fixture's own startup preference, never a user's profile.
    const activeProfilePath = path.join(sandbox.userDataDir, 'active-profile.json')
    writeFileSync(activeProfilePath, JSON.stringify({ profile: initialProfile }), 'utf8')

    const env = buildAppEnv(sandbox, {
      MOCK_API_KEY: 'e2e-mock-key',
      HERMES_DESKTOP_CWD: workspace,
      HOME: osHome,
      USERPROFILE: osHome,
    })

    // A caller can set HERMES_DESKTOP_PYTHON to a provisioned interpreter;
    // otherwise the app resolves the checkout's normal development venv.
    delete env.HERMES_DESKTOP_DEV_SERVER
    delete env.ELECTRON_RUN_AS_NODE

    let app: ElectronApplication | undefined
    let page!: Page
    let currentStep = 'launch'

    const step = async (name: string, run: () => Promise<void>) => {
      currentStep = name
      console.info(`[korean-persistence:${initialProfile}] ${name}`)
      await test.step(name, run)
    }

    const attachProfileState = async (name: string) => {
      await testInfo.attach(name, {
        body: JSON.stringify({
          step: currentStep,
          startup: JSON.parse(readFileSync(activeProfilePath, 'utf8')).profile,
          defaultLanguage: readConfig(sandbox.hermesHome).display.language,
          writerLanguage: readConfig(writerHome).display.language,
        }, null, 2),
        contentType: 'application/json',
      })
    }

    const launch = async () => {
      const launched = await launchDesktop(env)
      app = launched.app
      page = launched.page
      page.setDefaultTimeout(30_000)
      await waitForAppReady({ app, page } as MockBackendFixture, 120_000)
      await prepareWindowForInput(app, page)
      await expect(rail(page).getByRole('button', { name: 'writer', exact: true })).toBeVisible({ timeout: 60_000 })
    }

    const expectLanguage = async (language: 'en' | 'ko') => {
      await expect(page.locator('html')).toHaveAttribute('lang', language, { timeout: 60_000 })
      await expect(page.locator('html')).toHaveAttribute('dir', 'ltr')
    }

    const selectDefault = async () => {
      // The default home pill is translated after the Korean choice.
      await rail(page).getByRole('button', { name: /^(Switch to default|default\(으\)로 전환)$/ }).click()
      await expectLanguage('en')
    }

    const selectWriter = async () => {
      const writer = rail(page).getByRole('button', { name: 'writer', exact: true })
      await writer.click()
      await expect(writer).toHaveAttribute('aria-pressed', 'true', { timeout: 60_000 })
    }

    try {
      await step('Launch the isolated app and prepare native input', launch)

      if (initialProfile === 'default') {
        await step('Activate writer from the English default profile', async () => {
          await expectLanguage('en')
          await selectWriter()
        })
        await step('Search the language picker and save Korean', async () => {
          await page.evaluate(() => { window.location.hash = '#/settings?tab=config:appearance' })
          await page.getByRole('button', { name: 'Switch language', exact: true }).click()
          await page.getByPlaceholder('Search languages…', { exact: true }).fill('한국어')
          const korean = page.getByRole('option', { name: /한국어/ })
          await expect(korean).toBeVisible()
          // CommandList ignores a parked pointer until a real movement; the
          // locator's actionability check runs before its automatic movement.
          const bounds = await korean.boundingBox()
          expect(bounds).not.toBeNull()
          await page.mouse.move(bounds!.x + bounds!.width / 2, bounds!.y + bounds!.height / 2)
          await korean.click()
          await expectLanguage('ko')
          await expect(page.getByRole('button', { name: '언어 전환', exact: true })).toBeEnabled()
          await testInfo.attach('korean-appearance-after-save', {
            body: await page.screenshot({ timeout: 5_000 }),
            contentType: 'image/png',
          })
          await page.evaluate(() => { window.location.hash = '#/settings?tab=keybinds' })
          await expect(page.getByRole('heading', { name: '키보드 단축키', exact: true })).toBeVisible()
          await testInfo.attach('korean-keybinds', {
            body: await page.screenshot({ timeout: 5_000 }),
            contentType: 'image/png',
          })
          await page.evaluate(() => { window.location.hash = '#/skills?tab=toolsets' })
          await expect(page.getByPlaceholder('도구 세트 검색...', { exact: true })).toBeVisible()
          await testInfo.attach('korean-capabilities', {
            body: await page.screenshot({ timeout: 5_000 }),
            contentType: 'image/png',
          })
          await page.evaluate(() => { window.location.hash = '#/' })
        })
      } else {
        await step('Load the saved Korean alias from the startup writer profile', async () => {
          await expectLanguage('ko')
          await expect(rail(page).getByRole('button', { name: 'writer', exact: true })).toHaveAttribute('aria-pressed', 'true')
        })
      }

      await step('Verify profile languages and unrelated settings on disk', async () => {
        await expect.poll(() => readConfig(writerHome).display.language).toBe(initialProfile === 'writer' ? 'ko-KR' : 'ko')
        expect(readConfig(sandbox.hermesHome).display.language).toBe('en')

        for (const home of [sandbox.hermesHome, writerHome]) {
          expect(readConfig(home)).toMatchObject({ display: { skin: 'mono' }, terminal: { cwd: workspace } })
        }
      })

      // The real UI must reload each profile's preference, not merely retain
      // the language that happened to be mounted at app startup.
      await step('Switch to English default and back to Korean writer', async () => {
        await selectDefault()
        await selectWriter()
        await expectLanguage('ko')
        expect(await collectErrorBanners(page)).toEqual([])
      })
      await attachProfileState('profile-state-before-restart')

      // Quit the process, not just React or its page. Reuse both config and
      // Electron userData so the real persisted startup choice is exercised.
      await step('Quit Electron completely and restart with the same saved data', async () => {
        await expect.poll(() => JSON.parse(readFileSync(activeProfilePath, 'utf8')).profile).toBe('writer')
        await app!.close()
        app = undefined
        await launch()
      })
      await step('Verify Korean writer is restored after the full restart', async () => {
        await expect(rail(page).getByRole('button', { name: 'writer', exact: true })).toHaveAttribute('aria-pressed', 'true', { timeout: 60_000 })
        await expectLanguage('ko')
        await testInfo.attach('korean-writer-after-restart', {
          body: await page.screenshot({ timeout: 5_000 }),
          contentType: 'image/png',
        })
        expect(readConfig(sandbox.hermesHome).display.language).toBe('en')
        await selectDefault()
        await selectWriter()
        await expectLanguage('ko')
        expect(await collectErrorBanners(page)).toEqual([])
      })
    } catch (error) {
      // Inspect only the isolated UI and language/startup fields. Never attach
      // backend URLs, environment variables or complete configuration files.
      const diagnostics = await Promise.allSettled([
        attachProfileState('profile-state-at-failure'),
        ...(page && !page.isClosed() ? [
          page.screenshot({ timeout: 5_000 }).then(body => testInfo.attach('failure-window', { body, contentType: 'image/png' })),
          page.locator('body').ariaSnapshot({ timeout: 5_000 }).then(body =>
            testInfo.attach('failure-accessible-dom', { body, contentType: 'text/plain' })),
        ] : []),
      ])

      for (const result of diagnostics) {
        if (result.status === 'rejected') {
          console.info(`[korean-persistence:${initialProfile}] failure diagnostic unavailable`)
        }
      }

      throw error
    } finally {
      await app?.close().catch(() => undefined)
      await mock.close()
      sandbox.cleanup()
    }
  })
}
