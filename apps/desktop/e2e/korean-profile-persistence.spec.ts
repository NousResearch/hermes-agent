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
  writeMockProviderConfig
} from './fixtures'
import { collectErrorBanners, type ElectronApplication, expect, type Page, test } from './test'

const { load } = createRequire(import.meta.url)('js-yaml') as { load: (text: string) => unknown }

const { prepareWindowForInput } = createRequire(import.meta.url)(
  '../../../tests/install/e2e-assets/window-input.cjs'
) as { prepareWindowForInput: (app: ElectronApplication, page: Page) => Promise<void> }

test.use({ actionTimeout: 30_000, navigationTimeout: 30_000 })

interface SavedConfig {
  display: { language: string; skin: string }
  terminal: { cwd: string }
}

interface ProfileBridgeWindow {
  hermesDesktop: {
    getConnectionConfig: (profile: string) => Promise<unknown>
    getConnection: () => Promise<{ profile?: string; mode?: string; registryScoped?: boolean }>
    profile: { get: () => Promise<{ profile: string | null }> }
  }
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
      [writerHome, initialProfile === 'writer' ? 'ko-KR' : 'en']
    ]) {
      writeMockProviderConfig(
        home,
        mock.url,
        `  language: ${language}\n  skin: mono`,
        `terminal:\n  cwd: ${JSON.stringify(workspace)}\napprovals:\n  mode: smart`
      )
    }

    // This is the fixture's own startup preference, never a user's profile.
    const activeProfilePath = path.join(sandbox.userDataDir, 'active-profile.json')
    writeFileSync(activeProfilePath, JSON.stringify({ profile: initialProfile }), 'utf8')

    const env = buildAppEnv(sandbox, {
      MOCK_API_KEY: 'e2e-mock-key',
      HERMES_DESKTOP_CWD: workspace,
      HOME: osHome,
      USERPROFILE: osHome
    })

    // A caller can set HERMES_DESKTOP_PYTHON to a provisioned interpreter;
    // otherwise the app resolves the checkout's normal development venv.
    for (const key of Object.keys(env)) {
      if (
        [
          'HERMES_DESKTOP_DEV_SERVER',
          'HERMES_DESKTOP_REMOTE_URL',
          'HERMES_DESKTOP_FAKE_BOOT',
          'ELECTRON_RUN_AS_NODE',
          'NODE_OPTIONS'
        ].includes(key) ||
        key.startsWith('HERMES_DESKTOP_BOOT_FAKE')
      ) {
        delete env[key]
      }
    }

    let app: ElectronApplication | undefined
    let page!: Page
    let currentStep = 'launch'

    const step = async (name: string, run: () => Promise<void>) => {
      currentStep = name
      console.info(`[korean-persistence:${initialProfile}] ${name}`)
      await test.step(name, run)
    }

    const attachProfileState = async (name: string) => {
      const visibleState =
        page && !page.isClosed()
          ? await page
              .evaluate(async () => {
                const bridge = (window as unknown as ProfileBridgeWindow).hermesDesktop

                const native = await Promise.race([
                  Promise.all([bridge.profile.get(), bridge.getConnection()])
                    .then(([saved, connection]) => ({
                      savedProfile: saved.profile,
                      primaryProfile: connection.profile ?? null,
                      mode: connection.mode ?? null,
                      registryScoped: connection.registryScoped ?? false
                    }))
                    .catch(() => ({ unavailable: true })),
                  new Promise<{ unavailable: boolean }>(resolve =>
                    setTimeout(() => resolve({ unavailable: true }), 5000)
                  )
                ])

                return {
                  language: document.documentElement.lang,
                  selectedProfiles: Array.from(
                    document.querySelectorAll('[data-slot="profile-rail"] [aria-pressed="true"]')
                  ).map(element => element.getAttribute('aria-label')),
                  native
                }
              })
              .catch(() => ({ unavailable: true }))
          : null

      await testInfo.attach(name, {
        body: JSON.stringify(
          {
            step: currentStep,
            startup: JSON.parse(readFileSync(activeProfilePath, 'utf8')).profile,
            defaultLanguage: readConfig(sandbox.hermesHome).display.language,
            writerLanguage: readConfig(writerHome).display.language,
            visibleState
          },
          null,
          2
        ),
        contentType: 'application/json'
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
      await attachProfileState('profile-state-after-launch')
    }

    const expectLanguage = async (language: 'en' | 'ko') => {
      await expect(page.locator('html')).toHaveAttribute('lang', language, { timeout: 60_000 })
      await expect(page.locator('html')).toHaveAttribute('dir', 'ltr')
    }

    const expectPanelTitles = async (language: 'en' | 'ko') => {
      // Pane identities remain stable; only their visible tab titles change.
      await expect(page.locator('[data-tree-tab="sessions"]')).toHaveText(language === 'ko' ? '세션' : 'Sessions')
      await expect(page.locator('[data-tree-tab="hermes-bots:pane"]')).toHaveText(language === 'ko' ? '봇' : 'Bots')
    }

    const captureSurface = async (name: string) => {
      await testInfo.attach(name, {
        body: await page.screenshot({ timeout: 5_000 }),
        contentType: 'image/png'
      })
    }

    const expectKoreanComposer = async () => {
      const input = page.locator('[contenteditable="true"][data-placeholder]')

      // Check actual resting copy; translated startup/reconnect hints do not count.
      const starters = [
        '무엇을 만드시나요?',
        'Hermes에게 작업을 지시하세요',
        '무엇을 생각하고 계신가요?',
        '필요한 것을 설명하세요',
        '무엇을 처리할까요?',
        '무엇이든 물어보세요',
        '목표부터 시작하세요'
      ]

      await expect.poll(async () => starters.includes((await input.getAttribute('data-placeholder')) ?? '')).toBe(true)
    }

    const selectDefault = async () => {
      // The default home pill is translated after the Korean choice.
      await rail(page)
        .getByRole('button', { name: /^(Switch to default|default\(으\)로 전환)$/ })
        .click()
      await expectLanguage('en')
    }

    const selectWriter = async () => {
      const writer = rail(page).getByRole('button', { name: 'writer', exact: true })
      await writer.click()
      await expect(writer).toHaveAttribute('aria-pressed', 'true', { timeout: 60_000 })
    }

    const chooseLanguage = async (language: 'en' | 'ko') => {
      await page.evaluate(() => {
        window.location.hash = '#/settings?tab=config:appearance'
      })
      await page.getByRole('button', { name: /^(Switch language|언어 전환)$/ }).click()
      const label = language === 'ko' ? '한국어' : 'English'
      await page.getByRole('dialog').getByRole('combobox').fill(label)
      const option = page.getByRole('option', { name: new RegExp(label) })
      await expect(option).toBeVisible()
      const bounds = await option.boundingBox()
      expect(bounds).not.toBeNull()
      await page.mouse.move(bounds!.x + bounds!.width / 2, bounds!.y + bounds!.height / 2)
      await option.click()
      await expectLanguage(language)
      await expect.poll(() => readConfig(writerHome).display.language).toBe(language)
    }

    try {
      await step('Launch the isolated app and prepare native input', launch)
      await step('Observe concurrent profile configuration reads through the actual native bridge', async () => {
        const completionOrder = await page.evaluate(async () => {
          const completed: number[] = []
          await Promise.all(
            Array.from({ length: 12 }, async (_, index) => {
              await (window as unknown as ProfileBridgeWindow).hermesDesktop.getConnectionConfig(
                index % 2 === 0 ? 'writer' : 'default'
              )
              completed.push(index)
            })
          )

          return completed
        })

        await testInfo.attach('native-config-completion-order', {
          body: JSON.stringify(completionOrder),
          contentType: 'application/json'
        })
        // Completion order is evidence for the separate race investigation,
        // not a contract required of independent configuration reads.
        expect([...completionOrder].sort((a, b) => a - b)).toEqual(Array.from({ length: 12 }, (_, index) => index))
      })

      if (initialProfile === 'default') {
        await step('Activate writer from the English default profile', async () => {
          await expectLanguage('en')
          await selectWriter()
        })
        await step('Search the language picker and save Korean', async () => {
          await page.evaluate(() => {
            window.location.hash = '#/settings?tab=config:appearance'
          })
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
          await captureSurface('korean-appearance-after-save')
        })
        await step('Read Korean built-in theme descriptions without searching the Marketplace', async () => {
          // Reading installed cards must not trigger the external theme search
          // or change the profile's saved Mono skin.
          for (const [label, description] of [
            ['Nous', 'GitHub 스타일에 Nous 파란색 강조'],
            ['GitHub', 'GitHub 기본 밝은 테마와 어두운 테마'],
            ['Catppuccin', '차분한 파스텔 색상 — Latte와 Mocha']
          ]) {
            const card = page.getByRole('button').filter({ has: page.getByText(label, { exact: true }) })
            await card.scrollIntoViewIfNeeded()
            await expect(card.getByText(description, { exact: true })).toBeVisible()
            await captureSurface(`korean-theme-${label.toLowerCase()}`)
          }
        })
        await step('Verify Korean browser settings and the restored ConfigField explanation', async () => {
          await page.evaluate(() => {
            window.location.hash = '#/settings?tab=config:browser'
          })
          const browserNav = page.locator('[data-tour="nav-config:browser"]')
          await expect(browserNav).toHaveText('브라우저')
          await expect(browserNav).toBeVisible()
          await expect(page.locator('[data-tour="field-browser.use_real_profile"]')).toBeVisible()
          await captureSurface('korean-browser-settings')

          await page.evaluate(() => {
            window.location.hash = '#/settings?tab=config:safety'
          })
          const approvalTimeout = page.locator('[data-tour="field-approvals.timeout"]')
          await approvalTimeout.scrollIntoViewIfNeeded()
          await expect(approvalTimeout.getByText('승인 시간 초과', { exact: true })).toBeVisible()
          // The old ASCII-only duplicate filter dropped this distinct Korean
          // explanation. This assertion exercises the real schema + renderer.
          await expect(
            approvalTimeout.getByText('승인 프롬프트가 시간 초과되기까지의 대기 시간입니다.', { exact: true })
          ).toBeVisible()
          await captureSurface('korean-config-field-description')
        })
        await step('Read Korean keyboard and terminal toolset copy without running tools', async () => {
          await page.evaluate(() => {
            window.location.hash = '#/settings?tab=keybinds'
          })
          await expect(page.getByRole('heading', { name: '키보드 단축키', exact: true })).toBeVisible()
          await captureSurface('korean-keybinds')
          await page.evaluate(() => {
            window.location.hash = '#/skills?tab=toolsets'
          })
          const toolsetSearch = page.getByRole('textbox', { name: '도구 세트 검색...', exact: true })
          await expect(toolsetSearch).toBeVisible()
          await toolsetSearch.fill('터미널')
          await toolsetSearch.fill('터미널'.normalize('NFD'))

          const terminalRow = page
            .getByRole('button')
            .filter({ has: page.getByText('터미널 및 프로세스', { exact: true }) })

          await expect(terminalRow).toBeVisible({ timeout: 60_000 })
          await expect(terminalRow.getByText('도구 2개', { exact: true })).toBeVisible()
          await expect(terminalRow.getByText('터미널, 프로세스', { exact: true })).toBeVisible()
          await terminalRow.click()
          await expect(page.getByRole('heading', { name: '터미널 및 프로세스', exact: true })).toBeVisible()
          await expect(page.getByText('실행 백엔드', { exact: true })).toBeVisible({ timeout: 60_000 })
          await captureSurface('korean-capabilities')
          await page.evaluate(() => {
            window.location.hash = '#/'
          })
        })
        await step('Keep an explicit English choice on disk, then restore Korean', async () => {
          await chooseLanguage('en')
          await chooseLanguage('ko')
          await page.evaluate(() => {
            window.location.hash = '#/'
          })
          await expectPanelTitles('ko')
          await captureSurface('korean-session-and-bot-panel-titles')
        })
        await step('Read the Korean new-bot dialog and cancel without creating a profile', async () => {
          await page.locator('[data-tree-tab="hermes-bots:pane"]').click()
          await page.getByRole('button', { name: '새 봇 또는 그룹 대화', exact: true }).click()
          await page.getByRole('menuitem', { name: '새 봇', exact: true }).click()

          const dialog = page.getByRole('dialog', { name: '새 봇', exact: true })

          await expect(dialog.getByRole('heading', { name: '새 봇', exact: true })).toBeVisible()
          await expect(
            dialog.getByText(
              '자신만의 메모리, 스킬, 대화를 갖춘 동료입니다. 다른 에이전트와 메시지를 주고받을 수 있습니다.',
              { exact: true }
            )
          ).toBeVisible()
          await expect(dialog.getByPlaceholder('이 봇이 어떤 일을 도와주면 좋을까요?', { exact: true })).toBeVisible()

          for (const label of ['이름', '표시 제목', '설명']) {
            const field = dialog.getByRole('textbox', { name: label, exact: true })
            await expect(field).toBeVisible()
            await dialog
              .locator('label')
              .filter({ hasText: new RegExp(`^${label}$`) })
              .click()
            await expect(field).toBeFocused()
          }

          await expect(dialog.getByRole('button', { name: '얼굴 고정', exact: true })).toBeVisible()
          await expect(dialog.getByText('이름에 따라 얼굴이 바뀝니다.', { exact: true })).toBeVisible()
          await expect(dialog.getByText('자동', { exact: true })).toBeVisible()
          await captureSurface('korean-new-bot-dialog')
          // Advanced/General reads the capability catalog. Leave the name empty
          // and avoid the Skills tab, which can materialize a draft profile.
          await dialog.getByRole('button', { name: '고급', exact: true }).click()
          await expect(dialog.getByText('공급자', { exact: true })).toBeVisible()
          await expect(dialog.getByText('모델', { exact: true })).toBeVisible()
          await dialog.getByText('공급자', { exact: true }).scrollIntoViewIfNeeded()
          await captureSurface('korean-new-bot-model-options')
          const nativeWindow = await app!.browserWindow(page)
          const originalBounds = await nativeWindow.evaluate(window => window.getBounds())
          const originalZoom = await nativeWindow.evaluate(window => window.webContents.getZoomFactor())

          try {
            await nativeWindow.evaluate(window => {
              window.setSize(900, 700)
              window.webContents.setZoomFactor(1.25)
            })
            await expect.poll(() => page.evaluate(() => window.innerWidth)).toBe(720)
            await dialog.getByRole('textbox', { name: '설명', exact: true }).scrollIntoViewIfNeeded()
            const bounds = await dialog.boundingBox()
            const viewportWidth = await page.evaluate(() => window.innerWidth)

            expect(bounds).not.toBeNull()
            expect(bounds!.x).toBeGreaterThanOrEqual(0)
            expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(viewportWidth + 1)
            await testInfo.attach('narrow-window-metrics', {
              body: JSON.stringify({
                dialog: bounds,
                viewport: await page.evaluate(() => ({ width: window.innerWidth, height: window.innerHeight })),
                native: await nativeWindow.evaluate(window => ({
                  bounds: window.getBounds(),
                  zoom: window.webContents.getZoomFactor()
                }))
              }),
              contentType: 'application/json'
            })

            // Capture the native backing surface: Playwright's page screenshot
            // can crop the backing pixels at a non-default Electron zoom.
            const nativePng = await nativeWindow.evaluate(async window =>
              (await window.webContents.capturePage()).toPNG().toString('base64')
            )

            await testInfo.attach('korean-new-bot-narrow-zoom', {
              body: Buffer.from(nativePng, 'base64'),
              contentType: 'image/png'
            })
            await dialog.getByRole('button', { name: '취소', exact: true }).scrollIntoViewIfNeeded()
          } finally {
            await nativeWindow.evaluate(
              (window, original) => {
                window.webContents.setZoomFactor(original.zoom)
                window.setBounds(original.bounds)
              },
              { zoom: originalZoom, bounds: originalBounds }
            )
          }

          await dialog.getByRole('button', { name: '취소', exact: true }).click()
          await expect(dialog).not.toBeVisible()
          await page.locator('[data-tree-tab="sessions"]').click()
        })
      } else {
        await step('Load the saved Korean alias from the startup writer profile', async () => {
          await expectLanguage('ko')
          await expect(rail(page).getByRole('button', { name: 'writer', exact: true })).toHaveAttribute(
            'aria-pressed',
            'true'
          )
        })
      }

      await step('Verify profile languages and unrelated settings on disk', async () => {
        await expect
          .poll(() => readConfig(writerHome).display.language)
          .toBe(initialProfile === 'writer' ? 'ko-KR' : 'ko')
        expect(readConfig(sandbox.hermesHome).display.language).toBe('en')

        for (const home of [sandbox.hermesHome, writerHome]) {
          expect(readConfig(home)).toMatchObject({ display: { skin: 'mono' }, terminal: { cwd: workspace } })
        }
      })

      // The real UI must reload each profile's preference, not merely retain
      // the language that happened to be mounted at app startup.
      await step('Switch to English default and back to Korean writer', async () => {
        await selectDefault()
        await expectPanelTitles('en')
        await selectWriter()
        await expectLanguage('ko')
        await expectPanelTitles('ko')
        expect(await collectErrorBanners(page)).toEqual([])
        await expectKoreanComposer()
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
        await expect(rail(page).getByRole('button', { name: 'writer', exact: true })).toHaveAttribute(
          'aria-pressed',
          'true',
          { timeout: 60_000 }
        )
        await expectLanguage('ko')
        await expectKoreanComposer()
        await testInfo.attach('korean-writer-after-restart', {
          body: await page.screenshot({ timeout: 5_000 }),
          contentType: 'image/png'
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
        ...(page && !page.isClosed()
          ? [
              page
                .screenshot({ timeout: 5_000 })
                .then(body => testInfo.attach('failure-window', { body, contentType: 'image/png' })),
              page
                .locator('body')
                .ariaSnapshot({ timeout: 5_000 })
                .then(body => testInfo.attach('failure-accessible-dom', { body, contentType: 'text/plain' }))
            ]
          : [])
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
