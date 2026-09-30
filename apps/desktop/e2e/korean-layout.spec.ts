import { mkdirSync, writeFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import * as path from 'node:path'

import { writeMockProviderConfig } from '../../../tests-js/scripts/mock-provider-config'
import { startMockServer } from '../../../tests-js/scripts/mock-server'

import { buildAppEnv, createSandbox, launchDesktop, waitForAppReady } from './fixtures'
import { type ElectronApplication, expect, type Page, test } from './test'

const { prepareWindowForInput } = createRequire(import.meta.url)(
  '../../../tests/install/e2e-assets/window-input.cjs'
) as { prepareWindowForInput: (app: ElectronApplication, page: Page) => Promise<void> }

// eslint-disable-next-line no-empty-pattern -- Electron supplies its own page; Playwright supplies testInfo.
test('Korean Bot and Group drafts survive collapse at every zoom, and settings have accessible labels', async ({}, testInfo) => {
  const captures = testInfo.outputPath('captures')
  mkdirSync(captures, { recursive: true })

  const sandbox = createSandbox('ko-remediation'),
    mock = await startMockServer()

  const osHome = path.join(sandbox.root, 'os-home'),
    workspace = path.join(sandbox.root, '한글 작업 폴더')

  mkdirSync(osHome)
  mkdirSync(workspace)

  for (const name of ['', 'writer', 'reviewer']) {
    const home = name ? path.join(sandbox.hermesHome, 'profiles', name) : sandbox.hermesHome
    mkdirSync(home, { recursive: true })
    writeMockProviderConfig(
      home,
      mock.url,
      '  language: ko-KR\n  skin: mono',
      `terminal:\n  cwd: ${JSON.stringify(workspace)}\napprovals:\n  mode: smart`
    )
  }

  writeFileSync(path.join(sandbox.userDataDir, 'active-profile.json'), JSON.stringify({ profile: 'default' }), 'utf8')
  // Chromium needs existing Windows known-folder roots even in an isolated HOME.
  const appData = path.join(osHome, 'AppData', 'Roaming')
  const localAppData = path.join(osHome, 'AppData', 'Local')
  mkdirSync(appData, { recursive: true })
  mkdirSync(localAppData, { recursive: true })

  const env = buildAppEnv(sandbox, {
    MOCK_API_KEY: 'e2e-mock-key',
    HOME: osHome,
    APPDATA: appData,
    LOCALAPPDATA: localAppData,
    USERPROFILE: osHome,
    HERMES_DESKTOP_CWD: workspace
  })

  for (const k of Object.keys(env)) {
    if (
      (k !== 'MOCK_API_KEY' && /TOKEN|PASSWORD|SECRET|API_KEY|PAT_|CREDENTIAL|ACCESS_KEY|PRIVATE_KEY/i.test(k)) ||
      [
        'HERMES_DESKTOP_DEV_SERVER',
        'HERMES_DESKTOP_REMOTE_URL',
        'HERMES_DESKTOP_FAKE_BOOT',
        'ELECTRON_RUN_AS_NODE',
        'NODE_OPTIONS'
      ].includes(k) ||
      k.startsWith('HERMES_DESKTOP_BOOT_FAKE')
    ) {
      delete env[k]
    }
  }

  let desktop: Awaited<ReturnType<typeof launchDesktop>> | undefined
  let win: Awaited<ReturnType<ElectronApplication['browserWindow']>> | undefined
  const observations: Record<string, unknown>[] = []

  const save = () =>
    writeFileSync(path.join(captures, 'observations.json'), JSON.stringify(observations, null, 2) + '\n', 'utf8')

  const snap = async (id: string) => {
    if (!win) {
      return
    }
    const png = await win.evaluate(async w => (await w.webContents.capturePage()).toPNG().toString('base64'))
    writeFileSync(path.join(captures, id + '.png'), Buffer.from(png, 'base64'))
  }

  try {
    desktop = await launchDesktop(env)
    const { app, page } = desktop
    await waitForAppReady({ app, page, sandbox, cleanup: async () => {} }, 120000)
    await prepareWindowForInput(app, page)
    win = await app.browserWindow(page)
    await expect(page.locator('html')).toHaveAttribute('lang', 'ko', { timeout: 60000 })

    for (const kind of ['bot', 'group']) {
      for (const zoom of [1, 1.25, 1.5, 2]) {
        await win.evaluate(w => {
          w.webContents.setZoomFactor(1)
          w.setSize(1220, 800)
        })
        await page.evaluate(() => (location.hash = '#/'))
        await page.waitForTimeout(250)
        await page.locator('[data-tree-tab="hermes-bots:pane"]').click()
        await page.getByRole('button', { name: '새 봇 또는 그룹 대화', exact: true }).click()
        const menu = page.getByRole('menuitem', { name: kind === 'bot' ? '새 봇' : '새 그룹 대화', exact: true })
        await expect(menu).toBeEnabled({ timeout: 30000 })
        await menu.click()
        const dialog = page.getByRole('dialog')
        const field = dialog.getByRole('textbox', { name: kind === 'bot' ? /^설명$/ : /그룹.*이름|대화.*이름/ })
        const draft = kind === 'bot' ? '한글 작성중 👩‍💻\n둘째 줄 ' + '한글'.normalize('NFD') : '한국어 그룹 👩‍💻'
        await field.fill(draft)
        await win.evaluate((w, z) => w.webContents.setZoomFactor(z), zoom)

        for (const width of [641, 640, 639, 641]) {
          await win.evaluate(
            (w, size) => w.setContentSize(size[0], size[1]),
            [Math.round(width * zoom), Math.round(700 * zoom)]
          )
          await page.waitForTimeout(150)

          for (let n = 0; n < 4; n++) {
            const actual = await page.evaluate(() => window.innerWidth)

            if (actual === width) {
              break
            }
            await win.evaluate(
              (w, delta) => {
                const [x, y] = w.getContentSize()
                w.setContentSize(x + delta, y)
              },
              Math.round((width - actual) * zoom)
            )
            await page.waitForTimeout(150)
          }

          await expect.poll(() => page.evaluate(() => window.innerWidth)).toBe(width)
          await expect(dialog).toHaveCount(1)
          await expect(field).toHaveValue(draft)
          observations.push({ kind, zoom, width, open: true, valuePreserved: true })
          save()

          if (width === 639) {
            await snap(`${kind}-${zoom}-639`)
          }
        }

        await page.keyboard.press('Escape')
        await expect(dialog).toHaveCount(0)
      }
    }

    await win.evaluate(w => {
      w.webContents.setZoomFactor(1)
      w.setSize(1220, 800)
    })

    for (const item of [
      { tab: 'config:safety', role: 'combobox', name: '승인 모드' },
      { tab: 'config:safety', role: 'spinbutton', name: '승인 시간 초과' },
      { tab: 'config:voice&page=transcription', role: 'switch', name: '음성 인식' },
      { tab: 'config:voice&page=conversation', role: 'textbox', name: 'GPT-Live 페르소나' }
    ] as const) {
      await page.evaluate(hash => (location.hash = hash), '#/settings?tab=' + item.tab)
      const field = page.getByRole(item.role, { name: item.name, exact: true })
      await expect(field).toBeVisible({ timeout: 15000 })
      const description = await field.getAttribute('aria-describedby')
      expect(description).toBeTruthy()
      expect(await page.locator('[id=' + JSON.stringify(description) + ']').innerText()).not.toBe('')
      observations.push({ ...item, accessibleName: true, accessibleDescription: true })
      save()
    }

    await snap('settings-accessibility')
  } catch (e) {
    if (win) {
      await snap('failure').catch(() => {})
    }
    throw e
  } finally {
    save()

    if (desktop) {
      await desktop.app.close().catch(() => {})
    }
    await mock.close()
    sandbox.cleanup()
  }
})
