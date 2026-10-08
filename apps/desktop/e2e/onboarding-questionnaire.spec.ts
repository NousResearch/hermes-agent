/**
 * The static first-run questionnaire on a fresh install with the free-tier flag on.
 *
 * The home has a working mock provider but no `onboarding.run` key, so it counts as a fresh install
 * and the questionnaire opens as the first state of the first-run overlay. Start waits for inference
 * (the mock provider answers), writes `onboarding.run: false` to the root config, and hands one prompt
 * to a new chat: the ask on line one, then "About me". Skip setup writes the same flag and leaves.
 *
 * Prerequisite: `npm run build` must have been run so dist/ exists.
 */

import * as fs from 'node:fs'
import * as path from 'node:path'

import { writeEnvFile, writeMockProviderConfig } from '../../../tests-js/scripts/mock-provider-config'
import { startMockServer } from '../../../tests-js/scripts/mock-server'

import { buildAppEnv, createSandbox, launchDesktop, type Sandbox } from './fixtures'
import { type ElectronApplication, expect, type Page, test } from './test'

interface Running {
  app: ElectronApplication
  page: Page
  sandbox: Sandbox
  close: () => Promise<void>
}

let running: Running | null = null

test.afterEach(async () => {
  await running?.close()
  running = null
})

async function launchFresh(): Promise<Running> {
  const mock = await startMockServer()
  const sandbox = createSandbox('questionnaire')

  writeMockProviderConfig(sandbox.hermesHome, mock.url)
  writeEnvFile(sandbox.hermesHome)

  const { app, page } = await launchDesktop(buildAppEnv(sandbox, { HERMES_GUEST_ONBOARDING: '1' }))

  return {
    app,
    page,
    sandbox,
    close: async () => {
      await app.close().catch(() => undefined)
      await mock.close()
      sandbox.cleanup()
    }
  }
}

const rootConfig = (sandbox: Sandbox) => fs.readFileSync(path.join(sandbox.hermesHome, 'config.yaml'), 'utf8')

test.describe('first-run questionnaire', () => {
  test('Start hands one prompt to a new chat and marks setup done', async () => {
    running = await launchFresh()
    const { page, sandbox } = running

    await expect(page.getByText("Hi, I'm Hermes.")).toBeVisible({ timeout: 90_000 })
    // The status bar stays visible under the overlay.
    await expect(page.locator('[data-slot="statusbar"]')).toBeVisible()

    await page.getByLabel('Other answer').fill('Ada')
    await page.getByRole('button', { name: /Confirm and continue/ }).click()

    // Skip every remaining question until the review screen.
    while (!(await page.getByText('Ready, Ada.').isVisible())) {
      await page.getByRole('button', { exact: true, name: 'Skip' }).click()
    }

    await page.getByRole('button', { name: 'Start' }).click()

    await expect(page.getByText("Hi, I'm Hermes.")).toBeHidden({ timeout: 60_000 })
    await expect(page.getByText(/What can you help me with\?/).first()).toBeVisible({ timeout: 30_000 })
    expect(rootConfig(sandbox)).toMatch(/onboarding:\s*\n(?:.*\n)*?\s+run: false/)
  })

  test('Skip setup leaves the questionnaire for good', async () => {
    running = await launchFresh()
    const { page, sandbox } = running

    await expect(page.getByText("Hi, I'm Hermes.")).toBeVisible({ timeout: 90_000 })
    await page.getByRole('button', { name: 'Skip setup' }).click()

    await expect(page.getByText("Hi, I'm Hermes.")).toBeHidden({ timeout: 30_000 })
    expect(rootConfig(sandbox)).toMatch(/onboarding:\s*\n(?:.*\n)*?\s+run: false/)
  })
})
