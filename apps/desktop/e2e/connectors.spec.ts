import { expect, test } from './test'
import { createSandbox, buildAppEnv, launchDesktop } from './fixtures'

const GMAIL = 'Gmail'
const CALENDAR = 'Google Calendar'

test.describe('flagged connector onboarding', () => {
  test('shows the optional connector controls in a fresh flagged session', async () => {
    const sandbox = createSandbox('connectors-ui')
    const { app, page } = await launchDesktop(buildAppEnv(sandbox, {
      HERMES_GUEST_ONBOARDING: '1'
    }))

    try {
      await expect(page.getByText('Connect your apps')).toBeVisible({ timeout: 90_000 })
      await expect(page.getByText(/Connecting is optional/i)).toBeVisible()
      await expect(page.getByRole('button', { name: 'Connect', exact: true }).first()).toBeVisible()
      await expect(page.getByRole('button', { name: 'Not now', exact: true }).first()).toBeVisible()
      await expect(page.getByText(GMAIL)).toBeVisible()
      await expect(page.getByText(CALENDAR)).toBeVisible()
    } finally {
      await app.close().catch(() => undefined)
      sandbox.cleanup()
    }
  })

  test('keeps the default surface free of connector authorization controls', async () => {
    const sandbox = createSandbox('connectors-off')
    const { app, page } = await launchDesktop(buildAppEnv(sandbox))

    try {
      await expect(page.locator('[data-connector-offer]')).toHaveCount(0)
    } finally {
      await app.close().catch(() => undefined)
      sandbox.cleanup()
    }
  })
})
