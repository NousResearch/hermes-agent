import * as fs from 'node:fs'
import * as path from 'node:path'

import { type NoProviderFixture, setupNoProvider } from './fixtures'
import { expect, test } from './test'

let fixture: NoProviderFixture | null = null

test.beforeAll(async () => {
  fixture = await setupNoProvider()
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('native credential window saves outside the chat renderer', async () => {
  const app = fixture!.app
  const page = fixture!.page
  const credentialWindow = app.waitForEvent('window')

  const result = page.evaluate(() =>
    (window as unknown as {
      hermesDesktop: {
        secureCredential: {
          capture: (request: {
            envVar: string
            locale: string
            profile: string
            prompt: string
            requestId: string
          }) => Promise<{ status: 'busy' | 'cancelled' | 'saved' }>
        }
      }
    }).hermesDesktop.secureCredential.capture({
      envVar: 'HERMES_SECURE_CREDENTIAL_CANARY',
      locale: 'en',
      profile: 'default',
      prompt: 'Harmless end-to-end credential canary',
      requestId: 'e2e-credential-canary'
    })
  )

  const credentialPage = await credentialWindow

  await credentialPage.getByLabel('HERMES_SECURE_CREDENTIAL_CANARY').fill('  harmless-canary  ')
  await credentialPage.getByRole('button', { name: 'Save securely' }).click()

  await expect(result).resolves.toEqual({ status: 'saved' })

  const envText = fs.readFileSync(path.join(fixture!.sandbox.hermesHome, '.env'), 'utf8')
  const canaryLine = envText
    .split(/\r?\n/)
    .find(line => line.startsWith('HERMES_SECURE_CREDENTIAL_CANARY='))

  expect(canaryLine).toContain('  harmless-canary  ')
})
