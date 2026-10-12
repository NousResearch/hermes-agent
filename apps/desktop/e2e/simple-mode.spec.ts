/**
 * E2E: choosing Simple on a true first run writes the safety preset into the
 * user's own config.yaml without dropping a deny rule they already had, and
 * the Simple/Advanced question is asked once — never again after a reload.
 *
 * Prerequisite: `npm run build` must have been run so dist/ exists.
 */

import fs from 'node:fs'
import path from 'node:path'

import { parse } from 'yaml'

import { buildAppEnv, createSandbox, launchDesktop } from './fixtures'
import { expect, test } from './test'

const MODE_QUESTION = 'How do you want to use Hermes?'
const USER_DENY = 'npm publish*'

test('Simple at first run merges the safety preset and is asked only once', async () => {
  // A cold backend boot plus a renderer reload: two waits that can each take a while.
  test.setTimeout(180_000)
  const sandbox = createSandbox('simple-mode')
  const configPath = path.join(sandbox.hermesHome, 'config.yaml')

  fs.writeFileSync(configPath, `approvals:\n  deny:\n    - '${USER_DENY}'\n`, 'utf8')

  const { app, page } = await launchDesktop(buildAppEnv(sandbox))

  try {
    // The overlay opens on "Starting Hermes…" first; the question needs a live backend.
    await expect(page.getByText(MODE_QUESTION)).toBeVisible({ timeout: 90_000 })

    await page.getByRole('button', { name: /^Simple/ }).click()

    await expect
      .poll(() => parse(fs.readFileSync(configPath, 'utf8')) ?? {}, { timeout: 30_000 })
      .toMatchObject({
        approvals: { deny: expect.arrayContaining([USER_DENY, 'git *push*--force*', 'git *reset*--hard*']) },
        checkpoints: { enabled: true },
        agent: { coding_context: 'off' }
      })

    await page.reload()
    // Simple's own sign-in screen, not the question, is what a reload lands on.
    await expect(page.getByRole('button', { name: 'Switch to Advanced' })).toBeVisible({ timeout: 90_000 })
    await expect(page.getByText(MODE_QUESTION)).toHaveCount(0)
  } finally {
    await app.close().catch(() => undefined)
    sandbox.cleanup()
  }
})
