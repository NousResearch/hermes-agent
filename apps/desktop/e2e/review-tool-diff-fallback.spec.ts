/**
 * Regression E2E for #86334: a tool edit made outside a Git repository must
 * still open as a read-only diff from the transcript's Changed Files card.
 *
 * This drives the real Electron renderer, desktop backend, agent loop, mock
 * inference transport, and write_file tool. Git intentionally has no authority
 * over the sandbox project, so the Review pane must use the tool-result diff.
 */

import * as fs from 'node:fs'
import * as path from 'node:path'

import {
  buildAppEnv,
  createSandbox,
  launchDesktop,
  type MockBackendFixture,
  waitForAppReady,
  writeEnvFile,
  writeMockProviderConfig,
} from './fixtures'
import { REVIEW_TOOL_DIFF_QUESTION, REVIEW_TOOL_DIFF_TRIGGER, startMockServer } from './mock-server'
import { expect, test } from './test'

test('tool diff outside Git opens read-only Review instead of NO DIFFS', async () => {
  test.setTimeout(120_000)
  const sandbox = createSandbox('review-tool-diff-fallback')
  const projectRoot = path.join(sandbox.root, 'non-git-project')
  const changedFile = path.join(projectRoot, 'e2e-review-target.py')
  fs.mkdirSync(projectRoot)
  fs.writeFileSync(changedFile, 'def changed_by_e2e():\n    return "before"\n', 'utf8')

  const mock = await startMockServer({ verificationWritePath: changedFile })
  writeMockProviderConfig(sandbox.hermesHome, mock.url)
  fs.appendFileSync(path.join(sandbox.hermesHome, 'config.yaml'), `\nterminal:\n  cwd: ${JSON.stringify(projectRoot)}\n`, 'utf8')
  writeEnvFile(sandbox.hermesHome)
  const { app, page } = await launchDesktop(buildAppEnv(sandbox))
  const fixture: MockBackendFixture = {
    app,
    page,
    mock,
    mockUrl: mock.url,
    sandbox,
    cleanup: async () => {
      await app.close().catch(() => undefined)
      await mock.close()
      sandbox.cleanup()
    },
  }

  try {
    await waitForAppReady(fixture, 120_000)
    const composer = page.locator('[contenteditable="true"]').first()
    await composer.click()
    await composer.type(REVIEW_TOOL_DIFF_TRIGGER)
    await page.keyboard.press('Enter')

    await expect(page.getByText(REVIEW_TOOL_DIFF_QUESTION)).toBeVisible({ timeout: 60_000 })
    await page.locator('[data-slot="composer-root"] button[aria-label="Stop"]').click()

    await page.waitForFunction(() => {
      const card = document.querySelector('[data-slot="aui_changed-files"]')
      const reviewButton = [...(card?.querySelectorAll('button') ?? [])]
        .find(button => button.textContent?.trim() === 'Review')
      if (!reviewButton) {
        return false
      }

      reviewButton.click()
      return true
    }, undefined, { timeout: 30_000 })

    const review = page.getByRole('complementary', { name: 'Review' })
    await expect(review).toBeVisible()
    await expect(review).toContainText('e2e-review-target.py')
    await expect(review).not.toContainText('NO DIFFS')

    const reviewFile = review.locator('[aria-selected]').filter({ hasText: '+1' }).first()
    await expect(reviewFile).toBeVisible()
    await reviewFile.click()
    await expect(review).toContainText('return "before"')
    await expect(review).toContainText('return "changed"')

    await expect(review.getByRole('button', { name: /Stage|Unstage|Revert|Commit|Push|Create PR/i })).toHaveCount(0)
    await expect(review.getByText(/Commit changes|Create PR/i)).toHaveCount(0)
  } finally {
    await fixture.cleanup()
  }
})
