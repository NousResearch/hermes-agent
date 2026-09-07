import { execFileSync } from 'node:child_process'
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
import { startMockServer } from './mock-server'
import { expect, test } from './test'
import { expectVisualSnapshot } from './visual-snapshot'

const BRANCH_NAME = 'e2e-composer-branch'

/**
 * Enough branches to make both the base-branch popover and the convert-branch
 * list taller than their default height. That is the condition in which the
 * dialog's own scroll box clips the popover, and that regression is what the
 * visual snapshots here guard against.
 */
const EXTRA_BRANCHES = [
  'feature/alpha-one',
  'feature/beta-two',
  'feature/gamma-three',
  'fix/delta-four',
  'fix/epsilon-five',
  'chore/zeta-six',
  'chore/eta-seven',
  'spike/theta-eight',
  'spike/iota-nine',
  'release/kappa-ten',
]

function createGitRepo(root: string): string {
  const repo = path.join(root, 'repo')

  fs.mkdirSync(repo, { recursive: true })
  execFileSync('git', ['init', '--initial-branch=main'], { cwd: repo })
  execFileSync('git', ['config', 'user.email', 'e2e@example.com'], { cwd: repo })
  execFileSync('git', ['config', 'user.name', 'Hermes E2E'], { cwd: repo })
  fs.writeFileSync(path.join(repo, 'README.md'), '# E2E repo\n', 'utf8')
  execFileSync('git', ['add', 'README.md'], { cwd: repo })
  execFileSync('git', ['commit', '-m', 'initial'], { cwd: repo })

  for (const branch of EXTRA_BRANCHES) {
    execFileSync('git', ['branch', branch], { cwd: repo })
  }

  return repo
}

function configureRepoCwd(hermesHome: string, mockUrl: string, repo: string): void {
  writeMockProviderConfig(hermesHome, mockUrl)
  fs.appendFileSync(path.join(hermesHome, 'config.yaml'), `\nterminal:\n  cwd: ${repo}\n`, 'utf8')
  writeEnvFile(hermesHome)
}

let fixture: MockBackendFixture | null = null

/** A dialog renders as `[data-slot="dialog-content"]` (components/ui/dialog.tsx). */
const DIALOG = '[data-slot="dialog-content"]'

/** Open the worktree dialog with the global ⌘⇧B / ctrl+shift+B hotkey. */
async function openWorktreeDialog(): Promise<void> {
  const page = fixture!.page
  await page.keyboard.press('Control+Shift+B')
  await expect(page.locator(DIALOG)).toBeVisible()
}

/** Close the open dialog and wait until it leaves the DOM. */
async function closeDialog(): Promise<void> {
  const page = fixture!.page
  await page.keyboard.press('Escape')
  await expect(page.locator(DIALOG)).toHaveCount(0)
}

test.beforeAll(async () => {
  const sandbox = createSandbox('worktree-branch-status')
  const repo = createGitRepo(sandbox.root)
  const mock = await startMockServer()

  configureRepoCwd(sandbox.hermesHome, mock.url, repo)

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
    },
  }

  await waitForAppReady(fixture, 120_000)

  // The coding rail, and thus the ⌘⇧B worktree dialog, mounts only after the
  // session resolves a cwd that holds a repo. This happens on the first turn.
  const composer = page.locator('[contenteditable="true"]').first()
  await composer.click()
  await composer.type('create a repo-backed e2e session', { delay: 2 })
  await page.keyboard.press('Enter')
  await page.waitForFunction(
    prompt => (document.querySelector('[data-slot="aui_thread-viewport"]')?.textContent ?? '').includes(prompt),
    'create a repo-backed e2e session',
    { timeout: 15_000 },
  )
  await expect(page.locator('.coding-status-bar')).toContainText('main')
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('worktree dialog renders the base-branch picker over the dialog, not clipped by it', async () => {
  const page = fixture!.page

  await openWorktreeDialog()

  // Open the base-branch combobox. With 11 branches, the list is taller than
  // the space below the trigger. A popover that portals into the dialog's
  // `overflow-y-auto` box is therefore cut off. This snapshot catches that bug.
  await page.getByRole('button', { name: /branch off/i }).click()
  await expect(page.getByPlaceholder('Search branches…')).toBeVisible()
  await expect(page.getByRole('option', { name: 'feature/alpha-one' })).toBeVisible()

  await expectVisualSnapshot(page, { name: 'worktree-dialog-base-branch-picker', app: fixture!.app })

  // This check does not depend on pixels: the dialog's scroll box must not crop
  // the painted box of the popover. Measure the geometry, so a headless run
  // fails on this regression before a person looks at a diff image.
  const clipped = await page.evaluate(() => {
    const popover = document.querySelector('[data-slot="popover-content"]')
    const dialog = document.querySelector('[data-slot="dialog-content"]')

    if (!popover || !dialog) {
      return { reason: 'missing', clipped: true }
    }

    const p = popover.getBoundingClientRect()
    const d = dialog.getBoundingClientRect()
    const scrolls = window.getComputedStyle(dialog).overflowY

    return {
      reason: 'measured',
      // Only a clipping ancestor can crop the popover. The popover is cut when
      // the dialog scrolls its overflow AND the popover goes past the box of
      // the dialog.
      clipped: (scrolls === 'auto' || scrolls === 'scroll' || scrolls === 'hidden') &&
        (p.bottom > d.bottom + 1 || p.top < d.top - 1 || p.right > d.right + 1 || p.left < d.left - 1),
    }
  })

  expect(clipped.clipped, `base-branch popover is clipped by the dialog (${clipped.reason})`).toBe(false)

  await page.keyboard.press('Escape')
  await closeDialog()
})

test('worktree dialog convert-an-existing-branch sub-view lists the repo branches', async () => {
  const page = fixture!.page

  await openWorktreeDialog()
  await page.getByRole('button', { name: 'Convert an existing branch' }).click()

  await expect(page.getByPlaceholder('Search branches…')).toBeVisible()
  await expect(page.getByRole('option', { name: /feature\/alpha-one/ })).toBeVisible()

  await expectVisualSnapshot(page, { name: 'worktree-dialog-convert-branch', app: fixture!.app })

  await closeDialog()
})

test('creating a branch with ctrl-shift-b updates the composer git-status branch and leaves no dialog behind', async ({}, testInfo) => {
  const page = fixture!.page
  const codingRow = page.locator('.coding-status-bar')

  await openWorktreeDialog()
  // Exactly one dialog instance. A second dialog here, hidden or empty, is the
  // symptom of the double-open bug.
  await expect(page.locator(DIALOG)).toHaveCount(1)

  const branchInput = page.locator('input[placeholder="e.g. my-feature"]').first()
  await expect(branchInput).toBeVisible()
  await branchInput.fill(BRANCH_NAME)
  // Select a base branch, so this test uses the same path as the user: open the
  // picker, select a branch, then submit. It does not use the default value.
  // The keyboard drives this step. The dialog still clips the popover, so a
  // mouse click on an option is not reliable until that bug is corrected. The
  // double-open check below is therefore independent of the clipping bug.
  await page.getByRole('button', { name: /branch off/i }).click()
  await page.getByPlaceholder('Search branches…').fill('main')
  await expect(page.getByRole('option', { name: 'main' }).first()).toBeVisible()
  await page.keyboard.press('Enter')
  await expect(page.locator('[data-slot="popover-content"]')).toHaveCount(0)

  await page.getByRole('button', { name: 'New worktree' }).click()

  await expect(codingRow).toContainText(BRANCH_NAME, { timeout: 15_000 })

  // The dialog must close and stay closed. No empty second dialog can remain
  // after the new worktree session starts.
  await expect(page.locator(DIALOG)).toHaveCount(0)
  await page.waitForTimeout(2000)
  await expect(page.locator(DIALOG)).toHaveCount(0)

  await page.screenshot({ path: testInfo.outputPath('composer-branch-after-create.png') })
})

test('ctrl-shift-b opens exactly one worktree dialog when a second composer is on screen', async ({}, testInfo) => {
  const page = fixture!.page

  // ⌘T / ctrl+T stacks a second session tile. That gives a second live composer
  // and therefore a second coding rail. Each rail mounted its own
  // WorktreeDialog, and each rail subscribed to the same global token. One
  // keypress therefore opened two stacked dialogs, and the dialog the user
  // dismissed showed an identical empty one behind it. One mount in the sidebar
  // makes that impossible by structure.
  await page.keyboard.press('Control+T')
  await expect(page.locator('.coding-status-bar')).toHaveCount(2, { timeout: 20_000 })

  await page.keyboard.press('Control+Shift+B')
  await expect(page.locator(DIALOG).first()).toBeVisible()
  // Wait: let the effect of every subscriber flush before the count.
  await page.waitForTimeout(500)

  const count = await page.locator(DIALOG).count()
  await page.screenshot({ path: testInfo.outputPath('worktree-dialog-two-composers.png') })

  expect(count, 'one hotkey press must open exactly one worktree dialog').toBe(1)

  // A dismissal then leaves nothing behind.
  await closeDialog()
})
