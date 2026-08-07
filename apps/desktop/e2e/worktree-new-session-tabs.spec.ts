/**
 * Worktree A's "+" and worktree B's "+" must each open their OWN new session.
 *
 * Motivated by a report that, with a session open per worktree in separate
 * tabs, "+" on lane A opens a new session but "+" on lane B then does nothing.
 *
 * SCOPE — this drives the real two-lane scenario end to end, but it is
 * COVERAGE, not a regression test: it passes both with and without the
 * `homeFreshDraftToWorkspace` fix in store/session-states.ts (verified by
 * neutering that function and re-running). In other words the per-lane "+"
 * path is healthy on this build — two clicks reliably yield two distinct
 * stacked sessions. Keep it so a future change that collapses them is caught,
 * and so the (fiddly) worktree-lane fixture below stays exercised.
 *
 * Fixture notes (these were the hard part, don't regress them):
 *  - A desktop session deliberately does NOT adopt its launch directory as a
 *    workspace (`_LAUNCH_CWD_NOT_A_WORKSPACE` in tui_gateway/server.py), so
 *    `terminal.cwd` alone never stamps a session cwd and no lane can derive.
 *    The worktrees are registered as real projects via the folder-open flow
 *    (⌘O / Ctrl+O) with Electron's dialog stubbed.
 *  - Worktree lanes are session-derived: a lane appears once a PERSISTED turn
 *    has a cwd inside it, so each seed waits for the assistant reply.
 *  - `repo_scan_roots` is pinned to the sandbox so the host's real repos can't
 *    leak into the sidebar.
 *  - The `__HERMES_*` window hooks are DEV-only and absent from the packaged
 *    build these tests launch; use the `data-tree-tab` DOM hook instead.
 */
import { execFileSync } from 'node:child_process'
import * as fs from 'node:fs'
import * as path from 'node:path'

import { test, expect } from './test'

import {
  buildAppEnv,
  createSandbox,
  launchDesktop,
  writeEnvFile,
  writeMockProviderConfig,
  waitForAppReady,
  type MockBackendFixture,
} from './fixtures'
import { startMockServer } from './mock-server'

const WORKTREE_A = 'e2e-tree-a'
const WORKTREE_B = 'e2e-tree-b'

function git(cwd: string, ...args: string[]): string {
  return execFileSync('git', args, { cwd, encoding: 'utf8' })
}

function createRepoWithWorktrees(root: string): { repo: string; treeA: string; treeB: string } {
  const repo = path.join(root, 'repo')

  fs.mkdirSync(repo, { recursive: true })
  git(repo, 'init', '--initial-branch=main')
  git(repo, 'config', 'user.email', 'e2e@example.com')
  git(repo, 'config', 'user.name', 'Hermes E2E')
  fs.writeFileSync(path.join(repo, 'README.md'), '# E2E repo\n', 'utf8')
  git(repo, 'add', 'README.md')
  git(repo, 'commit', '-m', 'initial')

  const treeA = path.join(root, WORKTREE_A)
  const treeB = path.join(root, WORKTREE_B)

  git(repo, 'worktree', 'add', '-b', WORKTREE_A, treeA)
  git(repo, 'worktree', 'add', '-b', WORKTREE_B, treeB)

  return { repo, treeA, treeB }
}

let fixture: MockBackendFixture | null = null
let trees: { repo: string; treeA: string; treeB: string } | null = null

test.beforeAll(async () => {
  const sandbox = createSandbox('worktree-new-session-tabs')
  trees = createRepoWithWorktrees(sandbox.root)
  const mock = await startMockServer()

  writeMockProviderConfig(sandbox.hermesHome, mock.url)
  fs.appendFileSync(
    path.join(sandbox.hermesHome, 'config.yaml'),
    `\nterminal:\n  cwd: ${trees.treeA}\n` +
      `desktop:\n  repo_scan_enabled: true\n  repo_scan_roots:\n    - ${sandbox.root}\n`,
    'utf8',
  )
  writeEnvFile(sandbox.hermesHome)

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
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
  trees = null
})

test('"+" on each worktree lane opens its own distinct new session', async ({}, testInfo) => {
  const { app, page } = fixture!
  const { treeA, treeB } = trees!

  /** Session-tile tabs in the strip (production DOM hook). */
  const tabIds = () =>
    page.evaluate(() =>
      [...document.querySelectorAll('[data-tree-tab]')]
        .map(el => el.getAttribute('data-tree-tab') ?? '')
        .filter(id => id.startsWith('session-tile:')),
    )

  // The VISIBLE composer — a hidden tab's composer is still in the DOM.
  const composer = () => page.locator('[contenteditable="true"]:visible').first()

  /** Send a prompt and wait for the reply, so the turn PERSISTS (which is what
   *  gives the session a cwd and makes its worktree lane appear). */
  const seedSession = async (prompt: string) => {
    const input = composer()
    await input.click()
    await input.type(prompt, { delay: 2 })
    await page.keyboard.press('Enter')
    await page.waitForFunction(
      text =>
        [...document.querySelectorAll('[data-slot="aui_thread-viewport"]')].some(
          el => (el as HTMLElement).offsetParent !== null && (el.textContent ?? '').includes(text),
        ),
      prompt,
      { timeout: 20_000 },
    )
    await page.waitForFunction(
      () =>
        [...document.querySelectorAll('[data-slot="aui_assistant-message-root"]')].some(
          el => (el.textContent ?? '').includes('mock inference server'),
        ),
      undefined,
      { timeout: 30_000 },
    )
  }

  // ── Register both worktrees as projects via the real folder-open flow ────
  await app.evaluate(async ({ dialog }, paths) => {
    let i = 0
    ;(dialog as any).showOpenDialog = async () => ({ canceled: false, filePaths: [paths[i++]] })
  }, [treeA, treeB])

  // Ctrl+O = workspace.openFolder → open folder as a project + fresh session.
  await page.keyboard.press('Control+O')
  await page.waitForTimeout(4000)
  await seedSession('worktree A seed session')

  await page.keyboard.press('Control+O')
  await page.waitForTimeout(4000)
  await seedSession('worktree B seed session')

  // ── Both lanes must now be present in the projects view ─────────────────
  const showProjects = page.getByRole('button', { name: 'Show projects' }).first()

  if (await showProjects.isVisible().catch(() => false)) {
    await showProjects.click()
  }

  const plusA = page.locator(`[aria-label="New session in ${WORKTREE_A}"]`).first()
  const plusB = page.locator(`[aria-label="New session in ${WORKTREE_B}"]`).first()

  await expect(plusA).toBeAttached({ timeout: 30_000 })
  await expect(plusB).toBeAttached({ timeout: 30_000 })

  const before = await tabIds()
  await page.screenshot({ path: testInfo.outputPath('01-two-worktree-sessions.png') })

  // ── "+" on lane A ────────────────────────────────────────────────────────
  await plusA.click({ force: true })
  await expect.poll(async () => (await tabIds()).length, { timeout: 20_000 }).toBeGreaterThan(before.length)

  const afterA = await tabIds()
  await page.screenshot({ path: testInfo.outputPath('02-after-plus-worktree-a.png') })

  // ── "+" on lane B: MUST open ANOTHER new session, not reuse A's draft ────
  await plusB.click({ force: true })
  await expect.poll(async () => (await tabIds()).length, { timeout: 20_000 }).toBeGreaterThan(afterA.length)

  const afterB = await tabIds()
  await page.screenshot({ path: testInfo.outputPath('03-after-plus-worktree-b.png') })

  // Every open session must be distinct — no id appearing twice.
  expect(new Set(afterB).size).toBe(afterB.length)
})
