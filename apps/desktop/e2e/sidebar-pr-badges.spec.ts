/**
 * Screenshots + coverage for the sidebar PR badges.
 *
 * Builds a real git repo with real linked worktrees, creates REAL desktop
 * sessions in each worktree through the gateway (so each row records its own
 * git_branch — the column the badge joins on), and stubs `gh` with a script
 * that answers the GraphQL query the way GitHub does. Nothing about the join
 * is faked: the rows, the lanes, the store and the renderer are all real.
 *
 * Guards two things at once:
 *  - the gateway fix: a session created in a worktree records its branch, so
 *    its row badges. Before the fix every row here had a NULL branch.
 *  - the lane badge: a branch lane in a project shows the PR for its branch.
 */
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
  writeMockProviderConfig
} from './fixtures'
import { startMockServer } from './mock-server'
import { RealSessionBuilder } from './real-session-builder'
import { expect, test } from './test'
import { expectVisualSnapshot } from './visual-snapshot'

/** Branch lanes the screenshots show, with the PR state each one renders. */
const LANES = [
  {
    branch: 'ethie/title-gen',
    number: 8123,
    state: 'OPEN',
    title: 'fix(aux): translate response_format for Anthropic wires'
  },
  {
    branch: 'ethie/themepicker',
    number: 8098,
    state: 'MERGED',
    title: 'feat(desktop): live-preview themes from the palette'
  },
  {
    branch: 'ethie/browser-external',
    number: 8140,
    state: 'OPEN',
    draft: true,
    title: 'feat(desktop): open the in-app browser tab externally'
  },
  {
    branch: 'ethie/cli-env-deprecated',
    number: 8077,
    state: 'CLOSED',
    title: 'fix(cli): show the deprecated cwd hint on real lines'
  }
] as const

/** The project overview previews this many rows (PROJECT_PREVIEW_COUNT). */
const PREVIEW_COUNT = 3

function git(args: string[], cwd: string): void {
  execFileSync('git', args, { cwd })
}

/** A repo with one worktree per lane, each on its own branch. */
function createRepoWithWorktrees(root: string): { repo: string; worktrees: Map<string, string> } {
  const repo = path.join(root, 'hermes-agent')

  fs.mkdirSync(repo, { recursive: true })
  git(['init', '--initial-branch=main'], repo)
  git(['config', 'user.email', 'e2e@example.com'], repo)
  git(['config', 'user.name', 'Hermes E2E'], repo)
  // A GitHub remote, so `gh repo view` resolves an owner/name for the query.
  git(['remote', 'add', 'origin', 'https://github.com/NousResearch/hermes-agent.git'], repo)
  fs.writeFileSync(path.join(repo, 'README.md'), '# hermes-agent\n', 'utf8')
  git(['add', 'README.md'], repo)
  git(['commit', '-m', 'initial'], repo)

  const worktrees = new Map<string, string>()

  for (const { branch } of LANES) {
    const dir = path.join(repo, '.worktrees', branch.replace(/\//g, '-'))

    git(['worktree', 'add', '-b', branch, dir], repo)
    worktrees.set(branch, dir)
  }

  return { repo, worktrees }
}

/**
 * A `gh` stand-in on PATH. Answers the two calls reviewPrList makes:
 * `gh repo view --json nameWithOwner` and `gh api graphql -f query=...`.
 * The GraphQL reply mirrors the real shape — aliases `b<i>` per branch, each
 * holding `nodes`, with isCrossRepository so the fork guard is exercised.
 */
function writeGhStub(binDir: string): void {
  fs.mkdirSync(binDir, { recursive: true })

  const prs = LANES.map(lane => ({
    headRefName: lane.branch,
    isCrossRepository: false,
    isDraft: 'draft' in lane ? lane.draft : false,
    number: lane.number,
    state: lane.state,
    title: lane.title,
    url: `https://github.com/NousResearch/hermes-agent/pull/${lane.number}`
  }))

  const script = `#!/usr/bin/env node
const args = process.argv.slice(2)
const PRS = ${JSON.stringify(prs)}

if (args[0] === 'repo' && args[1] === 'view') {
  process.stdout.write('NousResearch/hermes-agent\\n')
  process.exit(0)
}

if (args[0] === 'api' && args[1] === 'graphql') {
  const query = (args.find(a => a.startsWith('query=')) || '').slice('query='.length)
  const repository = {}
  // Reply only to the aliases the caller actually asked for, exactly as the
  // API would — that is what proves the desktop asked about this branch.
  for (const [, alias, branch] of query.matchAll(/(b\\d+): pullRequests\\(headRefName: "([^"]+)"/g)) {
    const hit = PRS.find(pr => pr.headRefName === branch)
    repository[alias] = { nodes: hit ? [hit] : [] }
  }
  process.stdout.write(JSON.stringify({ data: { repository } }))
  process.exit(0)
}

process.exit(1)
`

  const ghPath = path.join(binDir, 'gh')

  fs.writeFileSync(ghPath, script, 'utf8')
  fs.chmodSync(ghPath, 0o755)
}

let fixture: MockBackendFixture | null = null

test.beforeAll(async () => {
  const sandbox = createSandbox('sidebar-pr-badges')
  const { repo, worktrees } = createRepoWithWorktrees(sandbox.root)
  const binDir = path.join(sandbox.root, 'bin')

  writeGhStub(binDir)

  const mock = await startMockServer()

  writeMockProviderConfig(sandbox.hermesHome, mock.url)
  fs.appendFileSync(path.join(sandbox.hermesHome, 'config.yaml'), `\nterminal:\n  cwd: ${repo}\n`, 'utf8')
  writeEnvFile(sandbox.hermesHome)

  // Real sessions, one per worktree, through the real gateway + agent loop.
  // Each row records the branch of the worktree it ran in.
  const builder = await RealSessionBuilder.start(sandbox.hermesHome)

  try {
    for (const { branch } of LANES) {
      await builder.createSession({
        cwd: worktrees.get(branch)!,
        title: branch,
        turns: [`Working on ${branch}`]
      })
    }
  } finally {
    await builder.close()
  }

  const env = buildAppEnv(sandbox, { PATH: `${binDir}:${process.env.PATH ?? ''}` })
  const { app, page } = await launchDesktop(env)

  fixture = {
    app,
    mock,
    mockUrl: mock.url,
    page,
    sandbox,
    cleanup: async () => {
      await app.close()
      await mock.close()
      sandbox.cleanup()
    }
  }

  await waitForAppReady(fixture, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

/** The sidebar subtree. Badge queries MUST be scoped to it: the composer's
 *  coding rail renders the same PrTag for the chat's own branch, so a
 *  page-wide query counts a badge this feature did not put there. */
function sidebar() {
  return fixture!.page.locator('[data-slot="sidebar"]').first()
}

/**
 * Put the sidebar into the view the badges live in: grouped by project (so
 * repos render their branch lanes) with the PR row metadata on, and back out
 * of any project a previous test entered.
 *
 * Written straight into the persisted atoms rather than walked through the
 * filter menu. The menu is three nested popovers deep, and driving it makes
 * the test assert on menu structure instead of on badges — the atoms ARE the
 * state the menu sets. `agentsGroupedByWorkspace` is the authority for the
 * project view (see the comment above $sidebarFlatGrouping); grouping `date`
 * vs `project` is not stored in the grouping atom at all.
 *
 * Every entry is rewritten on each call, and the page reloaded, because the
 * entered project persists: without the reset the second test opens already
 * inside the project and never finds the row it means to click.
 */
async function showProjectOverview(): Promise<void> {
  const page = fixture!.page

  await page.evaluate(() => {
    localStorage.setItem('hermes.desktop.agentsGroupedByWorkspace', 'true')
    localStorage.setItem('hermes.desktop.sidebarRowMeta', JSON.stringify(['pr', 'updated']))

    // Leave whichever project an earlier test entered.
    for (const key of Object.keys(localStorage)) {
      if (key.includes('enteredProject') || key.includes('projectScope')) {
        localStorage.removeItem(key)
      }
    }
  })
  await page.reload()
  await waitForAppReady(fixture!, 120_000)
}

/** Enter the repo's project. The label is path-disambiguated when a basename
 *  collides with another repo on this machine, so match the suffix. */
async function enterProject(): Promise<void> {
  await sidebar()
    .getByRole('button', { name: /^Open .*hermes-agent$/ })
    .first()
    .click()
}

test('session rows badge the PR of the branch their session recorded', async () => {
  // Setup is billed to the first test: four worktrees, four real agent turns
  // and an Electron boot do not fit the default per-test budget.
  test.slow()

  const page = fixture!.page

  await showProjectOverview()

  // These are session-ROW badges, which render only when the row recorded a
  // git_branch — the gateway half of this change. The overview previews
  // PROJECT_PREVIEW_COUNT rows newest-first, so assert the count rather than
  // naming which PRs survive the cut: that ordering is not this test's claim.
  await expect(sidebar().getByRole('button', { name: /^Open pull request #/ })).toHaveCount(PREVIEW_COUNT)

  await expectVisualSnapshot(page, { app: fixture!.app, name: 'sidebar-pr-badges-rows' })
})

test('every branch lane badges the PR for its own branch', async () => {
  test.slow()

  const page = fixture!.page

  await showProjectOverview()
  await enterProject()

  // The lane badge — the new surface. One assertion per PR state, so a
  // regression in the state→glyph mapping fails here too.
  for (const lane of LANES) {
    await expect(
      sidebar()
        .getByRole('button', { name: `Open pull request #${lane.number}` })
        .first()
    ).toBeVisible()
  }

  await expectVisualSnapshot(page, { app: fixture!.app, name: 'sidebar-pr-badges-lanes' })
})

test('an expanded lane and the session row inside it carry the same PR', async () => {
  test.slow()

  const lane = LANES[0]

  await showProjectOverview()
  await enterProject()

  // A lane that holds sessions defaults OPEN, so the row is already there —
  // clicking the label would collapse it. Two badges: the lane header (from
  // its label) and the session row inside it (from the branch its session
  // recorded). Both halves of this change, in one assertion.
  await expect(sidebar().getByRole('button', { name: `Open pull request #${lane.number}` })).toHaveCount(2)

  await expectVisualSnapshot(fixture!.page, { app: fixture!.app, name: 'sidebar-pr-badges-lane-expanded' })
})
