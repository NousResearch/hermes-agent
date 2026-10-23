/**
 * Linked Kanban work follows the conversation that ordered it — real app, real
 * `hermes serve`, only the LLM faked; no dispatcher runs and no task executes.
 *
 * The fixture is built by the PRODUCERS, not by hand: `kanban_create` runs under the origin's own
 * session id (the compression ROOT for one task, the TIP for the rest) on several boards, so the
 * origin index in state.db is exactly what the tool wrote; claims, heartbeats and the worker spawn
 * come from `claim_task` / `heartbeat_worker` / `_set_worker_pid`. Evidence is then aged or removed
 * (a stale spawn, a deleted task, a removed board, and a board whose kanban.db path is a directory —
 * a storage fault no permission repair can heal) so a gap is a real gap.
 *
 * Lineage here is SEEDED (a compression root + tip, as `end_session(..., 'compression')` leaves
 * them); a real in-app compaction is covered by lineage-rotation.spec.ts and is not repeated here.
 *
 *  - the origin row's sidebar badge reports the loudest state (needs input), with no board scan;
 *  - after the conversation's own turn completes — and after a reload — the composer strip still
 *    lists every linked task, each state distinct (evidence-backed background, reserved-only,
 *    stale, needs-input, done) plus three gaps shown as gaps (task / board missing, board
 *    unreadable), and "worker log" opens THAT task's board drawer on the log section;
 *  - the foreground turn is independent: one prompt, one completion, no kanban text in the transcript;
 *  - A → B → A: a profile whose store holds the same literal session id shows nothing, and the
 *    origin's links are back on return.
 */

import { spawnSync } from 'node:child_process'
import * as path from 'node:path'

import { expect, type Page, test } from '@playwright/test'

import {
  composer,
  type CoreSandbox,
  coreAppEnv,
  createCoreSandbox,
  currentSessionId,
  launchCoreApp,
  recordWebSockets,
  REPO_ROOT,
  send,
  waitForInteractive,
  writeProviderHome
} from './harness'
import { startScriptedProvider } from './provider'
import { messageRows, python } from './remote-helpers'

const nonce = Math.random()
  .toString(36)
  .slice(2, 8)
  .replace(/[^a-z0-9]/g, 'x')
  .padEnd(4, 'q')

const SEED = String.raw`
import json, os, sys, time
from pathlib import Path

home, nonce = Path(sys.argv[1]), sys.argv[2]
os.environ["HERMES_HOME"] = str(home)

from gateway.session_context import scoped_current_session_id
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_state import SessionDB
from tools import kanban_tools as kt

root, tip = "origin-root-" + nonce, "origin-tip-" + nonce


def sessions(path, rows):
    db = SessionDB(db_path=path / "state.db")
    for sid, title, parent, end in rows:
        db.create_session(sid, source="cli", **({"parent_session_id": parent} if parent else {}))
        db.append_message(sid, "user", "hello " + sid)
        db.append_message(sid, "assistant", "hi " + sid)
        db.set_session_title(sid, title)
        if end:
            db.end_session(sid, end)
    db.close()


sessions(home, [(root, "Origin conversation", None, "compression"), (tip, "Origin conversation (cont.)", root, None)])
named = home / "profiles" / "b"
named.mkdir(parents=True, exist_ok=True)
(named / "config.yaml").write_text("")
sessions(named, [(root, "Other profile same id", None, None)])  # the SAME literal id, another store

kb.init_db()
for slug in ("alpha", "gone", "broken"):
    kb.create_board(slug)


def create(board, title, session):
    args = {"title": title, "assignee": "default", **({"board": board} if board else {})}
    with scoped_current_session_id(session):
        out = json.loads(kt._handle_create(args))
    assert out["ok"], out
    return out["task_id"]


def sql(board, statement, *params):
    with kbc.connect_closing(board=board) as conn:
        conn.execute(statement, params)
        conn.commit()


def claim(board, task_id, *, beat=False, spawn=False):
    with kbc.connect_closing(board=board) as conn:
        assert kb.claim_task(conn, task_id) is not None  # opens a real run
        if spawn:
            kbd._set_worker_pid(conn, task_id, os.getpid())  # real pid + process fingerprint + event
        if beat:
            kbd.heartbeat_worker(conn, task_id)


ids = {
    "background": create("alpha", "Background job " + nonce, root),
    "done": create("alpha", "Finished report " + nonce, tip),
    "missing": create("alpha", "Deleted later " + nonce, tip),
    "reserved": create(None, "Reserved only " + nonce, tip),
    "ask": create(None, "Needs answer " + nonce, tip),
    "stale": create(None, "Quiet worker " + nonce, tip),
    "gone": create("gone", "Board removed " + nonce, tip),
    "broken": create("broken", "Board unreadable " + nonce, tip),
}
claim("alpha", ids["background"], beat=True)
sql("alpha", "UPDATE tasks SET status = 'done', completed_at = ? WHERE id = ?", int(time.time()), ids["done"])
claim(None, ids["reserved"])
sql(None, "UPDATE tasks SET status = 'blocked', block_kind = 'needs_input' WHERE id = ?", ids["ask"])
claim(None, ids["stale"], spawn=True)
old = int(time.time()) - 5 * 3600
sql(None, "UPDATE task_runs SET started_at = ? WHERE task_id = ?", old, ids["stale"])
sql(None, "UPDATE task_events SET created_at = ? WHERE task_id = ? AND kind = 'spawned'", old, ids["stale"])

# Controlled evidence loss AFTER indexing: the index keeps the ref, the boards no longer confirm it.
sql("alpha", "DELETE FROM tasks WHERE id = ?", ids["missing"])
kb.remove_board("gone", archive=False)
# A genuine, stable storage fault on the sandbox's own broken board: a DIRECTORY where kanban.db
# should be. SQLite cannot open it, and neither permission tightening nor a repair path can turn it
# back into a database. Every seeded connection is closed by now.
broken_db = Path(kb.kanban_db_path(board="broken"))
for suffix in ("", "-wal", "-shm"):
    Path(str(broken_db) + suffix).unlink(missing_ok=True)
broken_db.mkdir()

log = kb.worker_log_path(ids["background"], board="alpha")
log.parent.mkdir(parents=True, exist_ok=True)
log.write_text("LOG-LINE-" + nonce + "\n")
print(json.dumps(ids))
`

type Ids = Record<'ask' | 'background' | 'broken' | 'done' | 'gone' | 'missing' | 'reserved' | 'stale', string>

function seed(sandbox: CoreSandbox): Ids {
  const out = spawnSync(python(), ['-c', SEED, sandbox.hermesHome, nonce], {
    cwd: REPO_ROOT,
    encoding: 'utf8',
    env: {
      HERMES_HOME: sandbox.hermesHome,
      HOME: sandbox.home,
      PATH: process.env.PATH ?? '',
      PYTHONPATH: REPO_ROOT
    }
  })

  if (out.status !== 0) {
    throw new Error(`fixture seeding failed:\n${out.stderr}\n${out.stdout}`)
  }

  return JSON.parse(out.stdout.trim().split('\n').pop()!)
}

const ROW = '*:has(> [data-row-actions])'

const rowTitled = (page: Page, title: string) =>
  page
    .locator(`${ROW}:not(${ROW} *)`)
    .filter({ visible: true })
    .filter({ hasText: title })

const originRow = (page: Page) => rowTitled(page, 'Origin conversation')
const strip = (page: Page) => page.locator('[data-slot="kanban-origin-strip"]').filter({ visible: true })
const rail = (page: Page) => page.locator('[data-slot="profile-rail"]')

async function shot(page: Page, name: string) {
  const file = test.info().outputPath(`${name}.png`)

  await page.screenshot({ path: file })
  await test.info().attach(name, { contentType: 'image/png', path: file })
}

/** The expanded list: every linked task, each with the state the board's own evidence supports. */
async function expectLinkedTasks(page: Page, ids: Ids, where: string) {
  await expect(strip(page), `${where}: one strip`).toHaveCount(1, { timeout: 60_000 })

  const toggle = strip(page).getByRole('button', { name: 'Show linked tasks' })

  if (await toggle.count()) {
    await toggle.click()
  }

  const state = (id: string) => strip(page).locator(`[data-kanban-origin-ref="${id}"]`)

  await expect(state(ids.background), where).toHaveAttribute('data-kanban-origin-state', 'background')
  await expect(state(ids.reserved), where).toHaveAttribute('data-kanban-origin-state', 'reserved')
  await expect(state(ids.stale), where).toHaveAttribute('data-kanban-origin-state', 'stale')
  await expect(state(ids.ask), where).toHaveAttribute('data-kanban-origin-state', 'needs-input')
  await expect(state(ids.done), where).toHaveAttribute('data-kanban-origin-state', 'done')

  // Gaps stay visible as gaps — never dropped, never painted as idle or healthy, each with its own cause.
  const gaps: [string, string][] = [
    [ids.missing, 'Task not found'],
    [ids.gone, 'Board not found'],
    [ids.broken, 'Board unreadable']
  ]

  for (const [gap, cause] of gaps) {
    await expect(state(gap), `${where}: gap ${gap}`).toHaveAttribute('data-kanban-origin-state', 'unavailable')
    await expect(state(gap), `${where}: gap ${gap} cause`).toContainText(cause)
  }
}

test('origin kanban: linked work follows the conversation across turn end, reload, lineage, boards and profiles', async () => {
  const provider = await startScriptedProvider()
  const sandbox = createCoreSandbox('origin-kanban')
  writeProviderHome(sandbox.hermesHome, provider.url)
  const ids = seed(sandbox)
  // The seed creates profile b with an empty config, which the app answers with the no-provider
  // onboarding overlay. Give b the SAME synthetic provider (fake endpoint, fake key) so it is a
  // usable profile and "shows nothing" below is a real observation, not an overlay hiding the UI.
  writeProviderHome(path.join(sandbox.hermesHome, 'profiles', 'b'), provider.url)
  const { app, page } = await launchCoreApp(coreAppEnv(sandbox))
  const ws = recordWebSockets(page)
  const marker = `U1-${nonce}`

  try {
    await waitForInteractive(app, page)
    // The Kanban plugin ships off; opt in the way the Plugins page does.
    await page.evaluate(() =>
      window.localStorage.setItem('hermes.desktop.pluginDecisions.v2', JSON.stringify({ kanban: true }))
    )
    await page.reload()
    await waitForInteractive(app, page)

    await test.step('the origin row carries one badge: the loudest linked state, across every board', async () => {
      const badge = originRow(page).locator('[data-kanban-origin]')

      await expect(badge).toHaveCount(1, { timeout: 60_000 })
      await expect(badge).toHaveAttribute('data-kanban-origin', 'needs-input')
    })

    await test.step('after the conversation’s own turn ends, every linked task is still listed with its own state', async () => {
      await originRow(page).first().click()
      expect(await currentSessionId(page)).toContain('origin-')

      provider.script(marker, [{ text: [`A1-${nonce} done`] }])
      await send(page, `${marker} status check`, 'Enter', ws)
      await expect
        .poll(() => provider.completions.some(c => c.marker === marker && c.finished), { timeout: 120_000 })
        .toBe(true)
      await expect
        .poll(() => page.locator('[data-slot="composer-root"] button[aria-label="Stop"]').count(), { timeout: 60_000 })
        .toBe(0)
      await expect(page.getByText(`A1-${nonce} done`).first()).toBeVisible()

      await expectLinkedTasks(page, ids, 'after the turn')
      await shot(page, '1-after-turn-linked-tasks')
    })

    await test.step('the foreground turn was independent of the linked work', async () => {
      // One user prompt, one provider completion for it, and nothing Kanban in the transcript.
      expect(ws.sent.filter(frame => frame.method === 'prompt.submit')).toHaveLength(1)
      expect(provider.completions.filter(c => c.marker === marker)).toHaveLength(1)

      const stored = messageRows(sandbox, await currentSessionId(page))

      expect(stored.some(row => row.role === 'user' && row.content.includes(marker))).toBe(true)
      expect(stored.some(row => row.role === 'assistant' && row.content.includes(`A1-${nonce}`))).toBe(true)
      expect(stored.some(row => /Kanban t_|kanban/i.test(row.content))).toBe(false)
    })

    await test.step('a reload keeps the same links, states and gaps', async () => {
      await page.reload()
      await waitForInteractive(app, page)
      await expect(originRow(page).locator('[data-kanban-origin]')).toHaveCount(1, { timeout: 60_000 })
      await expectLinkedTasks(page, ids, 'after reload')
    })

    await test.step('"worker log" opens the task’s own board drawer on the log section', async () => {
      await strip(page).getByRole('button', { name: `Open logs: Background job ${nonce}` }).click()

      const drawer = page.getByRole('dialog')

      await expect(drawer).toContainText(`Background job ${nonce}`, { timeout: 30_000 })
      await expect(drawer).toContainText(`LOG-LINE-${nonce}`)
      await shot(page, '2-canonical-log-drawer')
      await page.keyboard.press('Escape')
    })

    await test.step('A → B → A: another profile holding the same literal session id shows nothing', async () => {
      const named = rail(page).getByRole('button', { name: /^b(?: · .*)?$/ }).first()

      await named.click()
      await expect(named).toHaveAttribute('aria-pressed', 'true', { timeout: 60_000 })
      // B must be a usable, interactive profile (no onboarding overlay) before absence means anything.
      await waitForInteractive(app, page)
      await rowTitled(page, 'Other profile same id').first().click()
      await expect(page.locator('[data-slot="aui_thread-viewport"]').filter({ visible: true }).first()).toContainText(
        `hi origin-root-${nonce}`,
        { timeout: 60_000 }
      )
      await expect(composer(page)).toBeVisible()
      await expect(page.locator('[data-kanban-origin], [data-slot="kanban-origin-strip"]')).toHaveCount(0)

      // On a named profile the home pill (the rail's only home glyph) is the way back to default.
      await rail(page).locator('button:has(.codicon-home)').first().click()
      await waitForInteractive(app, page)
      await expect(originRow(page).locator('[data-kanban-origin]')).toHaveCount(1, { timeout: 60_000 })
      await originRow(page).first().click()
      await expectLinkedTasks(page, ids, 'back on A')
      await shot(page, '3-profile-a-returned')
    })
  } finally {
    await app.close().catch(() => undefined)
    await provider.close()
    sandbox.cleanup()
  }
})
