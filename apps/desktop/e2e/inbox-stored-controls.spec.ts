/**
 * Stored-session controls — the Action Center acting on a session that is NOT running.
 *
 * Pause/resume are persisted-state writes, so the panel must keep them usable without a
 * live runtime: the click travels as `session.control` with only the stored session key,
 * the gateway applies it to the session's saved state, and the panel reflects the change.
 *
 * This spec seeds a stored session (row + goal/loop/heartbeat metas) directly into the
 * sandbox database — it is never opened, so it has no runtime — then drives the panel:
 * every Pause flips to Resume, the state is asserted in the database, and every Resume
 * flips back.
 *
 * Output: .inbox-work/live-approval-evidence/ (shared with the other live specs).
 */

import { execSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

import { MOCK_REPLY } from '../../../tests-js/scripts/mock-server'
import { setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test, type Page } from './test'

const OUTPUT = path.resolve(import.meta.dirname, '../../../.inbox-work/live-approval-evidence')

const SEED_SOURCE = [
  'import json, sqlite3, sys, time',
  '',
  'db_path, key = sys.argv[1], sys.argv[2]',
  'now = time.time()',
  'con = sqlite3.connect(db_path)',
  'con.execute(',
  '    "INSERT OR REPLACE INTO sessions (id, session_key, source, started_at, title, last_activity_at) VALUES (?, ?, ?, ?, ?, ?)",',
  '    (key, key, "desktop", now, "Stored automation session", now),',
  ')',
  'metas = {',
  '    "goal:" + key: {',
  '        "goal": "Ship from the stored session",',
  '        "status": "active",',
  '        "turns_used": 2,',
  '        "max_turns": 12,',
  '        "created_at": now,',
  '        "last_turn_at": now,',
  '    },',
  '    "loop:" + key: {',
  '        "prompt": "Stored loop check",',
  '        "status": "active",',
  '        "mode": "interval",',
  '        "interval_seconds": 900,',
  '        "current_delay": 900,',
  '        "created_at": now,',
  '        "next_due_at": now + 900,',
  '    },',
  '    "heartbeat:" + key: {',
  '        "prompt": "Stored heartbeat check",',
  '        "interval_seconds": 600,',
  '        "status": "active",',
  '        "created_at": now,',
  '        "last_fired_at": 0.0,',
  '        "fire_count": 0,',
  '    },',
  '}',
  'for meta_key, value in metas.items():',
  '    con.execute("INSERT OR REPLACE INTO state_meta (key, value) VALUES (?, ?)", (meta_key, json.dumps(value)))',
  'con.commit()',
  'con.close()',
  'print("seeded", key)',
  ''
].join('\n')

const CHECK_SOURCE = [
  'import json, sqlite3, sys',
  '',
  'db_path, key, *pairs = sys.argv[1:]',
  'con = sqlite3.connect(db_path)',
  'ok = True',
  'for i in range(0, len(pairs), 2):',
  '    kind, want = pairs[i], pairs[i + 1]',
  '    row = con.execute("SELECT value FROM state_meta WHERE key = ?", (kind + ":" + key,)).fetchone()',
  '    status = (json.loads(row[0]).get("status") if row else None)',
  '    print(kind, "=", status)',
  '    if status != want:',
  '        ok = False',
  'sys.exit(0 if ok else 1)',
  ''
].join('\n')

function pythonBinary(): string {
  const python = process.env.HERMES_DESKTOP_PYTHON

  if (!python) {
    throw new Error('HERMES_DESKTOP_PYTHON must be set to seed and check the sandbox database')
  }

  return python
}

function runPython(hermesHome: string, source: string, fileName: string, args: string[]): string {
  const script = path.join(hermesHome, fileName)

  fs.writeFileSync(script, source, 'utf8')

  return execSync(`"${pythonBinary()}" "${script}" ${args.map(arg => `"${arg}"`).join(' ')}`, {
    encoding: 'utf8'
  })
}

async function shot(page: Page, name: string): Promise<void> {
  fs.mkdirSync(OUTPUT, { recursive: true })
  await page.screenshot({ path: path.join(OUTPUT, name) })
}

test('a stored session pauses and resumes every automation from the panel', async () => {
  test.setTimeout(420_000)

  const fixture = await setupMockBackend()

  try {
    await waitForAppReady(fixture, 120_000)
    const page = fixture.page
    const key = `stored-${Date.now().toString(36)}`
    const dbPath = path.join(fixture.sandbox.hermesHome, 'state.db')

    runPython(fixture.sandbox.hermesHome, SEED_SOURCE, 'seed-stored-controls.py', [dbPath, key])

    // Tap outgoing session.control frames: the requests the panel actually sends are the witness.
    await page.evaluate(() => {
      const send = WebSocket.prototype.send

      ;(window as any).__sessionControlSends = [] as string[]

      WebSocket.prototype.send = function (data) {
        try {
          const frame = JSON.parse(String(data))

          if (frame?.method === 'session.control') {
            ;(window as any).__sessionControlSends.push(JSON.stringify(frame.params))
          }
        } catch {
          /* non-JSON frame */
        }

        return send.call(this, data)
      }
    })

    await page.keyboard.press('Escape')
    await page.waitForTimeout(200)
    await page.getByRole('button', { name: /^(Action Center|Action Center — \d+ need attention)$/ }).first().click()
    await expect(page.getByRole('heading', { name: 'Action Center' })).toBeVisible()

    const panel = page.locator('[data-overlay-surface]')

    const row = page.locator('[data-panel-row]').filter({ hasText: 'Stored automation session' }).first()

    await expect(row).toBeVisible({ timeout: 15_000 })
    await row.click()

    // No live runtime: the panel says so, and every control stays usable.
    const hint = panel.getByText("session isn't running — applies to stored state")

    await expect(panel.getByRole('button', { name: 'Pause goal', exact: true })).toBeVisible({ timeout: 15_000 })
    await expect(panel.getByRole('button', { name: 'Pause loop', exact: true })).toBeVisible({ timeout: 15_000 })
    await expect(panel.getByRole('button', { name: 'Pause heartbeat', exact: true })).toBeVisible({ timeout: 15_000 })
    await expect(hint).toHaveCount(3)
    await shot(page, 'stored-1-controls-usable.png')

    // Pause all three; each button flips only after the gateway confirmed the write.
    await panel.getByRole('button', { name: 'Pause goal', exact: true }).click()
    await expect(panel.getByRole('button', { name: 'Resume goal', exact: true })).toBeVisible({ timeout: 15_000 })

    try {
      await panel.getByRole('button', { name: 'Pause loop', exact: true }).click()
      await expect(panel.getByRole('button', { name: 'Resume loop', exact: true })).toBeVisible({ timeout: 15_000 })
    } catch (error) {
      // Diagnose: which requests left the panel, and what did the panel show at this point.
      console.log('SENDS:', await page.evaluate(() => (window as any).__sessionControlSends))
      console.log('PANEL TEXT:', (await panel.innerText()).slice(0, 1200))
      await shot(page, 'stored-failure-loop.png')
      throw error
    }

    await panel.getByRole('button', { name: 'Pause heartbeat', exact: true }).click()
    await expect(panel.getByRole('button', { name: 'Resume heartbeat', exact: true })).toBeVisible({ timeout: 15_000 })
    await shot(page, 'stored-2-all-paused.png')

    // The panel is not the witness — the database is.
    runPython(fixture.sandbox.hermesHome, CHECK_SOURCE, 'check-stored-controls.py', [
      dbPath, key, 'goal', 'paused', 'loop', 'paused', 'heartbeat', 'paused'
    ])

    // And every one of those clicks must have travelled as a stored-key session.control send.
    const sends = await page.evaluate(() => ((window as any).__sessionControlSends as string[]).map(s => JSON.parse(s)))
    const actions = sends.map(send => send.action)

    for (const action of ['goal.pause', 'loop.pause', 'heartbeat.pause']) {
      expect(actions).toContain(action)
      const frame = sends.find(send => send.action === action)

      expect(frame.session_key).toBe(key)
      expect(frame.session_id ?? '').toBe('')
    }

    // And back.
    await panel.getByRole('button', { name: 'Resume goal', exact: true }).click()
    await expect(panel.getByRole('button', { name: 'Pause goal', exact: true })).toBeVisible({ timeout: 15_000 })
    await panel.getByRole('button', { name: 'Resume loop', exact: true }).click()
    await expect(panel.getByRole('button', { name: 'Pause loop', exact: true })).toBeVisible({ timeout: 15_000 })
    await panel.getByRole('button', { name: 'Resume heartbeat', exact: true }).click()
    await expect(panel.getByRole('button', { name: 'Pause heartbeat', exact: true })).toBeVisible({ timeout: 15_000 })
    await shot(page, 'stored-3-all-resumed.png')

    runPython(fixture.sandbox.hermesHome, CHECK_SOURCE, 'check-stored-controls.py', [
      dbPath, key, 'goal', 'active', 'loop', 'active', 'heartbeat', 'active'
    ])

    // The seeded session never ran: no turn was ever requested for it.
    expect(fixture.mock.receivedPrompts.some(prompt => prompt.includes('Stored loop check'))).toBe(false)
  } finally {
    await fixture.cleanup()
  }
})
