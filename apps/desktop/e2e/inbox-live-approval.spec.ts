/**
 * Live approval loop — real gateway, real terminal tool, real Agent Inbox.
 *
 * Mock inference (scripted tool call) + `approvals: mode: "manual"`: the mock asks for
 * `rm -rf` through the REAL terminal tool, the real backend parks the turn on an approval,
 * and this spec drives the REAL Agent Inbox panel: the request must surface under Needs
 * attention and be answerable in place. The expiry case additionally proves that a request
 * that dies unanswered is still visible afterwards, with a Redo that re-raises it.
 *
 * Exactly ONE Electron app runs per test: `withApp` creates it, closes it, and copies the
 * sandbox's own approval lifecycle log either way. Never share a sandbox between tests and
 * never leave an app behind.
 *
 * Output: .inbox-work/live-approval-evidence/
 */

import fs from 'node:fs'
import path from 'node:path'
import { APPROVAL_COMMAND_TRIGGER, MOCK_REPLY } from '../../../tests-js/scripts/mock-server'
import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test, type Page } from './test'

const OUTPUT = path.resolve(import.meta.dirname, '../../../.inbox-work/live-approval-evidence')

const MANUAL_APPROVALS = 'approvals:\n  mode: "manual"\n'
const SHORT_TIMEOUT_APPROVALS = 'approvals:\n  mode: "manual"\n  timeout: 20\n'

test.describe.configure({ mode: 'serial' })

/**
 * One app per test. Copies the sandbox's agent/gui log before cleanup — the approval
 * lifecycle lines (approval.wait start/end, approval.resolve, approval.withdraw) are the
 * only durable trace of a request that ends without a click.
 */
async function withApp(
  name: string,
  extraConfig: string,
  body: (fixture: MockBackendFixture, page: Page) => Promise<void>
): Promise<void> {
  fs.mkdirSync(OUTPUT, { recursive: true })
  const fixture = await setupMockBackend({ extraConfig })
  const tag = name.replace(/[^a-z0-9]+/gi, '-').slice(0, 48)

  try {
    await waitForAppReady(fixture, 120_000)
    await body(fixture, fixture.page)
  } finally {
    try {
      const logs = path.join(fixture.sandbox.hermesHome, 'logs')

      for (const log of ['agent.log', 'gui.log']) {
        const src = path.join(logs, log)

        if (fs.existsSync(src)) {
          fs.copyFileSync(src, path.join(OUTPUT, `${tag}-${log}`))
        }
      }
    } catch {
      /* diagnostics only */
    }

    await fixture.cleanup()
  }
}

async function shot(page: Page, name: string): Promise<void> {
  await page.screenshot({ path: path.join(OUTPUT, name) })
}

async function installRespondTap(page: Page): Promise<void> {
  await page.evaluate(() => {
    const send = WebSocket.prototype.send

    ;(window as any).__approvalResponds = [] as string[]

    WebSocket.prototype.send = function (data) {
      try {
        const frame = JSON.parse(String(data))

        if (frame?.method === 'approval.respond') {
          ;(window as any).__approvalResponds.push(JSON.stringify(frame.params))
        }
      } catch {
        /* non-JSON frame */
      }

      return send.call(this, data)
    }
  })
}

async function sendPrompt(page: Page, text: string): Promise<void> {
  const composer = page.locator('[contenteditable="true"]').first()

  await composer.waitFor({ state: 'visible', timeout: 20_000 })
  await composer.click()
  await composer.type(text, { delay: 5 })
  await composer.press('Enter')
}

async function openInbox(page: Page): Promise<void> {
  await page.keyboard.press('Escape')
  await page.waitForTimeout(200)
  await page.getByRole('button', { name: /inbox/i }).first().click()
  await expect(page.getByRole('heading', { name: 'Agent Inbox' })).toBeVisible()
}

test('a pending approval surfaces in the inbox and can be approved there', async () => {
  test.setTimeout(300_000)

  await withApp('approve-loop', MANUAL_APPROVALS, async (_fixture, page) => {
    await installRespondTap(page)
    await sendPrompt(page, `Run the maintenance probe now. ${APPROVAL_COMMAND_TRIGGER}`)
    await openInbox(page)

    const row = page.locator('[data-panel-row]').first()
    const approveOnce = page.getByRole('button', { name: 'Approve once', exact: true })

    // The panel polls every 15s; the contract is "within one poll".
    await expect(row).toBeVisible({ timeout: 15_000 })
    await shot(page, 'live-1-list.png')

    await page.getByRole('button', { name: /needs attention/i }).first().click()
    await expect(row).toBeVisible({ timeout: 15_000 })

    await row.click()
    await expect(approveOnce).toBeVisible({ timeout: 15_000 })
    await shot(page, 'live-3-expanded-controls.png')

    await approveOnce.click()
    await expect.poll(() => page.evaluate(() => (window as any).__approvalResponds.length), {
      timeout: 15_000,
    }).toBe(1)
    await shot(page, 'live-4-answered.png')

    // The blocked command resumes: the turn finishes with the canned reply.
    await page.keyboard.press('Escape')
    await expect(page.getByText(MOCK_REPLY, { exact: true }).first()).toBeVisible({ timeout: 120_000 })
    await shot(page, 'live-5-command-ran.png')
  })
})

test('a pending approval can be denied in the inbox and the command never runs', async () => {
  test.setTimeout(300_000)

  await withApp('deny-loop', MANUAL_APPROVALS, async (_fixture, page) => {
    await installRespondTap(page)
    await sendPrompt(page, `Run the maintenance probe now. ${APPROVAL_COMMAND_TRIGGER}`)
    await openInbox(page)

    const row = page.locator('[data-panel-row]').first()

    await expect(row).toBeVisible({ timeout: 15_000 })
    await row.click()

    const deny = page.getByRole('button', { name: 'Deny', exact: true })

    await expect(deny).toBeVisible({ timeout: 15_000 })
    await deny.click()
    await expect.poll(() => page.evaluate(() => (window as any).__approvalResponds.length), {
      timeout: 15_000,
    }).toBe(1)
    await shot(page, 'live-6-denied.png')

    await page.keyboard.press('Escape')
    await expect(page.getByText(MOCK_REPLY, { exact: true }).first()).toBeVisible({ timeout: 120_000 })
  })
})

test('a pending approval stays actionable while it waits (no silent deny)', async () => {
  test.setTimeout(600_000)

  await withApp('sit-unanswered', MANUAL_APPROVALS, async (_fixture, page) => {
    await installRespondTap(page)
    await sendPrompt(page, `Run the maintenance probe now. ${APPROVAL_COMMAND_TRIGGER}`)
    await openInbox(page)

    const row = page.locator('[data-panel-row]').first()
    const approveOnce = page.getByRole('button', { name: 'Approve once', exact: true })

    await expect(row).toBeVisible({ timeout: 15_000 })
    await row.click()
    await expect(approveOnce).toBeVisible({ timeout: 15_000 })

    // Sit untouched for longer than the live failure (159s, "denied by user" with no
    // click). Nothing may resolve it: no client answer, no backend resolution.
    await page.waitForTimeout(200_000)

    expect(await page.evaluate(() => (window as any).__approvalResponds)).toHaveLength(0)
    await expect(approveOnce).toBeVisible()
    await shot(page, 'live-7-still-pending-after-200s.png')

    await approveOnce.click()
    await expect.poll(() => page.evaluate(() => (window as any).__approvalResponds.length), {
      timeout: 15_000,
    }).toBe(1)
    await page.keyboard.press('Escape')
    await expect(page.getByText(MOCK_REPLY, { exact: true }).first()).toBeVisible({ timeout: 120_000 })
  })
})

test('an expired request persists in the inbox and the redo re-raises it', async () => {
  test.setTimeout(480_000)

  // Short approval timeout: the request must EXPIRE inside the test.
  await withApp('expiry-redo', SHORT_TIMEOUT_APPROVALS, async (_fixture, page) => {
    await installRespondTap(page)
    await sendPrompt(page, `Run the maintenance probe now. ${APPROVAL_COMMAND_TRIGGER}`)
    await openInbox(page)

    const row = page.locator('[data-panel-row]').first()
    const approveOnce = page.getByRole('button', { name: 'Approve once', exact: true })

    // While pending: answerable.
    await expect(row).toBeVisible({ timeout: 15_000 })
    await row.click()
    await expect(approveOnce).toBeVisible({ timeout: 15_000 })

    // Let it expire untouched. The panel re-reads the row's request state on its poll, so the
    // live controls give way to the persisted record: what it was for, and a Redo.
    await expect(approveOnce).toHaveCount(0, { timeout: 90_000 })
    await expect(page.getByText(/Expired .* timed out without an answer/)).toBeVisible({ timeout: 30_000 })
    await shot(page, 'live-8-expired-persisted.png')

    // Redo re-raises: a fresh approval surfaces, and approving it runs the command.
    await page.getByRole('button', { name: 'Redo', exact: true }).click()
    await expect(approveOnce).toBeVisible({ timeout: 90_000 })
    await shot(page, 'live-9-redo-raised.png')

    await approveOnce.click()
    await expect.poll(() => page.evaluate(() => (window as any).__approvalResponds.length), {
      timeout: 15_000,
    }).toBe(1)
    await page.keyboard.press('Escape')
    await expect(page.getByText(MOCK_REPLY, { exact: true }).first()).toBeVisible({ timeout: 120_000 })
    await shot(page, 'live-10-redo-approved-command-ran.png')
  })
})
