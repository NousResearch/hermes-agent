import { MOCK_REPLY } from '../../../tests-js/scripts/mock-server'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test } from './test'

const FIRST_REPLY = 'FIRST_REPLY: initial work completed'
const FOLLOWUP_REPLY = 'FOLLOWUP_REPLY: subsequent work completed'
let fixture: MockBackendFixture | null = null

async function publicLog(page: MockBackendFixture['page']) {
  return page.evaluate(() => {
    const rooms = JSON.parse(localStorage.getItem('hermes.plugin.hermes-bots.group-chats') || '{}')

    return (rooms['Programmer, Reviewer']?.log || []).map((entry: any) => ({
      from: entry.from.name, text: entry.text, thread: entry.thread
    })) as { from: string; text: string; thread: string }[]
  })
}

async function openBots(page: MockBackendFixture['page']): Promise<void> {
  const tab = page.getByRole('button', { name: 'Bots', exact: true }).or(page.getByRole('tab', { name: 'Bots', exact: true })).first()
  await tab.click()
  await expect(page.getByRole('button', { name: 'New bot or group chat' })).toBeVisible()
}

async function createAgent(page: MockBackendFixture['page'], name: string, title: string): Promise<void> {
  await page.getByRole('button', { name: 'New bot or group chat' }).click()
  await page.getByRole('menuitem', { name: 'New Bot' }).click()

  const dialog = page.getByRole('dialog', { name: 'New Bot' })
  await dialog.getByPlaceholder('inbox-triage').fill(name)
  await dialog.getByPlaceholder('Inbox Triage').fill(title)
  await dialog.getByRole('button', { name: 'Create Bot' }).click()
  await expect(dialog).toBeHidden({ timeout: 30_000 })
  await expect(page.getByRole('button', { name: new RegExp(`^${title}\\b`) }).first()).toBeVisible({ timeout: 30_000 })
}

async function createRoom(page: MockBackendFixture['page']) {
  await openBots(page)
  await createAgent(page, 'programmer', 'Programmer')
  await createAgent(page, 'reviewer', 'Reviewer')

  await page.getByRole('button', { name: 'New bot or group chat' }).click()
  await page.getByRole('menuitem', { name: 'New Group Chat' }).click()

  const dialog = page.getByRole('dialog', { name: 'New Group Chat' })

  for (const title of ['Programmer', 'Reviewer']) {
    await dialog.getByText(title, { exact: true }).locator('xpath=ancestor::label').getByRole('checkbox').click()
  }

  await dialog.getByRole('textbox', { name: 'Group name' }).fill('Programmer, Reviewer')
  await dialog.getByRole('button', { name: 'Create Group (2)' }).click()

  const groupTab = page.getByRole('tab', { name: /Programmer, Reviewer Close/ })
  const groupComposer = page.getByRole('textbox', { name: 'Message Programmer, Reviewer' }).filter({ visible: true })
  await expect(groupTab).toBeVisible({ timeout: 20_000 })
  await expect(groupTab).toHaveAttribute('aria-selected', 'true')
  await expect(groupComposer).toBeVisible()


  return groupComposer
}

test.beforeEach(async () => {
  fixture = await setupMockBackend({ mockServer: {
    holdFirstCompletionContaining: 'LANE_A_FIRST',
    replyForPrompt: prompt => prompt.includes('LANE_A_FOLLOWUP') ? FOLLOWUP_REPLY : prompt.includes('LANE_A_FIRST') ? FIRST_REPLY : MOCK_REPLY
  } })
  await waitForAppReady(fixture, 120_000)
})

test.afterEach(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('group follow-up waits for its active member and retains the reply', async () => {
  test.setTimeout(240_000)
  const page = fixture!.page
  const groupComposer = await createRoom(page)
  await groupComposer.fill('@programmer LANE_A_FIRST')
  await groupComposer.press('Enter')
  await fixture!.mock.waitForHeldCompletion()
  console.log('PRODUCTION CLOCK: first inference held; no second submit before provider release.')
  await page.screenshot({ path: '/tmp/botmode-campaign/lane-a-held.png' })
  await page.getByRole('button', { name: 'Reply in thread', exact: true }).click()
  const replyComposer = page.getByRole('textbox', { name: 'Reply in thread', exact: true })
  await replyComposer.fill('@programmer LANE_A_FOLLOWUP')
  await replyComposer.press('Enter')
  await page.waitForTimeout(3000)
  const overlapping = fixture!.mock.receivedPrompts.filter(p => p.includes('LANE_A_FOLLOWUP'))
  console.log('OVERLAPPING', overlapping)
  fixture!.mock.releaseHeldStream()
  expect(overlapping).toHaveLength(0)
  await expect.poll(() => fixture!.mock.receivedPrompts.some(p => p.includes('LANE_A_FOLLOWUP')), { timeout: 60000 }).toBe(true)
  const delivered = fixture!.mock.receivedPrompts.find(p => p.includes('LANE_A_FOLLOWUP'))!
  expect(delivered).toContain(FIRST_REPLY)
  expect(delivered).not.toContain('LANE_A_FIRST')
  expect(delivered.indexOf('LANE_A_FOLLOWUP')).toBeLessThan(delivered.indexOf(FIRST_REPLY))
  await expect(page.getByText(FIRST_REPLY, { exact: true }).filter({ visible: true })).toHaveCount(1)
  await expect(page.getByText(FOLLOWUP_REPLY, { exact: true }).filter({ visible: true })).toHaveCount(1)
  await expect(page.getByRole('button', { name: 'Stop', exact: true })).toHaveCount(0)
  const log = await publicLog(page)
  expect(log.map(({ from, text }) => [from, text])).toEqual([
    ['You', '@programmer LANE_A_FIRST'], ['You', '@programmer LANE_A_FOLLOWUP'],
    ['programmer', FIRST_REPLY], ['programmer', FOLLOWUP_REPLY]
  ])
  expect(new Set(log.map(entry => entry.thread)).size).toBe(1)
  expect(fixture!.mock.receivedPrompts.filter(p => p.includes('LANE_A_FIRST'))).toHaveLength(1)
  expect(fixture!.mock.receivedPrompts.filter(p => p.includes('LANE_A_FOLLOWUP'))).toHaveLength(1)
  console.log('PRODUCTION CLOCK: released; exact public log', JSON.stringify(log))
  await page.screenshot({ path: '/tmp/botmode-campaign/lane-a-followup-after.png' })
})

test('quiet group still harvests a late answer after sixty observation ticks', async () => {
  test.setTimeout(240_000)
  const page = fixture!.page
  const groupComposer = await createRoom(page)

  await groupComposer.fill('@programmer LANE_A_FIRST')
  await groupComposer.press('Enter')
  await fixture!.mock.waitForHeldCompletion()
  // Jump only the deadline clock, not WebSocket heartbeats or reconnect timers.
  await page.evaluate(() => {
    const now = Date.now
    Date.now = () => now() + 21 * 60_000

    const timeout = window.setTimeout.bind(window)

    ;(window as any).__harvestTicks = 0
    window.setTimeout = ((handler: TimerHandler, delay?: number, ...args: any[]) => {
      if (delay === 5000) {
        return timeout(() => {
          (window as any).__harvestTicks++

          if (typeof handler === 'function') { handler(...args) }
        }, 100)
      }

      return timeout(handler, delay, ...args)
    }) as typeof window.setTimeout
  })
  await expect.poll(() => page.evaluate(() => (window as any).__harvestTicks), { timeout: 60000 }).toBeGreaterThanOrEqual(60)
  await new Promise(resolve => setTimeout(resolve, 1500))
  console.log('Harvest ticks before release:', await page.evaluate(() => (window as any).__harvestTicks))
  console.log('Activity before release:', await page.getByRole('button', { name: /^Activity/ }).textContent())
  fixture!.mock.releaseHeldStream()
  await expect(page.getByText(FIRST_REPLY, { exact: true }).filter({ visible: true })).toHaveCount(1, { timeout: 5000 })
  await page.waitForTimeout(1000)
  expect((await publicLog(page)).filter(entry => entry.from === 'programmer').map(entry => entry.text)).toEqual([FIRST_REPLY])
  console.log('ACCELERATED DEADLINE/OBSERVATION CLOCK ONLY: exact late public log', JSON.stringify(await publicLog(page)))
  await page.screenshot({ path: '/tmp/botmode-campaign/lane-a-late-after.png' })
})

test('a rejected member turn stays visible when the room settles', async () => {
  test.setTimeout(240_000)
  const page = fixture!.page
  await page.evaluate(() => {
    const send = WebSocket.prototype.send

    WebSocket.prototype.send = function(data) {
      const frame = JSON.parse(String(data))

      if (frame.method === 'prompt.submit' && JSON.stringify(frame.params).includes('LANE_A_FAILURE')) {
        (window as any).__rejected = ((window as any).__rejected || 0) + 1
        const reject = () => this.dispatchEvent(new MessageEvent('message', { data: JSON.stringify({ jsonrpc: '2.0', id: frame.id, error: { code: 4003, message: 'Controlled member admission refusal' } }) }))

        if ((window as any).__rejected <= 3) { (window as any).__releaseRefusal = reject } else { queueMicrotask(reject) }
      } else {
        send.call(this, data)
      }
    }
  })
  const groupComposer = await createRoom(page)

  await groupComposer.fill('@programmer LANE_A_FAILURE')
  await groupComposer.press('Enter')
  await expect.poll(() => page.evaluate(() => (window as any).__rejected), { timeout: 30000 }).toBe(1)
  await page.getByRole('button', { name: 'Reply in thread', exact: true }).click()
  const replyComposer = page.getByRole('textbox', { name: 'Reply in thread', exact: true })
  await replyComposer.fill('@programmer prequeued LANE_A_FAILURE')
  await replyComposer.press('Enter')
  await groupComposer.fill('@programmer cross-thread LANE_A_FAILURE')
  await groupComposer.press('Enter')
  console.log('PRODUCTION CLOCK / CONTROLLED TRANSPORT: two sends queued before refusal released')
  await page.evaluate(() => (window as any).__releaseRefusal())
  await expect(page.getByRole('button', { name: 'Stop', exact: true })).toHaveCount(0)
  await page.waitForTimeout(2000)
  expect(await page.evaluate(() => (window as any).__rejected)).toBe(1)
  expect((await publicLog(page)).filter(entry => entry.from !== 'You')).toEqual([])
  await expect(page.getByRole('button', { name: /^Activity/ })).toContainText('Programmer hit an error')
  await page.screenshot({ path: '/tmp/botmode-campaign/lane-a-error-after.png' })

  // One epoch: A fails, B waits, the user retries A, then B and A fail.
  // The retry is explicit and after A's failure, unlike the prequeued sends above.
  await groupComposer.fill('@programmer LANE_A_FAILURE recency')
  await groupComposer.press('Enter')
  await expect.poll(() => page.evaluate(() => (window as any).__rejected)).toBe(2)
  await groupComposer.fill('@reviewer LANE_A_FAILURE recency')
  await groupComposer.press('Enter')
  await page.evaluate(() => (window as any).__releaseRefusal())
  await expect.poll(() => page.evaluate(() => (window as any).__rejected)).toBe(3)
  await groupComposer.fill('@programmer LANE_A_FAILURE explicit retry')
  await groupComposer.press('Enter')
  await page.evaluate(() => (window as any).__releaseRefusal())
  await expect.poll(() => page.evaluate(() => (window as any).__rejected), { timeout: 30000 }).toBe(4)
  await expect(page.getByRole('button', { name: 'Stop', exact: true })).toHaveCount(0)
  await expect(page.getByRole('button', { name: /^Activity/ })).toContainText('Programmer hit an error')
  console.log('CONTROLLED REFUSAL: A failed, B failed, A failed again; newest unresolved summary:', await page.getByRole('button', { name: /^Activity/ }).textContent())
})

test('Stop clears a queued follow-up and a direct mention resumes the held member', async () => {
  test.setTimeout(240_000)
  const page = fixture!.page
  const groupComposer = await createRoom(page)
  await groupComposer.fill('@programmer LANE_A_FIRST')
  await groupComposer.press('Enter')
  await fixture!.mock.waitForHeldCompletion()
  await page.getByRole('button', { name: 'Reply in thread', exact: true }).click()
  const replyComposer = page.getByRole('textbox', { name: 'Reply in thread', exact: true })
  await replyComposer.fill('@programmer LANE_A_CANCELLED')
  await replyComposer.press('Enter')
  await page.getByRole('button', { name: 'Stop', exact: true }).click()
  fixture!.mock.releaseHeldStream()
  await expect(page.getByRole('button', { name: 'Stop', exact: true })).toHaveCount(0)
  await page.waitForTimeout(2000)
  expect(fixture!.mock.receivedPrompts.some(p => p.includes('LANE_A_CANCELLED'))).toBe(false)
  console.log('Stop: queued inference count = 0; paused status:', await page.getByText(/Paused:/).allTextContents())
  await groupComposer.fill('@programmer LANE_A_RESUME')
  await groupComposer.press('Enter')
  await expect.poll(() => fixture!.mock.receivedPrompts.some(p => p.includes('LANE_A_RESUME')), { timeout: 60000 }).toBe(true)
  await expect(page.getByText(MOCK_REPLY, { exact: true }).first()).toBeVisible()
  await page.screenshot({ path: '/tmp/botmode-campaign/lane-a-stop-resume.png' })
})
