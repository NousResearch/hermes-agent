/**
 * C2 core: a refreshed newest page merged onto a long chat's live turn.
 *
 * Every settled turn re-reads the newest stored page (120 rows) and merges it
 * onto what the live stream drew. Once a chat holds more rows than one page,
 * that page starts inside an older tool turn, so its first stored row is not
 * the first row the window holds and the merge pairs rows by stored id. The
 * streamed copy of the turn that just finished has stored ids on its settled
 * reply only: the prompt and the narration sealed before each tool were drawn
 * from frames. Those rows must be recognised as the stored rows the page
 * brings (#139741: the whole turn, prompt included, rendered twice; #141771:
 * the turn's narration re-rendered under its final answer).
 *
 * The chat is seeded with wide tool turns (one call fanning out to many tool
 * rows) until it spans more than a page, then each checked turn narrates
 * before every one of three tools. Each checked turn is sampled for a while
 * after it settles, so a duplicate that a later read heals still fails.
 */

import { type ElectronApplication, expect, type Page, test } from '@playwright/test'

import {
  coreAppEnv,
  type CoreSandbox,
  createCoreSandbox,
  launchCoreApp,
  recordWebSockets,
  send,
  waitForInteractive,
  writeProviderHome,
  type WsRecorder
} from './harness'
import { gate, type ScriptedProvider, startScriptedProvider } from './provider'

const nonce = Math.random()
  .toString(36)
  .slice(2, 8)
  .replace(/[^a-z0-9]/g, 'x')
  .padEnd(4, 'q')

const U = (n: number) => `U${n}-${nonce}`
const A = (n: number) => `A${n}-${nonce}`
const N = (n: number, k: number) => `A${n}i${k}-${nonce}`
const MARKER_RE = new RegExp(`\\b[UA]\\d+(?:i\\d+)?-${nonce}\\b`, 'g')

const words = (marker: string, ...rest: string[]) => [
  `${marker} `,
  ...rest.map((w, i) => (i === rest.length - 1 ? w : `${w} `))
]

const echo = (tag: string) => ({ name: 'terminal', args: { command: `echo core-${tag}` } })

// One seed turn = prompt + SEED_ROUNDS x (narration row with a tool call + tool row) + reply.
const SEED_ROUNDS = 10
const SEED_TURNS = 6
const CHECKED_TURNS = 3
const OBSERVE_MS = 13_000
// The app's own line between a reader at the bottom and one in history
// (RUN_START_SNAP_THRESHOLD_PX in thread/list.tsx).
const OFF_BOTTOM_PX = 64

const SHOTS = process.env.HERMES_E2E_SHOT_DIR

function viewport(page: Page) {
  return page.locator('[data-slot="aui_thread-viewport"]').filter({ visible: true }).first()
}

function stopButton(page: Page) {
  return page.locator('[data-slot="composer-root"] button[aria-label="Stop"]')
}

async function settled(page: Page, ws: WsRecorder, marker: string) {
  await expect
    .poll(() => ws.events.some(e => e.type === 'message.complete' && String(e.payload?.text ?? '').includes(marker)), {
      timeout: 120_000,
      message: `message.complete for ${marker}`
    })
    .toBe(true)
  await expect(stopButton(page)).toHaveCount(0, { timeout: 30_000 })
}

async function withApp(
  label: string,
  body: (ctx: {
    app: ElectronApplication
    page: Page
    provider: ScriptedProvider
    sandbox: CoreSandbox
    ws: WsRecorder
  }) => Promise<void>
) {
  const provider = await startScriptedProvider()
  const sandbox = createCoreSandbox(label)
  writeProviderHome(sandbox.hermesHome, provider.url)
  const { app, page } = await launchCoreApp(coreAppEnv(sandbox))
  const ws = recordWebSockets(page)

  try {
    await waitForInteractive(app, page)
    await body({ app, page, provider, sandbox, ws })
  } finally {
    await app.close().catch(() => undefined)
    await provider.close()
    sandbox.cleanup()
  }
}

/**
 * How often each marker shows in the visible thread, whether they read in
 * order, and whether the thread still follows its bottom. innerText skips the
 * off-screen turns content-visibility leaves unrendered, so the marker count is
 * only meaningful while the view sits at the newest turn: nothing here scrolls,
 * and a view that stopped following is the symptom itself (a settled turn's
 * row-height jump read as a scroll-up), reported as such.
 */
async function turnShape(
  page: Page,
  markers: string[]
): Promise<{ counts: number[]; doubled: string[]; offBottom: number; ordered: boolean }> {
  const { offBottom, text } = await page.evaluate(() => {
    const el = ([...document.querySelectorAll('[data-slot="aui_thread-viewport"]')] as HTMLElement[]).find(
      node => node.getClientRects().length > 0
    )

    return {
      offBottom: el ? Math.round(el.scrollHeight - el.clientHeight - el.scrollTop) : -1,
      text: el ? el.innerText.replace(/\s+/g, ' ') : ''
    }
  })

  const counts = markers.map(marker => text.split(marker).length - 1)
  const positions = markers.map(marker => text.indexOf(marker))
  const seen = new Map<string, number>()

  for (const marker of text.match(MARKER_RE) ?? []) {
    seen.set(marker, (seen.get(marker) ?? 0) + 1)
  }

  return {
    counts,
    doubled: [...seen].filter(([, count]) => count > 1).map(([marker, count]) => `${marker}x${count}`),
    offBottom,
    ordered: positions.every((at, i) => at >= 0 && (i === 0 || at > positions[i - 1]))
  }
}

async function seedLongChat(page: Page, provider: ScriptedProvider, ws: WsRecorder) {
  for (let n = 1; n <= SEED_TURNS; n += 1) {
    provider.script(U(n), [
      ...Array.from({ length: SEED_ROUNDS }, (_, k) => ({
        text: words(N(n, k + 1), 'seeding'),
        toolCalls: [echo(`seed-${n}-${k}`)]
      })),
      { text: words(A(n), 'seeded') }
    ])
    await send(page, `${U(n)} seed`, 'Enter', ws)
    await settled(page, ws, A(n))
  }
}

/**
 * Sample the newest turn for OBSERVE_MS after it settled while the window-return
 * backstop re-reads the newest page (the app gates it to one read per 5 s; the
 * sessions.changed tick does the same on its own cadence). Returns every sample
 * where a marker was not rendered exactly once, in order.
 */
async function observeTurn(page: Page, markers: string[], label: string): Promise<string[]> {
  const bad: string[] = []
  const deadline = Date.now() + OBSERVE_MS
  let shot = false
  let nextReturn = 0
  let previousOffBottom = 0

  while (Date.now() < deadline) {
    if (Date.now() >= nextReturn) {
      nextReturn = Date.now() + 5_500
      await page.evaluate(() => document.dispatchEvent(new Event('visibilitychange')))
    }

    const shape = await turnShape(page, markers)

    // Off the bottom on two samples in a row: a follow that lags one frame
    // behind a resize is not a view that stopped following.
    const lostBottom = shape.offBottom > OFF_BOTTOM_PX && previousOffBottom > OFF_BOTTOM_PX
    previousOffBottom = shape.offBottom

    if (shape.counts.some(count => count !== 1) || !shape.ordered || shape.doubled.length || lostBottom) {
      bad.push(
        `${markers.map((m, i) => `${m}x${shape.counts[i]}`).join(' ')} ordered=${shape.ordered} doubled=${shape.doubled.join(',')} offBottom=${shape.offBottom}px`
      )

      if (SHOTS && !shot) {
        shot = true
        await page.screenshot({ path: `${SHOTS}/${label}-symptom.png` })
      }
    }

    await page.waitForTimeout(200)
  }

  if (SHOTS) {
    await page.screenshot({ path: `${SHOTS}/${label}-settled.png` })

    const probe = await page.evaluate(async () => {
      const sid = decodeURIComponent(location.hash.replace(/^#\/?/, '').split('?')[0] ?? '')

      const latest = await (window as any).hermesDesktop.api({
        path: `/api/sessions/${sid}/messages?order=latest&limit=120&include_compacted=true`
      })

      const rows = latest?.messages ?? []

      return {
        head: rows.slice(0, 2).map((m: any) => `${m.id}:${m.role}`),
        bubbles: [
          ...document.querySelectorAll('[data-slot="aui_user-message-root"], [data-slot="aui_assistant-message-root"]')
        ]
          .slice(-14)
          .map(el => (el as HTMLElement).innerText.replace(/\s+/g, ' ').slice(0, 80))
      }
    })

    console.log(`[probe ${label}] ${JSON.stringify(probe)}`)
  }

  return bad
}

test('a long chat keeps each finished tool turn once, in order, after the newest page merges', async () => {
  test.setTimeout(900_000)

  await withApp('tail-resync', async ({ page, provider, ws }) => {
    await test.step(`seed ${SEED_TURNS} wide tool turns (more than one page of rows)`, () =>
      seedLongChat(page, provider, ws))

    // Every checked turn so far must keep rendering once, in order, under later merges.
    const shown: string[] = []

    for (let c = 0; c < CHECKED_TURNS; c += 1) {
      const n = SEED_TURNS + 1 + c

      await test.step(`turn ${n}: narration before each of three tools, then the answer`, async () => {
        provider.script(U(n), [
          { text: words(N(n, 1), 'first', 'look'), toolCalls: [echo(`n${n}-a`)] },
          { text: words(N(n, 2), 'second', 'look'), toolCalls: [echo(`n${n}-b`)] },
          { text: words(N(n, 3), 'third', 'look'), toolCalls: [echo(`n${n}-c`)] },
          { text: words(A(n), 'final', 'answer') }
        ])
        await send(page, `${U(n)} narrate tools`, 'Enter', ws)
        await settled(page, ws, A(n))

        shown.push(U(n), N(n, 1), N(n, 2), N(n, 3), A(n))
        const bad = await observeTurn(page, shown, `narrated-${n}`)

        expect(bad.slice(0, 5), `turn ${n} rendered once and in order after the page merge`).toEqual([])
      })
    }
  })
})

test('a long chat keeps a stopped turn and its resend once after the newest page merges', async () => {
  test.setTimeout(900_000)

  await withApp('tail-resync-stop', async ({ page, provider, ws }) => {
    await test.step(`seed ${SEED_TURNS} wide tool turns (more than one page of rows)`, () =>
      seedLongChat(page, provider, ws))

    const shown: string[] = []

    for (let c = 0; c < CHECKED_TURNS; c += 1) {
      const n = SEED_TURNS + 1 + 2 * c

      await test.step(`turn ${n}: stopped mid-answer, then turn ${n + 1} resends`, async () => {
        const hold = gate()
        provider.script(U(n), [
          { text: words(N(n, 1), 'looking'), toolCalls: [echo(`s${n}`)] },
          { text: words(A(n), 'partial', 'answer', 'that', 'never', 'ends'), holdAfterFirstChunk: hold }
        ])
        await send(page, `${U(n)} stop me`, 'Enter', ws)
        await provider.streamStarted(U(n), 1)
        await expect(viewport(page)).toContainText(A(n))
        await stopButton(page).filter({ visible: true }).first().click()
        hold.open()
        await expect(stopButton(page)).toHaveCount(0, { timeout: 30_000 })

        shown.push(U(n), N(n, 1), A(n))
        const stopped = await observeTurn(page, shown, `stopped-${n}`)

        expect(stopped.slice(0, 5), `stopped turn ${n} rendered once after the page merge`).toEqual([])

        provider.script(U(n + 1), [
          { text: words(N(n + 1, 1), 'again'), toolCalls: [echo(`r${n}`)] },
          { text: words(A(n + 1), 'resent', 'answer') }
        ])
        await send(page, `${U(n + 1)} resend`, 'Enter', ws)
        await settled(page, ws, A(n + 1))

        shown.push(U(n + 1), N(n + 1, 1), A(n + 1))
        const resent = await observeTurn(page, shown, `resent-${n + 1}`)

        expect(resent.slice(0, 5), `resent turn ${n + 1} rendered once after the page merge`).toEqual([])
      })
    }
  })
})

test('a long chat keeps a steered tool turn once, in order, after the newest page merges', async () => {
  test.setTimeout(900_000)

  await withApp('tail-resync-steer', async ({ page, provider, ws }) => {
    await test.step(`seed ${SEED_TURNS} wide tool turns (more than one page of rows)`, () =>
      seedLongChat(page, provider, ws))

    const shown: string[] = []

    for (let c = 0; c < CHECKED_TURNS; c += 1) {
      const n = SEED_TURNS + 1 + 2 * c

      await test.step(`turn ${n}: narration and a tool, steered by turn ${n + 1} during the answer`, async () => {
        const hold = gate()
        provider.script(U(n), [
          { text: words(N(n, 1), 'first', 'look'), toolCalls: [echo(`t${n}-a`)] },
          { text: words(N(n, 2), 'second', 'look'), toolCalls: [echo(`t${n}-b`)] },
          { text: words(A(n), 'answer', 'cut', 'by', 'steer'), holdAfterFirstChunk: hold }
        ])
        provider.script(U(n + 1), [
          { text: words(N(n + 1, 1), 'steered', 'look'), toolCalls: [echo(`t${n}-c`)] },
          { text: words(A(n + 1), 'steered', 'answer') }
        ])
        await send(page, `${U(n)} narrate then steer`, 'Enter', ws)
        await provider.streamStarted(U(n), 2)
        await expect(viewport(page)).toContainText(A(n))
        await send(page, `${U(n + 1)} change course`, 'Enter', ws)
        hold.open()
        await settled(page, ws, A(n + 1))

        shown.push(U(n), N(n, 1), N(n, 2), U(n + 1), N(n + 1, 1), A(n + 1))
        const bad = await observeTurn(page, shown, `steered-${n}`)

        expect(bad.slice(0, 5), `steered turn ${n} rendered once and in order after the page merge`).toEqual([])
      })
    }
  })
})
