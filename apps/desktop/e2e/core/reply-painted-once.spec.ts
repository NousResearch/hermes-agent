/**
 * C2 core: a reply is painted ONCE inside its bubble, whatever tool round
 * the turn ends on.
 *
 * Report: "Hermes Desktop sometimes duplicates the same agent output twice
 * consecutively within a single message bubble, with no line break or UI
 * elements between them."
 *
 * The backend has one path that ends a turn on text the user has already seen.
 * The model answers beside a housekeeping tool call (memory / todo_list /
 * skill_manage / session_search), then sends an empty follow-up. The agent
 * then reuses that answer as the final reply (turn_empty_response.py,
 * `fallback_prior_turn_content`). The answer was streamed BEFORE the tool, so
 * the final names text that sits ahead of the bubble's last tool row. A
 * renderer that bounds "the current response" at the last tool row appends
 * it again. A silent tool (todo_list) draws no row between the copies, which
 * is the reported "twice, back to back".
 *
 * Every step runs the real Electron app against a real `hermes serve`; only
 * the LLM is scripted (provider.ts). The transcript oracle checks each
 * marker is rendered exactly once, live and after a reload. An in-page
 * sampler fails on any transient double paint. A bubble-scoped check also
 * names the exact symptom (one bubble holding the reply twice).
 *
 * The scenarios:
 * - answer + housekeeping tool + empty follow-up, with message.interim
 *   sealing on (the default) and off (display.interim_assistant_messages:
 *   false: no seal, so the deltas and the final land in ONE bubble). The
 *   silent `todo_list` and the visible `memory` row are both covered;
 * - a substantive tool round first, then the same ending (several rounds in
 *   one bubble, the shape a last-tool-only bound misses);
 * - control: a different final after a tool keeps both responses.
 */

import { type ElectronApplication, expect, type Page, test } from '@playwright/test'

import {
  coreAppEnv,
  type CoreSandbox,
  createCoreSandbox,
  currentSessionId,
  launchCoreApp,
  recordWebSockets,
  send,
  waitForInteractive,
  writeProviderHome,
  type WsRecorder
} from './harness'
import { assertTranscriptOracle, installDuplicateSampler, type OracleTarget } from './oracle'
import { type ScriptedProvider, startScriptedProvider, type Step } from './provider'

const nonce = Math.random()
  .toString(36)
  .slice(2, 8)
  .replace(/[^a-z0-9]/g, 'x')
  .padEnd(4, 'q')

const U = (n: number) => `U${n}-${nonce}`
const A = (n: number) => `A${n}-${nonce}`
const AI = (n: number) => `A${n}i-${nonce}`

const words = (marker: string, ...rest: string[]) => [
  `${marker} `,
  ...rest.map((w, i) => (i === rest.length - 1 ? w : `${w} `))
]

const todo = (tag: string) => ({
  name: 'todo_list',
  args: { todos: [{ id: `core-${tag}`, content: `core ${tag}`, status: 'completed' }] }
})

const memory = (tag: string) => ({
  name: 'memory',
  args: { action: 'add', target: 'memory', content: `core e2e ${tag} ${nonce}` }
})

const echo = (tag: string) => ({ name: 'terminal', args: { command: `echo core-${tag}` } })

// interim_assistant_messages toggles the live seal; review off keeps the
// background memory fork from adding rows that are not part of the turn;
// tool search off keeps todo_list directly callable (it is deferred otherwise).
const config = (interim: boolean) => `display:
  interim_assistant_messages: ${interim}
memory:
  nudge_interval: 0
tools:
  tool_search:
    enabled: off
`

function viewport(page: Page) {
  return page.locator('[data-slot="aui_thread-viewport"]').filter({ visible: true }).first()
}

/** The turn as the wire and the provider saw it, for failure messages. */
function wireTail(ws: WsRecorder, provider?: ScriptedProvider): string {
  const frames = ws.events
    .filter(e => /^(message|tool)\./.test(e.type))
    .slice(-14)
    .map(e => `${e.type} ${JSON.stringify(e.payload?.text ?? e.payload?.name ?? '').slice(0, 80)}`)

  const calls = (provider?.completions ?? [])
    .slice(-6)
    .map(
      c =>
        `${c.marker}#${c.step} finished=${c.finished} tools=${c.toolCalls.join(',')} text=${JSON.stringify(c.sentText).slice(0, 60)}`
    )

  return `wire:\n  ${frames.join('\n  ')}\nprovider:\n  ${calls.join('\n  ')}`
}

async function settled(page: Page, ws: WsRecorder, marker: string, provider?: ScriptedProvider) {
  try {
    await expect
      .poll(
        () => ws.events.some(e => e.type === 'message.complete' && String(e.payload?.text ?? '').includes(marker)),
        {
          timeout: 120_000,
          message: `message.complete for ${marker}`
        }
      )
      .toBe(true)
  } catch (error) {
    throw new Error(`${(error as Error).message}\n${wireTail(ws, provider)}`)
  }

  await expect(page.locator('[data-slot="composer-root"] button[aria-label="Stop"]')).toHaveCount(0, {
    timeout: 30_000
  })
}

/** Assistant bubbles that contain `marker`, with how often each holds it. */
async function bubblesHolding(page: Page, marker: string): Promise<{ count: number; text: string }[]> {
  return page.evaluate(m => {
    const re = new RegExp(`\\b${m}\\b`, 'g')

    return (
      [
        ...document.querySelectorAll('[data-slot="aui_thread-viewport"] [data-slot="aui_assistant-message-root"]')
      ] as HTMLElement[]
    )
      .filter(el => el.getClientRects().length > 0 && el.innerText.includes(m))
      .map(el => ({
        count: (el.innerText.match(re) ?? []).length,
        text: el.innerText.replace(/\s+/g, ' ').slice(0, 300)
      }))
  }, marker)
}

async function expectPaintedOnce(page: Page, marker: string, label: string) {
  await expect
    .poll(() => bubblesHolding(page, marker), { timeout: 30_000, message: `${label}: ${marker} in one bubble, once` })
    .toEqual([expect.objectContaining({ count: 1 })])
}

async function withApp(
  label: string,
  interim: boolean,
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
  writeProviderHome(sandbox.hermesHome, provider.url, config(interim))
  const { app, page } = await launchCoreApp(coreAppEnv(sandbox))
  const ws = recordWebSockets(page)

  try {
    await waitForInteractive(app, page)
    await installDuplicateSampler(page)
    await body({ app, page, provider, sandbox, ws })
  } finally {
    await app.close().catch(() => undefined)
    await provider.close()
    sandbox.cleanup()
  }
}

interface Turn {
  n: number
  label: string
  steps: Step[]
  /** Markers whose text must be on screen exactly once, in one bubble. */
  once: string[]
}

const TURNS: Turn[] = [
  {
    n: 1,
    label: 'answer beside a silent todo_list, empty follow-up',
    steps: [{ text: words(A(1), 'answered', 'beside', 'a', 'todo'), toolCalls: [todo('t1')] }, { text: [] }],
    once: [A(1)]
  },
  {
    n: 2,
    label: 'answer beside memory, empty follow-up',
    steps: [{ text: words(A(2), 'answered', 'beside', 'memory'), toolCalls: [memory('t2')] }, { text: [] }],
    once: [A(2)]
  },
  {
    n: 3,
    label: 'a substantive tool round, then the answer beside todo_list',
    steps: [
      { text: words(AI(3), 'looking', 'first'), toolCalls: [echo('t3')] },
      { text: words(A(3), 'answered', 'after', 'looking'), toolCalls: [todo('t3')] },
      { text: [] }
    ],
    once: [AI(3), A(3)]
  },
  {
    n: 4,
    label: 'control: a different final after a tool',
    steps: [
      { text: words(AI(4), 'checking'), toolCalls: [echo('t4')] },
      { text: words(A(4), 'the', 'real', 'answer') }
    ],
    once: [AI(4), A(4)]
  }
]

for (const interim of [true, false]) {
  test(`a reply already streamed before the last tool renders once (interim seal ${interim ? 'on' : 'off'})`, async () => {
    await withApp(`reply-once-${interim ? 'seal' : 'noseal'}`, interim, async ({ app, page, provider, ws }) => {
      const session: OracleTarget = { sessionId: '', expectUserMarkers: [] }

      for (const turn of TURNS) {
        await test.step(turn.label, async () => {
          provider.script(U(turn.n), turn.steps)
          await send(page, `${U(turn.n)} ${turn.label}`, 'Enter', ws)
          await settled(page, ws, turn.once.at(-1)!, provider)
          session.sessionId ||= await currentSessionId(page)
          session.expectUserMarkers.push(U(turn.n))

          for (const marker of turn.once) {
            await expectPaintedOnce(page, marker, `live: ${turn.label}`)
          }

          await assertTranscriptOracle(page, ws, provider, session, `live: ${turn.label}`)
        })
      }

      await test.step('reload: history renders every reply once', async () => {
        await page.reload()
        await waitForInteractive(app, page)
        await expect(viewport(page)).toContainText(A(4), { timeout: 60_000 })
        await installDuplicateSampler(page)

        for (const turn of TURNS) {
          for (const marker of turn.once) {
            await expectPaintedOnce(page, marker, `reload: ${turn.label}`)
          }
        }

        await assertTranscriptOracle(page, ws, provider, session, 'after reload')
      })
    })
  })
}
