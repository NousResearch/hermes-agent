/**
 * #121088 — an acknowledged Desktop prompt is appended a second time, after its
 * own reply, when refreshed history carries synthetic user-role rows written by
 * an in-place compaction inside that prompt's turn.
 *
 * Mechanism (apps/desktop/src/app/session/hooks/use-session-actions/utils.ts ::
 * preserveLocalPendingTurnMessages): an optimistic prompt without a row-id
 * receipt survives a refresh unless (a) its text equals `latestAuthoritativeUser`
 * — the LAST user-role row of the refreshed tail — or (b) the user row at the
 * same role ordinal pairs with it. Real processes produce both misses here:
 *
 *   U3 turn:  tool round (todo_list) pushes the request over threshold_tokens
 *             -> in-place compaction #1 appends the todo-snapshot user row S1
 *             -> the resumed stream is still running when …
 *   U4:       … the user redirects (busy submit = correction, optimistic row, no
 *             row-id receipt) -> tool round -> in-place compaction #2 appends
 *             snapshot S2 after the committed U4 -> final reply A6.
 *
 * The live view never shows S1/S2 (not streamed). On the next refresh the
 * stored history is [.., U3, A3i, S1, A5, U4, A4i, S2, A6]: S1 shifts U4's role
 * ordinal onto S1 (miss b) and S2 is the latest user row (miss a), so the
 * optimistic U4 is re-appended after A6.
 *
 * Compaction threshold is calibrated from a first throwaway session's real
 * request size, so the config (written before the scenario session's agent is
 * built) crosses threshold exactly at the two tool rounds regardless of how big
 * the system prompt / tool schemas are on this commit.
 */
import fs from 'node:fs'
import path from 'node:path'
import { DatabaseSync } from 'node:sqlite'

import { expect, type Page, test } from '@playwright/test'

import {
  coreAppEnv,
  createCoreSandbox,
  currentSessionId,
  launchCoreApp,
  recordWebSockets,
  renderedTranscript,
  send,
  waitForInteractive,
  writeProviderHome
} from './harness'
import { expectNoSymptom } from './known'
import { installDuplicateSampler } from './oracle'
import { gate, startScriptedProvider } from './provider'

const KNOWN: Record<string, string | undefined> = {
  dupPrompt:
    '#121088 acknowledged redirect prompt re-appended after its reply when in-place compaction wrote synthetic user rows inside its turn'
}

const nonce = Math.random()
  .toString(36)
  .slice(2, 8)
  .replace(/[^a-z0-9]/g, 'x')
  .padEnd(4, 'q')

const U = (n: number) => `U${n}-${nonce}`
const A = (n: number) => `A${n}-${nonce}`
const SNAPSHOT = '[Your active task list was preserved across context compression]'

// Token budget (rough estimator ≈ chars/4) above the calibrated base request:
//   history 2×24k chars ≈ 12k  < X=18k (turn-start preflight must NOT fire)
//   history + round ≈ 24k     >= X     (compaction #1 after the U3 tool round)
//   summary + tail(round) ≈ 13.5k < X  (no re-fire after #1 / at the redirect)
//   2 rounds ≈ 25.5k          >= X     (compaction #2 after the U4 tool round)
const HISTORY_CHARS = 24_000
const ROUND_CHARS = 48_000
const HEADROOM_TOKENS = 18_000

function filler(tag: string, chars: number): string {
  const out: string[] = []
  let len = 0

  for (let i = 0; len < chars; i++) {
    const word = `${tag}${i}`
    out.push(word)
    len += word.length + 1
  }

  return out.join(' ')
}

const compressionYaml = (thresholdTokens: number) =>
  [
    'compression:',
    '  in_place: true',
    `  threshold_tokens: ${thresholdTokens}`,
    '  tail_mode: legacy',
    '  protect_first_n: 0',
    '  protect_last_n: 2',
    // todo_list is deferred behind tool_search by default; the scripted model calls it directly.
    'tools:',
    '  tool_search:',
    '    defer: []',
    ''
  ].join('\n')

function viewport(page: Page) {
  return page.locator('[data-slot="aui_thread-viewport"]').filter({ visible: true }).first()
}

async function settled(page: Page) {
  await expect
    .poll(() => page.locator('[data-slot="composer-root"] button[aria-label="Stop"]').count(), {
      timeout: 90_000,
      message: 'turn never settled'
    })
    .toBe(0)
}

async function openSession(page: Page, id: string) {
  await page.evaluate(sid => {
    window.location.hash = sid ? `#/${encodeURIComponent(sid)}` : '#/'
  }, id)
  await expect.poll(() => currentSessionId(page)).toBe(id)
}

interface Row {
  active: number
  content: string
  id: number
  role: string
}

function sessionRows(hermesHome: string, sid: string): Row[] {
  const db = new DatabaseSync(path.join(hermesHome, 'state.db'), { readOnly: true })

  try {
    return db
      .prepare(
        "SELECT id, role, COALESCE(active, 1) AS active, COALESCE(content, '') AS content FROM messages WHERE session_id = ? ORDER BY id"
      )
      .all(sid) as unknown as Row[]
  } finally {
    db.close()
  }
}

/** U4 as more than one user bubble, or a U4 user bubble after the final reply A6. */
function dupShape(bubbles: { role: string; text: string }[]) {
  const lastA6 = bubbles.findLastIndex(b => b.role === 'assistant' && b.text.includes(A(6)))
  const afterReply = lastA6 >= 0 && bubbles.slice(lastA6 + 1).some(b => b.role === 'user' && b.text.includes(U(4)))
  const userCopies = bubbles.filter(b => b.role === 'user' && b.text.includes(U(4))).length

  return {
    dup: afterReply || userCopies > 1,
    afterReply,
    userCopies,
    tail: bubbles.slice(-4).map(b => `${b.role}: ${b.text.slice(0, 40)}`)
  }
}

async function samplerHits(page: Page, marker: string): Promise<{ bubbles: string[]; count: number }[]> {
  return page.evaluate(
    m =>
      ((window as any).__coreSampler?.violations ?? [])
        .filter((v: any) => v.marker === m)
        .map((v: any) => ({ bubbles: v.bubbles, count: v.count })),
    marker
  )
}

test('an acknowledged redirect prompt renders once after in-place compaction inside its turn (#121088)', async () => {
  test.setTimeout(180_000)
  const provider = await startScriptedProvider()
  const sandbox = createCoreSandbox('lcp')
  writeProviderHome(sandbox.hermesHome, provider.url, compressionYaml(250_000))
  const { app, page } = await launchCoreApp(coreAppEnv(sandbox))
  const ws = recordWebSockets(page)

  try {
    await waitForInteractive(app, page)
    await installDuplicateSampler(page)

    // ── Calibration session: measure the real base request, then write the threshold.
    provider.script(U(0), [{ text: [`${A(0)} `, 'calibrated'] }])
    await send(page, `${U(0)} calibrate`, 'Enter', ws)
    await expect(viewport(page)).toContainText(A(0), { timeout: 60_000 })
    await settled(page)
    const calibration = provider.completions.find(c => c.marker === U(0))
    expect(calibration, 'calibration request reached the provider').toBeTruthy()
    const base = Math.ceil(JSON.stringify({ m: calibration!.body.messages, t: calibration!.body.tools }).length / 4)
    writeProviderHome(sandbox.hermesHome, provider.url, compressionYaml(base + HEADROOM_TOKENS))

    // ── Scenario session (fresh agent reads the calibrated config).
    await openSession(page, '')
    await expect(viewport(page)).not.toContainText(A(0))

    // Summary requests are keyed by their first marker (U1 / U2 here), so these
    // text-only replies double as the compaction summaries.
    provider.script(U(1), [{ text: [`${A(1)} `, 'short ', 'reply'] }])
    provider.script(U(2), [{ text: [`${A(2)} `, 'short ', 'reply'] }])

    for (const n of [1, 2]) {
      await send(page, `${U(n)} ${filler(`f${n}w`, HISTORY_CHARS)}`, 'Enter', ws)
      await expect(viewport(page)).toContainText(A(n), { timeout: 60_000 })
      await settled(page)
    }

    const sid = await currentSessionId(page)
    expect(sid, 'scenario session id').not.toBe('')

    const hold = gate()
    provider.script(U(3), [
      {
        text: [`${A(3)}i `, filler('p', ROUND_CHARS)],
        toolCalls: [
          {
            name: 'todo_list',
            args: { todos: [{ id: '1', content: `${U(5)} draft the report`, status: 'in_progress' }] }
          }
        ]
      },
      { text: [`${A(3)} `, 'compaction ', 'one ', 'missed'] }
    ])
    // After compaction #1 the last user row is snapshot S1 (first marker U5).
    provider.script(U(5), [{ text: [`${A(5)} `, 'slow ', 'after ', 'compaction'], holdAfterFirstChunk: hold }])
    provider.script(U(4), [
      {
        text: [`${A(4)}i `, filler('q', ROUND_CHARS)],
        toolCalls: [
          {
            name: 'todo_list',
            args: { todos: [{ id: '1', content: `${U(6)} ship the report`, status: 'in_progress' }] }
          }
        ]
      },
      { text: [`${A(4)} `, 'compaction ', 'two ', 'missed'] }
    ])
    // After compaction #2 the last user row is snapshot S2 (first marker U6).
    provider.script(U(6), [{ text: [`${A(6)} `, 'after ', 'second ', 'compaction'] }])

    await send(page, `${U(3)} plan it`, 'Enter', ws)
    await provider.streamStarted(U(5))
    await expect(viewport(page)).toContainText(A(5), { timeout: 60_000 })

    // Busy submit: the redirect correction, an optimistic row with no row-id receipt.
    await send(page, `${U(4)} change course`, 'Enter', ws)
    await expect
      .poll(() => provider.completions.some(c => c.marker === U(6) && c.finished), {
        timeout: 90_000,
        message: 'post-compaction-#2 reply (keyed on snapshot S2) never finished'
      })
      .toBe(true)
    hold.open()
    await settled(page)
    await expect(viewport(page)).toContainText(A(6), { timeout: 30_000 })

    // ── Preconditions (ordinary): two in-place compactions committed inside the
    // turn; stored history = committed U4, synthetic S1 before it, S2 after it.
    const agentLog = fs.readFileSync(path.join(sandbox.hermesHome, 'logs', 'agent.log'), 'utf8')

    const committed = agentLog
      .split('\n')
      .filter(l => l.includes(sid) && l.includes('"commit_status":"committed"') && l.includes('in_place_committed'))

    expect(committed.length, 'two in-place compactions committed for the scenario session').toBe(2)

    const summaries = provider.completions.filter(c =>
      JSON.stringify(c.body.messages ?? '').includes('context checkpoint')
    )

    expect(
      summaries.map(c => c.finished && !c.aborted),
      'two summary requests answered'
    ).toEqual([true, true])
    expect(
      provider.completions.some(c => (c.marker === U(3) || c.marker === U(4)) && c.step === 1),
      'no un-compacted continuation'
    ).toBe(false)

    const active = sessionRows(sandbox.hermesHome, sid).filter(r => r.active === 1)
    const users = active.filter(r => r.role === 'user')
    const committedU4 = users.filter(r => r.content === `${U(4)} change course`)
    expect(committedU4.length, 'the correction is committed exactly once in active history').toBe(1)
    const s2 = users.find(r => r.content.startsWith(SNAPSHOT) && r.content.includes(U(6)))
    expect(s2, 'snapshot S2 (synthetic user row) is active').toBeTruthy()
    const a6 = active.find(r => r.role === 'assistant' && r.content.startsWith(A(6)))
    expect(a6, 'final reply A6 is stored').toBeTruthy()
    expect(committedU4[0].id < s2!.id && s2!.id < a6!.id, 'stored order: U4 < S2 < A6').toBe(true)
    const all = sessionRows(sandbox.hermesHome, sid)

    expect(
      all.some(r => r.role === 'user' && r.content.startsWith(SNAPSHOT) && r.content.includes(U(5))),
      'snapshot S1 from compaction #1 is stored'
    ).toBe(true)

    // Observation 1: the settled live view. A turn-settle refresh may already
    // have reconciled the stored history (timing-dependent), so it is part of
    // the symptom window rather than an ordinary expect.
    const live = dupShape((await renderedTranscript(page)).bubbles)

    // ── Refresh through preserveLocalPendingTurnMessages: switch away and back.
    await openSession(page, '')
    await expect(viewport(page)).not.toContainText(A(6))
    await openSession(page, sid)
    // Refreshed stored history landed (only it carries S2), so the reconcile ran.
    await expect(viewport(page).getByText(U(6)).first(), 'refreshed history renders snapshot S2').toBeVisible({
      timeout: 60_000
    })
    await expect(viewport(page)).toContainText(A(6))

    // Observation 2: the switch-back view; observation 3: every sampled frame
    // since the redirect (the in-page sampler records transient duplicates).
    const switched = dupShape((await renderedTranscript(page)).bubbles)
    const hits = await samplerHits(page, U(4))
    const symptom = live.dup || switched.dup || hits.some(h => h.bubbles.length > 1)
    // ── Reload (no local optimistic state): stored history alone renders U4 once.
    await page.reload()
    await waitForInteractive(app, page)
    await expect(viewport(page)).toContainText(A(6), { timeout: 60_000 })
    const reloaded = await renderedTranscript(page)
    expect(
      reloaded.bubbles.filter(b => b.role === 'user' && b.text.includes(U(4))).length,
      'after reload, stored history renders U4 once'
    ).toBe(1)
    expect(
      reloaded.bubbles
        .slice(reloaded.bubbles.findLastIndex(b => b.text.includes(A(6))) + 1)
        .some(b => b.role === 'user'),
      'after reload, no user bubble after the final reply'
    ).toBe(false)

    // Last: the KNOWN symptom (an expected failure while #121088 is open, a pass once fixed).
    expectNoSymptom(
      KNOWN.dupPrompt,
      symptom,
      'the redirect prompt U4 renders as two user bubbles / a user bubble after its reply A6',
      JSON.stringify({ live, switched, sampler: hits.slice(0, 2) })
    )
  } finally {
    await app.close().catch(() => undefined)
    await provider.close()
    sandbox.cleanup()
  }
})
