/**
 * Seam-scope pins for the room composer middleware (PR follow-up per a
 * non-author review of the render-vs-send boundary).
 *
 * What this file guards:
 *  1. Composer-originated user text reaches the room engine through exactly
 *     two `sendToGroupChat` call sites, and every one of them crosses
 *     `runRoomMiddleware` (which runs the core `runComposerMiddleware` chain)
 *     first — the invariant as SCOPED: the primitive is the single ingest
 *     path for composer-originated sends.
 *  2. The middleware chain appears in NO other production module of the
 *     plugin — so the seam's coverage is exactly the composer, not "every
 *     pixel that types".
 *  3. The known, deliberately scoped-out path is named rather than hidden:
 *     the clarify-card answer (`GroupClarifyCard` -> `answerGroupClarify`
 *     via the `clarify.lock` RPC in `group-turns.ts`, echoed to the room log
 *     with `appendGroupChatEntry({ kind: 'user' })` in
 *     `group-chat-parts.tsx`) emits user free-text WITHOUT crossing the
 *     primitive or the middleware. It never did; this PR does not change
 *     that, and this test pins the exception so a future reader cannot
 *     mistake the scoped invariant for a total one.
 *
 * Static source reads, the established style for enumeration pins in this
 * directory (`relay-deliver-budget.test.ts`). Line numbers in comments drift;
 * patterns do not.
 */

import { readFileSync } from 'node:fs'
import { join } from 'node:path'

import { describe, expect, it } from 'vitest'

const here = process.cwd() // apps/desktop when run via `vitest --project ui`
const src = (rel: string) => readFileSync(join(here, 'src/plugins/hermes-bots', rel), 'utf8')

const view = src('group-chat-view.tsx')
const rounds = src('group-rounds.ts')
const parts = src('group-chat-parts.tsx')
const turns = src('group-turns.ts')

function callLines(text: string, name: string): number[] {
  return text
    .split('\n')
    .map((line, i) => ({ line, n: i + 1 }))
    .filter(({ line }) => new RegExp(`\\b${name}\\(`).test(line))
    .map(({ n }) => n)
}

describe('room composer middleware — seam scope', () => {
  it('sendToGroupChat keeps exactly one definition and two production callers', () => {
    expect(callLines(rounds, 'export function sendToGroupChat')).toHaveLength(1)
    // The only non-test call sites are the room submit and the per-thread reply.
    const sites = callLines(view, 'sendToGroupChat')
    expect(sites).toHaveLength(2)
    const middleware = callLines(view, 'runRoomMiddleware')
    expect(middleware.length).toBeGreaterThanOrEqual(2)

    for (const site of sites) {
      // Every send happens inside a submit handler that ran the chain first:
      // the nearest preceding runRoomMiddleware is within the same handler.
      const before = middleware.filter(n => n < site)
      expect(before.length, `send at :${site} has no preceding runRoomMiddleware`).toBeGreaterThan(0)
      expect(site - before[before.length - 1]).toBeLessThan(60)
    }
  })

  it('runRoomMiddleware is the core chain, not a local invention', () => {
    expect(view).toMatch(/import\s*{[^}]*\brunComposerMiddleware\b[^}]*}\s*from\s*'@hermes\/plugin-sdk'/s)
    expect(view).toMatch(/return await runComposerMiddleware\(/)
  })

  it('the middleware crosses no other production module of the plugin', () => {
    for (const [name, text] of [
      ['group-rounds.ts', rounds],
      ['group-chat-parts.tsx', parts],
      ['group-turns.ts', turns]
    ] as const) {
      expect(callLines(text, 'runComposerMiddleware'), `${name} must not call the chain`).toHaveLength(0)
    }
  })

  it('names the scoped-out clarify path instead of pretending it is guarded', () => {
    // The card answers through answerGroupClarify + the clarify.lock RPC and
    // echoes free-text to the room log; none of it touches the seam.
    expect(callLines(parts, 'answerGroupClarify').length).toBeGreaterThan(0)
    expect(turns).toMatch(/'clarify\.lock'/)
    expect(callLines(parts, 'appendGroupChatEntry').length).toBeGreaterThan(0)
    expect(parts).toMatch(/kind: 'user'/)
    expect(callLines(parts, 'sendToGroupChat')).toHaveLength(0)
    expect(callLines(parts, 'runComposerMiddleware')).toHaveLength(0)
  })
})
