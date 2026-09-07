import { readFileSync } from 'node:fs'
import { join } from 'node:path'

import { describe, expect, it } from 'vitest'

const src = (p: string) => readFileSync(join(__dirname, p), 'utf8')

/**
 * Bugs found by using the feature, not by reading it. Both are "the indicator
 * lies" failures: the UI claimed a temporary chat where there was none, and
 * offered a sidebar row that could not be opened.
 *
 * These read source rather than mounting the tree because both live in
 * multi-hundred-line hooks whose full render needs the whole gateway. That is
 * a weaker test -- it pins the mechanism, not the behaviour -- so each one
 * asserts on a specific line that must exist, and was checked to fail when
 * that line is removed.
 */
describe('temporary chat: leaking into other sessions', () => {
  it('resuming a saved session clears the temporary flag before any await', () => {
    const s = src('../../session/hooks/use-session-actions/index.ts')

    const body = s.slice(s.indexOf('const resumeSession = useCallback'))
    const reset = body.indexOf('$currentSessionEphemeral.set(false)')
    // Strip comments first: the fix's own comment mentions `await`, and
    // matching that instead of real code made this assertion fail on correct
    // source. Compare against executable awaits only.
    const code = body.replace(/\/\/[^\n]*/g, '').replace(/\/\*[\s\S]*?\*\//g, '')
    const firstAwait = code.indexOf('await ')
    const resetInCode = code.indexOf('$currentSessionEphemeral.set(false)')

    expect(reset).toBeGreaterThan(-1)
    expect(body).toMatch(/\$newChatEphemeral\.set\(false\)/)

    // The point of the fix: clearing must happen in the synchronous block that
    // paints the click. Behind the first `await` the badge from the previous
    // temporary chat stays on screen for the whole gateway round trip, which
    // is the bug users actually reported.
    expect(resetInCode).toBeLessThan(firstAwait)
  })

  it('the sidebar filters temporary sessions out of every list', () => {
    const s = src('./index.tsx')

    // One funnel: Recents, the project tree overlay and lane previews all
    // derive from visibleSessions, so filtering here covers every surface.
    const memo = s.slice(s.indexOf('const visibleSessions'), s.indexOf('const sortedSessions'))
    expect(memo).toMatch(/filter\(s => !s\.ephemeral\)/)
  })
})
