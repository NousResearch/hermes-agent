/**
 * KNOWN bugs for the core suite: merge-order-safe expected failures, the
 * Playwright twin of tests/e2e/core/_pending_fixes.py::known_failure.
 *
 * A strict marker (`test.fail()` up front, or an XPASS error) turns main red
 * the moment the fix merges before this suite does, and accepts ANY failure
 * (a dead backend, a timeout) as "the bug is still there". Instead only the
 * final symptom assertion is wrapped (`expectNoSymptom`): when it fails with the bug's own
 * message the test is marked expected-failing at run time; any other failure
 * propagates; a clean pass stays a pass once the fix lands (then delete the
 * KNOWN entry). Every wait must have settled before the wrapped assertion.
 */

import { expect, test } from '@playwright/test'

/**
 * The symptom assertion of a scenario. `symptom` is true when the bug's
 * user-visible symptom was observed. With no KNOWN entry this is a plain
 * `expect(symptom).toBe(false)`; with one (`known` = issue ref + one line),
 * that same failure — and only it — becomes an expected failure.
 */
export function expectNoSymptom(known: string | undefined, symptom: boolean, what: string, detail = ''): void {
  const check = () => expect(symptom, `${what}${detail ? `\n${detail}` : ''}`).toBe(false)

  if (!known) {
    check()

    return
  }

  try {
    check()
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)

    if (!message.includes(what)) {
      throw error
    }

    test
      .info()
      .annotations.push({
        type: 'known-bug',
        description: `${known} [observed: ${what}${detail ? ` — ${detail.slice(0, 200)}` : ''}]`
      })
    test.fail(true, known)

    throw error
  }
}
