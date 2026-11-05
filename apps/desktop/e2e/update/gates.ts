/**
 * Message-gated gaps for open bugs (the Playwright twin of
 * tests/e2e/core/_pending_fixes.py::known_failure).
 *
 * Wrap ONLY a final assertion, after every wait has settled. When it fails and
 * the failure text matches the bug's signature, the test is marked
 * expected-failing at run time; any other failure (a timeout, a boot failure,
 * the opposite symptom) propagates; a clean pass stays a pass once the fix
 * lands, and the gate is then deleted. No static test.fail(), no tables.
 */

import { test } from '@playwright/test'

export interface KnownBug {
  /** `#N` of the open issue. */
  issue: string
  /** Matches the failing assertion's message ONLY when it shows this bug's symptom. */
  signature: RegExp
  summary: string
}

export const KNOWN = {
  /** Desktop "Update now" on a PM-managed install is cancelled: the state.db pre-flight looks for an in-tree venv PM deleted. */
  preflightPython: {
    issue: '#122991',
    signature: /state\.db pre-flight failed: Python not found/,
    summary: "Desktop update preflight 'Python not found' cancels every update on PM-managed installs"
  }
} satisfies Record<string, KnownBug>

export function gateMatches(bug: KnownBug, message: string): boolean {
  return bug.signature.test(message)
}

/** Run `check`; its failure becomes an expected failure only when it carries `bug`'s signature. */
export async function gatedOn(bug: KnownBug | undefined, check: () => void | Promise<void>): Promise<void> {
  try {
    await check()
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)

    if (bug && gateMatches(bug, message)) {
      test.info().annotations.push({ type: 'known-bug', description: `gated on ${bug.issue}: ${bug.summary}` })
      test.fail(true, `gated on ${bug.issue}`)
    }

    throw error
  }
}
