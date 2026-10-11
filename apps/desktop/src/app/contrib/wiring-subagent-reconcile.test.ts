import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

import { describe, expect, it } from 'vitest'

// Source-text guard — the same category as an ESLint rule, expressed as a
// vitest (see components/ui/__tests__/no-native-title.test.ts): the app wiring
// must fire the one-shot subagent reconcile on every connection publish.
// Mounting all of WiredDesktopRoot to observe the callback is not practical in
// jsdom, so the invariant is pinned at the call site; the helper's behavior is
// covered by ./status-stack/connection-publish-reconcile.test.ts.
describe('wiring.tsx connection-publish subagent reconcile', () => {
  const source = readFileSync(resolve(__dirname, 'wiring.tsx'), 'utf8')

  it('fires the one-shot reconcile from onConnectionReady with a swallowed rejection', () => {
    const start = source.indexOf('onConnectionReady:')
    const end = source.indexOf('onGatewayReady:')

    expect(start).toBeGreaterThan(-1)
    expect(end).toBeGreaterThan(start)
    expect(source.slice(start, end)).toContain('void reconcileSubagentsOnConnectionPublish().catch(() => {})')
  })

  it('imports the helper from the snapshot module that owns the owner/race guard', () => {
    expect(source).toMatch(
      /import \{[^}]*\breconcileSubagentsOnConnectionPublish\b[^}]*\} from '@\/app\/chat\/composer\/status-stack\/use-subagent-snapshot'/
    )
  })
})
