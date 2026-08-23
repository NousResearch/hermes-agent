import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

import { describe, expect, it } from 'vitest'

const source = readFileSync(resolve(process.cwd(), 'src/app/chat/session-tile.tsx'), 'utf8')

describe('SessionTilePane owner-scoped listing', () => {
  it('resolves a newly active tile on its persisted owner route', () => {
    expect(source).toContain('void resolveStoredSession(storedSessionId, ownerRoute)')
    expect(source).not.toMatch(/void resolveStoredSession\(storedSessionId\)\s*\n/)
  })
})
