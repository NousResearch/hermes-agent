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

describe('SessionTileChrome owner ladder', () => {
  // A tile opened without an explicit route (openSessionTile with no
  // workspaceScope — how a branch child is opened) has no tile ownerRoute, so
  // the tile route ALONE leaves ownerRoute undefined and requestForSessionProfile
  // falls back to the ambient socket. The session's own row/hint rung is what
  // keeps its model + composer RPCs on the backend that owns it.
  it('falls back to the session row owner when the tile carries no route', () => {
    expect(source).toContain(
      'sessionTileOwnerRoute(storedSessionId) ?? knownSessionOwner(ownerLookupSessionRows(), storedSessionId)'
    )
  })

  it('does not resolve the chrome owner from the tile route alone', () => {
    // The pre-fix shape: `const ownerRoute = sessionTileOwnerRoute(storedSessionId)`
    // with no fallback rung.
    expect(source).not.toMatch(/const ownerRoute = sessionTileOwnerRoute\(storedSessionId\)\s*\n/)
  })

  it('narrows a bare profile owner to an object route', () => {
    // knownSessionOwner may return a bare profile string, which carries no
    // connection and must not be handed to requestForSessionProfile as a route.
    expect(source).toContain("typeof resolvedOwner === 'object'")
  })
})
