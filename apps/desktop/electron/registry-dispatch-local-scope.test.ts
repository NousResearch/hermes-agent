import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'

// dispatchRegistryApiRequest lives in main.ts, which cannot be imported under
// vitest (electron imports at module scope). Follow the repo convention for
// main.ts internals (see backend-dial-claim.test.ts): assert the structural
// shape of the handler source so a regression deletes the marker and fails.

const here = path.dirname(fileURLToPath(import.meta.url))
const mainSource = fs.readFileSync(path.join(here, 'main.ts'), 'utf8')

function dispatchBody() {
  const start = mainSource.indexOf('async function dispatchRegistryApiRequest(')
  expect(start).toBeGreaterThan(-1)
  return mainSource.slice(start, start + 2_600)
}

describe('dispatchRegistryApiRequest local-profile scoping (#settings-chip-model-scope)', () => {
  // A delegated local profile (connectionId 'local', v1 route genuinely local)
  // resolves to the SHARED primary backend (ensureBackend's `sharedPrimary`
  // route): one home serving every profile. The path itself must carry the
  // profile scope exactly the way the v1 handler computes it. The previous
  // code always went through pathForRegistryBackendRequest, whose
  // translateSelfProfileQuery only rewrites an EXISTING ?profile= — so
  // renderer requests like GET /api/model/info left for the backend with no
  // profile query at all, the shared primary answered from its launch home,
  // and the Settings → 模型 page showed (and then applied) the primary
  // profile's model under every other profile's chip.

  it('scopes the path through the v1 route table when the connection is the shared primary', () => {
    const body = dispatchBody()

    expect(body).toContain('connection?.sharedPrimary')
    expect(body).toContain('resolveProfileApiRequest(')
    expect(body).toContain('profileRouteOptions(requestProfile, request)')
  })

  it('keeps isolated/ssh/shared-remote registry backends on pathForRegistryBackendRequest', () => {
    const body = dispatchBody()

    expect(body).toContain('pathForRegistryBackendRequest(request.path, requestProfile, connection)')
  })
})
