import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

import { pathForRegistryBackendRequest } from './connection-config'

const here = path.dirname(fileURLToPath(import.meta.url))
const mainSource = fs.readFileSync(path.join(here, 'main.ts'), 'utf8').replace(/\r\n/g, '\n')

describe('primary-remote descriptor reuse keeps profile scope', () => {
  it('scopes a shared-remote request with ?profile=<profile>', () => {
    expect(pathForRegistryBackendRequest('/api/skills', 'acme', { sharedRemote: true })).toBe(
      '/api/skills?profile=acme'
    )
  })

  it('does not add a profile query when the backend is not shared-remote', () => {
    // An isolated backend owns one profile; the router must not invent a scope.
    expect(pathForRegistryBackendRequest('/api/skills', 'acme', { sharedRemote: false, remoteProfile: null })).toBe(
      '/api/skills'
    )
  })

  it('marks the reused primary-remote descriptor sharedRemote so the router scopes it', () => {
    const branchStart = mainSource.indexOf("if (id === registry.primary && source.kind !== 'local'")
    expect(branchStart).toBeGreaterThan(-1)
    const branch = mainSource.slice(branchStart, branchStart + 800)

    // The reuse branch must decorate the ambient primary descriptor with
    // sharedRemote: true, matching the explicit shared-remote connection path.
    expect(branch).toContain('const primaryDescriptor = await ensureBackend(profile)')
    expect(branch).toContain('registrySourceOwnsPrimaryBackend(registry, id, primaryDescriptor)')
    expect(branch).toContain('sharedRemote: true')
  })
})
