import { readFileSync } from 'node:fs'

import { describe, expect, it } from 'vitest'

const readJson = path => JSON.parse(readFileSync(new URL(path, import.meta.url), 'utf8'))
const manifest = readJson('../package.json')
const tui = readJson('../ui-tui/package.json')
const lock = readJson('../package-lock.json')

// npm ci validates direct dependencies, but does not guarantee that a changed
// override has refreshed the existing lockfile's transitive resolutions.
describe('security overrides are reflected in every locked resolution', () => {
  it.each([
    ['undici', '6', manifest.overrides['undici@^6']],
    ['undici', '7', manifest.overrides['undici@^7']],
    ['ip-address', '10', manifest.overrides['ip-address']]
  ])('%s major %s matches the override and its registry metadata', (name, major, version) => {
    const entries = Object.entries(lock.packages).filter(([path, entry]) =>
      path.endsWith(`/node_modules/${name}`) || path === `node_modules/${name}`
    ).filter(([, entry]) => entry.version?.split('.')[0] === major)
    expect(entries.length).toBeGreaterThan(0)
    for (const [, entry] of entries) {
      expect(entry.version).toBe(version)
      expect(entry.resolved).toBe(`https://registry.npmjs.org/${name}/-/${name}-${version}.tgz`)
      expect(entry.integrity).toMatch(/^sha512-[A-Za-z0-9+/]+={0,2}$/)
    }
  })

  it('keeps the direct TUI edge and its locked resolution aligned', () => {
    const version = manifest.overrides['undici@^6']
    expect(tui.dependencies.undici).toBe(version)
    expect(lock.packages['ui-tui'].dependencies.undici).toBe(version)
    expect(lock.packages['ui-tui/node_modules/undici'].version).toBe(version)
  })
})
