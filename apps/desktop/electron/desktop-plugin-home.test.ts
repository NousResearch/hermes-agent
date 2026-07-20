import { describe, expect, it } from 'vitest'

import { desktopPluginHome } from './desktop-plugin-home'

describe('desktopPluginHome', () => {
  const root = '/Users/test/.hermes'

  it('uses the selected local profile home', () => {
    expect(desktopPluginHome(root, 'work')).toBe('/Users/test/.hermes/profiles/work')
  })

  it('keeps the default profile at the local root', () => {
    expect(desktopPluginHome(root, 'default')).toBe(root)
  })

  it('honors the sticky profile when Desktop has no explicit selection', () => {
    expect(desktopPluginHome(root, null, () => 'personal\n')).toBe('/Users/test/.hermes/profiles/personal')
  })

  it('fails closed to the default home for an invalid sticky profile', () => {
    expect(desktopPluginHome(root, null, () => '../other')).toBe(root)
  })
})
