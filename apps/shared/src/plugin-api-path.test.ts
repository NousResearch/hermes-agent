import { describe, expect, it } from 'vitest'

import { buildPluginApiPath, isValidPluginManifestId, normalizePluginRelativePath } from './plugin-api-path'

describe('plugin API paths', () => {
  it('builds a plugin namespace from a safe relative path', () => {
    expect(buildPluginApiPath('kanban', '/board?archived=1')).toBe('/api/plugins/kanban/board?archived=1')
    expect(buildPluginApiPath('kanban', 'tasks')).toBe('/api/plugins/kanban/tasks')
  })

  it('rejects malformed manifest ids', () => {
    expect(isValidPluginManifestId('kanban')).toBe(true)
    expect(isValidPluginManifestId('../kanban')).toBe(false)
    expect(isValidPluginManifestId('kanban/other')).toBe(false)
    expect(() => buildPluginApiPath('kanban/other', '/board')).toThrow(/manifest id/)
  })

  it('rejects traversal, encoded traversal, schemes, and fragments', () => {
    expect(() => normalizePluginRelativePath('/../other')).toThrow(/Unsafe/)
    expect(() => normalizePluginRelativePath('/%2e%2e/other')).toThrow(/Unsafe/)
    expect(() => normalizePluginRelativePath('/%252e%252e/other')).toThrow(/Unsafe/)
    expect(() => normalizePluginRelativePath('https://example.test/x')).toThrow(/Unsafe/)
    expect(() => normalizePluginRelativePath('/tasks#fragment')).toThrow(/Invalid/)
  })
})
