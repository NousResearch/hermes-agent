import { describe, expect, it } from 'vitest'

import { type LayoutNode, migratePersistedTree } from './model'

// A stored `headerHidden: true` is ambiguous — a deliberate "Hide header" and
// an accidental double-tap wrote the same byte — and it is the state people got
// stuck in, because hiding removed the only control offering to unhide. The
// migration therefore drops it rather than translating it to `tabStrip: 'never'`.

const persisted = (node: unknown) => migratePersistedTree(node as LayoutNode) as never as Record<string, unknown>

describe('migratePersistedTree', () => {
  it('returns a hidden zone to auto instead of re-stranding it', () => {
    const migrated = persisted({
      active: 'workspace',
      headerHidden: true,
      id: 'g',
      panes: ['workspace'],
      type: 'group'
    })

    expect(migrated.headerHidden).toBeUndefined()
    expect(migrated.tabStrip).toBeUndefined()
    expect(migrated.panes).toEqual(['workspace'])
  })

  it('drops a stored false too — the repair paths wrote most of them, not users', () => {
    expect(persisted({ headerHidden: false, id: 'g', panes: ['workspace'], type: 'group' }).tabStrip).toBeUndefined()
  })

  it('keeps a tabStrip choice, which only a user can have written', () => {
    expect(persisted({ id: 'g', panes: ['workspace'], tabStrip: 'never', type: 'group' }).tabStrip).toBe('never')
    expect(persisted({ id: 'g', panes: ['workspace'], tabStrip: 'always', type: 'group' }).tabStrip).toBe('always')
  })

  it('discards a tabStrip value outside the schema', () => {
    expect(persisted({ id: 'g', panes: ['workspace'], tabStrip: 'sometimes', type: 'group' }).tabStrip).toBeUndefined()
  })

  it('reaches groups nested in splits', () => {
    const migrated = persisted({
      children: [
        { active: 'workspace', headerHidden: true, id: 'a', panes: ['workspace'], type: 'group' },
        {
          children: [{ headerHidden: true, id: 'b', panes: ['terminal'], type: 'group' }],
          id: 'inner',
          orientation: 'column',
          type: 'split',
          weights: [1]
        }
      ],
      id: 'root',
      orientation: 'row',
      type: 'split',
      weights: [1, 1]
    })

    expect(JSON.stringify(migrated)).not.toContain('headerHidden')
  })
})
