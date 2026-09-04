import { beforeEach, describe, expect, it } from 'vitest'

import { group, migratePersistedTree, setGroupBackgroundTint, split, ZONE_BACKGROUND_TINTS } from './model'
import { $activePresetId, $layoutTree, setTreeGroupBackgroundTint } from './store'

describe('zone background tint model', () => {
  it('changes only the selected zone and can return it to the theme default', () => {
    const tree = split('row', [group(['sessions'], { id: 'sidebar' }), group(['workspace'], { id: 'workspace' })])

    const tinted = setGroupBackgroundTint(tree, 'workspace', 'blue')

    expect(tinted.type).toBe('split')

    if (tinted.type !== 'split') {
      return
    }

    expect(tinted.children[0]).toHaveProperty('backgroundTint', undefined)
    expect(tinted.children[1]).toHaveProperty('backgroundTint', 'blue')

    const cleared = setGroupBackgroundTint(tinted, 'workspace', undefined)
    expect(cleared.type === 'split' && cleared.children[1]).not.toHaveProperty('backgroundTint')
  })

  it('keeps allowed persisted tints and drops untrusted values recursively', () => {
    const migrated = migratePersistedTree(
      split('row', [
        group(['sessions'], { backgroundTint: 'green', id: 'sidebar' }),
        group(['workspace'], { backgroundTint: 'url(https://example.com/tracker)', id: 'workspace' } as never)
      ])
    )

    expect(migrated.type).toBe('split')

    if (migrated.type !== 'split') {
      return
    }

    expect(migrated.children[0]).toHaveProperty('backgroundTint', 'green')
    expect(migrated.children[1]).not.toHaveProperty('backgroundTint')
    expect(ZONE_BACKGROUND_TINTS).toEqual(['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple'])
  })
})

describe('zone background tint persistence', () => {
  beforeEach(() => {
    window.localStorage.clear()
    $layoutTree.set(group(['workspace'], { id: 'workspace' }))
    $activePresetId.set('default')
  })

  it('persists the tint with the layout and marks the arrangement custom', () => {
    setTreeGroupBackgroundTint('workspace', 'purple')

    expect($layoutTree.get()).toHaveProperty('backgroundTint', 'purple')
    expect($activePresetId.get()).toBe('custom')
    expect(JSON.parse(window.localStorage.getItem('hermes.desktop.layoutTree.v2') ?? '{}')).toHaveProperty(
      'backgroundTint',
      'purple'
    )
  })
})
