import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { onPersistenceEvent } from '@/lib/storage'

import { $projectSessionSort, setProjectSessionSort } from './layout'

describe('project session sort preference', () => {
  let persisted: Array<{ key: string; value: null | string }> = []
  let stopObserving: () => void = () => undefined

  beforeEach(() => {
    stopObserving()
    setProjectSessionSort('recent')
    persisted = []
    stopObserving = onPersistenceEvent(event => {
      if (event.key === 'hermes.desktop.projectSessionSort') {
        persisted.push({ key: event.key, value: event.value })
      }
    })
  })

  afterEach(() => stopObserving())

  it('defaults to recent activity and persists title selection', () => {
    expect($projectSessionSort.get()).toBe('recent')

    setProjectSessionSort('title-desc')

    expect($projectSessionSort.get()).toBe('title-desc')
    expect(persisted).toContainEqual({ key: 'hermes.desktop.projectSessionSort', value: 'title-desc' })
  })
})
