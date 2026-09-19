// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest'

import {
  $ideDirtyPaths,
  $ideEditor,
  activateIdeFile,
  clearIdeFileDirty,
  closeIdeFile,
  IDE_EDITOR_STORAGE_KEY,
  openIdeFile,
  setIdeFileDirty
} from './tabs'

beforeEach(() => {
  window.localStorage.clear()
  $ideEditor.set({ activePath: null, openPaths: [] })
  $ideDirtyPaths.set([])
})

describe('ide editor tabs', () => {
  it('opens files as tabs and activates the newest', () => {
    openIdeFile('/repo/a.ts')
    openIdeFile('/repo/b.ts')

    expect($ideEditor.get().openPaths).toEqual(['/repo/a.ts', '/repo/b.ts'])
    expect($ideEditor.get().activePath).toBe('/repo/b.ts')
  })

  it('re-opening an existing file activates instead of duplicating', () => {
    openIdeFile('/repo/a.ts')
    openIdeFile('/repo/b.ts')
    openIdeFile('/repo/a.ts')

    expect($ideEditor.get().openPaths).toEqual(['/repo/a.ts', '/repo/b.ts'])
    expect($ideEditor.get().activePath).toBe('/repo/a.ts')

    activateIdeFile('/repo/b.ts')
    expect($ideEditor.get().activePath).toBe('/repo/b.ts')
  })

  it('closing the active tab falls back to the last remaining tab', () => {
    openIdeFile('/repo/a.ts')
    openIdeFile('/repo/b.ts')
    closeIdeFile('/repo/b.ts')

    expect($ideEditor.get().activePath).toBe('/repo/a.ts')
    expect($ideEditor.get().openPaths).toEqual(['/repo/a.ts'])
    expect($ideDirtyPaths.get()).toEqual([])
  })

  it('persists the open set under the IDE-owned key', () => {
    openIdeFile('/repo/a.ts')

    expect(window.localStorage.getItem(IDE_EDITOR_STORAGE_KEY)).toContain('/repo/a.ts')
  })

  it('tracks dirty paths per file and clears them individually', () => {
    openIdeFile('/repo/a.ts')
    openIdeFile('/repo/b.ts')

    setIdeFileDirty('/repo/a.ts', true)
    expect($ideDirtyPaths.get()).toEqual(['/repo/a.ts'])
    setIdeFileDirty('/repo/a.ts', true)
    expect($ideDirtyPaths.get()).toEqual(['/repo/a.ts'])

    clearIdeFileDirty('/repo/a.ts')
    expect($ideDirtyPaths.get()).toEqual([])
  })

  it('drops a closed file from the dirty set', () => {
    openIdeFile('/repo/a.ts')
    setIdeFileDirty('/repo/a.ts', true)
    closeIdeFile('/repo/a.ts')

    expect($ideDirtyPaths.get()).toEqual([])
  })
})
