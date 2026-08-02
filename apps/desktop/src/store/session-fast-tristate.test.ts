import { beforeEach, describe, expect, it } from 'vitest'

import { readKey, writeKey } from '@/lib/storage'

import {
  $currentFastMode,
  clearCurrentFastMode,
  getComposerModeSource,
  hasPersistedComposerFastMode,
  setCurrentFastMode,
  setCurrentFastModeFromDefault,
  setCurrentFastModeTransient,
  snapshotPersistedComposerFastMode
} from './session'

const FAST_KEY = 'hermes.desktop.composer.fast'
const FAST_SOURCE_KEY = 'hermes.desktop.composer.fast-source'

describe('composer Fast tri-state (absent ≠ false ≠ true)', () => {
  beforeEach(() => {
    writeKey(FAST_KEY, null)
    writeKey(FAST_SOURCE_KEY, null)
    $currentFastMode.set(false)
  })

  it('starts with absent durable Fast (no key)', () => {
    expect(hasPersistedComposerFastMode()).toBe(false)
    expect(snapshotPersistedComposerFastMode()).toBeNull()
    expect(getComposerModeSource('fast')).toBe('')
  })

  it('manual true persists true and marks source manual', () => {
    setCurrentFastMode(true)
    expect($currentFastMode.get()).toBe(true)
    expect(hasPersistedComposerFastMode()).toBe(true)
    expect(snapshotPersistedComposerFastMode()).toBe(true)
    expect(getComposerModeSource('fast')).toBe('manual')
    expect(readKey(FAST_KEY)).toBe('true')
  })

  it('manual false is distinct from absent', () => {
    setCurrentFastMode(false)
    expect($currentFastMode.get()).toBe(false)
    expect(hasPersistedComposerFastMode()).toBe(true)
    expect(snapshotPersistedComposerFastMode()).toBe(false)
    expect(getComposerModeSource('fast')).toBe('manual')
    expect(readKey(FAST_KEY)).toBe('false')
  })

  it('clear returns to absent without leaving a false key', () => {
    setCurrentFastMode(true)
    clearCurrentFastMode()
    expect(hasPersistedComposerFastMode()).toBe(false)
    expect(snapshotPersistedComposerFastMode()).toBeNull()
    expect(getComposerModeSource('fast')).toBe('')
    expect(readKey(FAST_KEY)).toBeNull()
  })

  it('transient paint does not claim durable ownership', () => {
    setCurrentFastMode(true)
    expect(hasPersistedComposerFastMode()).toBe(true)

    setCurrentFastModeTransient(false)
    expect($currentFastMode.get()).toBe(false)
    // Durable intent remains true under the key
    expect(snapshotPersistedComposerFastMode()).toBe(true)
    expect(getComposerModeSource('fast')).toBe('manual')
  })

  it('profile default seed does not mark Fast as manual', () => {
    setCurrentFastModeFromDefault(true)
    expect($currentFastMode.get()).toBe(true)
    expect(hasPersistedComposerFastMode()).toBe(true)
    expect(getComposerModeSource('fast')).toBe('default')
  })

  it('manual ownership blocks treating a later default as a wipe of explicit false', () => {
    setCurrentFastMode(false)
    // A later "default seed" path must check source !== manual before calling
    // setCurrentFastModeFromDefault. Prove explicit false stays owned.
    expect(getComposerModeSource('fast')).toBe('manual')
    expect(snapshotPersistedComposerFastMode()).toBe(false)
  })
})
