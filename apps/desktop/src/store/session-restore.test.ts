import { beforeEach, describe, expect, it } from 'vitest'

import {
  $sessionRestorePromptVisible,
  $sessionRestoreSnapshot,
  clearSessionRestorePrompt,
  showSessionRestorePrompt
} from './session-restore'

function mockSnapshot(entries = 1) {
  return {
    schemaVersion: 1,
    createdAt: Date.now(),
    entries: Array.from({ length: entries }, (_, i) => ({
      sessionId: `session-${i + 1}`,
      watch: false,
      bounds: { x: 0, y: 0, width: 1200, height: 800 }
    }))
  }
}

describe('session-restore store', () => {
  beforeEach(() => {
    $sessionRestoreSnapshot.set(null)
    $sessionRestorePromptVisible.set(false)
  })

  it('defaults to null snapshot and invisible prompt', () => {
    expect($sessionRestoreSnapshot.get()).toBeNull()
    expect($sessionRestorePromptVisible.get()).toBe(false)
  })

  it('showSessionRestorePrompt sets snapshot and makes prompt visible', () => {
    const snapshot = mockSnapshot(3)

    showSessionRestorePrompt(snapshot)

    expect($sessionRestoreSnapshot.get()).toEqual(snapshot)
    expect($sessionRestorePromptVisible.get()).toBe(true)
  })

  it('showSessionRestorePrompt with null clears the prompt', () => {
    const snapshot = mockSnapshot(1)

    showSessionRestorePrompt(snapshot)
    showSessionRestorePrompt(null)

    expect($sessionRestoreSnapshot.get()).toBeNull()
    expect($sessionRestorePromptVisible.get()).toBe(false)
  })

  it('showSessionRestorePrompt with empty entries does not show prompt', () => {
    const snapshot = mockSnapshot(0)

    showSessionRestorePrompt(snapshot)

    expect($sessionRestoreSnapshot.get()).toEqual(snapshot)
    expect($sessionRestorePromptVisible.get()).toBe(false)
  })

  it('clearSessionRestorePrompt resets both atoms', () => {
    showSessionRestorePrompt(mockSnapshot(2))

    clearSessionRestorePrompt()

    expect($sessionRestoreSnapshot.get()).toBeNull()
    expect($sessionRestorePromptVisible.get()).toBe(false)
  })

  it('clearSessionRestorePrompt is a no-op on default state', () => {
    clearSessionRestorePrompt()

    expect($sessionRestoreSnapshot.get()).toBeNull()
    expect($sessionRestorePromptVisible.get()).toBe(false)
  })
})
