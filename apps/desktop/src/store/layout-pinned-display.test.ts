// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest'

import {
  $sidebarPinnedCardRows,
  $sidebarPinnedInProjects,
  setSidebarPinnedCardRows,
  setSidebarPinnedInProjects
} from '@/store/layout'

// The two pinned-display preferences default to the deliberate behavior:
// pinned rows stay compact (not inbox cards) and a pinned chat lives in the
// Pinned section only (not duplicated into project/worktree lanes). Each
// setting is an opt-in override, persisted per the storage keys below.
const CARD_ROWS_KEY = 'hermes.desktop.sidebarPinnedCardRows'
const IN_PROJECTS_KEY = 'hermes.desktop.sidebarPinnedInProjects'

beforeEach(() => {
  window.localStorage.removeItem(CARD_ROWS_KEY)
  window.localStorage.removeItem(IN_PROJECTS_KEY)
  setSidebarPinnedCardRows(false)
  setSidebarPinnedInProjects(false)
})

describe('pinned display atoms', () => {
  it('default to off (preserve compact one-line Pinned rows)', () => {
    expect($sidebarPinnedCardRows.get()).toBe(false)
    expect($sidebarPinnedInProjects.get()).toBe(false)
  })

  it('setSidebarPinnedCardRows flips the atom and persists it', () => {
    setSidebarPinnedCardRows(true)
    expect($sidebarPinnedCardRows.get()).toBe(true)
    expect(window.localStorage.getItem(CARD_ROWS_KEY)).toBe('true')

    setSidebarPinnedCardRows(false)
    expect($sidebarPinnedCardRows.get()).toBe(false)
    expect(window.localStorage.getItem(CARD_ROWS_KEY)).toBe('false')
  })

  it('setSidebarPinnedInProjects flips the atom and persists it', () => {
    setSidebarPinnedInProjects(true)
    expect($sidebarPinnedInProjects.get()).toBe(true)
    expect(window.localStorage.getItem(IN_PROJECTS_KEY)).toBe('true')

    setSidebarPinnedInProjects(false)
    expect($sidebarPinnedInProjects.get()).toBe(false)
    expect(window.localStorage.getItem(IN_PROJECTS_KEY)).toBe('false')
  })
})
