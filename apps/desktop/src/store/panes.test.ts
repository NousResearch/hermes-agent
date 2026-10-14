import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import {
  $paneOpen,
  $paneStates,
  $paneWidthOverride,
  clearPaneWidthOverride,
  ensurePaneRegistered,
  getPaneStateSnapshot,
  setPaneHeightOverride,
  setPaneOpen,
  setPaneWidthOverride,
  togglePane
} from './panes'

const STORAGE_KEY = 'hermes.desktop.paneStates.v1'

describe('panes store', () => {
  beforeEach(() => {
    $paneStates.set({})
    window.localStorage.clear()
  })

  afterEach(() => {
    $paneStates.set({})
    window.localStorage.clear()
  })

  describe('ensurePaneRegistered', () => {
    it('is a no-op when the pane already exists', () => {
      ensurePaneRegistered('files', { open: false })
      ensurePaneRegistered('files', { open: true })

      expect(getPaneStateSnapshot('files')?.open).toBe(false)
    })

    it('preserves an existing widthOverride when re-registering', () => {
      ensurePaneRegistered('files', { open: true })
      setPaneWidthOverride('files', 360)
      ensurePaneRegistered('files', { open: false })

      expect(getPaneStateSnapshot('files')?.widthOverride).toBe(360)
    })
  })

  describe('setPaneOpen / togglePane', () => {
    it('togglePane flips the current value', () => {
      ensurePaneRegistered('files', { open: false })
      togglePane('files')
      togglePane('files')
      togglePane('files')

      expect(getPaneStateSnapshot('files')?.open).toBe(true)
    })

    it('togglePane on an unregistered id starts from false', () => {
      togglePane('ephemeral')

      expect(getPaneStateSnapshot('ephemeral')?.open).toBe(true)
    })

    it('preserves widthOverride across open/close changes', () => {
      ensurePaneRegistered('files', { open: true })
      setPaneWidthOverride('files', 280)
      setPaneOpen('files', false)
      setPaneOpen('files', true)

      expect(getPaneStateSnapshot('files')?.widthOverride).toBe(280)
    })
  })

  describe('width overrides', () => {
    it('a resize NEVER closes the pane it resized', () => {
      // A sash drag (and a double-click on a sash) reaches EVERY shown pane of
      // the zone. An unregistered one used to be invented as `{ open: false }`,
      // which closed the pane under the pointer — and a hidden zone stops
      // rendering, so everything stacked beside it re-divided its heights.
      setPaneWidthOverride('subagent-panes:live', 320)

      expect($paneOpen('subagent-panes:live').get()).toBe(true)
      expect(getPaneStateSnapshot('subagent-panes:live')).toEqual({ open: true, widthOverride: 320 })
    })

    it('the same holds on the Y axis', () => {
      setPaneHeightOverride('terminal', 220)

      expect($paneOpen('terminal').get()).toBe(true)
      expect(getPaneStateSnapshot('terminal')?.heightOverride).toBe(220)
    })

    it('an override leaves a pane the user had closed closed', () => {
      ensurePaneRegistered('sessions', { open: false })
      setPaneWidthOverride('sessions', 237)

      expect(getPaneStateSnapshot('sessions')?.open).toBe(false)
      expect(getPaneStateSnapshot('sessions')?.widthOverride).toBe(237)
    })

    it('clearPaneWidthOverride removes the override', () => {
      ensurePaneRegistered('files', { open: true })
      setPaneWidthOverride('files', 300)
      clearPaneWidthOverride('files')

      expect(getPaneStateSnapshot('files')?.widthOverride).toBeUndefined()
    })

    it('width override is NOT in-memory only, and is persisted across reloads', () => {
      ensurePaneRegistered('files', { open: true })
      setPaneWidthOverride('files', 300)

      const persisted = window.localStorage.getItem(STORAGE_KEY)

      expect(persisted).not.toBeNull()
      expect(JSON.parse(persisted ?? '{}')).toEqual({ files: { open: true, widthOverride: 300 } })
    })

    it('open flag is persisted across changes', () => {
      ensurePaneRegistered('files', { open: false })
      setPaneOpen('files', true)

      const persisted = window.localStorage.getItem(STORAGE_KEY)

      expect(persisted).not.toBeNull()
      expect(JSON.parse(persisted ?? '{}')).toEqual({ files: { open: true } })
    })
  })

  describe('derived atoms', () => {
    it('$paneOpen reflects the pane state', () => {
      const open$ = $paneOpen('files')
      expect(open$.get()).toBe(false)

      ensurePaneRegistered('files', { open: true })
      expect(open$.get()).toBe(true)

      setPaneOpen('files', false)
      expect(open$.get()).toBe(false)
    })

    it('$paneWidthOverride reflects the width', () => {
      const width$ = $paneWidthOverride('files')
      expect(width$.get()).toBeUndefined()

      ensurePaneRegistered('files', { open: true })
      setPaneWidthOverride('files', 240)
      expect(width$.get()).toBe(240)
    })
  })
})
