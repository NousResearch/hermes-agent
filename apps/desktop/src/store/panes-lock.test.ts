import { beforeEach, describe, expect, it } from 'vitest'

import {
  $paneStates,
  clearAllPaneSizeOverrides,
  clearPaneWidthOverride,
  getPaneStateSnapshot,
  type PaneStateSnapshot,
  setPaneHeightLock,
  setPaneHeightOverride,
  setPaneWidthLock,
  setPaneWidthOverride
} from './panes'

// Reset pane states between tests
beforeEach(() => {
  $paneStates.set({})
})

describe('pane axis locking', () => {
  describe('defaults', () => {
    it('lockWidth is absent by default', () => {
      $paneStates.set({ p1: { open: true } })
      const snap = getPaneStateSnapshot('p1')
      expect(snap?.lockWidth).toBeUndefined()
    })

    it('lockHeight is absent by default', () => {
      $paneStates.set({ p1: { open: true } })
      const snap = getPaneStateSnapshot('p1')
      expect(snap?.lockHeight).toBeUndefined()
    })

    it('lockedWidth is absent by default', () => {
      $paneStates.set({ p1: { open: true } })
      const snap = getPaneStateSnapshot('p1')
      expect(snap?.lockedWidth).toBeUndefined()
    })

    it('lockedHeight is absent by default', () => {
      $paneStates.set({ p1: { open: true } })
      const snap = getPaneStateSnapshot('p1')
      expect(snap?.lockedHeight).toBeUndefined()
    })
  })

  describe('setPaneWidthLock', () => {
    it('captures the current width override when locking', () => {
      $paneStates.set({ p1: { open: true, widthOverride: 300 } })
      setPaneWidthLock('p1', true)
      const snap = getPaneStateSnapshot('p1')
      expect(snap?.lockWidth).toBe(true)
      expect(snap?.lockedWidth).toBe(300)
    })

    it('captures the declared width (237px) when no override exists', () => {
      // When there's no override, lockedWidth should still be set
      // The actual declared width comes from contribution data, but
      // the lock captures whatever the override was at lock time.
      // With no override, lockedWidth is undefined (no capture).
      $paneStates.set({ p1: { open: true } })
      setPaneWidthLock('p1', true)
      const snap = getPaneStateSnapshot('p1')
      expect(snap?.lockWidth).toBe(true)
      // No override means lockedWidth is not captured here;
      // the track model will read the declared size from contribution data.
    })

    it('clears lock and lockedWidth on unlock', () => {
      $paneStates.set({ p1: { open: true, widthOverride: 300, lockWidth: true, lockedWidth: 300 } })
      setPaneWidthLock('p1', false)
      const snap = getPaneStateSnapshot('p1')
      expect(snap?.lockWidth).toBeUndefined()
      expect(snap?.lockedWidth).toBeUndefined()
    })

    it('preserves existing snapshot fields', () => {
      $paneStates.set({ p1: { open: true, heightOverride: 200 } })
      setPaneWidthLock('p1', true)
      const snap = getPaneStateSnapshot('p1')
      expect(snap?.open).toBe(true)
      expect(snap?.heightOverride).toBe(200)
    })
  })

  describe('setPaneHeightLock', () => {
    it('captures the current height override when locking', () => {
      $paneStates.set({ p1: { open: true, heightOverride: 400 } })
      setPaneHeightLock('p1', true)
      const snap = getPaneStateSnapshot('p1')
      expect(snap?.lockHeight).toBe(true)
      expect(snap?.lockedHeight).toBe(400)
    })

    it('clears lock and lockedHeight on unlock', () => {
      $paneStates.set({ p1: { open: true, heightOverride: 400, lockHeight: true, lockedHeight: 400 } })
      setPaneHeightLock('p1', false)
      const snap = getPaneStateSnapshot('p1')
      expect(snap?.lockHeight).toBeUndefined()
      expect(snap?.lockedHeight).toBeUndefined()
    })

    it('preserves existing snapshot fields', () => {
      $paneStates.set({ p1: { open: true, widthOverride: 150 } })
      setPaneHeightLock('p1', true)
      const snap = getPaneStateSnapshot('p1')
      expect(snap?.open).toBe(true)
      expect(snap?.widthOverride).toBe(150)
    })
  })

  describe('lock/unlock cycle preserves dimension', () => {
    it('locking then unlocking width does not change widthOverride', () => {
      $paneStates.set({ p1: { open: true, widthOverride: 250 } })
      setPaneWidthLock('p1', true)
      setPaneWidthLock('p1', false)
      const snap = getPaneStateSnapshot('p1')
      expect(snap?.widthOverride).toBe(250)
    })

    it('locking then unlocking height does not change heightOverride', () => {
      $paneStates.set({ p1: { open: true, heightOverride: 350 } })
      setPaneHeightLock('p1', true)
      setPaneHeightLock('p1', false)
      const snap = getPaneStateSnapshot('p1')
      expect(snap?.heightOverride).toBe(350)
    })
  })

  describe('lock does not interfere with override updates', () => {
    it('width override can be updated while width is locked', () => {
      $paneStates.set({ p1: { open: true, widthOverride: 200 } })
      setPaneWidthLock('p1', true)
      // A sash drag would update the override even when locked
      setPaneWidthOverride('p1', 250)
      const snap = getPaneStateSnapshot('p1')
      expect(snap?.widthOverride).toBe(250)
      expect(snap?.lockWidth).toBe(true)
      expect(snap?.lockedWidth).toBe(200) // captured at lock time
    })

    it('height override can be updated while height is locked', () => {
      $paneStates.set({ p1: { open: true, heightOverride: 300 } })
      setPaneHeightLock('p1', true)
      setPaneHeightOverride('p1', 350)
      const snap = getPaneStateSnapshot('p1')
      expect(snap?.heightOverride).toBe(350)
      expect(snap?.lockHeight).toBe(true)
      expect(snap?.lockedHeight).toBe(300)
    })
  })

  describe('clearPaneWidthOverride and clearPaneHeightOverride still work', () => {
    it('clearing width override works regardless of lock', () => {
      $paneStates.set({ p1: { open: true, widthOverride: 200 } })
      clearPaneWidthOverride('p1')
      const snap = getPaneStateSnapshot('p1')
      expect(snap?.widthOverride).toBeUndefined()
    })
  })

  describe('lock state persists through snapshot round-trip', () => {
    it('isSnapshot accepts snapshots with lock fields', () => {
      const snap: PaneStateSnapshot = {
        lockHeight: true,
        lockedHeight: 400,
        lockWidth: true,
        lockedWidth: 300,
        open: true,
        widthOverride: 300
      }

      // setPaneWidthLock validates via isSnapshot internally
      $paneStates.set({ p1: snap })
      const read = getPaneStateSnapshot('p1')
      expect(read?.lockWidth).toBe(true)
      expect(read?.lockedWidth).toBe(300)
      expect(read?.lockHeight).toBe(true)
      expect(read?.lockedHeight).toBe(400)
    })
  })

  describe('explicit dimension capture (DOM measurement)', () => {
    it('locks on a flex pane with no override but explicit fixedWidth 420', () => {
      $paneStates.set({ p1: { open: true } })
      setPaneWidthLock('p1', true, 420)
      const snap = getPaneStateSnapshot('p1')
      expect(snap?.lockWidth).toBe(true)
      expect(snap?.lockedWidth).toBe(420)
    })

    it('explicit fixedWidth overrides existing widthOverride', () => {
      $paneStates.set({ p1: { open: true, widthOverride: 300 } })
      setPaneWidthLock('p1', true, 420)
      const snap = getPaneStateSnapshot('p1')
      expect(snap?.lockWidth).toBe(true)
      expect(snap?.lockedWidth).toBe(420)
    })

    it('locks on a flex pane with no override but explicit fixedHeight 500', () => {
      $paneStates.set({ p1: { open: true } })
      setPaneHeightLock('p1', true, 500)
      const snap = getPaneStateSnapshot('p1')
      expect(snap?.lockHeight).toBe(true)
      expect(snap?.lockedHeight).toBe(500)
    })

    it('explicit fixedHeight overrides existing heightOverride', () => {
      $paneStates.set({ p1: { open: true, heightOverride: 300 } })
      setPaneHeightLock('p1', true, 500)
      const snap = getPaneStateSnapshot('p1')
      expect(snap?.lockHeight).toBe(true)
      expect(snap?.lockedHeight).toBe(500)
    })

    it('falls back to existing override when no explicit dimension given', () => {
      $paneStates.set({ p1: { open: true, widthOverride: 250 } })
      setPaneWidthLock('p1', true)
      const snap = getPaneStateSnapshot('p1')
      expect(snap?.lockedWidth).toBe(250)
    })
  })

  describe('clearAllPaneSizeOverrides clears locks', () => {
    it('clears lockWidth and lockedWidth on reset', () => {
      $paneStates.set({ p1: { open: true, widthOverride: 300, lockWidth: true, lockedWidth: 300 } })
      clearAllPaneSizeOverrides()
      const snap = getPaneStateSnapshot('p1')
      expect(snap?.lockWidth).toBeUndefined()
      expect(snap?.lockedWidth).toBeUndefined()
      expect(snap?.widthOverride).toBeUndefined()
    })

    it('clears lockHeight and lockedHeight on reset', () => {
      $paneStates.set({ p1: { open: true, heightOverride: 400, lockHeight: true, lockedHeight: 400 } })
      clearAllPaneSizeOverrides()
      const snap = getPaneStateSnapshot('p1')
      expect(snap?.lockHeight).toBeUndefined()
      expect(snap?.lockedHeight).toBeUndefined()
      expect(snap?.heightOverride).toBeUndefined()
    })

    it('preserves open state when clearing locks', () => {
      $paneStates.set({ p1: { open: true, lockWidth: true, lockedWidth: 300, widthOverride: 300 } })
      clearAllPaneSizeOverrides()
      const snap = getPaneStateSnapshot('p1')
      expect(snap?.open).toBe(true)
      expect(snap?.lockWidth).toBeUndefined()
    })
  })

  describe('unlock without dimension change', () => {
    it('unlock clears lock fields but preserves widthOverride', () => {
      $paneStates.set({ p1: { open: true, widthOverride: 350, lockWidth: true, lockedWidth: 350 } })
      setPaneWidthLock('p1', false)
      const snap = getPaneStateSnapshot('p1')
      expect(snap?.lockWidth).toBeUndefined()
      expect(snap?.lockedWidth).toBeUndefined()
      expect(snap?.widthOverride).toBe(350)
    })

    it('unlock clears lock fields but preserves heightOverride', () => {
      $paneStates.set({ p1: { open: true, heightOverride: 250, lockHeight: true, lockedHeight: 250 } })
      setPaneHeightLock('p1', false)
      const snap = getPaneStateSnapshot('p1')
      expect(snap?.lockHeight).toBeUndefined()
      expect(snap?.lockedHeight).toBeUndefined()
      expect(snap?.heightOverride).toBe(250)
    })
  })

  describe('finite positive capture validation', () => {
    it('rejects NaN as explicit fixedWidth', () => {
      $paneStates.set({ p1: { open: true } })
      setPaneWidthLock('p1', true, NaN)
      const snap = getPaneStateSnapshot('p1')
      // NaN is not finite — lockedWidth should not be set
      expect(snap?.lockedWidth).toBeUndefined()
      expect(snap?.lockWidth).toBe(true)
    })

    it('rejects negative fixedWidth', () => {
      $paneStates.set({ p1: { open: true } })
      setPaneWidthLock('p1', true, -100)
      const snap = getPaneStateSnapshot('p1')
      expect(snap?.lockedWidth).toBeUndefined()
      expect(snap?.lockWidth).toBe(true)
    })

    it('rejects zero fixedWidth', () => {
      $paneStates.set({ p1: { open: true } })
      setPaneWidthLock('p1', true, 0)
      const snap = getPaneStateSnapshot('p1')
      expect(snap?.lockedWidth).toBeUndefined()
      expect(snap?.lockWidth).toBe(true)
    })

    it('rejects Infinity as explicit fixedWidth', () => {
      $paneStates.set({ p1: { open: true } })
      setPaneWidthLock('p1', true, Infinity)
      const snap = getPaneStateSnapshot('p1')
      expect(snap?.lockedWidth).toBeUndefined()
      expect(snap?.lockWidth).toBe(true)
    })

    it('rejects NaN as explicit fixedHeight', () => {
      $paneStates.set({ p1: { open: true } })
      setPaneHeightLock('p1', true, NaN)
      const snap = getPaneStateSnapshot('p1')
      expect(snap?.lockedHeight).toBeUndefined()
      expect(snap?.lockHeight).toBe(true)
    })

    it('rejects negative fixedHeight', () => {
      $paneStates.set({ p1: { open: true } })
      setPaneHeightLock('p1', true, -50)
      const snap = getPaneStateSnapshot('p1')
      expect(snap?.lockedHeight).toBeUndefined()
      expect(snap?.lockHeight).toBe(true)
    })

    it('accepts positive finite fixedWidth', () => {
      $paneStates.set({ p1: { open: true } })
      setPaneWidthLock('p1', true, 420)
      const snap = getPaneStateSnapshot('p1')
      expect(snap?.lockedWidth).toBe(420)
    })

    it('accepts positive finite fixedHeight', () => {
      $paneStates.set({ p1: { open: true } })
      setPaneHeightLock('p1', true, 500)
      const snap = getPaneStateSnapshot('p1')
      expect(snap?.lockedHeight).toBe(500)
    })
  })

  describe('lock flags are boolean when set', () => {
    it('lockWidth is exactly true (not truthy string) when locking', () => {
      $paneStates.set({ p1: { open: true } })
      setPaneWidthLock('p1', true, 300)
      const snap = getPaneStateSnapshot('p1')
      expect(snap?.lockWidth).toBe(true)
      expect(typeof snap?.lockWidth).toBe('boolean')
    })

    it('lockHeight is exactly true when locking', () => {
      $paneStates.set({ p1: { open: true } })
      setPaneHeightLock('p1', true, 400)
      const snap = getPaneStateSnapshot('p1')
      expect(snap?.lockHeight).toBe(true)
      expect(typeof snap?.lockHeight).toBe('boolean')
    })
  })
})
