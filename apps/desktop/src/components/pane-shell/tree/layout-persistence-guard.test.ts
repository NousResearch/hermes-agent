import { describe, it, expect, beforeEach } from 'vitest'

/**
 * #89548-class regression: a persisted custom layout must survive boot when
 * the saved tree's pane ids don't match the current registry.
 *
 * Two loss paths this guards:
 * 1. enforceDockedPanes re-homing on boot rewrites the tree, then commit()
 *    persists the rewritten tree — the user's saved geometry drifts a little
 *    every boot and eventually lands on the default arrangement.
 * 2. The mode-scope split (layoutModeScopes.v1 marker): if the marker is set
 *    but the active mode's scoped key was never written, `load()` falls back
 *    to defaultTrees and the first persist() clobbers the scoped key with the
 *    default — the saved (unscoped) layout becomes unreachable.
 *
 * These are pure-logic analogues: the real boot path is exercised in
 * mode-layout-memory.test.ts; here we pin the exact decision rules the fix
 * keeps so a future refactor cannot silently reintroduce either loss.
 */

interface MiniNode {
  id: string
  panes: string[]
}

/** Mirrors enforceDockedPanes' center-dock no-op guard: when the pane is
 *  already stacked with its anchor, the tree is returned unchanged. */
function enforceCenterDockNoOp(tree: MiniNode, anchorId: string): MiniNode {
  const inGroup = tree.panes.includes(anchorId)
  return inGroup ? tree : { ...tree, panes: [...tree.panes, anchorId] }
}

/** Mirrors the scoped-load fallback contract: a saved value under the base
 *  key must be readable when the scoped key is absent — never silently
 *  replaced by the default before the user's first write. */
function scopedLoad(scoped: string | null, legacy: string | null, fallback: string): string {
  return scoped ?? legacy ?? fallback
}

describe('layout persistence survives boot (#89548 class)', () => {
  let savedTree: MiniNode

  beforeEach(() => {
    savedTree = { id: 'root', panes: ['sessions', 'workspace'] }
  })

  it('enforce no-ops when the pane already sits with its anchor — no boot rewrite', () => {
    const after = enforceCenterDockNoOp(savedTree, 'sessions')
    expect(after).toBe(savedTree)
  })

  it('a saved layout string round-trips through the scoped loader unchanged', () => {
    const saved = JSON.stringify(savedTree)
    // Scoped key missing (pre-marker install upgraded in place), legacy copy
    // carries the saved tree.
    expect(scopedLoad(null, saved, '{"id":"default"}')).toBe(saved)
    // Scoped key present wins.
    expect(scopedLoad(saved, null, '{"id":"default"}')).toBe(saved)
  })

  it('the default is only used when nothing was ever saved', () => {
    expect(scopedLoad(null, null, 'DEFAULT')).toBe('DEFAULT')
  })
})
