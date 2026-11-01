import { computed } from 'nanostores'
import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { group, split } from '@/components/pane-shell/tree/model'
import {
  $dismissedPanes,
  $hiddenTreePanes,
  $layoutTree,
  bindPaneVisibility,
  isPaneVisible
} from '@/components/pane-shell/tree/store'
import { registry } from '@/contrib/registry'

import { $reviewOpen, closeReview, revealReview, REVIEW_PANE_ID } from './review'
import { $currentCwd } from './session'

// requestOneShot reaches the gateway; stub it like review.test.ts does. The
// tree/registry harness mirrors the pane-shell tests' setup so the real
// revealReview/closeReview pair runs against the real pane tree.
vi.mock('@/lib/oneshot', () => ({ requestOneShot: vi.fn(async () => '') }))
vi.mock('./coding-status', () => ({ refreshRepoStatus: vi.fn(), repoStatusForCwd: () => ({ get: () => null }) }))

const disposers: (() => void)[] = []

beforeAll(() => {
  for (const [id, data] of [
    ['workspace', { placement: 'main', uncloseable: true }],
    ['review', { placement: 'right', collapsible: true }]
  ] as const) {
    disposers.push(registry.register({ area: 'panes', data, id, render: () => null, title: id }))
  }

  // The controller's binding: the pane's visibility is `open && hasWorkspace`
  // — the workspace gate that #135463 turns into an asymmetric close.
  bindPaneVisibility(
    'review',
    computed([$reviewOpen, $currentCwd], (open, cwd) => open && Boolean(cwd.trim())),
    () => closeReview(),
    () => revealReview()
  )
})

beforeEach(() => {
  window.localStorage.clear()
  $dismissedPanes.set(new Set())
  $hiddenTreePanes.set(new Set())
  $reviewOpen.set(false)
  $currentCwd.set('')

  $layoutTree.set(
    split('row', [
      group(['workspace'], { active: 'workspace', id: 'g-main' }),
      group(['review'], { active: 'review', id: 'g-review' })
    ])
  )
})

describe('closing the review pane after the reveal path (#135463)', () => {
  it('the ✕ hides the pane even with no cwd to gate the visibility binding', () => {
    // The changed-files card's route: revealReview fronts the pane through
    // the tree, bypassing the workspace-gated binding's computed.
    revealReview()

    expect($reviewOpen.get()).toBe(true)
    expect(isPaneVisible(REVIEW_PANE_ID)).toBe(true)

    // The pane header's ✕ — a store flip whose binding listener never fires
    // (the computed is pinned false by the empty cwd), so the close must
    // reach the tree on its own.
    closeReview()

    expect(isPaneVisible(REVIEW_PANE_ID)).toBe(false)
  })

  it('the ✕ still hides the pane with a cwd (binding-driven path unchanged)', () => {
    $currentCwd.set('/repo')

    revealReview()
    expect(isPaneVisible(REVIEW_PANE_ID)).toBe(true)

    closeReview()
    expect(isPaneVisible(REVIEW_PANE_ID)).toBe(false)
  })
})
