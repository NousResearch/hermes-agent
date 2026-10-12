/**
 * Docked-path probe for issue #122761 (NOT part of the shipped suite — a
 * one-shot verification script run from the fix worktree's vitest config).
 *
 * Boots the real tree store like mode-layout-memory.test.ts, writes a sash
 * drag override on the bots pane (the sessions|bots zone), then drives the
 * full chat-switch churn: session-tile open/close through paneMirror
 * unregister → removeTreePane → re-adopt, routines pane unregister/re-register
 * across botChatOwnsWorkspace() edges, and preset re-adoption. Asserts the
 * persisted widthOverride and the resolved fixed track survive every edge.
 */
import { beforeEach, expect, it, vi } from 'vitest'

beforeEach(() => {
  window.localStorage.clear()
  vi.resetModules()
})

async function boot() {
  const panes = await import('@/store/panes')
  const tree = await import('@/components/pane-shell/tree/store')
  const { registry } = await import('@/contrib/registry')

  const { registerLayoutPresets, DEFAULT_TREE, BASIC_TREE } = await import('@/app/contrib/layout-presets')

  const { fixedTrackSize } = await import('@/components/pane-shell/tree/renderer/track-model')

  for (const [id, placement] of [
    ['sessions', 'left'],
    ['workspace', 'main'],
    ['files', 'right']
  ] as const) {
    registry.register({
      id,
      area: 'panes',
      data: {
        placement,
        width: id === 'sessions' ? '237px' : undefined
      },
      render: () => null
    })
  }

  registry.register({
    id: 'bots',
    area: 'panes',
    render: () => null,
    data: {
      placement: 'left',
      width: '260px',
      dock: { pane: 'sessions', pos: 'center', enforce: true },
      collapsible: true,
      hideOnly: true
    }
  })
  registry.register({
    id: 'routines',
    area: 'panes',
    render: () => null,
    data: {
      placement: 'main',
      width: '250px',
      defaultCollapsed: true,
      dock: { pane: 'workspace', pos: 'right', enforce: true }
    }
  })
  registerLayoutPresets()
  tree.declareDefaultTree(DEFAULT_TREE, BASIC_TREE)
  tree.watchContributedPanes()

  return { panes, tree, registry, fixedTrackSize }
}

it('keeps the user-dragged bots zone width across full chat-switch churn', { timeout: 120_000 }, async () => {
  const { panes, tree, registry, fixedTrackSize } = await boot()

  // Sash drag: user narrows the shared zone to 170px — the sash writes the
  // same px override to EVERY shown pane of the zone (see commitPlan in
  // tree-split.tsx), so the probe writes both like a real drag would.
  panes.setPaneWidthOverride('sessions', 170)
  panes.setPaneWidthOverride('bots', 170)

  const zoneOf = (id: string) => {
    const t = tree.$layoutTree.get()!

    const find = (node: unknown): unknown => {
      const n = node as { type?: string; panes?: string[]; children?: unknown[] }

      if (n?.type === 'group' && n.panes?.includes(id)) {
        return n
      }

      return n?.children?.map(find).find(Boolean) ?? null
    }

    return find(t) as never
  }

  const ctx = {
    paneFor: (id: string) => registry.getArea('panes').find(p => p.id === id),
    paneGone: (id: string) => tree.$dismissedPanes.get().has(id),
    overrides: panes.$paneStates.get()
  }

  const trackBefore = fixedTrackSize(zoneOf('bots'), 'row', ctx)
  expect(trackBefore).toBe('170px')

  // ── Chat-switch churn #1: routines pane re-register (botChatOwnsWorkspace edge)
  const routinesDispose = registry.register({
    id: 'routines',
    area: 'panes',
    render: () => null,
    data: {
      placement: 'main',
      width: '250px',
      defaultCollapsed: true,
      dock: { pane: 'workspace', pos: 'right', enforce: true }
    }
  })

  routinesDispose()

  // ── Churn #2: remove + re-adopt a session tile pane (the workspace churn path)
  tree.removeTreePane('workspace')

  const tileDispose = registry.register({
    id: 'workspace',
    area: 'panes',
    render: () => null,
    data: { placement: 'main', uncloseable: true }
  })

  tileDispose()
  registry.register({
    id: 'workspace',
    area: 'panes',
    render: () => null,
    data: { placement: 'main', uncloseable: true }
  })

  // ── Churn #3: the bots pane itself re-registers (enforced dock re-adoption)
  const botsDispose = registry.register({
    id: 'bots',
    area: 'panes',
    render: () => null,
    data: {
      placement: 'left',
      width: '260px',
      dock: { pane: 'sessions', pos: 'center', enforce: true },
      collapsible: true,
      hideOnly: true
    }
  })

  botsDispose()
  registry.register({
    id: 'bots',
    area: 'panes',
    render: () => null,
    data: {
      placement: 'left',
      width: '260px',
      dock: { pane: 'sessions', pos: 'center', enforce: true },
      collapsible: true,
      hideOnly: true
    }
  })

  // After ALL churn: the override survived in the store, and the docked zone
  // still resolves the user's width, not the declared 260px.
  expect(panes.$paneStates.get().sessions?.widthOverride).toBe(170)
  expect(panes.$paneStates.get().bots?.widthOverride).toBe(170)

  const freshCtx = {
    paneFor: (id: string) => registry.getArea('panes').find(p => p.id === id),
    paneGone: (id: string) => tree.$dismissedPanes.get().has(id),
    overrides: panes.$paneStates.get()
  }

  const trackAfter = fixedTrackSize(zoneOf('bots'), 'row', freshCtx)
  expect(trackAfter).toBe('170px')
})
