import { beforeEach, describe, expect, it, vi } from 'vitest'

import type * as TreeModel from '@/components/pane-shell/tree/model'
import type * as TreeStore from '@/components/pane-shell/tree/store'
import { NEW_SESSION_DRAG } from '@/components/pane-shell/tree/store'

import { type NewSessionPlacement, startNewSessionDrag } from './new-session-drag'

// ---------------------------------------------------------------------------
// The drag MACHINERY (startDragSession) is exercised by the pane-shell's own
// tests; here we capture the spec the resolver hands it and drive resolveMove /
// onCommit directly. Geometry helpers are mocked so targeting is deterministic
// without real layout — the assertions are about the DROP LANGUAGE (stack /
// split / deny) and the create-on-commit contract, which is what's new here.
// ---------------------------------------------------------------------------

const captured: { spec: null | Record<string, any> } = { spec: null }

// vi.mock factories are hoisted above these declarations, so the mocks they
// reference must be created in a vi.hoisted() block (available at hoist time)
// rather than as plain top-level consts (a TDZ error when the factory runs).
const { findGroup, setTreeDragging, slotBefore, subZonePosition } = vi.hoisted(() => ({
  findGroup: vi.fn(),
  setTreeDragging: vi.fn(),
  slotBefore: vi.fn(() => ({ before: 'workspace' })),
  subZonePosition: vi.fn()
}))

vi.mock('@/components/pane-shell/tree/renderer/drag-session', () => ({
  rectContains: (rect: { bottom: number; left: number; right: number; top: number }, x: number, y: number) =>
    x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom,
  slotBefore,
  snapshotStrips: () => [
    { groupId: 'g1', rect: { bottom: 40, left: 0, right: 800, top: 0 }, slots: [{ id: 'workspace', mid: 400 }] }
  ],
  snapshotZones: () => [{ id: 'g1', rect: { bottom: 600, left: 0, right: 800, top: 0 } }],
  startDragSession: (_e: unknown, spec: Record<string, any>) => {
    captured.spec = spec
  },
  subZonePosition
}))

vi.mock('@/components/pane-shell/tree/store', async importOriginal => {
  const actual = await importOriginal<typeof TreeStore>()

  return {
    ...actual,
    $layoutTree: { get: () => ({}) },
    $treeDragging: { set: setTreeDragging }
  }
})

vi.mock('@/components/pane-shell/tree/model', async importOriginal => {
  const actual = await importOriginal<typeof TreeModel>()

  return {
    ...actual,
    findGroup
  }
})

vi.mock('@/i18n', () => ({ translateNow: () => 'New session' }))

// A chat zone hosting the workspace pane — the only kind of zone a new session
// may land in.
findGroup.mockImplementation((_tree: unknown, groupId: string) =>
  groupId === 'g1' ? { panes: ['workspace'] } : null
)

const fakePointerEvent = () =>
  ({
    button: 0,
    clientX: 0,
    clientY: 0,
    currentTarget: { style: { opacity: '', setProperty: vi.fn() } },
    pointerId: 1
  }) as unknown as React.PointerEvent<HTMLElement>

function engage(onCreate: (placement: NewSessionPlacement) => void = vi.fn()) {
  startNewSessionDrag(onCreate, fakePointerEvent())
  const spec = captured.spec

  if (!spec) {throw new Error('startDragSession was not called')}
  spec.onEngage(0, 0)

  return spec
}

describe('startNewSessionDrag', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    captured.spec = null
    findGroup.mockImplementation((_tree: unknown, groupId: string) =>
      groupId === 'g1' ? { panes: ['workspace'] } : null
    )
  })

  it('advertises the distinct NEW_SESSION_DRAG sentinel on engage', () => {
    engage()
    expect(setTreeDragging).toHaveBeenCalledWith(NEW_SESSION_DRAG)
  })

  it('drops a ghost labelled like the New session row', () => {
    startNewSessionDrag(vi.fn(), fakePointerEvent())
    expect(captured.spec?.ghost).toEqual({ label: 'New session' })
  })

  it('stacks a fresh tab on a center drop (never links — nothing to link yet)', () => {
    const onCreate = vi.fn()
    subZonePosition.mockReturnValue('center')
    const spec = engage(onCreate)

    const hint = spec.resolveMove(400, 300, false)

    expect(hint).toMatchObject({ groupId: 'g1', pos: 'center' })
    expect(hint?.stack).toBeUndefined()

    spec.onCommit(hint)
    expect(onCreate).toHaveBeenCalledWith({ anchor: 'workspace', dir: 'center' } satisfies NewSessionPlacement)
  })

  it('splits a new tile off a zone edge', () => {
    const onCreate = vi.fn()
    subZonePosition.mockReturnValue('right')
    const spec = engage(onCreate)

    const hint = spec.resolveMove(780, 300, false)

    expect(hint).toMatchObject({ groupId: 'g1', pos: 'right' })

    spec.onCommit(hint)
    expect(onCreate).toHaveBeenCalledWith({ anchor: 'workspace', dir: 'right' } satisfies NewSessionPlacement)
  })

  it('stacks at a tab-strip slot, carrying the insertion point', () => {
    const onCreate = vi.fn()
    slotBefore.mockReturnValue({ before: 'workspace' })
    const spec = engage(onCreate)

    // Inside the strip band (top 40px of the zone).
    const hint = spec.resolveMove(150, 20, false)

    expect(hint?.stack).toEqual({ before: 'workspace' })

    spec.onCommit(hint)
    expect(onCreate).toHaveBeenCalledWith({
      anchor: 'workspace',
      before: 'workspace',
      dir: 'center'
    } satisfies NewSessionPlacement)
  })

  it('creates nothing when released over a deny zone', () => {
    const onCreate = vi.fn()
    const spec = engage(onCreate)

    // Far outside any zone.
    const hint = spec.resolveMove(5000, 5000, false)

    expect(hint).toBeNull()

    spec.onCommit(hint)
    expect(onCreate).not.toHaveBeenCalled()
  })

  it('creates nothing when the drag never resolves a target before commit', () => {
    const onCreate = vi.fn()
    const spec = engage(onCreate)

    // Commit straight after engage, with no move over a valid zone.
    spec.onCommit(null)
    expect(onCreate).not.toHaveBeenCalled()
  })

  it('restores the source row opacity on end', () => {
    const event = fakePointerEvent()
    const source = event.currentTarget as unknown as { style: { opacity: string; setProperty: ReturnType<typeof vi.fn> } }
    startNewSessionDrag(vi.fn(), event)
    const spec = captured.spec!

    spec.onEngage(0, 0)
    expect(source.style.setProperty).toHaveBeenCalledWith('opacity', '0.45')

    spec.onEnd()
    expect(source.style.opacity).toBe('')
  })
})
