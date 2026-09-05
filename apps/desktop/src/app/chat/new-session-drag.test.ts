import { beforeEach, describe, expect, it, vi } from 'vitest'

import type * as TreeModel from '@/components/pane-shell/tree/model'
import type * as TreeStore from '@/components/pane-shell/tree/store'
import { NEW_SESSION_DRAG } from '@/components/pane-shell/tree/store'

import { type NewSessionPlacement, startNewProjectDrag, startNewSessionDrag } from './new-session-drag'

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

vi.mock('@/i18n', () => ({
  // Key-aware so each resolver's ghost label is pinned to its own string.
  translateNow: (key: string) => (key === 'sidebar.projects.newButton' ? 'New project' : 'New session')
}))

// A chat zone hosting the workspace pane — the only kind of zone a new session
// may land in.
findGroup.mockImplementation((_tree: unknown, groupId: string) => (groupId === 'g1' ? { panes: ['workspace'] } : null))

const fakePointerEvent = () =>
  ({
    button: 0,
    clientX: 0,
    clientY: 0,
    currentTarget: { style: { opacity: '', setProperty: vi.fn() } },
    pointerId: 1
  }) as unknown as React.PointerEvent<HTMLElement>

function engage(
  onCreate: (placement: NewSessionPlacement) => void = vi.fn(),
  opts?: Parameters<typeof startNewSessionDrag>[2]
) {
  startNewSessionDrag(onCreate, fakePointerEvent(), opts)
  const spec = captured.spec

  if (!spec) {
    throw new Error('startDragSession was not called')
  }

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

    const source = event.currentTarget as unknown as {
      style: { opacity: string; setProperty: ReturnType<typeof vi.fn> }
    }

    startNewSessionDrag(vi.fn(), event)
    const spec = captured.spec!

    spec.onEngage(0, 0)
    expect(source.style.setProperty).toHaveBeenCalledWith('opacity', '0.45')

    spec.onEnd()
    expect(source.style.opacity).toBe('')
  })

  it('labels the ghost with the project name for a project-row drag', () => {
    startNewSessionDrag(vi.fn(), fakePointerEvent(), { cwd: '/repo', label: 'New session in Hermes Browser' })
    expect(captured.spec?.ghost).toEqual({ label: 'New session in Hermes Browser' })
  })

  it('pins the created session to the project cwd on a center drop', () => {
    const onCreate = vi.fn()
    subZonePosition.mockReturnValue('center')
    const spec = engage(onCreate, { cwd: '/repo/hermes-browser' })

    const hint = spec.resolveMove(400, 300, false)
    spec.onCommit(hint)

    expect(onCreate).toHaveBeenCalledWith({
      anchor: 'workspace',
      cwd: '/repo/hermes-browser',
      dir: 'center'
    } satisfies NewSessionPlacement)
  })

  it('pins the created session to the project cwd on an edge split', () => {
    const onCreate = vi.fn()
    subZonePosition.mockReturnValue('left')
    const spec = engage(onCreate, { cwd: '/repo/hermes-browser' })

    const hint = spec.resolveMove(20, 300, false)
    spec.onCommit(hint)

    expect(onCreate).toHaveBeenCalledWith({
      anchor: 'workspace',
      cwd: '/repo/hermes-browser',
      dir: 'left'
    } satisfies NewSessionPlacement)
  })

  it('pins the created session to a profile for a profile-group drag', () => {
    const onCreate = vi.fn()
    subZonePosition.mockReturnValue('center')
    const spec = engage(onCreate, { profile: 'reviewer' })

    const hint = spec.resolveMove(400, 300, false)
    spec.onCommit(hint)

    expect(onCreate).toHaveBeenCalledWith({
      anchor: 'workspace',
      cwd: undefined,
      dir: 'center',
      profile: 'reviewer'
    } satisfies NewSessionPlacement)
  })

  it('leaves cwd undefined for a plain New session drag', () => {
    const onCreate = vi.fn()
    subZonePosition.mockReturnValue('center')
    const spec = engage(onCreate)

    const hint = spec.resolveMove(400, 300, false)
    spec.onCommit(hint)

    expect(onCreate).toHaveBeenCalledWith({ anchor: 'workspace', cwd: undefined, dir: 'center' })
  })

  it('ignores an inactive tab surface when resolving the center-drop anchor', () => {
    const rect = { bottom: 600, height: 600, left: 0, right: 800, top: 0, width: 800, x: 0, y: 0, toJSON: () => ({}) }
    document.body.innerHTML = `
      <div data-pane-hidden>
        <div data-session-anchor="session-tile:hidden"></div>
      </div>
      <div data-session-anchor="workspace"></div>
    `

    for (const element of document.querySelectorAll<HTMLElement>('[data-session-anchor]')) {
      element.getBoundingClientRect = vi.fn(() => rect)
    }

    const onCreate = vi.fn()
    subZonePosition.mockReturnValue('center')
    const spec = engage(onCreate)
    const hint = spec.resolveMove(400, 300, false)
    spec.onCommit(hint)

    expect(onCreate).toHaveBeenCalledWith({ anchor: 'workspace', cwd: undefined, dir: 'center' })
  })

  it('ignores an inactive tab composer when resolving an edge split', () => {
    const rect = { bottom: 600, height: 600, left: 0, right: 800, top: 0, width: 800, x: 0, y: 0, toJSON: () => ({}) }
    document.body.innerHTML = `
      <div data-session-anchor="workspace"></div>
      <div data-pane-hidden>
        <div data-slot="composer-root"></div>
      </div>
    `
    document.querySelector<HTMLElement>('[data-session-anchor]')!.getBoundingClientRect = vi.fn(() => rect)
    document.querySelector<HTMLElement>('[data-slot="composer-root"]')!.getBoundingClientRect = vi.fn(() => rect)

    const onCreate = vi.fn()
    subZonePosition.mockReturnValue('right')
    const spec = engage(onCreate)
    const hint = spec.resolveMove(780, 300, false)
    spec.onCommit(hint)

    expect(onCreate).toHaveBeenCalledWith({ anchor: 'workspace', cwd: undefined, dir: 'right' })
  })
})

// ---------------------------------------------------------------------------
// startNewProjectDrag — the "New project" + variant. Same drop language, but
// the placement ARMS the dialog flow instead of creating anything at release:
// a valid commit arms + opens the dialog, a sub-threshold release is the plain
// click, and an aborted/deny-zone drag clears any stale arm.
// ---------------------------------------------------------------------------

describe('startNewProjectDrag', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    captured.spec = null
    findGroup.mockImplementation((_tree: unknown, groupId: string) =>
      groupId === 'g1' ? { panes: ['workspace'] } : null
    )
  })

  function engageProject(onArm: (placement: NewSessionPlacement | null) => void = vi.fn(), onTap?: () => void) {
    startNewProjectDrag(onArm, fakePointerEvent(), { onTap })
    const spec = captured.spec

    if (!spec) {
      throw new Error('startDragSession was not called')
    }

    spec.onEngage(0, 0)

    return spec
  }

  it('arms the dropped placement and opens the project dialog on a valid commit', () => {
    const onArm = vi.fn()
    const onTap = vi.fn()
    subZonePosition.mockReturnValue('center')
    const spec = engageProject(onArm, onTap)

    const hint = spec.resolveMove(400, 300, false)
    spec.onCommit(hint)

    expect(onArm).toHaveBeenCalledWith({ anchor: 'workspace', dir: 'center' })
    // The SAME dialog a plain click opens — the drag only adds the placement.
    expect(onTap).toHaveBeenCalledOnce()
  })

  it('opens NOTHING on a denied drop — same language as Escape', () => {
    const onArm = vi.fn()
    const onTap = vi.fn()
    const spec = engageProject(onArm, onTap)

    // Released far outside any zone: the drop reads as cancelled.
    const hint = spec.resolveMove(5000, 5000, false)

    expect(hint).toBeNull()

    spec.onCommit(hint)

    expect(onTap).not.toHaveBeenCalled()

    // Teardown still sweeps any stale arm (onEnd sees placement === null).
    spec.onEnd()
    expect(onArm).toHaveBeenCalledOnce()
    expect(onArm).toHaveBeenCalledWith(null)
  })

  it('carries the tab-strip slot in the armed placement', () => {
    const onArm = vi.fn()
    slotBefore.mockReturnValue({ before: 'session-tile:abc' })
    const spec = engageProject(onArm)

    const hint = spec.resolveMove(150, 20, false)
    spec.onCommit(hint)

    expect(onArm).toHaveBeenCalledWith({ anchor: 'workspace', before: 'session-tile:abc', dir: 'center' })
  })

  it('keeps a sub-threshold release an ordinary dialog click — nothing armed', () => {
    const onArm = vi.fn()
    const onTap = vi.fn()
    engageProject(onArm, onTap)

    captured.spec!.onTap!()

    expect(onTap).toHaveBeenCalledOnce()
    expect(onArm).not.toHaveBeenCalled()
  })

  it('clears the arm when released over a deny zone', () => {
    const onArm = vi.fn()
    const spec = engageProject(onArm)

    const hint = spec.resolveMove(5000, 5000, false)

    expect(hint).toBeNull()

    spec.onCommit(hint)
    spec.onEnd()

    expect(onArm).toHaveBeenLastCalledWith(null)
  })

  it('does not clear a committed arm on drag end', () => {
    const onArm = vi.fn()
    subZonePosition.mockReturnValue('right')
    const spec = engageProject(onArm)

    const hint = spec.resolveMove(780, 300, false)
    spec.onCommit(hint)
    spec.onEnd()

    expect(onArm).toHaveBeenCalledOnce()
    expect(onArm).not.toHaveBeenCalledWith(null)
  })

  it('drops a ghost labelled like the New project control', () => {
    engageProject()
    expect(captured.spec?.ghost).toEqual({ label: 'New project' })
  })
})
