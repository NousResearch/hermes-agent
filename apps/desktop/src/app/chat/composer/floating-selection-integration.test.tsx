import { useStore } from '@nanostores/react'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { ReactionBadge } from '@/components/assistant-ui/thread/message-reactions'
import { PaneGroupContext, PaneVisibleContext } from '@/components/pane-shell/pane-visibility'
import { findGroup, group, split } from '@/components/pane-shell/tree/model'
import {
  $activeTreeGroup,
  $hoveredTreeGroup,
  $layoutTree,
  activateTreePane,
  activateTreeTabSlot,
  cycleTreeTabInFocusedZone,
  trackActiveTreeGroup
} from '@/components/pane-shell/tree/store'
import { registry } from '@/contrib/registry'
import { $composerPopout, $composerPopoutGesturesEnabled } from '@/store/composer-popout'
import { $selectedStoredSessionId, setSessions } from '@/store/session'
import { $focusedStoredSessionId } from '@/store/session-focus'
import { $focusedWorkspaceCwd, $sessionTiles, focusOpenSession } from '@/store/session-states'

import { $floatingComposerOwner } from './floating-state'
import { FloatingComposerSurface } from './floating-surface'
import { getActiveComposer } from './focus'
import { ComposerScopeProvider, ComposerSurfaceProvider, MAIN_COMPOSER_SCOPE } from './scope'

let stop: () => void

function Surface({ id, groupId, visible = true }: { id: string; groupId: string; visible?: boolean }) {
  return (
    <div data-pane-host={`session-tile:${id}`}>
      <PaneGroupContext value={groupId}>
        <PaneVisibleContext value={visible}>
          <ComposerScopeProvider value={{ ...MAIN_COMPOSER_SCOPE, target: `tile:${id}` }}>
            <ComposerSurfaceProvider value={id}>
              <section
                data-chat-surface=""
                data-composer-surface-id={id}
                data-composer-target={`tile:${id}`}
                data-testid={`pane-${id}`}
                data-tree-group={groupId}
              >
                <p data-testid={`text-${id}`}>Transcript {id}</p>
                <button data-testid={`button-${id}`}>Action {id}</button>
                <ReactionBadge onRetract={() => {}} reactions={[{ author: 'user', emoji: '👍', at: 0 }]} />
                <FloatingComposerSurface>
                  <div
                    contentEditable
                    data-slot="composer-rich-input"
                    data-testid={`editor-${id}`}
                    suppressContentEditableWarning
                    tabIndex={0}
                  >
                    draft {id}
                  </div>
                </FloatingComposerSurface>
              </section>
            </ComposerSurfaceProvider>
          </ComposerScopeProvider>
        </PaneVisibleContext>
      </PaneGroupContext>
    </div>
  )
}

function Fixture({ readyC = true }: { readyC?: boolean }) {
  const tree = useStore($layoutTree)
  const active = tree && findGroup(tree, 'side')?.active

  return (
    <>
      <Surface groupId="main" id="A" />
      <Surface groupId="side" id="B" visible={active === 'session-tile:B'} />
      {readyC && <Surface groupId="side" id="C" visible={active === 'session-tile:C'} />}
    </>
  )
}

function initialFocus(floating: boolean, readyC = true) {
  const view = render(<Fixture readyC={readyC} />)
  act(() => $composerPopout.set({ poppedOut: floating, position: { bottom: 24, right: 24 } }))
  fireEvent.pointerDown(screen.getByTestId('editor-A'), { pointerType: 'mouse', button: 0, buttons: 1 })
  act(() => screen.getByTestId('editor-A').focus())
  fireEvent.pointerUp(screen.getByTestId('editor-A'), { pointerType: 'mouse' })
  expect(getActiveComposer()).toBe('tile:A')

  return view
}

beforeEach(() => {
  $composerPopoutGesturesEnabled.set(true)
  $composerPopout.set({ poppedOut: false, position: { bottom: 24, right: 24 } })
  $floatingComposerOwner.set(null)
  $activeTreeGroup.set(null)
  $hoveredTreeGroup.set(null)
  $selectedStoredSessionId.set('A')
  setSessions(['A', 'B', 'C'].map(id => ({ id, cwd: `/repo-${id}` })) as never)
  $layoutTree.set(
    split('row', [
      group(['session-tile:A'], { active: 'session-tile:A', id: 'main' }),
      group(['session-tile:B', 'session-tile:C'], { active: 'session-tile:B', id: 'side' })
    ])
  )

  for (const id of ['A', 'B', 'C']) {
    registry.register({
      area: 'panes',
      id: `session-tile:${id}`,
      data: { placement: 'main' },
      render: () => null,
      title: id
    })
  }

  stop = trackActiveTreeGroup()
})

afterEach(async () => {
  cleanup()
  stop()
  await act(async () => {})
  window.getSelection()?.removeAllRanges()
  $floatingComposerOwner.set(null)
  $sessionTiles.set([])
})

describe.each([false, true])('explicit selection integration (floating=%s)', floating => {
  it('moves typing focus with a clicked reaction that prevents default focus', async () => {
    initialFocus(floating)

    const reaction = screen
      .getByTestId('pane-B')
      .querySelector<HTMLButtonElement>('[data-slot="aui_msg-reactions"] button')!

    expect(fireEvent.pointerDown(reaction, { pointerType: 'mouse', button: 0, buttons: 1 })).toBe(false)
    fireEvent.pointerUp(reaction, { pointerType: 'mouse' })
    fireEvent.click(reaction)
    await act(async () => {})
    expect(getActiveComposer()).toBe('tile:B')
    expect($focusedWorkspaceCwd.get()).toBe('/repo-B')
    expect(globalThis.document.activeElement).toBe(screen.getByTestId('editor-B'))
  })

  it.each(['slot', 'cycle'])(
    'selects the other split and its typing recipient on an explicit %s command',
    async method => {
      initialFocus(floating)
      fireEvent.pointerOver(screen.getByTestId('text-B'))
      fireEvent.pointerMove(screen.getByTestId('text-B'), { pointerType: 'mouse', buttons: 0 })
      expect(getActiveComposer()).toBe('tile:A')

      await act(async () => {
        const activated = method === 'slot' ? activateTreeTabSlot(2) : cycleTreeTabInFocusedZone(1)
        expect(activated).toBe('session-tile:C')
      })

      expect(getActiveComposer()).toBe('tile:C')
      expect($focusedStoredSessionId.get()).toBe('C')
      expect($focusedWorkspaceCwd.get()).toBe('/repo-C')
      expect(globalThis.document.activeElement).toBe(screen.getByTestId('editor-C'))
    }
  )

  it('does not claim typing focus when a background action only activates a tree pane', async () => {
    initialFocus(floating)
    await act(async () => activateTreePane('side', 'session-tile:C'))
    expect(getActiveComposer()).toBe('tile:A')
    expect(globalThis.document.activeElement).toBe(screen.getByTestId('editor-A'))
  })

  it('aligns an explicitly reopened visible session with its composer', async () => {
    initialFocus(floating)
    act(() => $sessionTiles.set([{ storedSessionId: 'B', runtimeId: 'runtime-B' }]))
    await act(async () => expect(focusOpenSession('B')).toBe('tile'))
    expect(getActiveComposer()).toBe('tile:B')
    expect($focusedWorkspaceCwd.get()).toBe('/repo-B')
    expect(globalThis.document.activeElement).toBe(screen.getByTestId('editor-B'))
  })

  it('fulfills an explicitly selected delayed pane when no newer choice was made', async () => {
    const view = initialFocus(floating, false)
    fireEvent.pointerOver(screen.getByTestId('text-B'))
    await act(async () => expect(activateTreeTabSlot(2)).toBe('session-tile:C'))
    await act(async () => view.rerender(<Fixture readyC />))
    expect(getActiveComposer()).toBe('tile:C')
    expect(globalThis.document.activeElement).toBe(screen.getByTestId('editor-C'))
  })

  it.each(['button', 'editor'])(
    'does not let a delayed pane override a newer Tab choice of the same owner %s',
    async kind => {
      const view = initialFocus(floating, false)
      fireEvent.pointerOver(screen.getByTestId('text-B'))
      await act(async () => expect(activateTreeTabSlot(2)).toBe('session-tile:C'))
      const chosen = screen.getByTestId(`${kind}-A`)
      fireEvent.keyDown(globalThis.document.body, { key: 'Tab', shiftKey: floating })
      act(() => chosen.focus())
      fireEvent.keyUp(globalThis.document.body, { key: 'Tab', shiftKey: floating })
      await act(async () => view.rerender(<Fixture readyC />))
      expect(getActiveComposer()).toBe('tile:A')
      expect($focusedWorkspaceCwd.get()).toBe('/repo-A')
      expect(globalThis.document.activeElement).toBe(chosen)
    }
  )
})
