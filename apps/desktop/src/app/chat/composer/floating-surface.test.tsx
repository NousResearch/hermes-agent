import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { useState } from 'react'
import { afterEach, expect, it } from 'vitest'

import { PaneGroupContext, PaneVisibleContext } from '@/components/pane-shell/pane-visibility'
import { $composerPopout, setComposerPoppedOut } from '@/store/composer-popout'

import { FloatingComposerSurface } from './floating-surface'
import { claimFloatingComposer, pinFloatingComposerCapture } from './floating-target'
import { getActiveComposer, markActiveComposer } from './focus'
import { ComposerScopeProvider, ComposerSurfaceProvider, MAIN_COMPOSER_SCOPE } from './scope'

function Surface({ id, visible = true, groupId = id }: { id: string; visible?: boolean; groupId?: string }) {
  return (
    <PaneGroupContext value={groupId}>
      <PaneVisibleContext value={visible}>
        <ComposerScopeProvider value={{ ...MAIN_COMPOSER_SCOPE, target: id }}>
          <ComposerSurfaceProvider value={id}>
            <div
              data-chat-surface=""
              data-composer-surface-id={id}
              data-testid={`pane-${id}`}
              data-tree-group={groupId}
            >
              <button>Focus {id}</button>
              <FloatingComposerSurface>
                <input aria-label={`Draft ${id}`} defaultValue={id} />
              </FloatingComposerSurface>
            </div>
          </ComposerSurfaceProvider>
        </ComposerScopeProvider>
      </PaneVisibleContext>
    </PaneGroupContext>
  )
}

afterEach(async () => {
  cleanup()
  await act(async () => {})
  $composerPopout.set({ poppedOut: false, position: { bottom: 24, right: 24 } })
})

it('shares one visible composer while preserving each editor and draft across hover, focus and docking', () => {
  render(
    <>
      <Surface id="a" />
      <Surface id="b" />
    </>
  )
  const a = screen.getByLabelText('Draft a')
  const b = screen.getByLabelText('Draft b')
  fireEvent.change(a, { target: { value: 'draft for a' } })
  fireEvent.change(b, { target: { value: 'draft for b' } })
  const hostA = a.closest<HTMLElement>('[data-composer-owner]')!
  const hostB = b.closest<HTMLElement>('[data-composer-owner]')!
  markActiveComposer('b')
  act(() => claimFloatingComposer('a'))
  expect(getActiveComposer()).toBe('a')
  act(() => setComposerPoppedOut(true))
  expect(hostA.parentElement).toBe(globalThis.document.body)
  expect(hostA.style.display).toBe('contents')
  expect(hostB.style.display).toBe('none')
  fireEvent.pointerMove(screen.getByTestId('pane-b'), { clientX: 500, clientY: 100 })
  expect(hostA.style.display).toBe('none')
  expect(hostB.style.display).toBe('contents')
  fireEvent.pointerOver(b)
  expect(hostB.style.display).toBe('contents')
  let releaseCapture!: () => void
  act(() => {
    releaseCapture = pinFloatingComposerCapture('b')
  })
  fireEvent.pointerMove(screen.getByTestId('pane-a'), { clientX: 100, clientY: 100 })
  expect(hostB.style.display).toBe('contents')
  expect(getActiveComposer()).toBe('b')
  act(() => releaseCapture())
  fireEvent.keyDown(globalThis.document.body, { key: 'Tab' })
  fireEvent.focusIn(screen.getByRole('button', { name: 'Focus a' }))
  fireEvent.keyUp(globalThis.document.body, { key: 'Tab' })
  expect(hostA.style.display).toBe('contents')
  expect(hostB.style.display).toBe('none')
  act(() => setComposerPoppedOut(false))
  expect(screen.getByLabelText('Draft a')).toBe(a)
  expect(screen.getByLabelText('Draft b')).toBe(b)
  expect((a as HTMLInputElement).value).toBe('draft for a')
  expect((b as HTMLInputElement).value).toBe('draft for b')
  expect(hostA.closest('[data-chat-surface]')).toBe(screen.getByTestId('pane-a'))
  expect(hostB.closest('[data-chat-surface]')).toBe(screen.getByTestId('pane-b'))
})

it.each([false, true])('keeps pointer-selected ownership through delayed events (floating=%s)', floating => {
  render(
    <>
      <Surface id="a" />
      <Surface id="b" />
    </>
  )
  act(() => setComposerPoppedOut(floating))
  const paneB = screen.getByTestId('pane-b')
  fireEvent.pointerOver(paneB, { clientX: 500, clientY: 100 })
  fireEvent.pointerMove(paneB, { clientX: 500, clientY: 100 })
  expect(getActiveComposer()).toBe('b')

  fireEvent.pointerOver(screen.getByTestId('pane-a'), { clientX: 500, clientY: 100 })
  fireEvent.focusIn(screen.getByRole('button', { name: 'Focus a' }))
  expect(getActiveComposer()).toBe('b')
})

it('hands the floating recipient to a visible tab when its old tab hides or closes', async () => {
  function Tabs() {
    const [first, setFirst] = useState(true)

    return (
      <>
        <button onClick={() => setFirst(false)}>Switch</button>
        <Surface groupId="tabs" id="a" visible={first} />
        <Surface groupId="tabs" id="b" visible={!first} />
      </>
    )
  }

  render(<Tabs />)
  act(() => setComposerPoppedOut(true))
  await act(async () => fireEvent.click(screen.getByText('Switch')))
  expect(screen.getByLabelText('Draft b').closest<HTMLElement>('[data-composer-owner]')!.style.display).toBe('contents')
  expect(screen.queryByLabelText('Draft a')).not.toBeNull()
})
