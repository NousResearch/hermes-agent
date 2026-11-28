import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $activeTip } from '@/store/tips'

import { TipHost } from './index'

vi.mock('./use-tip-rotation', () => ({ useTipRotation: () => undefined }))
vi.mock('./tip-bubble', () => ({ TipBubble: ({ title }: { title: string }) => <p role="note">{title}</p> }))

const TIP = { id: 'model', side: 'top', targets: ['[data-tour="model-pill"]'], text: 'Pick a model', title: 'Models' }

/** What the browser reports on top of the pill (jsdom has no hit testing). */
let onTop: () => Element | null

beforeEach(() => {
  vi.useFakeTimers()
  vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue(new DOMRect(0, 0, 160, 32))
  window.document.elementFromPoint = () => onTop()
  onTop = () => screen.getByRole('button', { name: 'Model' })
})

afterEach(() => {
  cleanup()
  $activeTip.set(null)
  vi.useRealTimers()
  vi.restoreAllMocks()
})

function mount() {
  const view = render(
    <main>
      <button data-tour="model-pill">Model</button>
      <TipHost />
    </main>
  )

  act(() => $activeTip.set(TIP as never))

  return view
}

const aSecond = () => act(() => vi.advanceTimersByTime(1_000))

// A menu, a dialog or a toast over the anchor is not the anchor going away: the tip steps back while
// it is covered and returns, instead of being spent on a moment the user never saw it in.
it('steps back while its anchor is covered and returns when it is not', () => {
  const view = mount()
  expect(screen.getByRole('note').textContent).toBe('Models')

  // A Radix layer opens: the app root is aria-hidden and the layer is what a point on the pill hits.
  view.container.setAttribute('aria-hidden', 'true')
  onTop = () => window.document.body
  aSecond()
  expect(screen.queryByRole('note')).toBeNull()
  expect($activeTip.get()).not.toBeNull()

  view.container.removeAttribute('aria-hidden')
  onTop = () => screen.getByRole('button', { name: 'Model' })
  aSecond()
  expect(screen.getByRole('note').textContent).toBe('Models')
})

it('ends when its anchor leaves the page', () => {
  const view = mount()

  view.rerender(
    <main>
      <TipHost />
    </main>
  )
  aSecond()

  expect($activeTip.get()).toBeNull()
})

it('ends when its anchor is in a pane that is no longer the shown one', () => {
  const view = mount()

  view.container.querySelector('main')!.setAttribute('data-pane-hidden', '')
  aSecond()

  expect($activeTip.get()).toBeNull()
})
