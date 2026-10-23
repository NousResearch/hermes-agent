import { act, fireEvent, render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useScreenFullscreen } from './use-screen-fullscreen'

let current: Element | null
let request: ReturnType<typeof vi.fn>
let exit: ReturnType<typeof vi.fn>

function Fixture({ visible = true }: { visible?: boolean }) {
  const screen = useScreenFullscreen()

  return (
    <>
      <button onClick={() => void screen.enter()} ref={screen.trigger}>
        Enter
      </button>
      {visible ? (
        <div data-testid="viewport" ref={screen.viewportRef}>
          <input aria-label="Remote keyboard" />
          <div ref={screen.controls}>
            <button onClick={() => void screen.exit()}>Exit</button>
          </div>
        </div>
      ) : null}
      <output>{screen.active ? 'Fullscreen' : 'Normal'}</output>
      {screen.failed ? <div role="alert">Fullscreen failed</div> : null}
    </>
  )
}

function change(element: Element | null) {
  current = element
  window.document.dispatchEvent(new Event('fullscreenchange'))
}

beforeEach(() => {
  current = null
  request = vi.fn(async function (this: HTMLElement) {
    change(this)
  })
  exit = vi.fn(async () => {
    change(null)
  })
  Object.defineProperty(window.document, 'fullscreenElement', { configurable: true, get: () => current })
  Object.defineProperty(HTMLElement.prototype, 'requestFullscreen', { configurable: true, value: request })
  Object.defineProperty(window.document, 'exitFullscreen', { configurable: true, value: exit })
})

afterEach(() => {
  Reflect.deleteProperty(HTMLElement.prototype as Partial<HTMLElement>, 'requestFullscreen')
  Reflect.deleteProperty(window.document as Partial<Document>, 'exitFullscreen')
  Reflect.deleteProperty(window.document, 'fullscreenElement')
})

describe('useScreenFullscreen', () => {
  it('tracks native fullscreen changes, including exits it did not cause, and restores focus', async () => {
    request.mockResolvedValue(undefined)
    const view = render(<Fixture />)
    const enter = view.getByText('Enter')
    enter.focus()
    fireEvent.click(enter)
    await act(async () => {})
    expect(view.getByText('Normal')).toBeTruthy()
    act(() => change(view.getByTestId('viewport')))
    expect(view.getByText('Fullscreen')).toBeTruthy()
    view.getByLabelText('Remote keyboard').focus()
    act(() => change(null))
    expect(view.getByText('Normal')).toBeTruthy()
    expect(window.document.activeElement).toBe(enter)
  })

  it('reports unavailable APIs and rejected requests without inventing fullscreen state', async () => {
    Reflect.deleteProperty(HTMLElement.prototype as Partial<HTMLElement>, 'requestFullscreen')
    const view = render(<Fixture />)
    fireEvent.click(view.getByText('Enter'))
    expect(view.getByRole('alert')).toBeTruthy()
    Object.defineProperty(HTMLElement.prototype, 'requestFullscreen', {
      configurable: true,
      value: vi.fn(async () => {
        throw new Error('denied')
      })
    })
    fireEvent.click(view.getByText('Enter'))
    await act(async () => {})
    expect(view.getByRole('alert')).toBeTruthy()
    expect(current).toBeNull()
  })

  it('reserves Escape locally, including keyup after exit, but not ordinary input', async () => {
    const view = render(<Fixture />)
    fireEvent.click(view.getByText('Enter'))
    await waitFor(() => expect(view.getByText('Fullscreen')).toBeTruthy())
    const input = view.getByLabelText('Remote keyboard')
    const remote = vi.fn()
    input.addEventListener('keydown', remote)
    input.addEventListener('keyup', remote)
    fireEvent.keyDown(input, { key: 'a', code: 'KeyA' })
    fireEvent.keyUp(input, { key: 'a', code: 'KeyA' })
    expect(remote).toHaveBeenCalledTimes(2)
    expect(view.getByText('Fullscreen')).toBeTruthy()
    fireEvent.keyDown(input, { key: 'Escape', code: 'Escape' })
    await waitFor(() => expect(view.getByText('Normal')).toBeTruthy())
    expect(exit).toHaveBeenCalledOnce()
    fireEvent.keyUp(input, { key: 'Escape', code: 'Escape' })
    expect(remote).toHaveBeenCalledTimes(2)
  })

  it('focuses the first overlay control on F6 and keeps repeat presses off the remote surface', async () => {
    const view = render(<Fixture />)
    fireEvent.click(view.getByText('Enter'))
    await waitFor(() => expect(view.getByText('Fullscreen')).toBeTruthy())
    const input = view.getByLabelText('Remote keyboard')
    input.focus()
    fireEvent.keyDown(input, { key: 'F6', code: 'F6' })
    expect(window.document.activeElement).toBe(view.getByText('Exit'))
    // Repeating F6 must not fall back into the remote surface, which would hand
    // the keyboard back to noVNC and strand the controls behind a hover-only overlay.
    fireEvent.keyDown(view.getByText('Exit'), { key: 'F6', code: 'F6' })
    expect(window.document.activeElement).toBe(view.getByText('Exit'))
  })

  it('never exits or replaces another element’s fullscreen', async () => {
    const other = window.document.createElement('div')
    current = other
    const view = render(<Fixture />)
    fireEvent.click(view.getByText('Exit'))
    expect(exit).not.toHaveBeenCalled()
    expect(current).toBe(other)
    view.unmount()
  })

  it.each(['unmount', 'removal'])('cleans up its own fullscreen on %s', async action => {
    const view = render(<Fixture />)
    fireEvent.click(view.getByText('Enter'))
    await act(async () => {})

    if (action === 'unmount') {
      view.unmount()
    }

    if (action === 'removal') {
      view.rerender(<Fixture visible={false} />)
    }

    await waitFor(() => expect(current).toBeNull())
    expect(exit).toHaveBeenCalledOnce()
    view.unmount()
    const remote = vi.fn()
    window.document.addEventListener('keydown', remote)
    fireEvent.keyDown(window.document.body, { key: 'Escape', code: 'Escape' })
    expect(remote).toHaveBeenCalledOnce()
    window.document.removeEventListener('keydown', remote)
  })

  it('exits a late fullscreen grant whose viewport went away while the request was pending', async () => {
    let grant!: () => void
    request.mockImplementation(function (this: HTMLElement) {
      const target = this

      return new Promise<void>(resolve => {
        grant = () => {
          change(target)
          resolve()
        }
      })
    })
    const view = render(<Fixture />)
    fireEvent.click(view.getByText('Enter'))
    view.rerender(<Fixture visible={false} />)
    await act(async () => grant())
    await waitFor(() => expect(current).toBeNull())
    expect(exit).toHaveBeenCalledOnce()
    view.unmount()
  })
})
