import { afterEach, describe, expect, it, vi } from 'vitest'

import { flipSurface } from './flip'
import { flushBackworkspacePage } from './page'
import { $backworkspaceOpen, toggleBackworkspace } from './store'

vi.mock('./flip', () => ({ flipSurface: vi.fn() }))
vi.mock('./page', () => ({ flushBackworkspacePage: vi.fn(() => Promise.resolve()) }))

afterEach(() => {
  document.body.replaceChildren()
})

describe('toggleBackworkspace', () => {
  it('drops a toggle while a flip runs, and turning back saves the page and returns focus', async () => {
    const root = document.createElement('div')
    const composer = document.createElement('textarea')

    let finishFlip = () => {}

    root.id = 'root'
    document.body.append(root, composer)
    composer.focus()
    vi.mocked(flipSurface).mockImplementation((_element, swap) => {
      swap()

      return new Promise<void>(resolve => {
        finishFlip = resolve
      })
    })

    const opening = toggleBackworkspace()

    void toggleBackworkspace()
    expect(flipSurface).toHaveBeenCalledTimes(1)
    expect($backworkspaceOpen.get()).toBe(true)
    finishFlip()
    await opening

    composer.blur()
    const closing = toggleBackworkspace()

    finishFlip()
    await closing

    expect(flushBackworkspacePage).toHaveBeenCalledTimes(1)
    expect($backworkspaceOpen.get()).toBe(false)
    expect(document.activeElement).toBe(composer)
  })
})
