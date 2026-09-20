import { EditorState } from '@codemirror/state'
import { EditorView } from '@codemirror/view'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { flipSurface } from './flip'
import { liveEditor, TURNING_ATTRIBUTE } from './live-editor'
import { flushBackworkspacePage } from './page'
import { $backworkspaceOpen, toggleBackworkspace } from './store'

vi.mock('./flip', () => ({ flipSurface: vi.fn() }))
vi.mock('./page', () => ({ flushBackworkspacePage: vi.fn(() => Promise.resolve()) }))

afterEach(() => {
  document.body.replaceChildren()
  $backworkspaceOpen.set(false)
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

  it("keeps the caret out of sight while the window turns, and has the page's editor measure again once it lands", async () => {
    const root = document.createElement('div')

    root.id = 'root'
    document.body.append(root)

    // Mounted mid-turn, as the real one is: what it measures there is the
    // squashed box, which is where the caret in the margin came from.
    const view = new EditorView({ parent: root, state: EditorState.create({ extensions: liveEditor('owner') }) })
    const measure = vi.spyOn(view, 'requestMeasure')
    let askedWhileTurning = -1
    let markedWhileTurning = false

    vi.mocked(flipSurface).mockImplementation(async (_element, swap) => {
      swap()
      askedWhileTurning = measure.mock.calls.length
      markedWhileTurning = document.documentElement.hasAttribute(TURNING_ATTRIBUTE)
    })

    await toggleBackworkspace()

    expect(markedWhileTurning).toBe(true)
    expect(measure.mock.calls.length).toBeGreaterThan(askedWhileTurning)
    expect(document.documentElement.hasAttribute(TURNING_ATTRIBUTE)).toBe(false)
    view.destroy()
  })
})
