import { afterEach, describe, expect, it } from 'vitest'

import { $workspaceIsPage } from '@/app/routes'
import { closeSwitcher } from '@/store/session-switcher'

import { onComposerFocusRequest } from './focus'
import { handleComposerFocusChord } from './focus-chord'

// jsdom is not macOS, so the chord is Ctrl+L. isComposerChord reads
// ctrlKey on other systems and metaKey on macOS.
function chordEvent(init: KeyboardEventInit = {}): KeyboardEvent {
  return new KeyboardEvent('keydown', { bubbles: true, cancelable: true, ctrlKey: true, key: 'l', ...init })
}

/** The bus defers dispatch by one macrotask. This helper flushes it. */
const flushBus = () => new Promise(resolve => setTimeout(resolve, 1))

async function focusRequests(event: KeyboardEvent): Promise<string[]> {
  const targets: string[] = []
  const off = onComposerFocusRequest(({ target }) => targets.push(target))

  handleComposerFocusChord(event)
  await flushBus()
  off()

  return targets
}

afterEach(() => {
  $workspaceIsPage.set(false)
  closeSwitcher()
  document.body.replaceChildren()
})

describe('handleComposerFocusChord', () => {
  it('focuses the active composer on the bare chord and swallows the key', async () => {
    const event = chordEvent()

    expect(await focusRequests(event)).toEqual(['main'])
    expect(event.defaultPrevented).toBe(true)
  })

  it('ignores non-chord keys (plain letter, shifted chord)', async () => {
    expect(await focusRequests(chordEvent({ ctrlKey: false }))).toEqual([])
    // Shift+chord is the default for view.showBrowser, not for this handler.
    expect(await focusRequests(chordEvent({ shiftKey: true }))).toEqual([])
  })

  it('yields to a press someone else already claimed', async () => {
    const event = chordEvent()
    event.preventDefault()

    expect(await focusRequests(event)).toEqual([])
  })

  it('leaves the key with a focused terminal: no selection means clear-screen', async () => {
    const term = document.createElement('div')
    term.setAttribute('data-terminal', '')
    const inner = document.createElement('textarea')
    term.append(inner)
    document.body.append(term)
    inner.focus()

    const event = chordEvent()

    expect(await focusRequests(event)).toEqual([])
    expect(event.defaultPrevented).toBe(false)
  })

  it('stands down behind blocking surfaces (dialog, full page)', async () => {
    const dialog = document.createElement('div')
    dialog.setAttribute('role', 'dialog')
    document.body.append(dialog)
    expect(await focusRequests(chordEvent())).toEqual([])

    document.body.replaceChildren()
    $workspaceIsPage.set(true)
    expect(await focusRequests(chordEvent())).toEqual([])
  })
})
