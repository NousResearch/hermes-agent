import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { focusComposerInput, markActiveComposer, onComposerFocusRequest } from '@/app/chat/composer/focus'
import { RICH_INPUT_SLOT } from '@/app/chat/composer/rich-editor'
import { $selectedStoredSessionId, setActiveSessionId } from '@/store/session'

import { composerHost } from './composer'

const disposers: (() => void)[] = []

function mountComposer(target: string, hidden = false) {
  const surface = document.createElement('div')
  surface.dataset.composerTarget = target
  surface.toggleAttribute('data-pane-hidden', hidden)
  const input = document.createElement('div')
  input.dataset.slot = RICH_INPUT_SLOT
  input.contentEditable = 'true'
  input.tabIndex = 0
  surface.append(input)
  document.body.append(surface)
  disposers.push(onComposerFocusRequest(request => {
    if (request.target === target) {
      focusComposerInput(input)
    }
  }))

  return input
}

beforeEach(() => {
  vi.useFakeTimers()
  setActiveSessionId(null)
  $selectedStoredSessionId.set('shared-session')
  markActiveComposer('main')
})

afterEach(() => {
  disposers.splice(0).forEach(dispose => dispose())
  document.body.replaceChildren()
  $selectedStoredSessionId.set(null)
  markActiveComposer('main')
  vi.useRealTimers()
})

describe('plugin composer focus restoration', () => {
  it('focuses the visible tile when its session is also selected in a hidden primary', () => {
    mountComposer('main', true)
    const visible = mountComposer('tile:shared-session')

    composerHost.focus('shared-session')
    vi.runOnlyPendingTimers()

    expect(document.activeElement).toBe(visible)
  })

  it('keeps an explicitly focused tile when the same session is visible in both surfaces', () => {
    mountComposer('main')
    const tile = mountComposer('tile:shared-session')
    markActiveComposer('tile:shared-session')

    composerHost.focus('shared-session')
    vi.runOnlyPendingTimers()

    expect(document.activeElement).toBe(tile)
  })

  it('restores the normal primary when the same-session tile is hidden', () => {
    const primary = mountComposer('main')
    mountComposer('tile:shared-session', true)
    markActiveComposer('tile:shared-session')

    composerHost.focus('shared-session')
    vi.runOnlyPendingTimers()

    expect(document.activeElement).toBe(primary)
  })

  it('never sends an addressed session focus request to an unrelated active tile', () => {
    const primary = mountComposer('main')
    mountComposer('tile:other-session')
    markActiveComposer('tile:other-session')

    composerHost.focus('shared-session')
    vi.runOnlyPendingTimers()

    expect(document.activeElement).toBe(primary)
  })
})
