import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { clearSessionDraft, mainComposerScope, stashSessionDraft } from '@/store/composer'

import type { QueueEditState } from './composer-utils'
import { useComposerDraft } from './hooks/use-composer-draft'

// Regression for #49903: on desktop v0.17.0 the composer threw an uncaught
// `Error: Composer is not available` at startup and the input went
// unresponsive. @assistant-ui/core's composer mutators (setText/send/…) throw
// when the thread's composer core is not bound yet, and the mount-time draft
// restore pushes text through `aui.composer().setText` inside that window (the
// popout refactor, #49488, widened it). The real draft hook must swallow the
// unbound-core throw; the editor DOM + draftRef carry the text until it binds.
const composer = vi.hoisted(() => ({
  applied: [] as string[],
  bound: false,
  setText(value: string) {
    if (!composer.bound) {
      throw new Error('Composer is not available')
    }

    composer.applied.push(value)
  }
}))

vi.mock('@assistant-ui/react', () => ({
  useAui: () => ({ composer: () => composer }),
  useAuiState: (selector: (state: { composer: { text: string } }) => unknown) => selector({ composer: { text: '' } }),
  useComposerRuntime: () => ({
    getState: () => ({ text: '' }),
    subscribe: () => () => undefined
  })
}))

const SESSION = 'session-guard'

function Harness() {
  useComposerDraft({
    activeQueueSessionKey: SESSION,
    focusKey: null,
    inputDisabled: false,
    queueEditRef: { current: null as QueueEditState | null },
    sessionId: SESSION
  })

  return null
}

afterEach(() => {
  cleanup()
  mainComposerScope.clear()
  clearSessionDraft(SESSION)
  composer.applied = []
  composer.bound = false
})

describe('composer draft restore vs an unbound composer core (#49903)', () => {
  it('swallows the unbound-core throw at startup instead of crashing the renderer', () => {
    stashSessionDraft(SESSION, 'restored draft', [])

    expect(() => render(<Harness />)).not.toThrow()
    expect(composer.applied).toEqual([])
  })

  it('writes the restored draft through once the core is bound', () => {
    composer.bound = true
    stashSessionDraft(SESSION, 'restored draft', [])

    render(<Harness />)

    expect(composer.applied).toContain('restored draft')
  })
})
