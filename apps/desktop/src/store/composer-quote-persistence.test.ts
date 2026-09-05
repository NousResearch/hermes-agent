import { beforeEach, describe, expect, it, vi } from 'vitest'

import { encodeComposerQuote, expandComposerQuotes } from '@/lib/composer-quote'

const quoteDraft = (label: string, response: string) =>
  `@quote:\`${encodeComposerQuote({ body: `> ${label}`, label })}\`${response}`

describe('composer quote draft persistence', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  it('restores each session quote after a renderer reload and A to B to A switch', async () => {
    const first = await import('./composer')

    first.stashSessionDraft('session-a', quoteDraft('Alpha?', 'A response'), [])
    first.stashSessionDraft('session-b', quoteDraft('Beta!', 'B response'), [])

    vi.resetModules()
    const restored = await import('./composer')
    const expanded = ['session-a', 'session-b', 'session-a'].map(session =>
      expandComposerQuotes(restored.takeSessionDraft(session).text)
    )

    expect(expanded).toEqual([
      '> Alpha?\n\nA response',
      '> Beta!\n\nB response',
      '> Alpha?\n\nA response'
    ])
    expect(expanded.every(text => !text.includes('@quote:'))).toBe(true)
  })
})
