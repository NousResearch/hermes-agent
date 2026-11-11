import { act, renderHook } from '@testing-library/react'
import { expect, it } from 'vitest'

import { en } from '@/i18n/en'
import { NEW_SESSION_DRAFT_KEY } from '@/store/composer'

import { useComposerPlaceholder } from './use-composer-placeholder'

const starters: readonly string[] = en.composer.newSessionPlaceholders
const followUps: readonly string[] = en.composer.followUpPlaceholders
const fresh = `${NEW_SESSION_DRAFT_KEY}:draft`

interface Props {
  scope: null | string
  sessionId: null | string
}

function renderPlaceholder(initialProps: Props = { scope: fresh, sessionId: null }) {
  return renderHook(
    ({ scope, sessionId }: Props) => useComposerPlaceholder({ disabled: false, reconnecting: false, scope, sessionId }),
    {
      initialProps
    }
  )
}

it('re-rolls to a follow-up as soon as the landing routes to an existing chat, before its runtime binds', () => {
  const { rerender, result } = renderPlaceholder()

  expect(starters).toContain(result.current.placeholder)
  // Paint-first resume: the route owns the stored key while the runtime id is still null.
  rerender({ scope: 'stored-a', sessionId: null })
  expect(followUps).toContain(result.current.placeholder)
  rerender({ scope: 'stored-a', sessionId: 'runtime-a' })
  expect(followUps).toContain(result.current.placeholder)
})

it('opens a chat already routed at mount with a follow-up', () => {
  const { result } = renderPlaceholder({ scope: 'stored-a', sessionId: null })

  expect(followUps).toContain(result.current.placeholder)
})

it('keeps the starter when its own first send creates the chat, and not for the next chat', () => {
  const { rerender, result } = renderPlaceholder()
  const starter = result.current.placeholder

  act(() => result.current.sessionCreated('created'))
  rerender({ scope: 'created', sessionId: 'runtime-created' })
  expect(result.current.placeholder).toBe(starter)

  rerender({ scope: 'other', sessionId: 'runtime-other' })
  expect(followUps).toContain(result.current.placeholder)
  rerender({ scope: `${NEW_SESSION_DRAFT_KEY}:next`, sessionId: null })
  expect(starters).toContain(result.current.placeholder)
})
