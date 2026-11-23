import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { firstTimeCopy } from '@/app/discovery/first-time-copy'
import { en } from '@/i18n/en'
import { beginBackendTransition, resetBackendTransitionsForTests } from '@/store/backend-transition'
import { NEW_SESSION_DRAFT_KEY } from '@/store/composer'
import { $guidanceLedger } from '@/store/guidance-ledger'
import { $activeGatewayProfile } from '@/store/profile'
import { $shellChoice } from '@/store/shell-variant'

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
    ({ scope, sessionId }: Props) =>
      useComposerPlaceholder({ disabled: false, ideasBelow: false, reconnecting: false, scope, sessionId }),
    {
      initialProps
    }
  )
}

afterEach(() => {
  $shellChoice.set(null)
  $guidanceLedger.set({})
  resetBackendTransitionsForTests()
  $activeGatewayProfile.set('default')
})

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

// Classic is main's rule, keyed on the runtime id: an existing chat opened paint-first keeps the
// starter it showed while its runtime was unbound (null → id is read as its own send persisting).
describe('classic', () => {
  it('keeps the starter on an existing chat whose runtime binds after the route lands', () => {
    $shellChoice.set(false)
    const { rerender, result } = renderPlaceholder()
    const starter = result.current.placeholder

    expect(starters).toContain(starter)
    rerender({ scope: 'stored-a', sessionId: null })
    expect(result.current.placeholder).toBe(starter)
    rerender({ scope: 'stored-a', sessionId: 'runtime-a' })
    expect(result.current.placeholder).toBe(starter)
  })

  it('opens with a follow-up only when a runtime is already bound, and re-rolls on a move between runtimes', () => {
    $shellChoice.set(false)
    expect(starters).toContain(renderPlaceholder({ scope: 'stored-a', sessionId: null }).result.current.placeholder)

    const { rerender, result } = renderPlaceholder({ scope: 'stored-a', sessionId: 'runtime-a' })

    expect(followUps).toContain(result.current.placeholder)
    rerender({ scope: `${NEW_SESSION_DRAFT_KEY}:next`, sessionId: null })
    expect(starters).toContain(result.current.placeholder)
  })
})

describe('a newcomer’s new chat in the redesign', () => {
  const copy = firstTimeCopy('en').landing

  function renderNewcomer(ideasBelow: boolean, scope = fresh) {
    $shellChoice.set(true)
    $guidanceLedger.set({ renameSplash: 'not_applicable' })

    return renderHook(() => useComposerPlaceholder({ disabled: false, ideasBelow, reconnecting: false, scope, sessionId: null }))
  }

  it('points at the ideas under it while they are there, and asks for a task once they are gone', () => {
    const { result } = renderNewcomer(true)
    expect(result.current.placeholder).toBe(copy.placeholderIdeas)

    act(() => $guidanceLedger.set({ renameSplash: 'not_applicable', starterIdeas: 'hidden' }))
    expect(result.current.placeholder).toBe(copy.placeholderRetired)
  })

  it('never points at ideas that are not under it', () => {
    expect(renderNewcomer(false).result.current.placeholder).toBe(copy.placeholderRetired)
  })

  it('keeps an open conversation’s follow-up, and everyone else’s rotating starters', () => {
    expect(followUps).toContain(renderNewcomer(true, 'stored-a').result.current.placeholder)

    act(() => $guidanceLedger.set({ renameSplash: 'seen' }))

    const { result } = renderHook(() =>
      useComposerPlaceholder({ disabled: false, ideasBelow: true, reconnecting: false, scope: fresh, sessionId: null })
    )

    expect(starters).toContain(result.current.placeholder)
  })
})

type PlaceholderOptions = Parameters<typeof useComposerPlaceholder>[0]

const stored = (options: Partial<PlaceholderOptions>): PlaceholderOptions => ({
  disabled: false,
  ideasBelow: false,
  reconnecting: false,
  scope: 's1',
  sessionId: 's1',
  ...options
})

it("names the profile's sandbox restart instead of 'Reconnecting to Hermes…' while the socket is down", () => {
  $activeGatewayProfile.set('work')
  beginBackendTransition('work', 'restarting', { source: 'launcher' })

  const { result, rerender } = renderHook(props => useComposerPlaceholder(props), {
    initialProps: stored({ disabled: true, reconnecting: true })
  })

  expect(result.current.placeholder).toBe('Restarting sandbox…')

  rerender(stored({ disabled: true, profile: 'default', reconnecting: true }))
  expect(result.current.placeholder).toBe('Reconnecting to Hermes…')
})

it("names the profile's sandbox start while the connection is still open", () => {
  beginBackendTransition('work', 'starting', { source: 'switch' })

  const { result, rerender } = renderHook(props => useComposerPlaceholder(props), {
    initialProps: stored({ profile: 'work' })
  })

  expect(result.current.placeholder).toBe('Starting sandbox…')

  rerender(stored({ profile: 'default' }))
  expect(result.current.placeholder).not.toBe('Starting sandbox…')
})
