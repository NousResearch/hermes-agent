import { act, render } from '@testing-library/react'
import { atom } from 'nanostores'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { stubThreadEnvironment } from '@/components/assistant-ui/test-utils'
import { type TranscriptWindowValue, useTranscriptWindow } from '@/components/assistant-ui/thread/transcript-window'
import { $activeGatewayProfile } from '@/store/profile'
import { _resetSessionOwnerHintsForTests, setConnection, setSessionOwnerHint } from '@/store/session'

import { type ComposerScope, ComposerScopeProvider, MAIN_COMPOSER_SCOPE, useComposerScope } from './composer/scope'
import { PRIMARY_SESSION_VIEW, SessionViewProvider } from './session-view'

import { ChatRuntimeBoundary } from '.'

stubThreadEnvironment()

// A Bot Chat must speak with the voice configured on ITS OWNER's profile, no
// matter which profile the app has selected. The composer scope is what the
// voice hooks read (use-voice-live-conversation / use-voice-conversation /
// use-voice-recorder / use-auto-speak-replies all take `profile` from it), so
// the scope ChatRuntimeBoundary publishes is the contract these tests pin.

const STORED = 'bot-canonical-chat'
const OWNER_CONNECTION = 'local'
const OWNER_PROFILE = 'bot-adam'
const APP_PROFILE = 'ani'

function mount(parentScope: ComposerScope = MAIN_COMPOSER_SCOPE) {
  let published: ComposerScope | null = null

  function Observe() {
    published = useComposerScope()

    return null
  }

  const view = {
    ...PRIMARY_SESSION_VIEW,
    $messages: atom([]),
    $runtimeId: atom<string | null>('runtime'),
    $storedId: atom<string | null>(STORED)
  }

  const rendered = render(
    <ComposerScopeProvider value={parentScope}>
      <SessionViewProvider value={view}>
        <ChatRuntimeBoundary
          busy={false}
          onCancel={vi.fn()}
          onEdit={vi.fn()}
          onReload={vi.fn()}
          onThreadMessagesChange={vi.fn()}
          suppressMessages={false}
        >
          <Observe />
        </ChatRuntimeBoundary>
      </SessionViewProvider>
    </ComposerScopeProvider>
  )

  return {
    ...rendered,
    get scope(): ComposerScope {
      return published!
    }
  }
}

beforeEach(() => {
  _resetSessionOwnerHintsForTests({ storage: true })
  setConnection({ baseUrl: '', connectionId: OWNER_CONNECTION, mode: 'local' } as never)
  $activeGatewayProfile.set(APP_PROFILE)
})

describe('ChatRuntimeBoundary — the open chat owner wins the composer scope', () => {
  it("publishes the chat owner's profile while the app is on a different profile", () => {
    // The exact shape openStoredBotChat leaves behind: a recorded owner route
    // for the canonical chat on the LOCAL connection, while the app's active
    // profile stays on the launch profile. Clicking the bot's profile in the
    // app was the manual workaround because the owner lookup below was scoped
    // to the ambient (connection, active profile) pair and so missed.
    setSessionOwnerHint(STORED, { connectionId: OWNER_CONNECTION, mode: 'local', profile: OWNER_PROFILE })

    const mounted = mount()

    expect(mounted.scope.profile).toBe(OWNER_PROFILE)
    expect(mounted.scope.connectionId).toBe(OWNER_CONNECTION)
  })

  it('keeps the ambient scope for a chat with no resolvable owner', () => {
    const mounted = mount()

    expect(mounted.scope.profile).toBeUndefined()
    expect(mounted.scope.connectionId).toBeUndefined()
    expect(mounted.scope).toBe(MAIN_COMPOSER_SCOPE)
  })

  it('lets the owner override a parent scope that already names a profile', () => {
    setSessionOwnerHint(STORED, { connectionId: OWNER_CONNECTION, mode: 'local', profile: OWNER_PROFILE })
    const parent: ComposerScope = { ...MAIN_COMPOSER_SCOPE, connectionId: 'gw-bots', profile: APP_PROFILE, target: 'tile:t' }

    const mounted = mount(parent)

    expect(mounted.scope.profile).toBe(OWNER_PROFILE)
    expect(mounted.scope.connectionId).toBe(OWNER_CONNECTION)
    // Everything that is NOT owner identity is inherited untouched: the scope
    // still belongs to the same composer (attachments, target, message atom).
    expect(mounted.scope.target).toBe('tile:t')
    expect(mounted.scope.$messages).toBe(parent.$messages)
    expect(mounted.scope.attachments).toBe(parent.attachments)
  })

  it('prefers the target profile when the recorded route carries one', () => {
    setSessionOwnerHint(STORED, {
      connectionId: OWNER_CONNECTION,
      mode: 'local',
      profile: 'worker',
      targetProfile: 'backend-worker'
    })

    const mounted = mount()

    expect(mounted.scope.profile).toBe('backend-worker')
  })

  it('stays unresolved when the id has SEVERAL recorded owners', () => {
    // Two backends can each record a route for one id (the hint registry is
    // keyed by (connection, profile, id)). Ambiguity must not pick a voice.
    setSessionOwnerHint(STORED, { connectionId: 'source-a', mode: 'remote', profile: 'bot-adam' })
    setSessionOwnerHint(STORED, { connectionId: 'source-b', mode: 'remote', profile: 'bot-omar' })

    const mounted = mount()

    expect(mounted.scope).toBe(MAIN_COMPOSER_SCOPE)
  })

  it('keeps the scope reference stable across re-renders of the same owner', () => {
    // Every owner lookup hands back a fresh object; deriving the scope from a
    // route would mint a new scope object per render and re-render every
    // consumer. The scope must stay referentially stable while the owner does.
    setSessionOwnerHint(STORED, { connectionId: OWNER_CONNECTION, mode: 'local', profile: OWNER_PROFILE })

    const seen: ComposerScope[] = []

    function Observe() {
      seen.push(useComposerScope())

      return null
    }

    const view = {
      ...PRIMARY_SESSION_VIEW,
      $messages: atom([]),
      $runtimeId: atom<string | null>('runtime'),
      $storedId: atom<string | null>(STORED)
    }

    const boundary = (busy: boolean) => (
      <SessionViewProvider value={view}>
        <ChatRuntimeBoundary
          busy={busy}
          onCancel={vi.fn()}
          onEdit={vi.fn()}
          onReload={vi.fn()}
          onThreadMessagesChange={vi.fn()}
          suppressMessages={false}
        >
          <Observe />
        </ChatRuntimeBoundary>
      </SessionViewProvider>
    )

    const { rerender } = render(boundary(false))
    rerender(boundary(true))

    expect(seen.length).toBe(2)
    expect(new Set(seen).size).toBe(1)
  })

  it('does not touch the transcript/history backend scope', async () => {
    // The card's hard constraint: voice precedence is CLIENT-SIDE only. The
    // history window keeps reading the ambient (connection, active profile)
    // scope even when the composer scope now names the owner — otherwise this
    // change would reroute backend transcript reads, which is out of scope.
    const api = vi.fn(async (_request: { path: string }) => ({
      messages: [],
      pagination: { has_newer: true, has_older: false, limit: 120, offset: 0, order: 'oldest', returned: 0 },
      session_id: STORED
    }))

    vi.stubGlobal('hermesDesktop', { ...window.hermesDesktop, api })

    setSessionOwnerHint(STORED, { connectionId: OWNER_CONNECTION, mode: 'local', profile: OWNER_PROFILE })

    let transcriptWindow!: Required<TranscriptWindowValue>
    let scopeNow: ComposerScope | null = null

    function Observe() {
      transcriptWindow = useTranscriptWindow()
      scopeNow = useComposerScope()

      return null
    }

    const view = {
      ...PRIMARY_SESSION_VIEW,
      $messages: atom([]),
      $runtimeId: atom<string | null>('runtime'),
      $storedId: atom<string | null>(STORED)
    }

    render(
      <SessionViewProvider value={view}>
        <ChatRuntimeBoundary
          busy={false}
          onCancel={vi.fn()}
          onEdit={vi.fn()}
          onReload={vi.fn()}
          onThreadMessagesChange={vi.fn()}
          suppressMessages={false}
        >
          <Observe />
        </ChatRuntimeBoundary>
      </SessionViewProvider>
    )

    // The composer scope now names the Bot…
    expect(scopeNow!.profile).toBe(OWNER_PROFILE)

    await act(async () => {
      await transcriptWindow.revealRow(10, new AbortController().signal)
    })

    // …while the history read still goes to the ambient profile it always did.
    const url = new URL(api.mock.calls[0]![0].path, 'http://test')

    expect(url.searchParams.get('profile')).toBe(APP_PROFILE)
  })
})
