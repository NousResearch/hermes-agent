import { act, cleanup, render } from '@testing-library/react'
import { useLayoutEffect } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { type ComposerAttachment, mainComposerScope, stashSessionDraft } from '@/store/composer'

import type { QueueEditState } from '../composer-utils'

import { useComposerDraft } from './use-composer-draft'

const mockComposerApi = { setText: vi.fn() }

vi.mock('@assistant-ui/react', () => ({
  useAui: () => ({ composer: () => mockComposerApi }),
  useAuiState: (selector: (state: { composer: { text: string } }) => unknown) => selector({ composer: { text: '' } }),
  useComposerRuntime: () => ({
    getState: () => ({ text: '' }),
    subscribe: () => () => undefined
  })
}))

interface ProbeHarnessProps {
  activeQueueSessionKey: string | null
  onLayoutSnapshot: (attachments: ComposerAttachment[]) => void
  sessionId: string
}

function ProbeHarness({ activeQueueSessionKey, onLayoutSnapshot, sessionId }: ProbeHarnessProps) {
  useComposerDraft({
    activeQueueSessionKey,
    focusKey: null,
    inputDisabled: false,
    queueEditRef: { current: null as QueueEditState | null },
    sessionId
  })

  // useLayoutEffect fires synchronously right after the DOM commit, BEFORE
  // the hook's per-thread scope-swap useEffect (a passive effect) has a
  // chance to swap attachmentScope.$attachments over to the new session. A
  // synchronous read here — the same read ChatBar's `attachments` prop
  // performs at render time — observes the OUTGOING session's attachments.
  useLayoutEffect(() => {
    onLayoutSnapshot(mainComposerScope.$attachments.get())
  })

  return null
}

describe('useComposerDraft — attachment scope stays coherent with the committed session on switch (#59305)', () => {
  afterEach(() => {
    cleanup()
    mainComposerScope.clear()
  })

  it('clears the outgoing session attachments by the layout phase right after switching sessions', () => {
    const attachmentA: ComposerAttachment = { id: 'url-A', kind: 'url', label: 'A' }
    stashSessionDraft('session-A', 'hi from A', [attachmentA])

    const snapshots: ComposerAttachment[][] = []

    const { rerender } = render(
      <ProbeHarness activeQueueSessionKey="session-A" onLayoutSnapshot={s => snapshots.push(s)} sessionId="session-A" />
    )

    // Mount loads session A's stashed attachment into the (module-level) main
    // scope — confirms the fixture actually seeded the leak precondition.
    expect(mainComposerScope.$attachments.get()).toEqual([attachmentA])

    snapshots.length = 0 // drop the initial-mount snapshot; only the switch matters

    act(() => {
      rerender(
        <ProbeHarness
          activeQueueSessionKey="session-B"
          onLayoutSnapshot={s => snapshots.push(s)}
          sessionId="session-B"
        />
      )
    })

    // By the layout phase the scope must already be B's (empty) — a submit
    // fired the instant B renders must never ship session A's attachment.
    expect(snapshots[0]).toEqual([])
  })
})

describe('useComposerDraft — rehydrate diagnostic log stays redacted', () => {
  afterEach(() => {
    cleanup()
    mainComposerScope.clear()
    vi.restoreAllMocks()
  })

  it('logs counts/kinds/scope on restore but never the raw url, refText, or label', () => {
    const secretUrl = 'https://secret.example.com/private-workspace-path'

    const attachment: ComposerAttachment = {
      id: 'url-secret',
      kind: 'url',
      label: 'do-not-leak-label',
      refText: `@url:${secretUrl}`
    }

    stashSessionDraft('session-secret', '', [attachment])

    const debugSpy = vi.spyOn(console, 'debug').mockImplementation(() => undefined)

    render(
      <ProbeHarness
        activeQueueSessionKey="session-secret"
        onLayoutSnapshot={() => undefined}
        sessionId="session-secret"
      />
    )

    const rehydrateCalls = debugSpy.mock.calls.filter(call => call[0] === '[composer-rehydrate]')
    expect(rehydrateCalls.length).toBeGreaterThan(0)

    const serialized = JSON.stringify(rehydrateCalls)
    expect(serialized).not.toContain(secretUrl)
    expect(serialized).not.toContain(attachment.label)
    expect(serialized).not.toContain(attachment.refText)

    expect(rehydrateCalls[0]?.[1]).toMatchObject({
      attachmentCount: 1,
      attachmentKinds: ['url'],
      scope: 'session-secret'
    })
  })
})
