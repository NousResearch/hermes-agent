import { act, render, renderHook, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { describe, expect, it } from 'vitest'

import { type SessionTranscript, type SessionView } from '@/app/chat/session-view'
import { PaneVisibleContext } from '@/components/pane-shell/pane-visibility'
import type { ChatMessage } from '@/lib/chat-messages'

import { useVisibleTranscriptSnapshot } from './visible-transcript'

const message = (path: string): ChatMessage =>
  ({
    id: 'same-assistant-id',
    role: 'assistant',
    parts: [{ type: 'text', text: path }]
  }) as unknown as ChatMessage

function viewWith(source: SessionTranscript) {
  const transcript = atom(source)
  const cwd = atom(source.identity.cwd)
  const messages = atom(source.messages)
  const runtimeId = atom(source.identity.runtimeId)

  const view: SessionView = {
    ...({} as SessionView),
    $cwd: cwd,
    $messages: messages,
    $runtimeId: runtimeId,
    $transcript: transcript,
    kind: 'primary'
  }

  return { cwd, messages, runtimeId, transcript, view }
}

describe('useVisibleTranscriptSnapshot', () => {
  it('never pairs destination identity with outgoing rows during an equal-signature switch', () => {
    const source = {
      identity: { cwd: '/project-a', runtimeId: 'session-a' },
      messages: [message('/project-a/source.html')]
    }

    const destination = {
      identity: { cwd: '/project-b', runtimeId: 'session-b' },
      messages: [message('/project-b/destination.html')]
    }

    const { cwd, messages, runtimeId, transcript, view } = viewWith(source)

    const observed: SessionTranscript[] = []

    const { result } = renderHook(() => {
      const snapshot = useVisibleTranscriptSnapshot(view)

      observed.push(snapshot)

      return snapshot
    })

    // Route-facing identity/message atoms may move before assistant-ui adopts
    // the destination repository. The coherent transcript source deliberately
    // stays on A for this intermediate render.
    act(() => {
      runtimeId.set(destination.identity.runtimeId)
      cwd.set(destination.identity.cwd)
      messages.set(destination.messages)
    })

    expect(result.current).toBe(source)

    act(() => transcript.set(destination))

    expect(result.current).toBe(destination)
    expect(
      observed.some(
        snapshot => snapshot.identity.runtimeId === 'session-b' && snapshot.messages[0] === source.messages[0]
      )
    ).toBe(false)
  })

  it('freezes a hidden pane and adopts one coherent destination snapshot when revealed', async () => {
    const source = {
      identity: { cwd: '/project-a', runtimeId: 'session-a' },
      messages: [message('/project-a/source.html')]
    }

    const destination = {
      identity: { cwd: '/project-b', runtimeId: 'session-b' },
      messages: [message('/project-b/destination.html')]
    }

    const { transcript, view } = viewWith(source)
    const observed: SessionTranscript[] = []

    function Probe() {
      observed.push(useVisibleTranscriptSnapshot(view))

      return null
    }

    const { rerender } = render(
      <PaneVisibleContext.Provider value={false}>
        <Probe />
      </PaneVisibleContext.Provider>
    )

    act(() => transcript.set(destination))

    expect(observed.at(-1)).toBe(source)

    rerender(
      <PaneVisibleContext.Provider value>
        <Probe />
      </PaneVisibleContext.Provider>
    )

    await waitFor(() => expect(observed.at(-1)).toBe(destination))
    expect(
      observed.some(
        snapshot => snapshot.identity.runtimeId === 'session-b' && snapshot.messages[0] === source.messages[0]
      )
    ).toBe(false)
  })
})
