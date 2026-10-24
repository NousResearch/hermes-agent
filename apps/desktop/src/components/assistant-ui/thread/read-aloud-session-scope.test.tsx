// Read aloud is per SESSION, not per window.
//
// The desktop can show several sessions side by side (tiles) in ONE renderer,
// and the speaker is a single device: whoever spoke last owns it. Before this,
// clicking Read aloud in one chat flipped every open chat's message button into
// a Stop button and disabled every sibling's — the reported "one session speaks
// and all of them respond".
import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { PRIMARY_SESSION_VIEW, type SessionView, SessionViewProvider } from '@/app/chat/session-view'
import { en } from '@/i18n/en'
import { setVoicePlaybackState } from '@/store/voice-playback'

import { stubThreadEnvironment } from '../test-utils'

import { Thread } from '.'

const playSpeechText = vi.hoisted(() => vi.fn(async () => true))

vi.mock('@/lib/voice-playback', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  playSpeechText: (...args: unknown[]) => playSpeechText(...(args as []))
}))

stubThreadEnvironment()

const createdAt = new Date('2026-05-01T00:00:00.000Z')

afterEach(() => {
  cleanup()
  playSpeechText.mockClear()
  setVoicePlaybackState({
    audioElement: null,
    messageId: null,
    sequence: 0,
    sessionId: null,
    source: null,
    status: 'idle'
  })
})

function transcript(): ThreadMessage[] {
  return [
    {
      id: 'user-1',
      role: 'user',
      content: [{ type: 'text', text: 'question one' }],
      attachments: [],
      createdAt,
      metadata: { custom: {} }
    },
    {
      id: 'assistant-1',
      role: 'assistant',
      content: [{ type: 'text', text: 'the answer' }],
      status: { type: 'complete', reason: 'stop' },
      createdAt,
      metadata: { unstable_state: null, unstable_annotations: [], unstable_data: [], steps: [], custom: {} }
    }
  ] as unknown as ThreadMessage[]
}

function view(sessionId: string): SessionView {
  return { ...PRIMARY_SESSION_VIEW, kind: 'tile', $runtimeId: atom<string | null>(sessionId) }
}

function Harness({ sessionId }: { sessionId: string }) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: transcript(),
    isRunning: false,
    onNew: async () => {}
  })

  return (
    <SessionViewProvider value={view(sessionId)}>
      <AssistantRuntimeProvider runtime={runtime}>
        <Thread />
      </AssistantRuntimeProvider>
    </SessionViewProvider>
  )
}

const idle = { audioElement: null, messageId: null, sequence: 0, source: null, status: 'idle' as const }

const speakingIn = (sessionId: string) =>
  setVoicePlaybackState({ ...idle, messageId: 'assistant-1', sequence: 1, sessionId, source: 'read-aloud', status: 'speaking' })

/** The message action bar's Read aloud / Stop reading control. */
const readButton = () =>
  screen.getByRole('button', {
    name: (name: string) => name.startsWith(en.assistant.thread.readAloud) || name === en.assistant.thread.stopReading
  }) as HTMLButtonElement

describe('Read aloud session scope', () => {
  it('leaves a sibling session’s button idle while another session speaks', async () => {
    speakingIn('session-a')
    render(<Harness sessionId="session-b" />)

    await screen.findByText('the answer')

    // session-a owns the clip; session-b's button must still be its own
    // Read aloud control, enabled, and NOT a Stop button.
    const button = readButton()

    expect(button.disabled).toBe(false)
    expect(button.getAttribute('aria-label')).toContain(en.assistant.thread.readAloud)
    expect(button.getAttribute('aria-label')).not.toBe(en.assistant.thread.stopReading)
  })

  it('turns its own button into Stop reading when this session speaks', async () => {
    speakingIn('session-a')
    render(<Harness sessionId="session-a" />)

    await screen.findByText('the answer')

    const button = readButton()

    expect(button.getAttribute('aria-label')).toBe(en.assistant.thread.stopReading)
  })

  it('lets a sibling session take the speaker over instead of swallowing the click', async () => {
    // session-a is speaking; the user asks session-b to read too. Blocking the
    // click here was the "all sessions locked" half of the bug.
    speakingIn('session-a')
    render(<Harness sessionId="session-b" />)

    await screen.findByText('the answer')
    fireEvent.click(readButton())

    expect(playSpeechText).toHaveBeenCalledWith(
      'the answer',
      expect.objectContaining({ sessionId: 'session-b', source: 'read-aloud' })
    )
  })
})
