// The voice model's spoken reading of a Hermes reply (metadata.custom.spokenRelay)
// collapses under the reply by default, or hides, per voice.gpt_live.spoken_replies.
import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $gptLiveSpokenReplies } from '@/store/voice-prefs'

import { Thread } from '.'

const createdAt = new Date('2026-05-01T00:00:00.000Z')

class TestResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}
vi.stubGlobal('ResizeObserver', TestResizeObserver)
vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) =>
  window.setTimeout(() => callback(performance.now()), 0)
)
vi.stubGlobal('cancelAnimationFrame', (id: number) => window.clearTimeout(id))
vi.stubGlobal('CSS', { escape: (str: string) => str })

Element.prototype.scrollTo = function scrollTo() {}

Element.prototype.animate = function animate() {
  return { cancel() {}, finished: Promise.resolve() } as unknown as Animation
}

afterEach(() => {
  cleanup()
  $gptLiveSpokenReplies.set('collapse')
})

const assistantMetadata = { unstable_state: null, unstable_annotations: [], unstable_data: [], steps: [], custom: {} }

function user(id: string, text: string): ThreadMessage {
  return {
    id,
    role: 'user',
    content: [{ type: 'text', text }],
    attachments: [],
    createdAt,
    metadata: { custom: {} }
  } as ThreadMessage
}

function assistant(id: string, text: string, running: boolean): ThreadMessage {
  return {
    id,
    role: 'assistant',
    content: text ? [{ type: 'text', text }] : [],
    status: running ? { type: 'running' } : { type: 'complete', reason: 'stop' },
    createdAt,
    metadata: assistantMetadata
  } as ThreadMessage
}

function Harness({ messages }: { messages: ThreadMessage[] }) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages,
    isRunning: messages.at(-1)?.status?.type === 'running',
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}


function spoken(id: string, text: string, relay: 'differs' | 'repeat' = 'repeat'): ThreadMessage {
  return { ...assistant(id, text, false), metadata: { ...assistantMetadata, custom: { spokenRelay: relay } } } as ThreadMessage
}

const CALL = [
  user('u1', 'Tell me the time'),
  assistant('a1', 'It is 1:24 a.m. Eastern.', false),
  spoken('a2', 'Yeah, I will check. It is 1:24 a.m. Eastern.')
]

describe('spoken reading of a Hermes reply', () => {
  it('collapses to an expandable "Spoken aloud" line by default', async () => {
    render(<Harness messages={CALL} />)

    await screen.findByText('It is 1:24 a.m. Eastern.')
    expect(screen.getByText('Spoken aloud')).toBeTruthy()
    const details = screen.getByText('Spoken aloud').closest('details')
    expect(details?.open).toBe(false)
    expect(details?.textContent).toContain('Yeah, I will check.')
  })

  it('hides it when the preference says hide', async () => {
    $gptLiveSpokenReplies.set('hide')
    render(<Harness messages={CALL} />)

    await screen.findByText('It is 1:24 a.m. Eastern.')
    expect(screen.queryByText('Spoken aloud')).toBeNull()
    expect(screen.queryByText(/Yeah, I will check/)).toBeNull()
  })
})

describe('display-joined voice rows', () => {
  it('keeps a reading that differs from the reply visible, even in hide mode', async () => {
    $gptLiveSpokenReplies.set('hide')
    render(<Harness messages={[user('u1', 'Is the backup done?'), assistant('a1', 'The backup is complete.', false), spoken('a2', 'The backup is not complete.', 'differs')]} />)

    await screen.findByText('The backup is complete.')
    expect(screen.getByText('Spoken aloud')).toBeTruthy()
  })

  it('offers no edit on a user bubble joined from several saved rows', async () => {
    const joined = { ...user('u1', 'That was one of them'), metadata: { custom: { voiceJoined: true } } } as ThreadMessage
    render(<Harness messages={[joined, assistant('a1', 'Okay.', false)]} />)

    await screen.findByText('That was one of them')
    expect(screen.queryByLabelText('Edit message')).toBeNull()
  })
})

describe('joined voice reactions', () => {
  it('shows an existing reaction on a joined bubble without a remove control', async () => {
    const joined = {
      ...user('u1', 'That was one of them'),
      metadata: { custom: { reactions: [{ author: 'user', emoji: '👍' }], voiceJoined: true } }
    } as ThreadMessage

    render(<Harness messages={[joined, assistant('a1', 'Okay.', false)]} />)

    await screen.findByText('That was one of them')
    expect(screen.getByText('👍')).toBeTruthy()
    expect(screen.queryByLabelText('Remove 👍 reaction')).toBeNull()
  })
})
