import { act, cleanup, render, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { type SessionView, SessionViewProvider } from '@/app/chat/session-view'
import type { ChatMessage } from '@/lib/chat-messages'
import { $dismissedPreviewPublications, $previewStatusBySession, dismissPreviewArtifact } from '@/store/preview-status'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection, $sessions } from '@/store/session'

import { previewArtifactPublications, PreviewArtifactPublisher } from './preview-artifact-publisher'

const TARGET = '/tmp/generated-preview.html'

const previewMessage = (
  id: string,
  overrides: { isError?: boolean; result?: { success: boolean }; toolCallId?: string } = {}
): ChatMessage => ({
  id: `message-${id}`,
  parts: [
    {
      args: { path: TARGET },
      result: { success: true },
      toolCallId: id,
      toolName: 'write_file',
      type: 'tool-call',
      ...overrides
    }
  ],
  role: 'assistant'
})

const createSessionView = (
  runtimeId = atom<string | null>(null),
  storedId = atom<string | null>('lineage-root'),
  messages = atom<ChatMessage[]>([])
): SessionView => ({
  kind: 'primary',
  $awaitingResponse: atom(false),
  $busy: atom(false),
  $cwd: atom('/tmp'),
  $fast: atom(false),
  $lastVisibleIsUser: atom(false),
  $messages: messages,
  $messagesEmpty: atom(false),
  $model: atom(''),
  $provider: atom(''),
  $reasoningEffort: atom(''),
  $runtimeId: runtimeId,
  $storedId: storedId
})

const renderPublisher = (view: SessionView) =>
  render(
    <SessionViewProvider value={view}>
      <PreviewArtifactPublisher />
    </SessionViewProvider>
  )

beforeEach(() => {
  localStorage.clear()
  $activeGatewayProfile.set('default')
  $connection.set({ baseUrl: 'http://127.0.0.1:4000', mode: 'local' } as never)
  $dismissedPreviewPublications.set([])
  $previewStatusBySession.set({})
  $sessions.set([{ id: 'lineage-root' }] as never)
})

afterEach(cleanup)

describe('PreviewArtifactPublisher', () => {
  it('collects every completed publication in loaded history, not only rendered rows', () => {
    expect(
      previewArtifactPublications([
        previewMessage('old'),
        previewMessage('missing-id', { toolCallId: undefined }),
        previewMessage('pending', { result: undefined }),
        previewMessage('failed', { isError: true }),
        previewMessage('new')
      ])
    ).toEqual([
      { publicationId: 'old', target: TARGET },
      { publicationId: 'new', target: TARGET }
    ])
  })

  it('retries when the runtime becomes ready and dismisses every duplicate publication', async () => {
    const runtimeId = atom<string | null>(null)
    const messages = atom([previewMessage('old'), previewMessage('new')])
    const view = createSessionView(runtimeId, atom('lineage-root'), messages)

    const rendered = renderPublisher(view)

    expect($previewStatusBySession.get()['runtime-1']).toBeUndefined()

    act(() => runtimeId.set('runtime-1'))
    await waitFor(() => expect($previewStatusBySession.get()['runtime-1']).toHaveLength(1))

    const artifact = $previewStatusBySession.get()['runtime-1']![0]!
    act(() => dismissPreviewArtifact('runtime-1', artifact.id))
    expect($dismissedPreviewPublications.get()).toHaveLength(2)

    rendered.unmount()
    renderPublisher(view)
    expect($previewStatusBySession.get()['runtime-1']).toBeUndefined()

    act(() => messages.set([...messages.get(), previewMessage('fresh')]))
    await waitFor(() => expect($previewStatusBySession.get()['runtime-1']).toHaveLength(1))

    act(() => runtimeId.set('runtime-2'))
    await waitFor(() => expect($previewStatusBySession.get()['runtime-2']).toHaveLength(1))
    expect($previewStatusBySession.get()['runtime-1']).toBeUndefined()
  })

  it('waits for authoritative compression lineage before registering a new publication', async () => {
    const runtimeId = atom<string | null>('runtime-1')
    const storedId = atom<string | null>('lineage-root')
    const messages = atom([previewMessage('before-compression')])
    const view = createSessionView(runtimeId, storedId, messages)

    renderPublisher(view)
    await waitFor(() => expect($previewStatusBySession.get()['runtime-1']).toHaveLength(1))

    const artifact = $previewStatusBySession.get()['runtime-1']![0]!
    act(() => dismissPreviewArtifact('runtime-1', artifact.id))

    act(() => {
      storedId.set('compression-tip')
      messages.set([...messages.get(), previewMessage('after-compression')])
    })
    expect($previewStatusBySession.get()['runtime-1']).toBeUndefined()

    act(() => $sessions.set([{ id: 'compression-tip', _lineage_root_id: 'lineage-root' }] as never))
    await waitFor(() => expect($previewStatusBySession.get()['runtime-1']).toHaveLength(1))
  })

  it('clears the concrete runtime on a null readiness gap and unmount', async () => {
    const runtimeId = atom<string | null>('runtime-1')
    const storedId = atom<string | null>('lineage-root')
    const messages = atom([previewMessage('gap')])
    const view = createSessionView(runtimeId, storedId, messages)

    const rendered = renderPublisher(view)

    await waitFor(() => expect($previewStatusBySession.get()['runtime-1']).toHaveLength(1))

    act(() => runtimeId.set(null))
    expect($previewStatusBySession.get()['runtime-1']).toBeUndefined()

    act(() => runtimeId.set('runtime-2'))
    await waitFor(() => expect($previewStatusBySession.get()['runtime-2']).toHaveLength(1))

    rendered.unmount()
    expect($previewStatusBySession.get()['runtime-2']).toBeUndefined()
  })
})
