import { AssistantRuntimeProvider, ExportedMessageRepository, type ThreadMessage } from '@assistant-ui/react'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useMemo } from 'react'
import { afterEach, describe, expect, it } from 'vitest'

import { useIncrementalExternalStoreRuntime } from '@/lib/incremental-external-store-runtime'
import { $previewStatusBySession } from '@/store/preview-status'

import { stubThreadEnvironment, stubThreadViewportSize } from '../test-utils'

import { type TranscriptIdentity, transcriptIdentityFromRuntimeExtras } from './transcript-identity'

import { Thread } from '.'

stubThreadEnvironment()
stubThreadViewportSize()

const noopAsync = async () => {}

function previewMessage(path: string): ThreadMessage {
  return {
    id: 'assistant-1',
    role: 'assistant',
    content: [
      {
        args: { path },
        argsText: JSON.stringify({ path }),
        result: { path },
        toolCallId: 'write-1',
        toolName: 'write_file',
        type: 'tool-call'
      }
    ],
    status: { reason: 'stop', type: 'complete' },
    createdAt: new Date('2026-08-19T00:00:00.000Z'),
    metadata: { custom: {}, steps: [], unstable_annotations: [], unstable_data: [], unstable_state: null }
  } as ThreadMessage
}

function Harness({ identity, path }: { identity: TranscriptIdentity; path: string }) {
  const repository = useMemo(() => ExportedMessageRepository.fromArray([previewMessage(path)]), [path])
  const extras = useMemo(() => ({ transcriptIdentity: identity }), [identity])

  const runtime = useIncrementalExternalStoreRuntime<ThreadMessage>({
    extras,
    isRunning: false,
    messageRepository: repository,
    onCancel: noopAsync,
    onEdit: noopAsync,
    onNew: noopAsync,
    onReload: noopAsync,
    setMessages: () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread cwd={identity.cwd} sessionId={identity.runtimeId} sessionKey={identity.runtimeId} />
    </AssistantRuntimeProvider>
  )
}

afterEach(() => {
  cleanup()
  $previewStatusBySession.set({})
})

describe('transcript identity adoption', () => {
  it('adopts a destination session whose rendered messages have the same structural signature', async () => {
    const source = { cwd: '/project-a', runtimeId: 'session-a' }
    const destination = { cwd: '/project-b', runtimeId: 'session-b' }
    const { rerender } = render(<Harness identity={source} path="/project-a/source.html" />)

    await waitFor(() => {
      expect($previewStatusBySession.get()['session-a']?.map(item => item.label)).toEqual(['source.html'])
    })

    await act(async () => {
      rerender(<Harness identity={destination} path="/project-b/destination.html" />)
    })

    await waitFor(() => {
      expect($previewStatusBySession.get()['session-b']?.map(item => item.label)).toEqual(['destination.html'])
    })

    expect($previewStatusBySession.get()['session-a']?.map(item => item.label)).toEqual(['source.html'])
  })

  it('adopts a cwd change inside the same transcript session', async () => {
    const { rerender } = render(
      <Harness identity={{ cwd: '/project/old', runtimeId: 'session-a' }} path="/project/old/source.html" />
    )

    await waitFor(() => {
      expect($previewStatusBySession.get()['session-a']?.map(item => item.cwd)).toEqual(['/project/old'])
    })

    await act(async () => {
      rerender(
        <Harness identity={{ cwd: '/project/new', runtimeId: 'session-a' }} path="/project/new/destination.html" />
      )
    })

    await waitFor(() => {
      expect($previewStatusBySession.get()['session-a']?.map(item => item.cwd)).toEqual([
        '/project/old',
        '/project/new'
      ])
    })
  })
})

describe('transcriptIdentityFromRuntimeExtras', () => {
  it('rejects absent or malformed ownership metadata', () => {
    expect(transcriptIdentityFromRuntimeExtras(undefined)).toBeNull()
    expect(transcriptIdentityFromRuntimeExtras({ transcriptIdentity: { cwd: 42, runtimeId: 'session-a' } })).toBeNull()
    expect(transcriptIdentityFromRuntimeExtras({ transcriptIdentity: { cwd: '/project-a', runtimeId: 42 } })).toBeNull()
  })
})
