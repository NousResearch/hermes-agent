import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { cleanup, render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $visibleTranscriptSessionIds } from '@/store/thread-scroll'

import { Thread } from '.'

function Harness({ transcriptVisible }: { transcriptVisible: boolean }) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    isRunning: false,
    messages: [],
    onNew: async () => undefined
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread sessionKey="stored-session" transcriptVisible={transcriptVisible} />
    </AssistantRuntimeProvider>
  )
}

describe('Thread transcript visibility', () => {
  beforeEach(() => {
    vi.stubGlobal(
      'ResizeObserver',
      class {
        disconnect() {}
        observe() {}
        unobserve() {}
      }
    )
  })

  afterEach(() => cleanup())

  it('does not register a transcript hidden behind a failed-resume overlay', async () => {
    const view = render(<Harness transcriptVisible={false} />)

    expect($visibleTranscriptSessionIds.get()).toEqual([])

    view.rerender(<Harness transcriptVisible />)

    await waitFor(() => expect($visibleTranscriptSessionIds.get()).toEqual(['stored-session']))
  })
})
