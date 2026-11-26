// #145872: after Stop, the edit draft must survive two things: a stored re-read that re-derives the id
// of the row being edited, and the stopped turn's receipt binding its stored row id onto the same live
// message (its React key must not change). Both would otherwise close the edit box and drop what was typed.
import { AssistantRuntimeProvider, ExportedMessageRepository, type ThreadMessage } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { graftRefreshedTailOntoBackfill } from '@/app/chat/transcript-backfill'
import { preserveLocalPendingTurnMessages } from '@/app/session/hooks/use-session-actions/utils'
import { toChatMessages } from '@/lib/chat-messages'
import { useIncrementalExternalStoreRuntime } from '@/lib/incremental-external-store-runtime'

import { assistantMessage, stubThreadEnvironment, stubThreadViewportSize, userMessage } from '../test-utils'

import { Thread } from '.'

stubThreadEnvironment()
stubThreadViewportSize()

afterEach(cleanup)

const USER_ROW_ID = 7

function Harness({ userId, stored = true }: { userId: string; stored?: boolean }) {
  const user = {
    ...userMessage(userId, 'edit me please'),
    metadata: { custom: stored ? { rowId: USER_ROW_ID } : {} }
  } as ThreadMessage

  const runtime = useIncrementalExternalStoreRuntime<ThreadMessage>({
    messageRepository: ExportedMessageRepository.fromArray([user, assistantMessage()]),
    isRunning: false,
    setMessages: () => {},
    onNew: async () => {},
    onEdit: async () => {},
    onCancel: async () => {},
    onReload: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

// The live transcript holds the optimistic `user-<uuid>` row bound to its durable row id; the
// stored page carries the same row under a page-index id.
function idOfUserRowAfterStoredReread(): string {
  const stored = [
    { role: 'user', content: 'edit me please', timestamp: 200, row_id: USER_ROW_ID },
    { role: 'assistant', content: 'done', timestamp: 201, row_id: 8 }
  ] as never

  const live = [
    { id: 'user-abc', role: 'user', parts: [{ type: 'text', text: 'edit me please' }], rowId: USER_ROW_ID }
  ] as never

  const merged = preserveLocalPendingTurnMessages(graftRefreshedTailOntoBackfill(toChatMessages(stored), live), live)

  return merged.find(message => message.role === 'user')!.id
}

describe('edit draft across a stored re-read (#145872)', () => {
  it('keeps what was typed into the open edit box when the re-read re-derives the row', async () => {
    const { rerender } = render(<Harness userId="user-abc" />)

    fireEvent.click(await screen.findByRole('button', { name: 'Edit message' }))
    const editor = await screen.findByRole('textbox', { name: 'Edit message' })
    editor.tabIndex = 0
    editor.focus()

    await act(async () => {
      editor.textContent = 'edit me please plus more'
      fireEvent.input(editor)
    })

    await act(async () => {
      rerender(<Harness userId={idOfUserRowAfterStoredReread()} />)
    })

    await waitFor(() =>
      expect(screen.getByRole('textbox', { name: 'Edit message' }).textContent).toBe('edit me please plus more')
    )
  })

  it('keeps the open edit box when the stopped turn binds its stored row id to the same message', async () => {
    // Stop with no transcript read: the interrupted completion binds the durable row id onto the
    // live `user-<uuid>` message. Its id does not change, but its React key must not either.
    const { rerender } = render(<Harness stored={false} userId="user-abc" />)

    fireEvent.click(await screen.findByRole('button', { name: 'Edit message' }))
    const editor = await screen.findByRole('textbox', { name: 'Edit message' })
    editor.tabIndex = 0
    editor.focus()

    await act(async () => {
      editor.textContent = 'edit me please plus more'
      fireEvent.input(editor)
    })

    await act(async () => {
      rerender(<Harness userId="user-abc" />)
    })

    const after = await screen.findByRole('textbox', { name: 'Edit message' })

    expect(after).toBe(editor)
    expect(editor.ownerDocument.activeElement).toBe(editor)
    expect(after.textContent).toBe('edit me please plus more')
  })
})
