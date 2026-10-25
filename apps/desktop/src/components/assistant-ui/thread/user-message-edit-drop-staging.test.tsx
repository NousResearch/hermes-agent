// An OS drop into the message-edit composer that the shell cannot expose as a
// native path is staged as File bytes (the Webapp bridge uploads it). When that
// staging is rejected, the reason the server or a proxy gave (an nginx 413, a
// 502) must reach the user; the drop used to vanish without a word.
import {
  type AppendMessage,
  AssistantRuntimeProvider,
  ExportedMessageRepository,
  type ThreadMessage
} from '@assistant-ui/react'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { HermesGateway } from '@/hermes'
import { useIncrementalExternalStoreRuntime } from '@/lib/incremental-external-store-runtime'
import { $notifications, clearNotifications } from '@/store/notifications'

import { assistantMessage, stubThreadEnvironment, stubThreadViewportSize, userMessage } from '../test-utils'

import { Thread } from '.'

stubThreadEnvironment()
stubThreadViewportSize()

const UPLOAD_413 = 'File upload failed (413): the file is larger than the server or a proxy in front of it accepts'

afterEach(() => {
  cleanup()
  clearNotifications()
  Reflect.deleteProperty(window, 'hermesDesktop')
})

function Harness({ gateway }: { gateway: HermesGateway }) {
  const repository = ExportedMessageRepository.fromArray([userMessage(), assistantMessage()])

  const runtime = useIncrementalExternalStoreRuntime<ThreadMessage>({
    messageRepository: repository,
    isRunning: false,
    setMessages: () => {},
    onNew: async () => {},
    onEdit: async (_message: AppendMessage) => {},
    onCancel: async () => {},
    onReload: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread cwd="/workspace" gateway={gateway} sessionId="session-1" />
    </AssistantRuntimeProvider>
  )
}

// A browser drop: a File with no native path (the Webapp shell has no getPathForFile).
function fileDrop(file: File) {
  return {
    files: { item: (index: number) => (index === 0 ? file : null), length: 1 },
    getData: () => '',
    items: [{ getAsFile: () => file, kind: 'file', webkitGetAsEntry: () => null }],
    types: ['Files']
  }
}

describe('edit composer drop staging failures', () => {
  it('reports why a dropped file could not be staged', async () => {
    const stageFileForAttach = vi.fn(async () => {
      throw new Error(UPLOAD_413)
    })

    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { stageFileForAttach } })
    const request = vi.fn(async () => ({}))
    render(<Harness gateway={{ request } as unknown as HermesGateway} />)

    fireEvent.click(await screen.findByRole('button', { name: 'Edit message' }))
    const editor = await screen.findByRole('textbox', { name: 'Edit message' })

    await act(async () => {
      fireEvent.drop(editor, { dataTransfer: fileDrop(new File(['payload'], 'notes.txt', { type: 'text/plain' })) })
    })

    await waitFor(() => expect(stageFileForAttach).toHaveBeenCalledOnce())
    await waitFor(() =>
      expect($notifications.get()).toEqual([expect.objectContaining({ kind: 'error', message: UPLOAD_413 })])
    )
    expect(request).not.toHaveBeenCalledWith('file.attach', expect.anything())
  })
})
