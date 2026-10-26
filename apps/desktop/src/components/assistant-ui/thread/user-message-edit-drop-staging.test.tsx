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

// A browser drop: Files with no native path (the Webapp shell has no getPathForFile).
function fileDrop(...files: File[]) {
  return {
    files: { item: (index: number) => files[index] ?? null, length: files.length },
    getData: () => '',
    items: files.map(file => ({ getAsFile: () => file, kind: 'file', webkitGetAsEntry: () => null })),
    types: ['Files']
  }
}

async function openEditor(gateway: HermesGateway) {
  render(<Harness gateway={gateway} />)
  fireEvent.click(await screen.findByRole('button', { name: 'Edit message' }))

  return screen.findByRole('textbox', { name: 'Edit message' })
}

describe('edit composer drop staging failures', () => {
  it('reports why a dropped file could not be staged', async () => {
    const stageFileForAttach = vi.fn(async () => {
      throw new Error(UPLOAD_413)
    })

    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { stageFileForAttach } })
    const request = vi.fn(async () => ({}))
    const editor = await openEditor({ request } as unknown as HermesGateway)

    await act(async () => {
      fireEvent.drop(editor, { dataTransfer: fileDrop(new File(['payload'], 'notes.txt', { type: 'text/plain' })) })
    })

    await waitFor(() => expect(stageFileForAttach).toHaveBeenCalledOnce())
    await waitFor(() =>
      expect($notifications.get()).toEqual([
        expect.objectContaining({ kind: 'error', message: `Could not attach notes.txt: ${UPLOAD_413}` })
      ])
    )
    expect(request).not.toHaveBeenCalledWith('file.attach', expect.anything())
  })

  // The oversize reason leaves out the name, so the edit composer must add it, or several
  // oversized files in one drop give the same error. A long name is shortened; the limit stays.
  it('names each oversized file of a mixed drop, bounded, and keeps the limit', async () => {
    const limit = 'The file is larger than 16 MiB, the largest file the browser can attach.'
    const longName = `${'a'.repeat(150)}.zip`

    const stageFileForAttach = vi.fn(async (file: File) => {
      if (file.name === 'notes.txt') {
        return '/srv/uploads/web-notes.txt'
      }

      throw new Error(limit)
    })

    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { stageFileForAttach } })
    const editor = await openEditor({ request: vi.fn(async () => ({})) } as unknown as HermesGateway)

    await act(async () => {
      fireEvent.drop(editor, {
        dataTransfer: fileDrop(
          new File(['payload'], 'notes.txt', { type: 'text/plain' }),
          new File(['x'], longName, { type: 'application/zip' }),
          new File(['x'], 'big.zip', { type: 'application/zip' })
        )
      })
    })

    await waitFor(() => expect(stageFileForAttach).toHaveBeenCalledTimes(3))

    const warning = await waitFor(() => {
      const found = $notifications.get().find(notice => notice.message.includes('big.zip'))
      expect(found).toBeDefined()

      return found!
    })

    expect(warning.kind).toBe('error')
    // A reason another item follows loses its period; the detail keeps it whole.
    expect(warning.message).toContain(`Could not attach ${'a'.repeat(24)}…${'a'.repeat(19)}.zip: ${limit.slice(0, -1)}; `)
    expect(warning.message).toContain(`Could not attach big.zip: ${limit}`)
    expect(warning.message).not.toContain(longName)
    expect(warning.message).not.toContain('notes.txt')
    expect(warning.detail).toContain(`Could not attach ${longName}: ${limit}`)
  })
})
