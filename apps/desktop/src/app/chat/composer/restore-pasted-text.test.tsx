import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'

import { AssistantRuntimeProvider, type ThreadMessageLike, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { type ComposerAttachment, mainComposerScope } from '@/store/composer'
import { $connection, $gatewayState } from '@/store/session'

import { writeComposerPaste } from '../../../../electron/composer-paste'
import { useComposerActions } from '../hooks/use-composer-actions'

import { composerPlainText, RICH_INPUT_SLOT } from './rich-editor'

import { ChatBar } from './index'

const pastePath = 'C:\\Users\\Example\\composer-pastes\\pasted_content_stamp_abc.txt'
const pasted = 'Проверить проект — preserve Unicode and all lines.\n'.repeat(90) + 'FINAL ACCEPTANCE LINE'
const onSubmit = vi.fn(async () => true)

function Composer({ sessionId }: { sessionId?: string }) {
  const actions = useComposerActions({ activeSessionId: null, currentCwd: '/workspace', requestGateway: vi.fn() })

  return (
    <ChatBar
      busy={false}
      disabled={false}
      gateway={null}
      onAttachPastedText={actions.attachPastedText}
      onCancel={vi.fn()}
      onRemoveAttachment={actions.removeAttachment}
      onSubmit={onSubmit}
      sessionId={sessionId}
      state={{
        model: { canSwitch: false, model: '', provider: '' },
        tools: { enabled: false, label: '' },
        voice: { enabled: false, active: false }
      }}
    />
  )
}

function Harness({ sessionId }: { sessionId?: string }) {
  const runtime = useExternalStoreRuntime({
    convertMessage: (message: ThreadMessageLike) => message,
    isRunning: false,
    messages: [] as ThreadMessageLike[],
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <MemoryRouter>
        <I18nProvider configClient={null} initialLocale="en">
          <Composer sessionId={sessionId} />
        </I18nProvider>
      </MemoryRouter>
    </AssistantRuntimeProvider>
  )
}

afterEach(() => {
  cleanup()
  mainComposerScope.clear()
  $connection.set(null)
  vi.unstubAllGlobals()
  vi.clearAllMocks()
  window.localStorage.clear()
})

it('restores a generated paste to full editable /goal text and submits without its file attachment', async () => {
  const home = await fs.mkdtemp(path.join(os.tmpdir(), 'hermes-restore-paste-'))

  const readFileText = vi.fn(async (filePath: string) => ({
    path: filePath,
    text: await fs.readFile(filePath, 'utf8'),
    truncated: false
  }))

  const originalBridge = window.hermesDesktop
  window.hermesDesktop = {
    savePastedText: (text: string) => writeComposerPaste(home, text),
    readFileText
  } as unknown as typeof window.hermesDesktop
  $gatewayState.set('open')
  $connection.set({ mode: 'remote', baseUrl: 'http://gateway.invalid' } as never)

  try {
    const view = render(<Harness />)
    const editor = view.container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!
    Object.defineProperty(editor, 'isContentEditable', { configurable: true, value: true })
    editor.focus()
    fireEvent.paste(editor, { clipboardData: { getData: () => pasted, files: [], items: [] } })
    await waitFor(() => expect(mainComposerScope.$attachments.get()).toHaveLength(1))
    expect(composerPlainText(editor)).toBe('')
    const attachment = mainComposerScope.$attachments.get()[0]!
    // A failed send can return a chip whose path was rewritten by remote staging.
    act(() => mainComposerScope.update({ ...attachment, path: '/remote/attachments/staged.txt' }))

    await act(async () => {
      editor.textContent = '/goal'
      fireEvent.input(editor)
    })
    fireEvent.click(view.getByRole('button', { name: 'Insert as text' }))
    await waitFor(() => expect(composerPlainText(editor)).toBe(`/goal\n\n${pasted}`))
    expect(readFileText).toHaveBeenCalledWith(attachment.pastedTextPath)
    expect(mainComposerScope.$attachments.get()).toEqual([])

    fireEvent.keyDown(editor, { key: 'Enter' })
    await waitFor(() =>
      expect(onSubmit).toHaveBeenCalledWith(`/goal\n\n${pasted}`, expect.objectContaining({ attachments: [] }))
    )
  } finally {
    window.hermesDesktop = originalBridge
    await fs.rm(home, { recursive: true, force: true })
  }
})

it.each(['rejected', 'truncated', 'binary', 'empty', 'removed', 'replaced', 'switched', 'unmounted'] as const)(
  'keeps the draft and attachment intact instead of applying an invalid or stale read: %s',
  async failure => {
    let resolveRead!: (value: { path: string; text: string; truncated?: boolean; binary?: boolean }) => void
    let rejectRead!: (reason: Error) => void

    const read = new Promise<{ path: string; text: string; truncated?: boolean; binary?: boolean }>(
      (resolve, reject) => {
        resolveRead = resolve
        rejectRead = reject
      }
    )

    const originalBridge = window.hermesDesktop
    window.hermesDesktop = { readFileText: vi.fn(() => read) } as unknown as typeof window.hermesDesktop
    $gatewayState.set('open')

    try {
      const view = render(<Harness sessionId="first-session" />)
      const editor = view.container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!

      const attachment: ComposerAttachment = {
        id: 'paste',
        kind: 'file',
        label: 'Pasted content',
        path: pastePath,
        occurrenceId: 'first'
      }

      act(() => mainComposerScope.add(attachment))
      act(() => {
        editor.textContent = '/goal keep this draft'
        fireEvent.input(editor)
      })
      fireEvent.click(view.getByRole('button', { name: 'Insert as text' }))

      if (failure === 'removed') {act(() => mainComposerScope.remove(attachment.id))}

      if (failure === 'replaced')
        {act(() => mainComposerScope.$attachments.set([{ ...attachment, occurrenceId: 'replacement' }]))}

      if (failure === 'switched') {view.rerender(<Harness sessionId="second-session" />)}

      if (failure === 'unmounted') {view.unmount()}
      const remaining = mainComposerScope.$attachments.get()
      const draft = composerPlainText(editor)

      await act(async () => {
        if (failure === 'rejected') {rejectRead(new Error('read denied'))}
        else
          {resolveRead({
            path: pastePath,
            text: failure === 'empty' ? '' : pasted,
            truncated: failure === 'truncated',
            binary: failure === 'binary'
          })}

        await read.catch(() => {})
      })
      expect(composerPlainText(editor)).toBe(draft)
      expect(mainComposerScope.$attachments.get()).toEqual(remaining)
      expect(onSubmit).not.toHaveBeenCalled()
    } finally {
      window.hermesDesktop = originalBridge
    }
  }
)
