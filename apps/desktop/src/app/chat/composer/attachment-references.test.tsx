import { AssistantRuntimeProvider, type ThreadMessageLike, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render, waitFor } from '@testing-library/react'
import { useMemo } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import {
  clearSessionDraft,
  createComposerAttachmentScope,
  mainComposerScope,
  stashSessionDraft
} from '@/store/composer'
import { $gatewayState } from '@/store/session'

import { useComposerActions } from '../hooks/use-composer-actions'

import { composerPlainText, placeCaretAtOffset, renderComposerContents, RICH_INPUT_SLOT } from './rich-editor'
import { type ComposerScope, ComposerScopeProvider, MAIN_COMPOSER_SCOPE, useComposerScope } from './scope'
import type { ChatBarProps } from './types'

import { ChatBar } from './index'

let actions: ReturnType<typeof useComposerActions>
const actionsBySession = new Map<string, ReturnType<typeof useComposerActions>>()

interface ComposerProps {
  session: string
  onSubmit: ChatBarProps['onSubmit']
}

function Composer({ session, onSubmit }: ComposerProps) {
  const { attachments, target } = useComposerScope()
  const scope = useMemo(() => ({ ...attachments, target }), [attachments, target])
  actions = useComposerActions({ activeSessionId: session, currentCwd: '/project', requestGateway: vi.fn(), scope })
  actionsBySession.set(session, actions)

  return (
    <ChatBar
      busy={false}
      disabled={false}
      gateway={null}
      onAttachDroppedItems={actions.attachDroppedItems}
      onAttachImageBlob={actions.attachImageBlob}
      onCancel={vi.fn()}
      onSubmit={onSubmit}
      sessionId={session}
      state={{
        model: { canSwitch: false, model: '', provider: '' },
        tools: { enabled: false, label: '' },
        voice: { enabled: false, active: false }
      }}
    />
  )
}

interface HarnessProps {
  session?: string
  onSubmit?: ChatBarProps['onSubmit']
  scope?: ComposerScope
}

function Harness({
  session = 'references-A',
  onSubmit = vi.fn(async () => true),
  scope = MAIN_COMPOSER_SCOPE
}: HarnessProps) {
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
          <ComposerScopeProvider value={scope}>
            <Composer onSubmit={onSubmit} session={session} />
          </ComposerScopeProvider>
        </I18nProvider>
      </MemoryRouter>
    </AssistantRuntimeProvider>
  )
}

function editorIn(container: HTMLElement) {
  const editor = container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!
  Object.defineProperty(editor, 'isContentEditable', { configurable: true, value: true })

  return editor
}

function typeDraft(editor: HTMLElement, text: string, caret = text.length) {
  editor.focus()
  renderComposerContents(editor, text)
  placeCaretAtOffset(editor, caret)
  fireEvent.input(editor)
}

afterEach(() => {
  cleanup()
  mainComposerScope.clear()
  clearSessionDraft('references-A')
  clearSessionDraft('references-B')
  actionsBySession.clear()
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

it('inserts attachments at the caret, supports undo/redo, and submits the references with the original attachments', async () => {
  $gatewayState.set('open')
  const onSubmit = vi.fn(async () => true)
  const { container } = render(<Harness onSubmit={onSubmit} />)
  const editor = editorIn(container)
  typeDraft(editor, 'first quote\nsecond quote', 11)
  act(() => {
    actions.attachContextFilePath('/project/layout.txt')
  })
  expect(composerPlainText(editor)).toBe('first quote [layout.txt]\nsecond quote')
  expect(editor.querySelector('[data-attachment-reference]')?.textContent).toBe('[layout.txt]')
  act(() => {
    editor.dispatchEvent(new InputEvent('beforeinput', { bubbles: true, cancelable: true, inputType: 'historyUndo' }))
  })
  expect(composerPlainText(editor)).toBe('first quote\nsecond quote')
  act(() => {
    editor.dispatchEvent(new InputEvent('beforeinput', { bubbles: true, cancelable: true, inputType: 'historyRedo' }))
  })
  expect(composerPlainText(editor)).toBe('first quote [layout.txt]\nsecond quote')
  fireEvent.submit(editor.closest('form')!)
  await waitFor(() =>
    expect(onSubmit).toHaveBeenCalledWith(
      'first quote [layout.txt]\nsecond quote',
      expect.objectContaining({
        attachments: [expect.objectContaining({ kind: 'file', label: 'layout.txt', path: '/project/layout.txt' })]
      })
    )
  )
})

it('rehydrates references without inserting duplicates, skips links, and drops chip styling when an attachment is removed', async () => {
  $gatewayState.set('open')
  const attachment = { id: 'file:report', kind: 'file' as const, label: 'Report.txt', path: '/project/Report.txt' }
  const text = '[ report.TXT ] and [Report.txt](https://example.com)'
  stashSessionDraft('references-A', text, [attachment])
  stashSessionDraft('references-B', 'unrelated draft', [])
  const view = render(<Harness />)
  const editor = editorIn(view.container)
  await waitFor(() => expect(editor.querySelectorAll('[data-attachment-reference]')).toHaveLength(1))
  expect(composerPlainText(editor)).toBe(text)
  view.rerender(<Harness session="references-B" />)
  expect(composerPlainText(editor)).toBe('unrelated draft')
  expect(editor.querySelector('[data-attachment-reference]')).toBeNull()
  view.rerender(<Harness session="references-A" />)
  expect(composerPlainText(editor)).toBe(text)
  expect(editor.querySelectorAll('[data-attachment-reference]')).toHaveLength(1)
  await act(async () => {
    await actions.removeAttachment(attachment.id)
  })
  expect(composerPlainText(editor)).toBe(text)
  expect(editor.querySelector('[data-attachment-reference]')).toBeNull()
})

it.each(['picker', 'paste', 'drop'] as const)(
  'keeps filenames and reference positions through the %s attachment path',
  async mode => {
    $gatewayState.set('open')
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        selectPaths: vi.fn(async () => ['/project/first.txt', '/project/second.txt']),
        saveImageBuffer: vi.fn(async (_data, _extension, name) => `/cache/${name}`)
      }
    })
    const { container } = render(<Harness />)
    const editor = editorIn(container)
    typeDraft(editor, 'before after', 6)

    if (mode === 'picker') {
      await act(async () => {
        await actions.pickContextPaths('file')
      })
      expect(composerPlainText(editor)).toBe('before [first.txt] [second.txt] after')
    } else {
      const file = new File(['image'], 'screen[1].png', { type: 'image/png' })
      Object.defineProperty(file, 'arrayBuffer', { value: async () => new ArrayBuffer(4) })

      if (mode === 'paste') {
        fireEvent.paste(editor, {
          clipboardData: {
            files: [file],
            items: [{ type: 'image/png', kind: 'file', getAsFile: () => file }],
            getData: () => ''
          }
        })
      } else {
        fireEvent.drop(editor, {
          dataTransfer: {
            types: ['Files'],
            files: { length: 1, item: () => file },
            items: [{ kind: 'file', type: 'image/png', getAsFile: () => file }],
            getData: () => ''
          }
        })
      }

      await waitFor(() => expect(composerPlainText(editor)).toBe('before [screen[1].png] after'))
      expect(mainComposerScope.$attachments.get()[0]).toMatchObject({
        label: 'screen[1].png',
        path: '/cache/screen[1].png'
      })
    }

    expect(editor.querySelectorAll('[data-attachment-reference]').length).toBe(mode === 'picker' ? 2 : 1)
  }
)

it('defers reference insertion until IME commits and retains the reference after a staged image rename', async () => {
  $gatewayState.set('open')
  const { container } = render(<Harness />)
  const editor = editorIn(container)
  typeDraft(editor, '你好')
  fireEvent.compositionStart(editor)
  await act(async () => {
    await actions.attachImagePath('/project/screen.png', new Blob(['image'], { type: 'image/png' }))
  })
  expect(composerPlainText(editor)).toBe('你好')
  fireEvent.compositionEnd(editor)
  expect(composerPlainText(editor)).toBe('你好 [screen.png] ')
  const attachment = mainComposerScope.$attachments.get()[0]
  act(() => {
    mainComposerScope.update({ ...attachment, label: 'staged-screen.png', path: '/staged/staged-screen.png' })
  })
  expect(editor.querySelector('[data-attachment-reference]')?.textContent).toBe('[screen.png]')
  expect(composerPlainText(editor)).toBe('你好 [screen.png] ')
})

it('discards a delayed picker result after switching sessions, including a round trip back to the original session', async () => {
  $gatewayState.set('open')
  let finish!: (paths: string[]) => void
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      selectPaths: vi.fn(
        () =>
          new Promise<string[]>(resolve => {
            finish = resolve
          })
      )
    }
  })
  const view = render(<Harness />)
  const editor = editorIn(view.container)
  typeDraft(editor, 'original draft')
  let picking!: Promise<void>
  act(() => {
    picking = actions.pickContextPaths('file')
  })
  view.rerender(<Harness session="references-B" />)
  view.rerender(<Harness session="references-A" />)
  await act(async () => {
    finish(['/project/late.txt'])
    await picking
  })
  expect(mainComposerScope.$attachments.get()).toEqual([])
  expect(composerPlainText(editor)).toBe('original draft')
})

it('keeps simultaneous composers and their attachment reference names isolated', () => {
  $gatewayState.set('open')
  const scope = { ...MAIN_COMPOSER_SCOPE, attachments: createComposerAttachmentScope(), target: 'tile:references' }
  const main = render(<Harness />)
  const tile = render(<Harness scope={scope} session="references-B" />)
  const mainEditor = editorIn(main.container)
  const tileEditor = editorIn(tile.container)
  typeDraft(mainEditor, '[tile.txt] is unrelated')
  typeDraft(tileEditor, 'tile draft')
  act(() => {
    actionsBySession.get('references-B')!.attachContextFilePath('/project/tile.txt')
  })
  expect(composerPlainText(tileEditor)).toBe('tile draft [tile.txt] ')
  expect(composerPlainText(mainEditor)).toBe('[tile.txt] is unrelated')
  expect(mainEditor.querySelector('[data-attachment-reference]')).toBeNull()
  expect(mainComposerScope.$attachments.get()).toEqual([])
  expect(scope.attachments.$attachments.get()).toHaveLength(1)
})
