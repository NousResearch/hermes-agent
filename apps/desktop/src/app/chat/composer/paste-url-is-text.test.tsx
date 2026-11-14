// @vitest-environment jsdom
import { AssistantRuntimeProvider, useExternalStoreRuntime } from '@assistant-ui/react'
import type { ThreadMessageLike } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { clearSessionDraft, mainComposerScope } from '@/store/composer'

import type { ClipboardFilePathsResult } from '../../../../electron/clipboard-files'

import { markActiveComposer } from './focus'
import { handleWindowPaste } from './paste-to-focus'
import { composerPlainText, RICH_INPUT_SLOT } from './rich-editor'
import type { ChatBarState } from './types'

import { ChatBar } from './index'

afterEach(() => {
  cleanup()
  mainComposerScope.clear()
  clearSessionDraft(null)
  markActiveComposer('main')
})

// THE INVARIANT: a pasted URL is never swallowed.
//
// A GitHub PR-comment deep link (`…/pull/1#issuecomment-2`) used to be
// special-cased inside ChatBar's paste handler: it called
// `onAttachPrCommentUrl`, ran `event.preventDefault()`, and returned — so the
// clipboard payload never reached the editor (proven: with this harness the
// editor ends up EMPTY and the hook is called once) and the user saw only an
// attachment pill above the composer. Removing that interception is what makes
// the two URLs behave identically again.
//
// The handler is driven through the REAL ChatBar with a real paste event on the
// contentEditable, focused first (an unfocused ⌘V routes through paste-to-focus,
// which shares this insertion path). jsdom has no clipboard, so the event
// carries the fake DataTransfer shape the sibling paste-to-focus tests use.
//
// `onAttachPrCommentUrl` is spread in under a cast deliberately: the prop was
// deleted along with the interception, and passing it anyway is what proves the
// handler no longer consults it. On the pre-fix tree it is called once and the
// payload is dropped; here it must never be called and the URL must survive.
//
// `composerPlainText` round-trips a `@url:` chip to its directive text, so the
// assertions hold whether the link lands chipped or raw — the chip form is
// url-refs.test.ts's contract, this file's is only that the paste survives.
const PR_COMMENT_URL = 'https://github.com/o/r/pull/1#issuecomment-2'

const state: ChatBarState = {
  model: { canSwitch: false, model: '', provider: '' },
  tools: { enabled: false, label: '' },
  voice: { enabled: false, active: false }
}

function Harness({ onAttachPrCommentUrl, onAttachDroppedItems, onAttachImageBlob }: {
  onAttachPrCommentUrl?: (url: string) => boolean
  onAttachDroppedItems?: Parameters<typeof ChatBar>[0]['onAttachDroppedItems']
  onAttachImageBlob?: Parameters<typeof ChatBar>[0]['onAttachImageBlob']
}) {
  // The adapter's message generic infers to `never` from an empty array, so the
  // empty store is typed explicitly. The runtime itself is only here to satisfy
  // ChatBar's provider; nothing in this test reads it.
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
          <ChatBar
            busy={false}
            disabled={false}
            gateway={null}
            onAttachDroppedItems={onAttachDroppedItems}
            onAttachImageBlob={onAttachImageBlob}
            onCancel={vi.fn()}
            onSubmit={vi.fn(async () => true)}
            state={state}
            {...({ onAttachPrCommentUrl } as Record<string, unknown>)}
          />
        </I18nProvider>
      </MemoryRouter>
    </AssistantRuntimeProvider>
  )
}

/** Focus `editor` and fire the paste event a ⌘V into the composer produces. */
function pasteInto(editor: HTMLElement, text: string, files: File[] = [], route: 'focused' | 'unfocused' = 'focused') {
  // jsdom does not implement isContentEditable, so the app's window-level paste
  // router would treat this target as page chrome and insert through the
  // composer bus instead of the editor's own handler. Pin it.
  Object.defineProperty(editor, 'isContentEditable', { configurable: true, value: true })
  editor.focus()

  const event = new Event('paste', { bubbles: true, cancelable: true }) as ClipboardEvent

  const clipboard = {
    getData: (type: string) => (type === 'text' || type === 'text/plain' ? text : ''),
    files: { item: (i: number) => files[i] ?? null, length: files.length },
    items: files.map(file => ({ kind: 'file', type: file.type, getAsFile: () => file }))
  }

  Object.defineProperty(event, 'clipboardData', { value: clipboard })

  act(() => {
    if (route === 'unfocused') {
      editor.blur()
      editor.ownerDocument.body.addEventListener('paste', handleWindowPaste, { once: true })
      fireEvent(editor.ownerDocument.body, event)
    } else {
      fireEvent(editor, event)
    }
  })

  // The IPC reply must use the captured payload, never the expired event data.
  Object.defineProperties(clipboard, {
    files: { get: () => { throw new Error('DataTransfer detached') } },
    items: { get: () => { throw new Error('DataTransfer detached') } }
  })

  clipboard.getData = () => { throw new Error('DataTransfer detached') }

  return event
}

describe('a pasted URL survives the paste', () => {
  it('keeps a GitHub PR-comment deep link in the composer and attaches nothing', () => {
    const onAttachPrCommentUrl = vi.fn(() => true)

    const { container } = render(<Harness onAttachPrCommentUrl={onAttachPrCommentUrl} />)

    const editor = container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!

    const event = pasteInto(editor, PR_COMMENT_URL)

    // The payload reached the editor instead of being consumed by the handler…
    expect(composerPlainText(editor)).toContain(PR_COMMENT_URL)
    // …nothing was attached above the composer…
    expect(mainComposerScope.$attachments.get()).toEqual([])
    // …and the interception hook was never consulted.
    expect(onAttachPrCommentUrl).not.toHaveBeenCalled()
    expect(event.defaultPrevented).toBe(true)
  })
})

describe('a paste of OS files preserves the original paths', () => {
  it('recovers original paths for cloned paste files and keeps screenshots on the image pipeline', async () => {
    const onAttachDroppedItems = vi.fn(async () => true)
    const onAttachImageBlob = vi.fn(async () => { })
    // webUtils.getPathForFile returns '' for cloned paste File objects (#118181).
    const getPathForFile = vi.fn(() => '')

    const readClipboardFilePaths = vi.fn().mockResolvedValue({
      status: 'files',
      files: [{ path: 'C:/original/a.pdf', isDirectory: false }]
    })

    const originalBridge = window.hermesDesktop
    window.hermesDesktop = {
      ...originalBridge,
      readClipboardFilePaths,
      getPathForFile
    } as typeof window.hermesDesktop

    try {
      const { container } = render(
        <Harness onAttachDroppedItems={onAttachDroppedItems} onAttachImageBlob={onAttachImageBlob} />
      )

      const editor = container.querySelector(`[data-slot="${RICH_INPUT_SLOT}"]`) as HTMLElement
      Object.defineProperty(editor, 'isContentEditable', { configurable: true, value: true })
      editor.focus()

      // jsdom's plain arrays lack FileList.item(); coerce so extractDroppedFiles
      // can read .item(i) the same way a real paste does.
      const makeFileList = (files: File[]) => {
        const list = files as unknown as FileList & File[]

        list.item = (index: number) => list[index] ?? null

        return list
      }

      const pasteFile = async (file: File) => {
        const event = new Event('paste', { bubbles: true, cancelable: true }) as ClipboardEvent

        Object.defineProperty(event, 'clipboardData', {
          value: {
            getData: () => '',
            files: makeFileList([file]),
            items: [{ kind: 'file', type: file.type, getAsFile: () => file }]
          }
        })

        await act(async () => {
          fireEvent(editor, event)
        })

        // The handler claims the event — no default insertion, no leak into the editor.
        expect(event.defaultPrevented).toBe(true)
      }

      const pdf = new File(['pdf'], 'a.pdf', { type: 'application/pdf' })

      await pasteFile(pdf)

      // Original path recovered via the native clipboard read, paired with the cloned File.
      expect(onAttachDroppedItems).toHaveBeenCalledWith([
        { file: pdf, path: 'C:/original/a.pdf', isDirectory: false }
      ])
      expect(readClipboardFilePaths).toHaveBeenCalled()

      // A screenshot-only paste (no native paths) falls through to the image pipeline.
      readClipboardFilePaths.mockResolvedValue({ status: 'empty', files: [] })

      const screenshot = new File(['png'], 'shot.png', { type: 'image/png' })

      await pasteFile(screenshot)
      expect(onAttachImageBlob).toHaveBeenCalledWith(screenshot)
      // The file pipeline was not invoked for the screenshot.
      expect(onAttachDroppedItems).toHaveBeenCalledTimes(1)
    } finally {
      window.hermesDesktop = originalBridge
    }
  })
})

describe.each(['focused', 'unfocused'] as const)('%s mixed file paste waits for native resolution', route => {
  it.each([
    { name: 'restores image and text when no native paths exist', paths: [], accepted: true },
    { name: 'attaches only the native file when accepted', paths: ['C:/original/shot.png'], accepted: true },
    { name: 'restores image and text when native attachment is declined', paths: ['C:/original/shot.png'], accepted: false },
    { name: 'keeps the snapshot when a later clipboard list has a different count', paths: ['C:/wrong/a.pdf', 'C:/wrong/b.txt'], accepted: true }
  ])('$name', async ({ paths, accepted }) => {
    let resolveNative!: (result: ClipboardFilePathsResult) => void
    const nativeRead = new Promise<ClipboardFilePathsResult>(resolve => { resolveNative = resolve })
    const readClipboardFilePaths = vi.fn(() => nativeRead)
    const onAttachDroppedItems = vi.fn(async () => accepted)
    const onAttachImageBlob = vi.fn(async () => {})
    const originalBridge = window.hermesDesktop
    window.hermesDesktop = {
      ...originalBridge,
      getPathForFile: () => '',
      readClipboardFilePaths
    } as typeof window.hermesDesktop

    try {
      const { container } = render(
        <Harness onAttachDroppedItems={onAttachDroppedItems} onAttachImageBlob={onAttachImageBlob} />
      )

      const editor = container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!
      const file = new File(['png'], 'shot.png', { type: 'image/png' })
      const text = 'look at https://example.com/docs and @/tmp/source.txt'

      const event = pasteInto(editor, ` \n[200~${text}[201~\n `, [file], route)
      await act(async () => { await new Promise(resolve => setTimeout(resolve, 0)) })

      expect(event.defaultPrevented).toBe(true)
      expect(readClipboardFilePaths).toHaveBeenCalledTimes(1)
      expect(onAttachDroppedItems).not.toHaveBeenCalled()
      expect(onAttachImageBlob).not.toHaveBeenCalled()
      expect(composerPlainText(editor)).toBe('')

      // Another composer can become active while IPC is pending; the fallback
      // must still address the composer that received this paste.
      if (route === 'unfocused') {
        markActiveComposer('tile:other')
      }

      await act(async () => {
        resolveNative({ status: paths.length ? 'files' : 'empty', files: paths.map(path => ({ path, isDirectory: false })) })
        await nativeRead
      })
      await act(async () => { await new Promise(resolve => setTimeout(resolve, 0)) })

      if (paths.length === 1) {
        expect(onAttachDroppedItems).toHaveBeenCalledExactlyOnceWith([
          { file, path: paths[0], isDirectory: false }
        ])
      } else {
        expect(onAttachDroppedItems).not.toHaveBeenCalled()
      }

      if (paths.length === 1 && accepted) {
        expect(onAttachImageBlob).not.toHaveBeenCalled()
        expect(composerPlainText(editor)).toBe('')
      } else {
        expect(onAttachImageBlob).toHaveBeenCalledExactlyOnceWith(file)
        expect(composerPlainText(editor)).toBe('look at @url:`https://example.com/docs` and @file:`/tmp/source.txt`')
      }
    } finally {
      window.hermesDesktop = originalBridge
    }
  })
})
