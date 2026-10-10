// @vitest-environment jsdom
import { AssistantRuntimeProvider, useExternalStoreRuntime } from '@assistant-ui/react'
import type { ThreadMessageLike } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { mainComposerScope } from '@/store/composer'

import { RICH_INPUT_SLOT } from './rich-editor'
import type { ChatBarProps, ChatBarState } from './types'

import { ChatBar } from './index'

afterEach(() => {
  cleanup()
  mainComposerScope.clear()
  vi.unstubAllGlobals()
})

// THE INVARIANT: a pasted file is never swallowed.
//
// A clipboard carrying `text/uri-list` (any file copied in WeChat or a file
// manager) reaches a contenteditable paste as `files` with NO text — Blink
// classifies the whole offer as Files. The handler used to see empty text, no
// image blobs, and fall into a silent image fallback, so Ctrl+V did nothing at
// all while the same clipboard pasted fine into a file manager. The path is
// recovered from the main-process clipboard and handed to the drop pipeline
// (`onAttachDroppedItems`), the same path a file drag takes.
const FILE_PATH = '/home/user/Documents/WeChat_Data/xwechat_files/wxid_x/temp/RWTemp/吉祥物 男女(2).fbx'

const state: ChatBarState = {
  model: { canSwitch: false, model: '', provider: '' },
  tools: { enabled: false, label: '' },
  voice: { enabled: false, active: false }
}

function Harness({ onAttachDroppedItems }: Pick<ChatBarProps, 'onAttachDroppedItems'>) {
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
            onCancel={vi.fn()}
            onSubmit={vi.fn(async () => true)}
            state={state}
          />
        </I18nProvider>
      </MemoryRouter>
    </AssistantRuntimeProvider>
  )
}

/** A paste whose clipboard is a file: no text, one non-image File. */
function pasteFileInto(editor: HTMLElement) {
  Object.defineProperty(editor, 'isContentEditable', { configurable: true, value: true })
  editor.focus()

  const file = { name: '吉祥物 男女(2).fbx', type: 'application/octet-stream', size: 116_345_292 }
  const files = Object.assign([file], { item: (i: number) => ([file][i] ?? null) })

  const event = new Event('paste', { bubbles: true, cancelable: true }) as ClipboardEvent
  Object.defineProperty(event, 'clipboardData', {
    value: {
      getData: () => '',
      files,
      items: []
    }
  })

  act(() => {
    fireEvent(editor, event)
  })

  return event
}

describe('a pasted file is attached, not swallowed', () => {
  it('recovers the path from the main-process clipboard and hands it to the drop pipeline', async () => {
    vi.stubGlobal('window', Object.assign(window, { hermesDesktop: { readClipboard: vi.fn().mockResolvedValue(FILE_PATH) } }))

    const onAttachDroppedItems = vi.fn(async () => true)
    const { container } = render(<Harness onAttachDroppedItems={onAttachDroppedItems} />)
    const editor = container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!

    const event = pasteFileInto(editor)
    await act(async () => {})

    expect(event.defaultPrevented).toBe(true)
    expect(onAttachDroppedItems).toHaveBeenCalledTimes(1)
    expect(onAttachDroppedItems).toHaveBeenCalledWith([{ path: FILE_PATH }])
  })

  it('decodes a file:// URI the way WeChat/wl-copy publish it', async () => {
    const uri = 'file:///home/user/xwechat_files/wxid_x/temp/RWTemp/%E5%90%89%E7%A5%A5%E7%89%A9%20%E7%94%B7%20%E6%8B%86%E4%BB%B6(2).fbx'
    vi.stubGlobal('window', Object.assign(window, { hermesDesktop: { readClipboard: vi.fn().mockResolvedValue(uri) } }))

    const onAttachDroppedItems = vi.fn(async () => true)
    const { container } = render(<Harness onAttachDroppedItems={onAttachDroppedItems} />)
    const editor = container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!

    pasteFileInto(editor)
    await act(async () => {})

    expect(onAttachDroppedItems).toHaveBeenCalledWith([
      { path: '/home/user/xwechat_files/wxid_x/temp/RWTemp/吉祥物 男 拆件(2).fbx' }
    ])
  })
})
