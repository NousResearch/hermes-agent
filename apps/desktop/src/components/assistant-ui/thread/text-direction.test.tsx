// Appearance → Text direction. Auto keeps the per-block resolver (not the
// browser's first-strong vote): boxes and prose carry its resolved dir and
// must keep doing so; RTL/LTR stamp one `dir` on each chat prose surface and
// both composers, replacing that vote, while inline code keeps its own
// `dir="ltr"`.
// jsdom neither resolves `dir` nor applies styles.css, so these pin the
// attribute contract; the stylesheet half is checked in real Chromium.
import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import type { ThreadMessageLike } from '@assistant-ui/react'
import { act, cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ChatBar } from '@/app/chat/composer'
import { RICH_INPUT_SLOT } from '@/app/chat/composer/rich-editor'
import type { ChatBarState } from '@/app/chat/composer/types'
import { I18nProvider } from '@/i18n'
import { $textDirection, setTextDirection, type TextDirection } from '@/store/text-direction'

import { stubThreadEnvironment, stubThreadViewportSize } from '../test-utils'

import { Thread } from '.'

const createdAt = new Date('2026-06-01T00:00:00.000Z')
stubThreadEnvironment()
stubThreadViewportSize()

afterEach(() => {
  cleanup()
  setTextDirection('auto')
})

// #100280's Latin-majority Arabic quote: first-strong resolves it LTR, which
// is exactly the case a reader needs to override.
const QUOTE =
  'Ergander و Aspalta وردت عليه. الضبط Care إلى Ozul + Daphne هو Seyir وفريق — Seyir + Ozul + Daphne + Saverio هو Daphne dragonheir.info فريق في'

const ASSISTANT_TEXT = [
  QUOTE,
  '',
  '1. `npm install` ثم Seyir',
  '2. Daphne مع Ozul',
  '',
  '> Ultimate - Desperate Beast يضرب مرتين'
].join('\n')

// #83868's English-leading Arabic line, then an Arabic-leading one.
const USER_TEXT = 'Hello مرحبا كيفك\nمرحبا Hello'

function userMessage(): ThreadMessage {
  return {
    id: 'user-1',
    role: 'user',
    content: [{ type: 'text', text: USER_TEXT }],
    attachments: [],
    createdAt,
    metadata: { custom: {} }
  } as ThreadMessage
}

function assistantMessage(): ThreadMessage {
  return {
    id: 'assistant-1',
    role: 'assistant',
    content: [{ type: 'text', text: ASSISTANT_TEXT }],
    status: { type: 'complete', reason: 'stop' },
    createdAt,
    metadata: { unstable_state: null, unstable_annotations: [], unstable_data: [], steps: [], custom: {} }
  } as ThreadMessage
}

function ThreadHarness() {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [userMessage(), assistantMessage()],
    isRunning: false,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

const chatBarState: ChatBarState = {
  model: { canSwitch: false, model: '', provider: '' },
  tools: { enabled: false, label: '' },
  voice: { enabled: false, active: false }
}

function ComposerHarness() {
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
            onCancel={vi.fn()}
            onSubmit={vi.fn(async () => true)}
            state={chatBarState}
          />
        </I18nProvider>
      </MemoryRouter>
    </AssistantRuntimeProvider>
  )
}

async function renderThread() {
  const view = render(<ThreadHarness />)
  const quote = await screen.findByText(/Aspalta/)
  const root = quote.closest('.aui-md')
  const list = (await screen.findByText(/Daphne مع Ozul/)).closest('ol')
  const blockquote = (await screen.findByText(/Desperate Beast/)).closest('blockquote')
  const code = await screen.findByText('npm install')
  const userText = view.container.querySelector<HTMLElement>('[data-slot="aui_user-inline-text"]')

  expect(root).not.toBeNull()
  expect(list).not.toBeNull()
  expect(blockquote).not.toBeNull()
  expect(userText?.textContent).toBe(USER_TEXT)

  return { blockquote: blockquote!, code, list: list!, quote, root: root!, userText: userText!, view }
}

function composerEditor(container: HTMLElement) {
  const editor = container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)

  expect(editor).not.toBeNull()

  return editor!
}

function dirAttributes(container: HTMLElement) {
  return [...container.querySelectorAll('[dir]')].map(el => `${el.tagName.toLowerCase()}=${el.getAttribute('dir')}`)
}

describe('Text direction: Auto', () => {
  it('keeps the per-block resolver DOM: prose and boxes vote, code opts out', async () => {
    const { blockquote, code, list, quote, root, userText, view } = await renderThread()

    expect($textDirection.get()).toBe('auto')
    expect(root.hasAttribute('dir')).toBe(false)
    // The resolver replaces the browser's first-strong vote: a paragraph opening with a
    // Latin name ("Ergander و Aspalta …") still resolves RTL, and the list/quote boxes
    // carry that same resolved direction for their markers/border. The quote body here
    // resolves LTR (dominant strong chars) — that is the resolver's vote, not "auto".
    expect(quote.getAttribute('dir')).toBe('rtl')
    expect(userText.getAttribute('dir')).toBe('rtl')
    expect(list.getAttribute('dir')).toBe('rtl')
    expect(blockquote.getAttribute('dir')).toBe('ltr')
    expect(code.getAttribute('dir')).toBe('ltr')
    expect(new Set(dirAttributes(view.container))).toEqual(
      // p inside the quote body resolves LTR; the two list items split (ltr, rtl).
      new Set(['blockquote=ltr', 'code=ltr', 'li=ltr', 'li=rtl', 'ol=rtl', 'p=ltr', 'p=rtl', 'span=rtl'])
    )
  })

  it('keeps the composer on the resolver and stores nothing', () => {
    const { container } = render(<ComposerHarness />)

    // Auto keeps the composer's own vote — stamped imperatively from the draft
    // (empty draft → `dir="auto"`, typed text → the resolved direction).
    expect(composerEditor(container).getAttribute('dir')).toBe('auto')
    expect(window.localStorage.getItem('hermes.desktop.textDirection')).toBeNull()
  })

  it('returns to the identical DOM after a forced direction is cleared', async () => {
    const { view } = await renderThread()
    const autoHtml = view.container.innerHTML

    act(() => setTextDirection('rtl'))
    expect(view.container.innerHTML).not.toBe(autoHtml)

    act(() => setTextDirection('auto'))
    expect(view.container.innerHTML).toBe(autoHtml)
    expect(window.localStorage.getItem('hermes.desktop.textDirection')).toBeNull()
  })
})

describe.each<Exclude<TextDirection, 'auto'>>(['rtl', 'ltr'])('Text direction: %s', direction => {
  it('forces the prose root, list/quote boxes and user lines; inline code stays LTR', async () => {
    setTextDirection(direction)

    const { blockquote, code, list, quote, root, userText } = await renderThread()

    expect(root.getAttribute('dir')).toBe(direction)
    expect(list.getAttribute('dir')).toBe(direction)
    expect(blockquote.getAttribute('dir')).toBe(direction)
    expect(userText.getAttribute('dir')).toBe(direction)
    // Paragraphs carry the forced direction too (the resolver steps aside), so an
    // English-leading paragraph inside a forced RTL document cannot vote itself LTR.
    expect(quote.getAttribute('dir')).toBe(direction)
    expect(code.getAttribute('dir')).toBe('ltr')
    // Presentation only: the text itself carries no injected bidi controls.
    expect(quote.textContent).toBe(QUOTE)
    expect(userText.textContent).not.toMatch(/[\u200e\u200f\u202a-\u202e\u2066-\u2069]/)
  })

  it('forces the composer live and persists the choice', () => {
    const { container } = render(<ComposerHarness />)

    act(() => setTextDirection(direction))
    expect(composerEditor(container).getAttribute('dir')).toBe(direction)
    expect(window.localStorage.getItem('hermes.desktop.textDirection')).toBe(direction)

    act(() => setTextDirection('auto'))
    // React's Auto `dir={undefined}` drops the attribute; the next keystroke
    // restamps via syncElementTextDirection.
    expect(composerEditor(container).hasAttribute('dir')).toBe(false)
  })
})
