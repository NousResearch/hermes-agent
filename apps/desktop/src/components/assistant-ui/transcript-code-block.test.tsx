import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { type ReactNode, useState } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { requestComposerSubmitConfirmed } from '@/app/chat/composer/focus'
import { ComposerSurfaceProvider } from '@/app/chat/composer/scope'
import type { SessionView } from '@/app/chat/session-view'
import { registry } from '@/contrib'
import type { ChatMessage } from '@/lib/chat-messages'
import { TRANSCRIPT_CODE_BLOCK_AREA, type TranscriptCodeBlockProps } from '@/lib/transcript-code-blocks'

import { MarkdownTextContent } from './markdown-text'
import { TranscriptCodeBlock, TranscriptCodeBlockScope } from './transcript-code-block'

vi.mock('@/app/chat/composer/focus', () => ({ requestComposerSubmitConfirmed: vi.fn(async () => true) }))
vi.mock('@/store/session-states', () => ({ knownOwnerForSession: () => owner }))
vi.mock('@/store/read-only-transcript', () => ({ isStoredTranscriptReadOnly: () => readOnly }))
vi.mock('@/app/chat/session-view', () => ({ useSessionView: () => view }))

const source = '```my-form\nexample\n```'
const messages = atom<ChatMessage[]>([])

const view = {
  $runtimeId: atom<string | null>('runtime'),
  $storedId: atom<string | null>('stored'),
  $busy: atom(false),
  $messages: messages
} satisfies Pick<SessionView, '$runtimeId' | '$storedId' | '$busy' | '$messages'>

let owner: string | { connectionId: string; profile: string } | undefined
let readOnly = false
let latest: TranscriptCodeBlockProps
let removeClaim: () => void

function Demo(props: TranscriptCodeBlockProps) {
  latest = props

  return (
    <button disabled={props.busy || props.readOnly} onClick={() => void props.submitInteraction('Use my selection.')}>
      Send example
    </button>
  )
}

function register() {
  removeClaim = registry.register({
    id: 'example:form',
    source: 'plugin:example',
    area: TRANSCRIPT_CODE_BLOCK_AREA,
    data: { language: 'my-form', render: Demo }
  })
}

function Scope({ children, text = source, partIndex = 0 }: { children: ReactNode; text?: string; partIndex?: number }) {
  return (
    <ComposerSurfaceProvider value="original-surface">
      <TranscriptCodeBlockScope messageId="answer" partIndex={partIndex} source={text}>
        {children}
      </TranscriptCodeBlockScope>
    </ComposerSurfaceProvider>
  )
}

beforeEach(() => {
  owner = { connectionId: 'original-host', profile: 'example-profile' }
  readOnly = false
  view.$runtimeId.set('runtime')
  view.$storedId.set('stored')
  view.$busy.set(false)
  messages.set([{ id: 'answer', role: 'assistant', parts: [{ type: 'text', text: source }] }])
  register()
})

afterEach(() => {
  cleanup()
  removeClaim()
  vi.clearAllMocks()
})

describe('message-owned fenced renderers', () => {
  it('remounts a hot-reloaded renderer with a different hook layout', () => {
    removeClaim()

    function FirstRenderer() {
      const [value] = useState('Old renderer')

      return <span>{value}</span>
    }

    function NextRenderer() {
      const [value] = useState('New renderer')
      const [extra] = useState('Second field')

      return (
        <span>
          {value}: {extra}
        </span>
      )
    }

    removeClaim = registry.register({
      id: 'example:form',
      source: 'plugin:example',
      area: TRANSCRIPT_CODE_BLOCK_AREA,
      data: { language: 'my-form', render: FirstRenderer }
    })
    render(
      <Scope>
        <TranscriptCodeBlock
          code="example"
          fallback={<span>Original code</span>}
          language="my-form"
          streaming={false}
        />
      </Scope>
    )
    expect(screen.getByText('Old renderer')).toBeTruthy()
    act(() => {
      removeClaim = registry.register({
        id: 'example:form',
        source: 'plugin:example',
        area: TRANSCRIPT_CODE_BLOCK_AREA,
        data: { language: 'my-form', render: NextRenderer }
      })
    })
    expect(screen.getByText('New renderer: Second field')).toBeTruthy()
    expect(screen.queryByText('Original code')).toBeNull()
  })

  it('dispatches a complete registered fence through the real Markdown pipeline, preserving prose', async () => {
    render(
      <Scope>
        <MarkdownTextContent isRunning={false} text={`Before\n\n${source}\n\nAfter`} />
      </Scope>
    )
    expect(await screen.findByText('Send example')).toBeTruthy()
    expect(screen.getByText('Before')).toBeTruthy()
    expect(screen.getByText('After')).toBeTruthy()
    expect(latest.origin).toMatchObject({
      profile: 'example-profile',
      storedSessionId: 'stored',
      messageId: 'answer',
      partIndex: 0,
      blockIndex: 0
    })
  })

  it.each(['streaming', 'unfinished', 'unclaimed', 'unscoped', 'ambiguous'] as const)(
    'preserves source fallback for %s content',
    kind => {
      if (kind === 'unclaimed') {
        removeClaim()
      }

      const text =
        kind === 'unfinished' ? '```my-form\nexample' : kind === 'ambiguous' ? `${source}\n${source}` : source

      const block = (
        <TranscriptCodeBlock
          code="example"
          fallback={<span>Original code</span>}
          language="my-form"
          streaming={kind === 'streaming'}
        />
      )

      render(kind === 'unscoped' ? block : <Scope text={text}>{block}</Scope>)
      expect(screen.getByText('Original code')).toBeTruthy()
      expect(screen.queryByText('Send example')).toBeNull()
    }
  )

  it('keeps identical fences readable instead of guessing from chunk-local positions', async () => {
    removeClaim()
    removeClaim = registry.register({
      id: 'example:form',
      source: 'plugin:example',
      area: TRANSCRIPT_CODE_BLOCK_AREA,
      data: {
        language: 'my-form',
        render: (props: TranscriptCodeBlockProps) => <span data-testid="fence-identity">{props.origin.blockIndex}</span>
      }
    })
    const text = `${source}\n\n${source}`
    messages.set([{ id: 'answer', role: 'assistant', parts: [{ type: 'text', text }] }])
    render(
      <Scope text={text}>
        <MarkdownTextContent isRunning={false} text={text} />
      </Scope>
    )
    await waitFor(() => expect(screen.queryAllByTestId('fence-identity')).toHaveLength(0))
    expect(screen.getAllByText('example', { exact: true })).toHaveLength(2)
    expect(requestComposerSubmitConfirmed).not.toHaveBeenCalled()
  })

  it('keeps malformed plugin output readable when validation refuses or rendering throws', () => {
    removeClaim()
    removeClaim = registry.register({
      id: 'bad:form',
      source: 'plugin:bad',
      area: TRANSCRIPT_CODE_BLOCK_AREA,
      data: {
        language: 'my-form',
        render: () => {
          throw new Error('Invalid program')
        }
      }
    })
    const error = vi.spyOn(console, 'error').mockImplementation(() => {})
    render(
      <Scope>
        <TranscriptCodeBlock
          code="example"
          fallback={<span>Original code</span>}
          language="my-form"
          streaming={false}
        />
      </Scope>
    )
    expect(screen.getByText('Original code')).toBeTruthy()
    error.mockRestore()
  })

  it('sends once through the original surface, waits for admission and never rewrites the prompt', async () => {
    let finish!: (accepted: boolean) => void
    vi.mocked(requestComposerSubmitConfirmed).mockImplementationOnce(
      () =>
        new Promise(resolve => {
          finish = resolve
        })
    )
    render(
      <Scope>
        <TranscriptCodeBlock code="example" fallback={null} language="my-form" streaming={false} />
      </Scope>
    )
    let pending!: Promise<boolean>
    act(() => {
      pending = latest.submitInteraction('Use my selection.')
    })
    await expect(latest.submitInteraction('Use my selection.')).resolves.toBe(false)
    expect(requestComposerSubmitConfirmed).toHaveBeenCalledExactlyOnceWith(
      'Use my selection.',
      'main',
      'original-surface'
    )
    await act(async () => finish(true))
    await expect(pending).resolves.toBe(true)
  })

  it.each([
    'busy',
    'read-only',
    'owner',
    'session',
    'edited',
    'removed',
    'disabled',
    'unmounted',
    'oversized'
  ] as const)('blocks a retained callback after %s', async kind => {
    const mounted = render(
      <Scope>
        <TranscriptCodeBlock code="example" fallback={null} language="my-form" streaming={false} />
      </Scope>
    )

    const submit = latest.submitInteraction
    act(() => {
      if (kind === 'busy') {
        view.$busy.set(true)
      }

      if (kind === 'read-only') {
        readOnly = true
      }

      if (kind === 'owner') {
        owner = { connectionId: 'different-host', profile: 'example-profile' }
      }

      if (kind === 'session') {
        view.$storedId.set('different-session')
      }

      if (kind === 'edited') {
        messages.set([{ id: 'answer', role: 'assistant', parts: [{ type: 'text', text: 'Changed' }] }])
      }

      if (kind === 'removed') {
        messages.set([])
      }

      if (kind === 'disabled') {
        removeClaim()
      }

      if (kind === 'unmounted') {
        mounted.unmount()
      }
    })
    await expect(submit(kind === 'oversized' ? 'x'.repeat(65537) : 'Use my selection.')).resolves.toBe(false)
    expect(requestComposerSubmitConfirmed).not.toHaveBeenCalled()
  })
})
