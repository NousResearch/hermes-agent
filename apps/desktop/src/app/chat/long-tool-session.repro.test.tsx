// @vitest-environment jsdom
import { AssistantRuntimeProvider, type ThreadMessage } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ChatBar } from '@/app/chat/composer'
import { useRuntimeMessageRepository } from '@/app/chat/runtime-repository'
import { Thread } from '@/components/assistant-ui/thread'
import type { ChatMessage, ChatMessagePart } from '@/lib/chat-messages'
import { useIncrementalExternalStoreRuntime } from '@/lib/incremental-external-store-runtime'
import { $previewStatusBySession, clearPreviewArtifacts, dismissPreviewArtifact } from '@/store/preview-status'

const TARGET_CONTENT_CHARS = 650_451
const MESSAGE_COUNT = 266
const TOOL_CALL_COUNT = 143

class TestResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

vi.stubGlobal('ResizeObserver', TestResizeObserver)
vi.stubGlobal('CSS', { escape: (value: string) => value })
vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) =>
  window.setTimeout(() => callback(performance.now()), 0)
)
vi.stubGlobal('cancelAnimationFrame', (id: number) => window.clearTimeout(id))

Element.prototype.scrollTo = function scrollTo() {}

Element.prototype.animate = function animate() {
  return { cancel: () => {}, finished: Promise.resolve() } as unknown as Animation
}

function partChars(part: ChatMessagePart): number {
  if (part.type === 'text' || part.type === 'reasoning') {
    return part.text.length
  }

  if (part.type === 'tool-call') {
    const output = (part.result as { output?: unknown } | undefined)?.output

    return typeof output === 'string' ? output.length : 0
  }

  return 0
}

function makeFixture(): ChatMessage[] {
  const messages: ChatMessage[] = []
  let toolIndex = 0

  for (let turn = 0; turn < MESSAGE_COUNT / 2; turn += 1) {
    messages.push({
      id: `fixture-user-${turn}`,
      role: 'user',
      parts: [
        { type: 'text', text: `Inspect module ${turn} and explain the observed behavior without omitting evidence.` }
      ],
      timestamp: turn * 2
    })

    const parts: ChatMessagePart[] = []
    const toolsThisTurn = turn < 10 ? 2 : 1

    for (let index = 0; index < toolsThisTurn; index += 1) {
      const id = `fixture-tool-${toolIndex++}`
      parts.push({
        type: 'tool-call',
        toolCallId: id,
        toolName: toolIndex % 3 === 0 ? 'terminal' : toolIndex % 3 === 1 ? 'search_files' : 'read_file',
        args: { path: `/workspace/synthetic/module-${turn}.ts`, query: `symbol-${toolIndex}` },
        argsText: JSON.stringify({ path: `/workspace/synthetic/module-${turn}.ts`, query: `symbol-${toolIndex}` }),
        result: { output: `${id}\n${'deterministic tool output '.repeat(165)}` },
        isError: false
      })
    }

    parts.push({
      type: 'text',
      text: `Turn ${turn} complete. The fixture keeps all durable history while exercising markdown and tool rendering.\n\n\`\`\`ts\nexport const turn = ${turn}\n\`\`\``
    })

    messages.push({
      id: `fixture-assistant-${turn}`,
      role: 'assistant',
      parts,
      timestamp: turn * 2 + 1,
      pending: false
    })
  }

  const chars = messages.reduce((sum, message) => sum + message.parts.reduce((n, part) => n + partChars(part), 0), 0)
  const pad = TARGET_CONTENT_CHARS - chars

  if (pad < 0) {
    throw new Error(`fixture exceeded target by ${-pad} chars`)
  }

  const last = messages.at(-1)
  const text = last?.parts.at(-1)

  if (!last || !text || text.type !== 'text') {
    throw new Error('fixture tail missing')
  }

  last.parts[last.parts.length - 1] = { ...text, text: `${text.text}${'x'.repeat(pad)}` }

  return messages
}

function makeSingleHeavyTurnFixture(): ChatMessage[] {
  const parts: ChatMessagePart[] = Array.from({ length: TOOL_CALL_COUNT }, (_, index) => {
    const id = `single-turn-tool-${index}`

    return {
      type: 'tool-call',
      toolCallId: id,
      toolName: index === 0 ? 'image_generate' : index % 2 === 0 ? 'read_file' : 'search_files',
      args: { path: `/workspace/synthetic/heavy-${index}.ts` },
      argsText: JSON.stringify({ path: `/workspace/synthetic/heavy-${index}.ts` }),
      result: { output: `${id}\n${'single-turn tool output '.repeat(180)}` },
      isError: false
    }
  })

  const usedChars = parts.reduce((sum, part) => sum + partChars(part), 0)

  parts.push({ type: 'text', text: `All tools complete.${'x'.repeat(TARGET_CONTENT_CHARS - usedChars - 19)}` })

  return [
    { id: 'single-turn-user', role: 'user', parts: [{ type: 'text', text: 'Inspect every result.' }], timestamp: 0 },
    { id: 'single-turn-assistant', role: 'assistant', parts, timestamp: 1, pending: false }
  ]
}

function makeInterleavedHeavyTurnFixture(toolCount = 160): ChatMessage[] {
  const messages: ChatMessage[] = [
    { id: 'interleaved-user', role: 'user', parts: [{ type: 'text', text: 'Inspect every result.' }], timestamp: 0 }
  ]

  let toolIndex = 0

  for (let assistantIndex = 0; assistantIndex < 4; assistantIndex += 1) {
    const parts: ChatMessagePart[] = []

    while (toolIndex < Math.ceil(((assistantIndex + 1) * toolCount) / 4)) {
      const id = `interleaved-tool-${toolIndex}`
      parts.push({
        type: 'tool-call',
        toolCallId: id,
        toolName: toolIndex % 2 === 0 ? 'read_file' : 'search_files',
        args: { path: `/workspace/synthetic/interleaved-${toolIndex}.ts` },
        argsText: JSON.stringify({ path: `/workspace/synthetic/interleaved-${toolIndex}.ts` }),
        result: { output: `${id} complete` },
        isError: false
      })
      parts.push({ type: 'text', text: `Narration after tool ${toolIndex}.` })
      toolIndex += 1
    }

    messages.push({
      id: `interleaved-assistant-${assistantIndex}`,
      role: 'assistant',
      parts,
      timestamp: assistantIndex + 1,
      pending: false
    })
  }

  return messages
}

function makeMinimalTurns(count: number, withLatestHeavyTools = 0): ChatMessage[] {
  return Array.from({ length: count }, (_, turn) => {
    const assistantParts: ChatMessagePart[] =
      turn === count - 1 && withLatestHeavyTools > 0
        ? Array.from({ length: withLatestHeavyTools }, (__, tool) => ({
            type: 'tool-call' as const,
            toolCallId: `minimal-tool-${tool}`,
            toolName: 'read_file',
            args: { path: `/workspace/minimal-${tool}.ts` },
            argsText: JSON.stringify({ path: `/workspace/minimal-${tool}.ts` }),
            result: { output: `tool ${tool}` },
            isError: false
          }))
        : [{ type: 'text', text: `Answer ${turn}` }]

    return [
      {
        id: `minimal-user-${turn}`,
        role: 'user' as const,
        parts: [{ type: 'text' as const, text: `Question ${turn}` }]
      },
      { id: `minimal-assistant-${turn}`, role: 'assistant' as const, parts: assistantParts, pending: false }
    ]
  }).flat()
}

function Harness({
  messages,
  onSubmit,
  sessionKey = 'stored-fixture'
}: {
  messages: ChatMessage[]
  onSubmit: (text: string) => boolean
  sessionKey?: string
}) {
  const repository = useRuntimeMessageRepository(messages)

  const runtime = useIncrementalExternalStoreRuntime<ThreadMessage>({
    messageRepository: repository,
    isRunning: false,
    onNew: async () => {},
    setMessages: () => {}
  })

  return (
    <MemoryRouter>
      <AssistantRuntimeProvider runtime={runtime}>
        <div style={{ height: 800, position: 'relative' }}>
          <Thread clampToComposer sessionId="runtime-fixture" sessionKey={sessionKey} />
          <ChatBar
            busy={false}
            cwd="/workspace/synthetic"
            disabled={false}
            focusKey="runtime-fixture"
            onCancel={() => {}}
            onSubmit={onSubmit}
            sessionId="runtime-fixture"
            state={{
              model: { canSwitch: true, model: 'synthetic-model', provider: 'custom' },
              tools: { enabled: true, label: 'Add context' },
              voice: { active: false, enabled: false }
            }}
          />
        </div>
      </AssistantRuntimeProvider>
    </MemoryRouter>
  )
}

afterEach(cleanup)

describe('representative long tool-heavy session contract (#68467)', () => {
  it('keeps the idle composer editable/sendable and bounds mounted history DOM', async () => {
    const messages = makeFixture()
    const toolCalls = messages.flatMap(message => message.parts).filter(part => part.type === 'tool-call')
    const chars = messages.reduce((sum, message) => sum + message.parts.reduce((n, part) => n + partChars(part), 0), 0)
    const onSubmit = vi.fn(() => true)

    expect(messages).toHaveLength(MESSAGE_COUNT)
    expect(toolCalls).toHaveLength(TOOL_CALL_COUNT)
    expect(chars).toBe(TARGET_CONTENT_CHARS)

    const view = render(<Harness messages={messages} onSubmit={onSubmit} />)

    const editor = screen.getByRole('textbox', { name: /message/i })
    expect(editor.getAttribute('contenteditable')).toBe('true')

    editor.textContent = 'continue this session'
    fireEvent.input(editor)
    fireEvent.keyDown(editor, { key: 'Enter', code: 'Enter' })

    await waitFor(() => expect(onSubmit).toHaveBeenCalledWith('continue this session', expect.anything()))
    await act(async () => new Promise(resolve => window.setTimeout(resolve, 25)))

    const mountedToolRows = view.container.querySelectorAll('[data-tool-row]').length
    const mountedTurnGroups = view.container.querySelectorAll('[data-slot="aui_turn-pair"]').length
    const showEarlier = screen.getByRole('button', { name: 'Show earlier messages' })

    expect(showEarlier).toBeTruthy()
    expect(showEarlier.getAttribute('type')).toBe('button')
    expect(mountedToolRows).toBeLessThanOrEqual(20)
    expect(mountedTurnGroups).toBeLessThanOrEqual(20)

    await act(async () => new Promise(resolve => window.setTimeout(resolve, 25)))
    expect(view.container.querySelectorAll('[data-tool-row]')).toHaveLength(mountedToolRows)
    expect(view.container.querySelectorAll('[data-slot="aui_turn-pair"]')).toHaveLength(mountedTurnGroups)

    fireEvent.click(showEarlier)

    await waitFor(() =>
      expect(view.container.querySelectorAll('[data-slot="aui_turn-pair"]').length).toBeGreaterThan(mountedTurnGroups)
    )
    expect(view.container.querySelectorAll('[data-tool-row]').length).toBeGreaterThan(mountedToolRows)
    expect(messages).toHaveLength(MESSAGE_COUNT)
    expect(toolCalls).toHaveLength(TOOL_CALL_COUNT)
  })

  it('keeps one tool-heavy turn bounded by the existing tool-group window', async () => {
    const messages = makeSingleHeavyTurnFixture()
    const view = render(<Harness messages={messages} onSubmit={() => true} />)

    await act(async () => new Promise(resolve => window.setTimeout(resolve, 25)))

    expect(screen.getByRole('textbox', { name: /message/i }).getAttribute('contenteditable')).toBe('true')
    const initialRows = view.container.querySelectorAll('[data-tool-row]').length
    const showEarlier = screen.getByRole('button', { name: 'Show earlier tool calls' })

    expect(initialRows).toBeLessThan(TOOL_CALL_COUNT)
    expect(showEarlier.getAttribute('type')).toBe('button')
    fireEvent.click(showEarlier)
    expect(view.container.querySelectorAll('[data-tool-row]').length).toBeGreaterThan(initialRows)

    while (screen.queryByRole('button', { name: 'Show earlier tool calls' })) {
      fireEvent.click(screen.getByRole('button', { name: 'Show earlier tool calls' }))
    }

    expect(view.container.querySelectorAll('[data-tool-row]')).toHaveLength(TOOL_CALL_COUNT - 1)
    expect(messages.flatMap(message => message.parts).filter(part => part.type === 'tool-call')).toHaveLength(
      TOOL_CALL_COUNT
    )
  })

  it('bounds an interleaved multi-message logical turn and progressively restores it in order', async () => {
    const messages = makeInterleavedHeavyTurnFixture()
    const view = render(<Harness messages={messages} onSubmit={() => true} />)

    await act(async () => new Promise(resolve => window.setTimeout(resolve, 25)))

    expect(view.container.querySelectorAll('[data-tool-page-key]')).toHaveLength(20)
    expect(view.container.querySelectorAll('[data-tool-row]')).toHaveLength(20)

    while (screen.queryByRole('button', { name: 'Show earlier tool calls' })) {
      fireEvent.click(screen.getByRole('button', { name: 'Show earlier tool calls' }))
    }

    const keys = Array.from(view.container.querySelectorAll<HTMLElement>('[data-tool-page-key]')).map(
      element => element.dataset.toolPageKey
    )

    expect(keys).toEqual(
      Array.from(
        { length: 160 },
        (_, index) => `interleaved-assistant-${Math.floor(index / 40)}:interleaved-tool-${index}`
      )
    )
    const firstTool = view.container.querySelector<HTMLElement>('[data-tool-page-key]')
    const firstToolToggle = firstTool?.querySelector<HTMLButtonElement>('button[aria-expanded]')

    if (!firstToolToggle) {
      throw new Error('first restored tool is not expandable')
    }

    fireEvent.click(firstToolToggle)
    expect(firstTool?.textContent).toContain('interleaved-tool-0 complete')
    expect(view.container.textContent).toContain('Narration after tool 159.')
    expect(messages.flatMap(message => message.parts).filter(part => part.type === 'tool-call')).toHaveLength(160)
  })

  it('keeps the oldest revealed tool stable when the same turn appends another call', async () => {
    const initial = makeInterleavedHeavyTurnFixture(30)
    const view = render(<Harness messages={initial} onSubmit={() => true} />)
    const pager = screen.getByRole('button', { name: 'Show earlier tool calls' })

    pager.focus()
    fireEvent.click(pager)

    await waitFor(() => expect(screen.queryByRole('button', { name: 'Show earlier tool calls' })).toBeNull())
    const oldest = view.container.querySelector<HTMLElement>('[data-tool-page-key]')

    expect(oldest?.dataset.toolPageKey).toBe('interleaved-assistant-0:interleaved-tool-0')
    await waitFor(() => expect(globalThis.document.activeElement).toBe(oldest))

    const appended = structuredClone(initial)
    const lastAssistant = appended.at(-1)

    if (!lastAssistant || lastAssistant.role !== 'assistant') {
      throw new Error('missing assistant tail')
    }

    lastAssistant.parts.push({
      type: 'tool-call',
      toolCallId: 'interleaved-tool-30',
      toolName: 'read_file',
      args: { path: '/workspace/synthetic/interleaved-30.ts' },
      argsText: JSON.stringify({ path: '/workspace/synthetic/interleaved-30.ts' }),
      result: { output: 'interleaved-tool-30 complete' },
      isError: false
    })
    view.rerender(<Harness messages={appended} onSubmit={() => true} />)

    await waitFor(() => expect(view.container.querySelectorAll('[data-tool-page-key]')).toHaveLength(31))
    expect(view.container.querySelector<HTMLElement>('[data-tool-page-key]')?.dataset.toolPageKey).toBe(
      'interleaved-assistant-0:interleaved-tool-0'
    )
  })

  it('keeps revealed history identity stable across append and transfers final pager focus', async () => {
    const initial = makeMinimalTurns(25)
    const view = render(<Harness messages={initial} onSubmit={() => true} />)
    const pager = screen.getByRole('button', { name: 'Show earlier messages' })

    pager.focus()
    fireEvent.click(pager)

    await waitFor(() => expect(screen.queryByRole('button', { name: 'Show earlier messages' })).toBeNull())
    const oldest = view.container.querySelector<HTMLElement>('[data-history-group-id]')

    expect(oldest?.dataset.historyGroupId).toBe('minimal-user-0')
    await waitFor(() => expect(globalThis.document.activeElement).toBe(oldest))

    const appended: ChatMessage[] = [
      ...initial,
      { id: 'minimal-user-25', role: 'user', parts: [{ type: 'text', text: 'Question 25' }] },
      { id: 'minimal-assistant-25', role: 'assistant', parts: [{ type: 'text', text: 'Answer 25' }] }
    ]

    view.rerender(<Harness messages={appended} onSubmit={() => true} />)

    await waitFor(() => expect(view.container.querySelectorAll('[data-history-group-id]')).toHaveLength(26))
    expect(view.container.querySelector<HTMLElement>('[data-history-group-id]')?.dataset.historyGroupId).toBe(
      'minimal-user-0'
    )
  })

  it('resets expanded history when the session key changes', async () => {
    const messages = makeMinimalTurns(25)
    const view = render(<Harness messages={messages} onSubmit={() => true} sessionKey="first-session" />)

    fireEvent.click(screen.getByRole('button', { name: 'Show earlier messages' }))
    await waitFor(() => expect(screen.queryByRole('button', { name: 'Show earlier messages' })).toBeNull())
    expect(view.container.querySelectorAll('[data-history-group-id]')).toHaveLength(25)

    view.rerender(<Harness messages={messages} onSubmit={() => true} sessionKey="second-session" />)

    expect(screen.getByRole('button', { name: 'Show earlier messages' })).toBeTruthy()
    expect(view.container.querySelectorAll('[data-history-group-id]')).toHaveLength(20)
  })

  it('keeps history and tool pagers uniquely named', () => {
    render(<Harness messages={makeMinimalTurns(25, 25)} onSubmit={() => true} />)

    expect(screen.getByRole('button', { name: 'Show earlier messages' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Show earlier tool calls' })).toBeTruthy()
  })

  it('registers a hidden preview artifact without mounting its heavy tool row', async () => {
    clearPreviewArtifacts('runtime-fixture')
    const messages = makeMinimalTurns(25)
    const hiddenAssistant = messages[1]

    hiddenAssistant.parts = [
      {
        type: 'tool-call',
        toolCallId: 'hidden-preview-tool',
        toolName: 'write_file',
        args: { path: '/workspace/hidden-preview.html' },
        argsText: JSON.stringify({ path: '/workspace/hidden-preview.html' }),
        result: { output: 'created' },
        isError: false
      }
    ]

    const view = render(<Harness messages={messages} onSubmit={() => true} />)

    await waitFor(() =>
      expect($previewStatusBySession.get()['runtime-fixture']?.map(item => item.target)).toContain(
        '/workspace/hidden-preview.html'
      )
    )
    expect(view.container.querySelector('[data-tool-page-key*="hidden-preview-tool"]')).toBeNull()
    expect(view.container.querySelectorAll('[data-slot="aui_turn-pair"]')).toHaveLength(20)
  })

  it('re-registers the same preview target after a restored timeline produces it again', async () => {
    clearPreviewArtifacts('runtime-fixture')
    const withPreview = makeMinimalTurns(25)
    withPreview[1].parts = [
      {
        type: 'tool-call',
        toolCallId: 'preview-before-restore',
        toolName: 'write_file',
        args: { path: '/workspace/restored-preview.html' },
        argsText: JSON.stringify({ path: '/workspace/restored-preview.html' }),
        result: { output: 'created' },
        isError: false
      }
    ]

    const view = render(<Harness messages={withPreview} onSubmit={() => true} />)

    await waitFor(() =>
      expect($previewStatusBySession.get()['runtime-fixture']?.map(item => item.target)).toContain(
        '/workspace/restored-preview.html'
      )
    )

    act(() => clearPreviewArtifacts('runtime-fixture'))
    expect($previewStatusBySession.get()['runtime-fixture']).toBeUndefined()

    const reproduced = structuredClone(withPreview)
    const reproducedPart = reproduced[1].parts[0]

    if (reproducedPart.type !== 'tool-call') {
      throw new Error('expected the reproduced preview tool call')
    }

    reproduced[1].parts = [{ ...reproducedPart, toolCallId: 'preview-after-restore' }]
    view.rerender(<Harness messages={reproduced} onSubmit={() => true} />)

    await waitFor(() =>
      expect($previewStatusBySession.get()['runtime-fixture']?.map(item => item.target)).toContain(
        '/workspace/restored-preview.html'
      )
    )
  })

  it('registers a large historical preview set with one bounded atom update', async () => {
    clearPreviewArtifacts('runtime-fixture')
    const messages = makeMinimalTurns(1)
    messages[1].parts = Array.from({ length: 101 }, (_, index) => ({
      type: 'tool-call' as const,
      toolCallId: `preview-${index}`,
      toolName: 'write_file',
      args: { path: `/workspace/preview-${index}.html` },
      argsText: JSON.stringify({ path: `/workspace/preview-${index}.html` }),
      result: { output: 'created' },
      isError: false
    }))
    let emissions = 0

    const unsubscribe = $previewStatusBySession.subscribe(() => {
      emissions += 1
    })

    render(<Harness messages={messages} onSubmit={() => true} />)

    await waitFor(() =>
      expect($previewStatusBySession.get()['runtime-fixture']?.map(item => item.target)).toEqual([
        '/workspace/preview-97.html',
        '/workspace/preview-98.html',
        '/workspace/preview-99.html',
        '/workspace/preview-100.html'
      ])
    )
    expect(emissions).toBe(2)
    unsubscribe()
  })

  it('does not resurrect a dismissed preview when another target is produced', async () => {
    clearPreviewArtifacts('runtime-fixture')
    const initial = makeMinimalTurns(1)
    initial[1].parts = [
      {
        type: 'tool-call',
        toolCallId: 'dismissed-preview',
        toolName: 'write_file',
        args: { path: '/workspace/dismissed.html' },
        argsText: JSON.stringify({ path: '/workspace/dismissed.html' }),
        result: { output: 'created' },
        isError: false
      }
    ]
    const view = render(<Harness messages={initial} onSubmit={() => true} />)

    await waitFor(() => expect($previewStatusBySession.get()['runtime-fixture']).toHaveLength(1))
    act(() => dismissPreviewArtifact('runtime-fixture', '/workspace/dismissed.html'))

    const appended = structuredClone(initial)
    appended[1].parts.push({
      type: 'tool-call',
      toolCallId: 'new-preview',
      toolName: 'write_file',
      args: { path: '/workspace/new.html' },
      argsText: JSON.stringify({ path: '/workspace/new.html' }),
      result: { output: 'created' },
      isError: false
    })
    view.rerender(<Harness messages={appended} onSubmit={() => true} />)

    await waitFor(() =>
      expect($previewStatusBySession.get()['runtime-fixture']?.map(item => item.target)).toEqual([
        '/workspace/new.html'
      ])
    )
  })
})
