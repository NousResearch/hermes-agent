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
vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => window.setTimeout(() => callback(performance.now()), 0))
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
      parts: [{ type: 'text', text: `Inspect module ${turn} and explain the observed behavior without omitting evidence.` }],
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

function Harness({ messages, onSubmit }: { messages: ChatMessage[]; onSubmit: (text: string) => boolean }) {
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
          <Thread clampToComposer sessionId="runtime-fixture" sessionKey="stored-fixture" />
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
    const showEarlier = screen.getByRole('button', { name: /earlier/i })

    expect(showEarlier).toBeTruthy()
    expect(showEarlier.getAttribute('type')).toBe('button')
    expect(mountedToolRows).toBeLessThan(TOOL_CALL_COUNT)
    expect(mountedTurnGroups).toBeLessThan(MESSAGE_COUNT / 2)

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
    const showEarlier = screen.getByRole('button', { name: /earlier/i })

    expect(initialRows).toBeLessThan(TOOL_CALL_COUNT)
    expect(showEarlier.getAttribute('type')).toBe('button')
    fireEvent.click(showEarlier)
    expect(view.container.querySelectorAll('[data-tool-row]').length).toBeGreaterThan(initialRows)

    while (screen.queryByRole('button', { name: /earlier/i })) {
      fireEvent.click(screen.getByRole('button', { name: /earlier/i }))
    }

    expect(view.container.querySelectorAll('[data-tool-row]')).toHaveLength(TOOL_CALL_COUNT - 1)
    expect(messages.flatMap(message => message.parts).filter(part => part.type === 'tool-call')).toHaveLength(
      TOOL_CALL_COUNT
    )
  })
})
