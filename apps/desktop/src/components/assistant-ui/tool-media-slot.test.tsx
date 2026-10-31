// @vitest-environment jsdom
import { MessagePrimitive, type ThreadMessage, ThreadPrimitive } from '@assistant-ui/react'
import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, type MockInstance, vi } from 'vitest'

import { $pluginRecords } from '@/contrib/plugins-store'
import { registry } from '@/contrib/registry'
import { renderMediaTags } from '@/lib/chat-messages/parts'
import {
  TOOL_MEDIA_RENDERER_AREA,
  type ToolMediaRendererContribution,
  type ToolMediaRendererProps
} from '@/lib/tool-media-renderers'
import { $toolMediaOwners, toolMediaOwnerKey } from '@/store/tool-media-owners'

import { MarkdownText, MessageTextContent } from './markdown-text'
import { createdAt, stubThreadEnvironment, ThreadRuntime, userMessage } from './test-utils'

// A real transcript: one turn where the probe plugin's tool and core TTS each
// produced an audio file, and the reply delivers both with MEDIA: lines.
const SFX_PATH = '/home/u/.hermes/cache/generated/audio/boom.mp3'
const TTS_PATH = '/home/u/.hermes/cache/audio/tts_reply.ogg'
const UPLOAD_PATH = '/home/u/Music/upload.mp3'

function turn(bridged = false): ThreadMessage[] {
  const sfxCall = bridged
    ? {
        type: 'tool-call',
        toolCallId: 'call-sfx',
        toolName: 'tool_call',
        args: { calls: [{ name: 'sfx_make', arguments: { prompt: 'boom' } }] },
        argsText: '{}',
        result: { file_path: SFX_PATH, media_tag: `MEDIA:${SFX_PATH}` }
      }
    : {
        type: 'tool-call',
        toolCallId: 'call-sfx',
        toolName: 'sfx_make',
        args: { prompt: 'boom' },
        argsText: '{"prompt":"boom"}',
        result: { file_path: SFX_PATH, media_tag: `MEDIA:${SFX_PATH}` }
      }

  return [
    userMessage('u1', 'make a boom and say hi'),
    {
      id: 'a1',
      role: 'assistant',
      content: [
        sfxCall,
        {
          type: 'tool-call',
          toolCallId: 'call-tts',
          toolName: 'text_to_speech',
          args: { text: 'hi' },
          argsText: '{"text":"hi"}',
          result: JSON.stringify({ file_path: TTS_PATH })
        },
        {
          type: 'text',
          text: renderMediaTags(`Here:\n\nMEDIA:${SFX_PATH}\n\nMEDIA:${TTS_PATH}\n\nMEDIA:${UPLOAD_PATH}`)
        }
      ],
      status: { type: 'complete', reason: 'stop' },
      createdAt,
      metadata: { unstable_state: null, unstable_annotations: [], unstable_data: [], steps: [], custom: {} }
    } as unknown as ThreadMessage
  ]
}

const Assistant = () => (
  <MessagePrimitive.Root>
    <MessagePrimitive.Parts components={{ Text: MarkdownText }} />
  </MessagePrimitive.Root>
)

function renderTurn(bridged = false) {
  return render(
    <ThreadRuntime messages={turn(bridged)}>
      <ThreadPrimitive.Root>
        <ThreadPrimitive.Messages components={{ AssistantMessage: Assistant, UserMessage: () => null }} />
      </ThreadPrimitive.Root>
    </ThreadRuntime>
  )
}

const seen: ToolMediaRendererProps[] = []

function claim(pluginId: string, tools: string[], render?: ToolMediaRendererContribution['render']) {
  return registry.register({
    id: `${pluginId}:media`,
    area: TOOL_MEDIA_RENDERER_AREA,
    source: `plugin:${pluginId}`,
    data: {
      tools,
      render:
        render ??
        (props => {
          seen.push(props)

          return <div data-testid={`card-${pluginId}`}>{props.file.name}</div>
        })
    } satisfies ToolMediaRendererContribution
  })
}

describe('tool media renderers', () => {
  const disposers: (() => void)[] = []
  let warn: MockInstance<typeof console.warn>

  beforeEach(() => {
    stubThreadEnvironment()
    seen.length = 0
    warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    $pluginRecords.set({
      sfx: { id: 'sfx', kind: 'disk', name: 'SFX', packageName: 'sfx', status: 'loaded' },
      thief: { id: 'thief', kind: 'disk', name: 'Thief', packageName: 'thief', status: 'loaded' }
    })
    // What the backend's plugins.manage list reported for this chat's owner.
    $toolMediaOwners.set({
      [toolMediaOwnerKey(null)]: {
        owners: new Map([
          ['sfx', { tools: new Set(['sfx_make']) }],
          ['thief', { tools: new Set(['thief_tool']) }]
        ]),
        request: 1,
        status: 'ready'
      }
    })
  })

  afterEach(() => {
    disposers.splice(0).forEach(dispose => dispose())
    cleanup()
    warn.mockRestore()
    $pluginRecords.set({})
  })

  it('renders the owning plugin component for the file its tool produced, and the built-in player for the rest', async () => {
    disposers.push(claim('sfx', ['sfx_make']))
    const error = vi.spyOn(console, 'error').mockImplementation(() => {})
    const { container } = renderTurn()

    await waitFor(() => expect(screen.getByTestId('card-sfx').textContent).toBe('boom.mp3'))
    // Core TTS and a file no tool in the turn produced keep the built-in <audio> player.
    await waitFor(() => expect(container.querySelectorAll('audio[controls]')).toHaveLength(2))
    expect(screen.getAllByTestId('card-sfx')).toHaveLength(1)

    const props = seen.at(-1)!
    expect(props.toolName).toBe('sfx_make')
    expect(props.toolCallId).toBe('call-sfx')
    expect(props.args).toEqual({ prompt: 'boom' })
    expect(props.result).toEqual({ file_path: SFX_PATH, media_tag: `MEDIA:${SFX_PATH}` })
    expect(props.file).toMatchObject({ kind: 'audio', name: 'boom.mp3', path: SFX_PATH })
    expect(typeof props.file.resolveUrl).toBe('function')
    expect(props.fallback).toBeTruthy()
    // The plugin's block card never lands inside a <p> (invalid DOM nesting).
    expect(screen.getByTestId('card-sfx').closest('p')).toBeNull()
    expect(error.mock.calls.map(call => String(call[0])).join('\n')).not.toMatch(/descendant/)
    error.mockRestore()
  })

  it('recognises a plugin tool the model ran through the deferred-tool bridge (tool_call)', async () => {
    disposers.push(claim('sfx', ['sfx_make']))
    renderTurn(true)

    await waitFor(() => expect(screen.getByTestId('card-sfx').textContent).toBe('boom.mp3'))
    expect(seen.at(-1)).toMatchObject({ args: { prompt: 'boom' }, toolName: 'sfx_make' })
  })

  it('refuses a claim on a core tool or another plugin’s tool, with a console warning', async () => {
    disposers.push(claim('thief', ['sfx_make', 'text_to_speech']))
    const { container } = renderTurn()

    await waitFor(() => expect(container.querySelectorAll('audio[controls]')).toHaveLength(3))
    expect(screen.queryByTestId('card-thief')).toBeNull()
    expect(warn.mock.calls.map(call => String(call[0])).join('\n')).toMatch(/plugin:thief.*refused.*"text_to_speech"/)
    expect(warn.mock.calls.map(call => String(call[0])).join('\n')).toMatch(/plugin:thief.*refused.*"sfx_make"/)
  })

  it('falls back to the built-in player when the plugin renderer throws', async () => {
    disposers.push(
      claim('sfx', ['sfx_make'], () => {
        throw new Error('boom')
      })
    )
    vi.spyOn(console, 'error').mockImplementation(() => {})
    const { container } = renderTurn()

    await waitFor(() => expect(container.querySelectorAll('audio[controls]')).toHaveLength(3))
  })

  it('keeps the built-in player when the plugin renderer draws nothing (null)', async () => {
    disposers.push(claim('sfx', ['sfx_make'], () => null))
    const { container } = renderTurn()

    await waitFor(() => expect(container.querySelectorAll('audio[controls]')).toHaveLength(3))
  })

  it('keeps the built-in player once the plugin is disabled (its contributions are gone)', async () => {
    const dispose = claim('sfx', ['sfx_make'])
    const { container } = renderTurn()

    await waitFor(() => expect(screen.getByTestId('card-sfx')).toBeTruthy())
    act(() => dispose())
    await waitFor(() => expect(container.querySelectorAll('audio[controls]')).toHaveLength(3))
    expect(screen.queryByTestId('card-sfx')).toBeNull()
  })

  it('keeps the built-in player outside a transcript turn (a Bot group room has no tool calls)', async () => {
    disposers.push(claim('sfx', ['sfx_make']))
    const { container } = render(<MessageTextContent text={`MEDIA:${SFX_PATH}`} />)

    await waitFor(() => expect(container.querySelector('audio[controls]')).not.toBeNull())
    expect(screen.queryByTestId('card-sfx')).toBeNull()
  })
})
