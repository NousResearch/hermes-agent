import type { GatewayEvent } from '@hermes/shared'
import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { MAIN_COMPOSER_SCOPE } from '@/app/chat/composer/scope'
import { useSessionTileActions } from '@/app/chat/session-tile-actions'
import { usePromptActions } from '@/app/session/hooks/use-prompt-actions'
import { useSubmitPrompt } from '@/app/session/hooks/use-prompt-actions/submit'
import type { ClientSessionState } from '@/app/types'
import { type ChatMessage, chatMessageText, textPart } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { ComposerAttachment } from '@/store/composer'
import { $providerWaitSessions, setSessionProviderWait } from '@/store/provider-wait'
import { $busy, $messages } from '@/store/session'
import { $sessionStates, setSessionTileDelegate } from '@/store/session-states'
import { $draftingToolSessions, setSessionDraftingTool } from '@/store/tool-drafting'

import { renderMessageStream } from './test-harness'
import { STREAM_DELTA_FLUSH_MS } from './utils'

const start = (execution: string, text: string, ref?: string): GatewayEvent => ({
  type: 'message.start',
  session_id: 's',
  turn: { id: execution, source: { kind: 'unknown' } },
  payload: { input: { role: 'user', text }, inputs: [{ id: `${execution}-input`, ref }] }
})

function harness() {
  const states = new Map<string, ClientSessionState>()

  const updateSessionState = (sid: string, update: (state: ClientSessionState) => ClientSessionState) => {
    const state = update(states.get(sid) ?? createClientSessionState())
    states.set(sid, state)
    $sessionStates.set({ ...$sessionStates.get(), [sid]: state })
    $messages.set(state.messages)

    return state
  }

  const h = renderMessageStream('s', { states, updateSessionState })

  return { ...h, updateSessionState }
}

function mountActions(
  h: ReturnType<typeof harness>,
  tile: boolean,
  requestGateway: Parameters<typeof usePromptActions>[0]['requestGateway']
) {
  if (tile) {
    setSessionTileDelegate({
      archiveSession: vi.fn(),
      branchSession: vi.fn(),
      deleteSession: vi.fn(),
      executeSlash: vi.fn(),
      interruptSession: vi.fn(),
      resumeTile: vi.fn(),
      submitToSession: vi.fn(),
      updateSession: h.updateSessionState
    })

    return renderHook(() =>
      useSessionTileActions({ requestGateway, runtimeId: 's', storedSessionId: 'stored', scope: MAIN_COMPOSER_SCOPE })
    ).result
  }

  return renderHook(() =>
    usePromptActions({
      activeSessionId: 's',
      activeSessionIdRef: { current: 's' },
      branchCurrentSession: async () => true,
      busyRef: { current: false },
      createBackendSessionForSend: async () => 's',
      getRoutedStoredSessionId: () => null,
      getRuntimeIdForStoredSession: () => null,
      getRouteToken: () => 'stable',
      handleSkinCommand: () => '',
      openMemoryGraph: () => undefined,
      refreshSessions: async () => undefined,
      requestGateway,
      resumeStoredSession: async () => undefined,
      runtimeIdByStoredSessionIdRef: { current: new Map() },
      selectedStoredSessionIdRef: { current: 'stored' },
      startFreshSessionDraft: () => undefined,
      sttEnabled: false,
      updateSessionState: h.updateSessionState
    })
  ).result
}

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  $sessionStates.set({})
  $messages.set([])
  $busy.set(false)
  $providerWaitSessions.set({})
  $draftingToolSessions.set({})
})

describe('shared input integration boundaries', () => {
  it('binds primary and tile regenerate, restore and edit to a fresh occurrence in either ACK order', async () => {
    for (const tile of [false, true]) {
      for (const verb of ['reload', 'restore', 'edit'] as const) {
        for (const ackFirst of [false, true]) {
          const h = harness()

          const source = {
            id: 'user-original',
            rowId: 17,
            inputIds: ['previous-input'],
            role: 'user' as const,
            parts: [textPart('Original')]
          }

          h.updateSessionState('s', state => ({
            ...state,
            messages: [source, { id: 'assistant-old', role: 'assistant', parts: [textPart('Old reply')] }]
          }))
          let acknowledge!: () => void

          const ack = new Promise<void>(resolve => {
            acknowledge = resolve
          })

          let params: Record<string, unknown> = {}

          const requestGateway = async <T,>(method: string, payload?: Record<string, unknown>): Promise<T> => {
            expect(method).toBe('prompt.submit')
            params = payload ?? {}
            await ack

            return { status: 'accepted', survivor_row_id_map: { 17: null } } as T
          }

          const actions = mountActions(h, tile, requestGateway)
          let pending!: Promise<void>
          await act(async () => {
            pending =
              verb === 'reload'
                ? actions.current.reloadFromMessage('assistant-old')
                : verb === 'restore'
                  ? actions.current.restoreToMessage(source.id)
                  : actions.current.editMessage({
                      role: 'user',
                      sourceId: source.id,
                      content: [{ type: 'text', text: 'Edited' }]
                    } as never)
          })
          expect(params.truncate_before_row_id).toBe(17)

          if (ackFirst) {
            await act(async () => {
              acknowledge()
              await pending
            })
          }

          act(() => {
            h.handleEvent(start('rewind', verb === 'edit' ? 'Edited' : 'Original', params.submission_ref as string))
            h.handleEvent(start('rewind', verb === 'edit' ? 'Edited' : 'Original', params.submission_ref as string))
          })

          if (!ackFirst) {
            await act(async () => {
              acknowledge()
              await pending
            })
          }

          const users = h.state().messages.filter(row => row.role === 'user')
          expect(users, `${tile ? 'tile' : 'primary'} ${verb}, ACK first: ${ackFirst}`).toHaveLength(1)
          expect(params.submission_ref).toEqual(expect.any(String))
          expect(params.submission_ref).not.toBe(source.id)
          expect(users[0]).toMatchObject({ inputIds: ['rewind-input'], id: params.submission_ref })
          expect(users[0]?.rowId).toBeUndefined()
          expect(chatMessageText(users[0]!)).toBe(verb === 'edit' ? 'Edited' : 'Original')
          cleanup()
          $busy.set(false)
        }
      }
    }
  })

  it('resolves observed start and correction rows through durable history before rewinding', async () => {
    for (const correction of [false, true]) {
      const h = harness()
      act(() => {
        h.handleEvent(start('peer', 'Peer prompt'))

        if (correction) {
          h.handleEvent({
            type: 'message.input',
            session_id: 's',
            turn: { id: 'peer', source: { kind: 'unknown' } },
            payload: {
              kind: 'redirect',
              input: { role: 'user', text: 'Peer correction', display_kind: 'steer' },
              inputs: [{ id: 'correction' }]
            }
          })
        }
      })
      h.updateSessionState('s', state => ({ ...state, busy: false, awaitingResponse: false }))
      const source = h.state().messages.at(-1)!
      const requests: { method: string; params?: Record<string, unknown> }[] = []

      const requestGateway = async <T,>(method: string, params?: Record<string, unknown>): Promise<T> => {
        requests.push({ method, params })

        return (
          method === 'session.history'
            ? { messages: [{ role: 'user', text: chatMessageText(source), row_id: 42 }] }
            : { status: 'accepted' }
        ) as T
      }

      const actions = mountActions(h, false, requestGateway)
      await act(async () => {
        await actions.current.restoreToMessage(source.id)
      })
      expect(requests.map(request => request.method)).toEqual(['session.history', 'prompt.submit'])
      expect(requests[1]?.params).toMatchObject({ truncate_before_row_id: 42, confirm_truncate: true })
      expect(requests[1]?.params).not.toHaveProperty('truncate_before_message_id')
      cleanup()
      $busy.set(false)
    }
  })

  it('reconciles thumbnails from two real submits with canonical local and peer images once, including replay', async () => {
    const h = harness()
    const users: ChatMessage[] = []
    const submits: Record<string, unknown>[] = []

    for (const name of ['first', 'second']) {
      const attachment: ComposerAttachment = {
        id: name,
        label: name,
        kind: 'image',
        thumbnailUrl: `data:image/png;base64,${name}`,
        refText: `@image:/server/${name}.png`
      }

      const { result } = renderHook(() =>
        useSubmitPrompt({
          activeSessionIdRef: { current: 's' },
          busyRef: { current: false },
          copy: {} as never,
          createBackendSessionForSend: async () => 's',
          getRoutedStoredSessionId: () => null,
          getRuntimeIdForStoredSession: () => null,
          getRouteToken: () => 'stable',
          requestGateway: async <T,>(_method: string, params?: Record<string, unknown>): Promise<T> => {
            submits.push(params ?? {})

            return { status: 'accepted' } as T
          },
          runtimeIdByStoredSessionIdRef: { current: new Map() },
          resumeStoredSession: async () => undefined,
          selectedStoredSessionIdRef: { current: null },
          syncAttachmentsForSubmit: async sessionId => ({ sessionId, attachments: [attachment] }),
          updateSessionState: h.updateSessionState,
          scope: {
            removeAttachments: () => undefined,
            readAttachments: () => [attachment],
            setAwaitingResponse: () => undefined,
            setBusy: () => undefined,
            setMessages: () => undefined
          }
        })
      )

      await act(async () => {
        expect(await result.current(name)).toBe(true)
      })
      users.push(h.state().messages.at(-1)!)
      expect(users.at(-1)?.attachmentRefs).toEqual([attachment.thumbnailUrl])
      h.updateSessionState('s', state => ({ ...state, messages: [], busy: false, awaitingResponse: false }))
    }

    h.updateSessionState('s', state => ({ ...state, messages: users }))

    const event: GatewayEvent = {
      ...start('merged', ''),
      payload: {
        input: {
          role: 'user',
          text: [...submits.map(submit => submit.text), '@image:/server/peer.png\n\nPeer'].join('\n\n')
        },
        inputs: [
          ...submits.map((submit, index) => ({ id: `local-${index}`, ref: submit.submission_ref as string })),
          { id: 'peer' }
        ]
      }
    }

    act(() => {
      h.handleEvent(event)
      h.handleEvent(event)
    })
    expect(h.state().messages).toHaveLength(1)
    expect(h.state().messages[0]?.attachmentRefs).toEqual([
      '@image:/server/first.png',
      '@image:/server/second.png',
      '@image:/server/peer.png'
    ])
  })

  it('seals every fresh execution before new output, regardless of visible input or optimistic correlation', async () => {
    vi.useFakeTimers()

    for (const kind of ['own', 'own-before-output', 'hidden', 'null', 'invalid', 'empty', 'peer']) {
      const h = harness()
      act(() => {
        h.handleEvent(start('old', 'Old'))
        h.handleEvent({ type: 'reasoning.delta', session_id: 's', payload: { text: 'Old reasoning' } })
        h.handleEvent({ type: 'message.delta', session_id: 's', payload: { text: 'Old output' } })
      })
      await act(async () => {
        await vi.advanceTimersByTimeAsync(STREAM_DELTA_FLUSH_MS)
      })
      act(() => {
        h.handleEvent({
          type: 'tool.start',
          session_id: 's',
          payload: { id: 'old-tool', name: 'read_file', args: { path: 'old.txt' } }
        })
      })
      const oldId = h.state().streamId
      const optimistic: ChatMessage = { id: 'user-own', role: 'user', parts: [textPart('New')] }
      h.updateSessionState('s', state => ({
        ...state,
        messages: [
          ...(kind === 'own-before-output'
            ? state.messages.flatMap(row => (row.id === oldId ? [optimistic, row] : [row]))
            : [...state.messages, optimistic]),
          { id: 'assistant-placeholder', role: 'assistant', pending: true, parts: [] }
        ]
      }))
      const event = start(`new-${kind}`, 'New', kind.startsWith('own') ? 'user-own' : undefined)
      const payload = event.payload as Record<string, unknown>

      if (kind === 'hidden') {
        event.payload = { ...payload, input: { role: 'user', text: 'Internal', display_kind: 'hidden' } }
      }

      if (kind === 'null') {
        event.payload = { ...payload, input: null }
      }

      if (kind === 'invalid') {
        event.payload = { ...payload, input: { role: 'assistant', text: 'Invalid' } }
      }

      if (kind === 'empty') {
        event.payload = { ...payload, input: { role: 'user', text: '' } }
      }

      act(() => {
        h.handleEvent(event)
        h.handleEvent({ type: 'message.delta', session_id: 's', payload: { text: 'New output' } })
      })
      await act(async () => {
        await vi.advanceTimersByTimeAsync(STREAM_DELTA_FLUSH_MS)
      })
      const old = h.state().messages.find(row => row.id === oldId)!
      expect(chatMessageText(old), kind).toBe('Old output')
      expect(old.pending, kind).toBe(false)
      expect(old.parts).toContainEqual(
        expect.objectContaining({ type: 'reasoning', text: 'Old reasoning', completedAt: expect.any(Number) })
      )
      expect(old.parts).toContainEqual(
        expect.objectContaining({ type: 'tool-call', toolCallId: 'old-tool', completedAt: expect.any(Number) })
      )
      expect(
        h.state().messages.some(row => row.id === 'assistant-placeholder'),
        kind
      ).toBe(false)
      expect(chatMessageText(h.state().messages.at(-1)!), kind).toBe('New output')
      expect(h.state().streamId, kind).not.toBe(oldId)

      if (kind.startsWith('own')) {
        const messages = h.state().messages
        expect(
          messages.findIndex(row => row.id === 'user-own'),
          kind
        ).toBeGreaterThan(messages.findIndex(row => row.id === oldId))
      }

      cleanup()
    }
  })

  it('rejects replayed and unscoped observed starts before clearing status stores or changing the unscoped stream pin', async () => {
    vi.useFakeTimers()
    const activeSessionIdRef = { current: 'original' }
    const h = renderMessageStream('original', { activeSessionIdRef })
    act(() => {
      h.handleEvent({ type: 'message.start' })
      h.handleEvent({ ...start('seen', 'Start'), session_id: 'original' })
    })
    setSessionDraftingTool('original', 'terminal')
    setSessionProviderWait('original', 'Waiting')
    const drafting = $draftingToolSessions.get()
    const waiting = $providerWaitSessions.get()
    act(() => {
      h.handleEvent({ ...start('seen', 'Start'), session_id: 'original' })
    })
    expect($draftingToolSessions.get()).toBe(drafting)
    expect($providerWaitSessions.get()).toBe(waiting)
    activeSessionIdRef.current = 'other'
    act(() => {
      h.handleEvent({ ...start('unscoped', 'Ignored'), session_id: undefined })
      h.handleEvent({ type: 'message.delta', payload: { text: 'Still original' } })
    })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(STREAM_DELTA_FLUSH_MS)
    })
    expect(h.state('original').messages.map(chatMessageText)).toContain('Still original')
    expect(h.state('other').messages).toEqual([])
  })
})
