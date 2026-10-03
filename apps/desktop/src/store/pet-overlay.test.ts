import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages/types'
import { setBusy, setMessages } from '@/store/session'

import {
  initPetOverlayBridge,
  type PetOverlayControl,
  type PetOverlayStatePayload,
  popInPet,
  popOutPet,
  setPetOverlayDictateHandler,
  setPetOverlayNewChatHandler,
  setPetOverlaySubmitHandler
} from './pet-overlay'

const text = (id: string, role: ChatMessage['role'], body: string): ChatMessage =>
  ({ id, parts: [{ text: body, type: 'text' }], role }) as ChatMessage

let pushed: PetOverlayStatePayload[] = []
let sendControl: ((payload: PetOverlayControl) => void) | null = null

let disposeBridge: () => void = () => {}

const lastPayload = () => pushed[pushed.length - 1]

beforeEach(() => {
  pushed = []
  window.localStorage.clear()
  setBusy(false)
  setMessages([])
  window.hermesDesktop = {
    petOverlay: {
      close: vi.fn(async () => undefined),
      onControl: (cb: (payload: PetOverlayControl) => void) => {
        sendControl = cb

        return () => {
          sendControl = null
        }
      },
      open: vi.fn(async () => ({})),
      pushState: (payload: PetOverlayStatePayload) => void pushed.push(payload)
    }
  } as unknown as typeof window.hermesDesktop
  disposeBridge = initPetOverlayBridge()
  popOutPet({ height: 60, width: 60, x: 10, y: 10 })
})

afterEach(() => {
  popInPet()
  disposeBridge()
  setPetOverlaySubmitHandler(null)
  setPetOverlayDictateHandler(null)
  setPetOverlayNewChatHandler(null)
})

describe('pet overlay companion thread', () => {
  it('keeps one final answer per question, dropping mid-turn narration', () => {
    setMessages([
      text('u1', 'user', 'any mail about X?'),
      text('a1', 'assistant', 'Let me check the inbox.'),
      text('a2', 'assistant', 'Two messages about X, both from Ana.'),
      text('u2', 'user', 'thanks')
    ])
    setBusy(true)

    const thread = lastPayload().thread

    expect(thread.map(turn => turn.id)).toEqual(['u1', 'a2', 'u2'])
  })

  it('withholds the running turn answer until the turn ends', () => {
    setMessages([text('u1', 'user', 'hi'), text('a1', 'assistant', 'Checking…')])
    setBusy(true)
    expect(lastPayload().thread.map(turn => turn.role)).toEqual(['user'])

    setBusy(false)
    expect(lastPayload().thread.map(turn => turn.id)).toEqual(['u1', 'a1'])
  })
})

describe('pet overlay companion controls', () => {
  it('routes new-chat to the registered handler', () => {
    const newChat = vi.fn()
    setPetOverlayNewChatHandler(newChat)

    sendControl?.({ type: 'new-chat' })

    expect(newChat).toHaveBeenCalledTimes(1)
  })

  it('transcribes dictation, echoes what was heard, then sends it', async () => {
    const submit = vi.fn()
    setPetOverlaySubmitHandler(submit)
    setPetOverlayDictateHandler(async () => '  hello there ')

    sendControl?.({ dataUrl: 'data:audio/webm;base64,AAAA', mime: 'audio/webm', type: 'dictate' })

    await vi.waitFor(() => expect(submit).toHaveBeenCalledWith('hello there'))
    expect(lastPayload().heard).toMatchObject({ text: 'hello there' })
    expect(lastPayload().heard?.error).toBeFalsy()
  })

  it('reports an empty transcription instead of sending it', async () => {
    const submit = vi.fn()
    setPetOverlaySubmitHandler(submit)
    setPetOverlayDictateHandler(async () => '   ')

    sendControl?.({ dataUrl: 'data:audio/webm;base64,AAAA', mime: 'audio/webm', type: 'dictate' })

    await vi.waitFor(() => expect(lastPayload().heard?.error).toBe(true))
    expect(submit).not.toHaveBeenCalled()
  })
})

describe('pet overlay notices', () => {
  it('surfaces a notice from the main process and lights the unread icon', () => {
    sendControl?.({ id: '42', text: 'Two emails need a reply', type: 'notice' })

    expect(lastPayload().notice).toEqual({ id: '42', text: 'Two emails need a reply' })
    expect(lastPayload().unread).toBe(true)
  })
})
