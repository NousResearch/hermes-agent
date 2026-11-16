import { type ChatMessage, textPart } from '@/lib/chat-messages'
import { $gateway } from '@/store/gateway'
import { $activeSessionId, setMessages, touchSessionActivity } from '@/store/session'
import { $sessionStates, requestForOwnedSession } from '@/store/session-states'

interface AppendOptions {
  clientId?: string
  role?: string
  sessionId?: null | string
  text?: string
}

interface AppendReceipt {
  id: number
  message_uid: string
  ok: true
}

/** The boolean acknowledges dispatch, not persistence. Retain clientId when
 * retrying after a disconnect; the owner returns the original durable receipt. */
export function appendMessage(opts: AppendOptions = {}): boolean {
  const body = String(opts.text ?? '').trim()

  const role = String(opts.role ?? 'assistant')
    .trim()
    .toLowerCase()

  const runtimeId = String(opts.sessionId || $activeSessionId.get() || '').trim()

  if (!body || !runtimeId || (role !== 'user' && role !== 'assistant')) {
    return false
  }

  const clientId = opts.clientId ?? randomUuid()
  void requestForOwnedSession<AppendReceipt>(
    runtimeId,
    async (method, params) => {
      const gateway = $gateway.get()

      if (!gateway) {
        throw new Error('Hermes gateway unavailable')
      }

      return gateway.request(method, params)
    },
    'session.append_message',
    { session_id: runtimeId, role, content: body, client_id: clientId }
  )
    .then(receipt => {
      if (receipt.ok !== true || !Number.isInteger(receipt.id) || !receipt.message_uid) {
        throw new Error('Invalid session.append_message receipt')
      }

      const message: ChatMessage = {
        id: `external-${receipt.message_uid}`,
        rowId: receipt.id,
        messageUid: receipt.message_uid,
        role,
        parts: [textPart(body)]
      }

      const append = (messages: ChatMessage[]) =>
        messages.some(item => item.messageUid === receipt.message_uid || item.rowId === receipt.id)
          ? messages : [...messages, message]

      // Re-read focus at acknowledgement time: the user may have switched chats.
      if ($activeSessionId.get() === runtimeId) {
        setMessages(append)
      }

      const states = $sessionStates.get()
      const target = states[runtimeId]

      if (target) {
        $sessionStates.set({ ...states, [runtimeId]: { ...target, messages: append(target.messages) } })
      }

      touchSessionActivity(runtimeId, { preview: body })
    })
    .catch(error => {
      console.warn('session.append_message failed', error)
    })

  return true
}

import { randomUuid } from '@hermes/shared'
