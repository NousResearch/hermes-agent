import { useEffect, useRef } from 'react'

import { isMissingRpcMethod } from '@/lib/gateway-rpc'
import {
  claimSideChatTask,
  initSideChatBridge,
  replyToSideChat,
  setSideChatAskHandler,
  type SideChatAsk
} from '@/store/side-chat'

interface SideChatBridgeParams {
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
}

/**
 * Wires the floating `/btw` side chat back into this window's gateway.
 *
 * A question typed there rides the SAME `prompt.btw` RPC the inline `/btw`
 * always called — the window is a view, not a second client — and the answer
 * arrives as the `btw.complete` event this window is already subscribed to
 * (see use-message-stream/gateway-event/status). The task id the RPC returns is
 * what joins the two halves: it is claimed here and consumed there, which is
 * also what keeps a side-chat answer OUT of the parent transcript.
 *
 * Deliberately NOT primary-only, unlike the Quick Entry bridge. `/btw` is
 * reachable from a popped-out session window and the HUD too, and main routes
 * asks back to the window that opened the side chat — the one actually holding
 * that session's runtime binding and profile-scoped socket. A window that never
 * opened one is simply never sent anything.
 *
 * The handler registers ONCE through a ref tracking the latest callback:
 * re-registering on identity churn leaves a nulled-handler window that can drop
 * an ask (the same bug shape use-pet-bridge guards).
 */
export function useSideChatBridge({ requestGateway }: SideChatBridgeParams): void {
  const requestGatewayRef = useRef(requestGateway)
  requestGatewayRef.current = requestGateway

  useEffect(() => {
    const runAsk = async (ask: SideChatAsk) => {
      try {
        const result = await requestGatewayRef.current<{ task_id?: string }>('prompt.btw', {
          session_id: ask.sessionId,
          text: ask.text
        })

        const taskId = String(result?.task_id ?? '').trim()

        if (!taskId) {
          // Without a task id there is nothing to match `btw.complete` against,
          // so the bubble could never settle. Say so now rather than never.
          replyToSideChat({ askId: ask.askId, error: 'The backend did not start this side question.', text: '' })

          return
        }

        claimSideChatTask(taskId, ask.askId)
      } catch (err) {
        // A gateway too old for the dedicated RPC has no side-agent path the
        // side window could use either (the slash-worker fallback prints its
        // answer past the capture window — #99065), so this is a dead end
        // rather than something to retry differently.
        const message = isMissingRpcMethod(err)
          ? 'This backend is too old for side questions.'
          : err instanceof Error
            ? err.message
            : String(err)

        replyToSideChat({ askId: ask.askId, error: message, text: '' })
      }
    }

    setSideChatAskHandler(ask => void runAsk(ask))

    const dispose = initSideChatBridge()

    return () => {
      setSideChatAskHandler(null)
      dispose()
    }
  }, [])
}
