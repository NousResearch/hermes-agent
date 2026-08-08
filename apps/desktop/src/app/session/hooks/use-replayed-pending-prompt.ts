import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { $replayedPendingPrompt } from '@/store/session'
import type { RpcEvent } from '@/types/hermes'

/** Dispatch a blocking prompt the backend replayed on reattach.
 *
 * The gateway emits clarify/sudo/secret/terminal.read exactly once, so a client
 * that disconnected before answering only learns the request is still open from
 * the resume/activate payload. The resume path cannot reach the gateway-event
 * handler, so it parks the request on an atom and this drains it.
 *
 * Replaying the original event rather than writing the prompt stores directly
 * keeps one implementation of every blocking bridge instead of a second copy
 * for the reconnect case. It is safe to re-dispatch: the prompt stores key by
 * session, so re-applying the same request id overwrites rather than stacks.
 */
export function useReplayedPendingPrompt(handleGatewayEvent: (event: RpcEvent) => void): void {
  const replayed = useStore($replayedPendingPrompt)

  useEffect(() => {
    if (!replayed) {
      return
    }

    // Clear before dispatching: the handler can synchronously trigger another
    // resume, and a still-armed atom would replay the same request forever.
    $replayedPendingPrompt.set(null)

    handleGatewayEvent({
      payload: replayed.pending.payload ?? {},
      session_id: replayed.sessionId,
      type: replayed.pending.event
    })
  }, [handleGatewayEvent, replayed])
}
