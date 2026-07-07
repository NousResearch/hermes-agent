import type { HermesGateway } from '@/hermes'
import type { Translations } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { type ClarifyRequest, clearClarifyRequest } from '@/store/clarify'
import { notifyError } from '@/store/notifications'

type ClarifyCopy = Translations['assistant']['clarify']

export async function respondToClarifyRequest({
  answer,
  copy,
  gateway,
  onBeforeSend,
  onError,
  request
}: {
  answer: string
  copy: ClarifyCopy
  gateway: HermesGateway | null | undefined
  onBeforeSend?: () => void
  onError?: (error: unknown) => void
  request: ClarifyRequest | null | undefined
}): Promise<boolean> {
  const trimmed = answer.trim()

  if (!trimmed) {
    return false
  }

  if (!request?.requestId) {
    notifyError(new Error(copy.notReady), copy.sendFailed)

    return false
  }

  if (!gateway) {
    notifyError(new Error(copy.gatewayDisconnected), copy.sendFailed)

    return false
  }

  onBeforeSend?.()

  try {
    await gateway.request<{ ok?: boolean }>('clarify.respond', {
      request_id: request.requestId,
      answer: trimmed
    })
    triggerHaptic('submit')
    clearClarifyRequest(request.requestId, request.sessionId)

    return true
  } catch (error) {
    notifyError(error, copy.sendFailed)
    onError?.(error)

    return false
  }
}
