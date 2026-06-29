import { getGlobalModelOptions, type HermesGateway, type ModelOptionsResponse } from '@/hermes'

interface ModelOptionsRequest {
  gateway?: HermesGateway
  refresh?: boolean
  sessionId?: null | string
}

export function requestModelOptions({ gateway, refresh = false, sessionId }: ModelOptionsRequest): Promise<ModelOptionsResponse> {
  if (gateway) {
    const params: Record<string, unknown> = {}

    if (sessionId) {
      params.session_id = sessionId
    }

    if (refresh) {
      params.refresh = true
    }

    return gateway.request<ModelOptionsResponse>('model.options', params)
  }

  return getGlobalModelOptions(refresh ? { refresh: true } : undefined)
}
