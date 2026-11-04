import { JsonRpcGatewayError, type SharedMetricsDesktopDailyParams } from '@hermes/shared'

import type { SharedMetricsRequester } from './shared-metrics'

/** Old gateways reject `via=state` before executing the daily report. Preserve legacy metrics there. */
export async function sendDesktopDaily(
  request: SharedMetricsRequester,
  params: SharedMetricsDesktopDailyParams,
  canRetry: () => boolean
) {
  try {
    return await request<{ recorded?: boolean }>('shared_metrics.desktop_daily', { ...params })
  } catch (error) {
    if (
      !(error instanceof JsonRpcGatewayError) ||
      ![4000, -32602].includes(error.code ?? 0) ||
      !params.actions?.some(action => action.via === 'state')
    ) {
      throw error
    }

    if (!canRetry()) {
      return
    }

    return request<{ recorded?: boolean }>('shared_metrics.desktop_daily', {
      ...params,
      actions: params.actions.filter(action => !action.action.startsWith('ia.'))
    })
  }
}
