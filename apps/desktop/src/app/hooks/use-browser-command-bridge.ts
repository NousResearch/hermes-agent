import type { GatewayEvent } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import {
  type BrowserBridgeCommand,
  resolveBrowserBridgeTargetTabId,
  runBrowserBridgeCommand
} from '@/store/browser-bridge'
import { $gateway } from '@/store/gateway'

import { useGatewayRequest } from '../gateway/hooks/use-gateway-request'

const RESPONSE_METHOD = 'browser.desktop.respond'
const REQUEST_EVENT = 'browser.command.request'

interface BrowserCommandRequestPayload {
  command?: unknown
  params?: unknown
  request_id?: unknown
  requestId?: unknown
  tab_id?: unknown
  tabId?: unknown
}

interface BrowserCommandResponse extends Record<string, unknown> {
  error?: string
  ok: boolean
  request_id: string
  result?: unknown
}

type GatewayRequester = (method: string, params?: Record<string, unknown>, timeoutMs?: number) => Promise<unknown>
type BrowserCommandRunner = (
  tabId: NonNullable<ReturnType<typeof resolveBrowserBridgeTargetTabId>>,
  command: BrowserBridgeCommand,
  params?: Record<string, unknown>
) => Promise<unknown>

export async function handleBrowserCommandRequest(
  event: GatewayEvent<BrowserCommandRequestPayload>,
  requestGateway: GatewayRequester,
  runCommand: BrowserCommandRunner = runBrowserBridgeCommand
): Promise<void> {
  const payload = event.payload ?? {}

  const requestId = typeof payload.request_id === 'string'
    ? payload.request_id
    : typeof payload.requestId === 'string'
      ? payload.requestId
      : ''

  if (!requestId) {
    return
  }

  const response: BrowserCommandResponse = { ok: false, request_id: requestId }

  try {
    const command = normalizeCommand(payload.command)
    const tabId = resolveBrowserBridgeTargetTabId(payload.tab_id ?? payload.tabId, event.session_id ?? null)

    if (!tabId) {
      throw new Error('No visible browser tab is bound for this session')
    }

    const params = isRecord(payload.params) ? payload.params : {}
    response.ok = true
    response.result = await runCommand(tabId, command, params)
  } catch (error) {
    response.error = error instanceof Error ? error.message : String(error)
  }

  await requestGateway(RESPONSE_METHOD, response, 10_000)
}

export function useBrowserCommandBridge(enabled = true): void {
  const gateway = useStore($gateway)
  const { requestGateway } = useGatewayRequest()

  useEffect(() => {
    if (!enabled || !gateway) {
      return undefined
    }

    return gateway.on<BrowserCommandRequestPayload>(REQUEST_EVENT, event => {
      void handleBrowserCommandRequest(event, requestGateway)
    })
  }, [enabled, gateway, requestGateway])
}

function normalizeCommand(command: unknown): BrowserBridgeCommand {
  if (
    command === 'accessibilityAudit'
    || command === 'clearConsole'
    || command === 'clearNetwork'
    || command === 'click'
    || command === 'clickRef'
    || command === 'designHandoff'
    || command === 'doubleClick'
    || command === 'doubleClickRef'
    || command === 'evaluate'
    || command === 'fillRef'
    || command === 'getConsole'
    || command === 'getImages'
    || command === 'getNetwork'
    || command === 'getState'
    || command === 'goBack'
    || command === 'goForward'
    || command === 'hover'
    || command === 'hoverRef'
    || command === 'inspectElement'
    || command === 'navigate'
    || command === 'press'
    || command === 'reload'
    || command === 'rightClick'
    || command === 'rightClickRef'
    || command === 'screenshot'
    || command === 'scroll'
    || command === 'selectElement'
    || command === 'snapshot'
    || command === 'stop'
    || command === 'type'
  ) {
    return command
  }

  throw new Error(`Unsupported visible browser command: ${String(command)}`)
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}
