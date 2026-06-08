import type { StatusbarMenuItem } from '@/app/shell/statusbar-controls'

const LOG_TAIL = 5

interface RpcEventLike {
  payload?: unknown
  type?: string
}

const SESSION_SCOPED_EVENT_TYPES = new Set([
  'approval.request',
  'clarify.request',
  'error',
  'message.complete',
  'message.delta',
  'message.start',
  'reasoning.available',
  'reasoning.delta',
  'secret.request',
  'status.update',
  'subagent.complete',
  'subagent.progress',
  'subagent.spawn_requested',
  'subagent.start',
  'subagent.thinking',
  'subagent.tool',
  'sudo.request',
  'thinking.delta'
])

function asRecord(payload: unknown): Record<string, unknown> {
  return payload && typeof payload === 'object' ? (payload as Record<string, unknown>) : {}
}

export function gatewayEventRequiresSessionId(eventType: string | undefined): boolean {
  if (!eventType) {
    return false
  }

  return SESSION_SCOPED_EVENT_TYPES.has(eventType) || eventType.startsWith('tool.')
}

export function gatewayEventCompletedFileDiff(event: RpcEventLike): boolean {
  if (event.type !== 'tool.complete') {
    return false
  }

  const diff = asRecord(event.payload).inline_diff

  return typeof diff === 'string' && diff.trim().length > 0
}

export function buildGatewayLogItems(lines: readonly string[]): readonly StatusbarMenuItem[] {
  if (lines.length === 0) {
    return [
      {
        className: 'text-muted-foreground',
        disabled: true,
        id: 'gateway-log-empty',
        label: 'No recent gateway log lines'
      }
    ]
  }

  return lines.slice(-LOG_TAIL).map((line, index) => ({
    className: 'font-mono text-[0.68rem] text-muted-foreground',
    disabled: true,
    id: `gateway-log:${index}`,
    label: line.trim().slice(0, 120) || '(blank log line)'
  }))
}
