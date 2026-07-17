'use client'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { useContributions } from '@/contrib/react/use-contributions'
import { cn } from '@/lib/utils'

export const CHAT_IMAGE_ACTIONS_AREA = 'chat.imageActions'

export interface ChatImageActionInput {
  src: string
  toolName?: string
  toolResult?: unknown
}

export interface ChatImageActionContribution {
  codicon?: string
  label: string
  onSelect: (input: ChatImageActionInput) => void
  when?: (input: ChatImageActionInput) => boolean
}

const TOOL_RESULT_ACTION_KEYS = new Set([
  'aspect_ratio', 'batch_size', 'count', 'created_at', 'createdAt', 'data', 'duration', 'height', 'id',
  'items', 'job_id', 'jobId', 'jobs', 'medias', 'minUrl', 'model', 'outputs', 'params', 'prompt', 'rawUrl',
  'resolution', 'result', 'result_url', 'resultUrl', 'results', 'role', 'status', 'structuredContent', 'type', 'url',
  'width'
])

interface ToolResultSanitizeBudget {
  remaining: number
}

export function toolResultForImageAction(
  value: unknown,
  depth = 0,
  seen = new Set<unknown>(),
  budget: ToolResultSanitizeBudget = { remaining: 256 }
): unknown {
  if (depth > 6 || budget.remaining <= 0 || value === null || value === undefined) {
    return null
  }

  budget.remaining -= 1

  if (typeof value === 'string') {
    return value.slice(0, 4096)
  }

  if (typeof value === 'number' || typeof value === 'boolean') {
    return value
  }

  if (Array.isArray(value)) {
    return value.slice(0, 20).map(item => toolResultForImageAction(item, depth + 1, seen, budget))
  }

  if (typeof value !== 'object' || seen.has(value)) {
    return null
  }

  seen.add(value)
  const result: Record<string, unknown> = {}

  for (const [key, entry] of Object.entries(value)) {
    if (budget.remaining <= 0) {
      break
    }

    if (!TOOL_RESULT_ACTION_KEYS.has(key)) {
      continue
    }

    result[key] = toolResultForImageAction(entry, depth + 1, seen, budget)
  }

  return result
}

function imageActionData(value: unknown): ChatImageActionContribution | null {
  if (!value || typeof value !== 'object') {
    return null
  }

  const candidate = value as Partial<ChatImageActionContribution>

  return typeof candidate.label === 'string' && typeof candidate.onSelect === 'function'
    ? (candidate as ChatImageActionContribution)
    : null
}

export function ChatImageActions({
  className,
  src,
  toolName,
  toolResult
}: ChatImageActionInput & { className?: string }) {
  const contributions = useContributions(CHAT_IMAGE_ACTIONS_AREA)

  const input: ChatImageActionInput = {
    src,
    ...(toolName ? { toolName } : {}),
    ...(toolResult === undefined ? {} : { toolResult: toolResultForImageAction(toolResult) })
  }

  const actions = contributions
    .map(contribution => ({ contribution, data: imageActionData(contribution.data) }))
    .filter(
      (entry): entry is { contribution: (typeof contributions)[number]; data: ChatImageActionContribution } =>
        Boolean(entry.data && (!entry.data.when || entry.data.when(input)))
    )

  if (!src || actions.length === 0) {
    return null
  }

  return (
    <span className={cn('absolute left-2 top-2 z-10 flex gap-1', className)} data-slot="chat-image-actions">
      {actions.map(({ contribution, data }) => (
        <Button
          aria-label={data.label}
          key={contribution.id}
          onClick={event => {
            event.stopPropagation()
            data.onSelect(input)
          }}
          size="icon-sm"
          title={data.label}
          type="button"
          variant="secondary"
        >
          {data.codicon ? <Codicon name={data.codicon} /> : data.label.slice(0, 1)}
        </Button>
      ))}
    </span>
  )
}
