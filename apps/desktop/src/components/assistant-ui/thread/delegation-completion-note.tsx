import { type FC, useId, useState } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { DisclosureCaret } from '@/components/ui/disclosure-caret'
import { LogView } from '@/components/ui/log-view'
import { useI18n } from '@/i18n'

const SINGLE_MARKER_RE = /^\[ASYNC DELEGATION COMPLETE — ([^\]\r\n]+)\](?:\r?\n|$)/
const BATCH_MARKER_RE = /^\[ASYNC DELEGATION BATCH COMPLETE — ([^\]\r\n]+)\](?:\r?\n|$)/
const RESULT_SEPARATOR = '--- RESULT ---'
const TASK_HEADER_RE = /^--- [✓✗] TASK \d+\/\d+(?:: (.*?))?\s+\(status=([^,)]+)[^)]*\) ---$/

interface DelegationTaskSummary {
  goal?: string
  status: string
  summary?: string
}

export interface DelegationCompletion {
  duration?: string
  goal?: string
  id: string
  kind: 'single' | 'batch'
  raw: string
  status?: string
  summary?: string
  tasks: DelegationTaskSummary[]
}

function fieldValue(lines: string[], label: string): string | undefined {
  const line = lines.find(candidate => candidate.startsWith(label))

  return line?.slice(label.length).trim() || undefined
}

function firstMeaningfulLine(lines: string[]): string | undefined {
  return lines
    .find(line => {
      const value = line.trim()

      return (
        value.length > 0 &&
        !value.startsWith('A background ') &&
        !value.startsWith('Dispatched:') &&
        !value.startsWith('Context you provided:') &&
        !value.startsWith('Toolsets:') &&
        !value.startsWith('Role:') &&
        !value.startsWith('Status:') &&
        value !== RESULT_SEPARATOR &&
        value !== 'Partial output:' &&
        !TASK_HEADER_RE.test(value)
      )
    })
    ?.trim()
}

function parseTasks(lines: string[]): DelegationTaskSummary[] {
  const tasks: DelegationTaskSummary[] = []

  for (let index = 0; index < lines.length; index += 1) {
    const match = lines[index]?.trim().match(TASK_HEADER_RE)

    if (!match) {
      continue
    }

    const followingLines = lines.slice(index + 1)
    tasks.push({
      goal: match[1]?.trim() || undefined,
      status: match[2]?.trim() || 'unknown',
      summary: firstMeaningfulLine(followingLines)
    })
  }

  return tasks
}

export function parseDelegationCompletion(text: string): DelegationCompletion | null {
  const batchMarker = text.match(BATCH_MARKER_RE)
  const singleMarker = text.match(SINGLE_MARKER_RE)
  const marker = batchMarker ?? singleMarker

  if (!marker) {
    return null
  }

  const kind = batchMarker ? 'batch' : 'single'
  const lines = text.replace(/\r\n/g, '\n').split('\n')
  const roleLine = lines.find(line => line.startsWith('Role:'))
  const statusLine = lines.find(line => line.startsWith('Status:'))
  const tasks = kind === 'batch' ? parseTasks(lines) : []
  const resultIndex = lines.findIndex(line => line.trim() === RESULT_SEPARATOR)
  const resultLines = resultIndex >= 0 ? lines.slice(resultIndex + 1) : lines.slice(1)

  const duration =
    statusLine?.match(/\bDuration:\s*([^\s]+)/)?.[1] ?? roleLine?.match(/\bTotal duration:\s*([^\s]+)/)?.[1]

  const status = statusLine?.match(/^Status:\s*([^\s]+)/)?.[1]

  return {
    duration,
    goal: kind === 'single' ? fieldValue(lines, 'Original goal:') : tasks.find(task => task.goal)?.goal,
    id: marker[1]!.trim(),
    kind,
    raw: text,
    status: kind === 'single' ? status : aggregateTaskStatus(tasks),
    summary: kind === 'batch' ? tasks.find(task => task.summary)?.summary : firstMeaningfulLine(resultLines),
    tasks
  }
}

function aggregateTaskStatus(tasks: DelegationTaskSummary[]): string | undefined {
  if (tasks.length === 0) {
    return undefined
  }

  const counts = new Map<string, number>()

  for (const task of tasks) {
    counts.set(task.status, (counts.get(task.status) ?? 0) + 1)
  }

  return [...counts].map(([status, count]) => `${count} ${status}`).join(' · ')
}

export const DelegationCompletionNote: FC<{ completion: DelegationCompletion }> = ({ completion }) => {
  const { t } = useI18n()
  const copy = t.assistant.thread
  const [expanded, setExpanded] = useState(false)
  const payloadId = useId()
  const title = completion.kind === 'batch' ? copy.subagentsCompleted : copy.subagentCompleted
  const toggleLabel = expanded ? copy.hideFullPayload : copy.showFullPayload

  return (
    <section
      aria-label={title}
      className="mx-auto flex w-full max-w-[44rem] flex-col gap-1.5 border-l border-(--ui-stroke-tertiary) px-3 py-1.5 text-[0.6875rem] leading-5 text-(--ui-text-secondary)"
      data-slot="delegation-completion-note"
    >
      <div className="flex min-w-0 items-center gap-1.5">
        <Codicon className="shrink-0 text-(--ui-text-tertiary)" name="check-all" size="0.75rem" />
        <strong className="text-(--ui-text-primary)">{title}</strong>
        {completion.status && <span className="truncate text-(--ui-text-tertiary)">{completion.status}</span>}
        {completion.duration && (
          <span className="ml-auto shrink-0 font-mono text-(--ui-text-tertiary)">{completion.duration}</span>
        )}
      </div>
      {completion.goal && <div className="truncate text-(--ui-text-secondary)">{completion.goal}</div>}
      {completion.summary && <div className="line-clamp-2 text-(--ui-text-tertiary)">{completion.summary}</div>}
      <button
        aria-controls={payloadId}
        aria-expanded={expanded}
        aria-label={toggleLabel}
        className="flex w-fit items-center gap-1 text-(--ui-text-tertiary) hover:text-(--ui-text-secondary)"
        onClick={() => setExpanded(value => !value)}
        type="button"
      >
        <DisclosureCaret aria-hidden open={expanded} />
        <span>{toggleLabel}</span>
      </button>
      {expanded && (
        <LogView className="max-h-64" id={payloadId}>
          {completion.raw}
        </LogView>
      )}
    </section>
  )
}
