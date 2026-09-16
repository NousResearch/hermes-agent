// Pure parsing for the async-delegation completion payloads that
// tools/process_registry.py injects as synthetic user turns. No React/DOM so
// timeline-data can reuse the marker classification.

const SINGLE_MARKER_RE = /^\[ASYNC DELEGATION COMPLETE — ([^\]\r\n]+)\](?:\r?\n|$)/
const BATCH_MARKER_RE = /^\[ASYNC DELEGATION BATCH COMPLETE — ([^\]\r\n]+)\](?:\r?\n|$)/
const RESULT_SEPARATOR = '--- RESULT ---'
const ERROR_SEPARATOR = '--- ERROR ---'
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

/** The exact marker classification: does this text START as a delegation
 *  completion payload? Mid-text mentions of the marker do not count. */
export function isDelegationCompletionText(text: string): boolean {
  return SINGLE_MARKER_RE.test(text) || BATCH_MARKER_RE.test(text)
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

  // A batch that failed before producing any result has no TASK headers: the
  // formatter emits an "--- ERROR ---" block instead (empty `results` with a
  // top-level `error`). Surface that failure so the collapsed card doesn't
  // render as an empty success.
  const errorIndex = tasks.length === 0 ? lines.findIndex(line => line.trim() === ERROR_SEPARATOR) : -1
  const batchError = errorIndex >= 0 ? firstMeaningfulLine(lines.slice(errorIndex + 1)) : undefined

  const duration =
    statusLine?.match(/\bDuration:\s*([^\s]+)/)?.[1] ?? roleLine?.match(/\bTotal duration:\s*([^\s]+)/)?.[1]

  const status = statusLine?.match(/^Status:\s*([^\s]+)/)?.[1]

  return {
    duration,
    goal: kind === 'single' ? fieldValue(lines, 'Original goal:') : tasks.find(task => task.goal)?.goal,
    id: marker[1]!.trim(),
    kind,
    raw: text,
    status: kind === 'single' ? status : (aggregateTaskStatus(tasks) ?? (batchError ? 'failed' : undefined)),
    summary: kind === 'batch' ? (tasks.find(task => task.summary)?.summary ?? batchError) : firstMeaningfulLine(resultLines),
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
