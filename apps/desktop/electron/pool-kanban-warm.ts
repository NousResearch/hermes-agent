/**
 * Skip idle-SIGTERM of a desktop backend whose profile still has a running
 * Kanban worker. Council specialists were dying at 10 min idle while a card
 * for that seat was about to dispatch; Playwright MCP is not kept all day —
 * only while the card is `running`.
 */

export function poolKeyProfile(key: string): string {
  const parts = String(key).split('::')
  return (parts[parts.length - 1] || key).trim().toLowerCase()
}

export function runningAssigneesFromKanbanList(payload: unknown): Set<string> {
  const rows = Array.isArray(payload)
    ? payload
    : payload && typeof payload === 'object' && Array.isArray((payload as { tasks?: unknown }).tasks)
      ? (payload as { tasks: unknown[] }).tasks
      : []
  const names = new Set<string>()
  for (const row of rows) {
    if (!row || typeof row !== 'object') {
      continue
    }
    const status = String((row as { status?: unknown }).status || '').toLowerCase()
    const assignee = String((row as { assignee?: unknown }).assignee || '').trim().toLowerCase()
    if (status === 'running' && assignee) {
      names.add(assignee)
    }
  }
  return names
}

export function shouldKeepPoolBackendWarm(poolKey: string, runningAssignees: ReadonlySet<string>): boolean {
  return runningAssignees.has(poolKeyProfile(poolKey))
}
