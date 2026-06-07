import { atom, computed } from 'nanostores'

export interface FileChange {
  additions: number
  diff: string
  path: string
  timestamp: number
  toolCallId: string
  toolName: string
}

export interface ToolCallSummary {
  count: number
  errorCount: number
  toolName: string
  totalDuration: number
}

export const $sessionChanges = atom<FileChange[]>([])

export const $sessionChangeSummary = computed($sessionChanges, changes => {
  let totalAdditions = 0
  let totalDeletions = 0
  for (const c of changes) {
    totalAdditions += c.additions
    for (const line of c.diff.split('\n')) {
      if (line.startsWith('-') && !line.startsWith('---')) totalDeletions++
    }
  }
  return { files: changes.length, totalAdditions, totalDeletions }
})

export const $toolCallSummary = computed($sessionChanges, changes => {
  const byTool = new Map<string, { count: number; totalDuration: number }>()
  for (const c of changes) {
    const existing = byTool.get(c.toolName)
    if (existing) {
      existing.count++
    } else {
      byTool.set(c.toolName, { count: 1, totalDuration: 0 })
    }
  }
  const result: ToolCallSummary[] = []
  for (const [toolName, stats] of byTool) {
    result.push({
      count: stats.count,
      errorCount: 0,
      toolName,
      totalDuration: stats.totalDuration
    })
  }
  return result.sort((a, b) => b.count - a.count)
})

export function addSessionChange(change: FileChange) {
  const current = $sessionChanges.get()
  $sessionChanges.set([...current, change])
}

export function clearSessionChanges() {
  $sessionChanges.set([])
}

export function parseDiffStats(diff: string): { additions: number; deletions: number } {
  let additions = 0
  let deletions = 0
  for (const line of diff.split('\n')) {
    if (line.startsWith('+') && !line.startsWith('+++')) additions++
    else if (line.startsWith('-') && !line.startsWith('---')) deletions++
  }
  return { additions, deletions }
}

export function extractFilePath(diff: string): string {
  const match = diff.match(/\+\+\+ b\/(.+)/)
  if (match) return match[1]
  const match2 = diff.match(/@@.*@@\s+(.+)/)
  return match2 ? match2[1] : 'unknown file'
}
