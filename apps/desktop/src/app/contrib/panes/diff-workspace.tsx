import { useStore } from '@nanostores/react'
import React from 'react'

import { $selectedStoredSessionId } from '@/store/session'

interface FileChange {
  path: string
  status: 'added' | 'modified' | 'deleted' | 'renamed'
  additions: number
  deletions: number
  hunks: DiffHunk[]
  oldPath?: string
}

interface DiffHunk {
  oldStart: number
  oldLines: number
  newStart: number
  newLines: number
  lines: DiffLine[]
}

interface DiffLine {
  type: 'context' | 'add' | 'remove'
  content: string
  oldLineNum?: number
  newLineNum?: number
}

export function DiffWorkspace() {
  const sessionId = useStore($selectedStoredSessionId)

  if (!sessionId) {
    return (
      <div className="flex h-full flex-col items-center justify-center p-8 text-center text-muted-foreground">
        <p className="text-sm">Select a session to view workspace</p>
      </div>
    )
  }

  // Mock data - in reality this would come from the agent's file operations
  const changes: FileChange[] = [
    {
      path: 'apps/desktop/src/app/contrib/panes/plan-pane.tsx',
      status: 'modified',
      additions: 145,
      deletions: 12,
      hunks: [
        {
          oldStart: 1,
          oldLines: 75,
          newStart: 1,
          newLines: 298,
          lines: [
            { type: 'context', content: "import React from 'react'", oldLineNum: 1, newLineNum: 1 },
            { type: 'context', content: "import { useStore } from '@nanostores/react'", oldLineNum: 2, newLineNum: 2 },
            {
              type: 'context',
              content: "import { $selectedStoredSessionId } from '@/store/session'",
              oldLineNum: 3,
              newLineNum: 3
            },
            {
              type: 'context',
              content: "import { $subagentsBySession } from '@/store/subagents'",
              oldLineNum: 4,
              newLineNum: 4
            },
            { type: 'add', content: '', newLineNum: 5 },
            { type: 'add', content: 'interface PlanStep {', newLineNum: 6 },
            { type: 'add', content: '  id: string', newLineNum: 7 },
            { type: 'add', content: '  title: string', newLineNum: 8 },
            { type: 'add', content: '  description?: string', newLineNum: 9 },
            {
              type: 'add',
              content: "  status: 'pending' | 'running' | 'completed' | 'failed' | 'blocked'",
              newLineNum: 10
            },
            { type: 'add', content: '  agentId?: string', newLineNum: 11 },
            { type: 'add', content: '  verifier?: {', newLineNum: 12 },
            { type: 'add', content: "    kind: 'test' | 'lint' | 'typecheck' | 'build' | 'custom'", newLineNum: 13 },
            { type: 'add', content: '    passed: boolean', newLineNum: 14 },
            { type: 'add', content: '    summary: string', newLineNum: 15 },
            { type: 'add', content: '    details?: string', newLineNum: 16 },
            { type: 'add', content: '  }', newLineNum: 17 },
            { type: 'add', content: '  budget?: {', newLineNum: 18 },
            { type: 'add', content: '    allocated: number', newLineNum: 19 },
            { type: 'add', content: '    used: number', newLineNum: 20 },
            { type: 'add', content: "    unit: 'tokens' | 'seconds' | 'turns'", newLineNum: 21 },
            { type: 'add', content: '  }', newLineNum: 22 },
            { type: 'add', content: '  outputContract?: string', newLineNum: 23 },
            { type: 'add', content: "  consumerDecision?: 'approve' | 'revise' | 'reject'", newLineNum: 24 },
            { type: 'add', content: '}', newLineNum: 25 }
          ]
        }
      ]
    },
    {
      path: 'apps/desktop/src/app/contrib/panes/background-task-panel.tsx',
      status: 'added',
      additions: 412,
      deletions: 0,
      hunks: [
        {
          oldStart: 0,
          oldLines: 0,
          newStart: 1,
          newLines: 50,
          lines: [
            { type: 'add', content: "import React from 'react'", newLineNum: 1 },
            { type: 'add', content: "import { useStore } from '@nanostores/react'", newLineNum: 2 },
            { type: 'add', content: "import { $subagentsBySession } from '@/store/subagents'", newLineNum: 3 },
            { type: 'add', content: "import { $selectedStoredSessionId } from '@/store/session'", newLineNum: 4 },
            { type: 'add', content: "import { $terminalTakeover } from '@/app/right-sidebar/store'", newLineNum: 5 },
            { type: 'add', content: '', newLineNum: 6 },
            { type: 'add', content: "type TaskTab = 'tasks' | 'subagents' | 'terminals'", newLineNum: 7 }
          ]
        }
      ]
    }
  ]

  const totalAdditions = changes.reduce((sum, c) => sum + c.additions, 0)
  const totalDeletions = changes.reduce((sum, c) => sum + c.deletions, 0)

  return (
    <div className="flex h-full flex-col bg-background">
      {/* Workspace Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border bg-card/50">
        <div className="flex items-center gap-3">
          <h2 className="text-sm font-bold uppercase tracking-wider opacity-60">Workspace</h2>
          <span className="text-xs font-mono bg-accent/20 text-accent px-2 py-0.5 rounded-full">
            {changes.length} files
          </span>
        </div>
        <div className="flex items-center gap-4 text-[10px] font-mono">
          <span className="text-green-500">+{totalAdditions}</span>
          <span className="text-red-500">-{totalDeletions}</span>
        </div>
      </div>

      {/* File Tree / Changes List */}
      <div className="flex-1 overflow-hidden">
        <div className="h-full overflow-y-auto p-3 space-y-2">
          {changes.map(change => (
            <FileChangeCard change={change} key={change.path} />
          ))}
        </div>
      </div>

      {/* Empty state */}
      {changes.length === 0 && (
        <div className="flex h-full items-center justify-center p-8 text-center text-muted-foreground">
          <div className="max-w-xs">
            <div className="mb-4 rounded-full bg-muted p-4 mx-auto">
              <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                />
              </svg>
            </div>
            <h3 className="text-sm font-medium text-foreground">No changes yet</h3>
            <p className="text-xs mt-1">File changes from agent operations will appear here</p>
          </div>
        </div>
      )}
    </div>
  )
}

function FileChangeCard({ change }: { change: FileChange }) {
  const [expanded, setExpanded] = React.useState(false)

  const statusConfig = {
    added: { color: 'text-green-500', bg: 'bg-green-500/10', label: 'A', icon: '+' },
    modified: { color: 'text-blue-500', bg: 'bg-blue-500/10', label: 'M', icon: '~' },
    deleted: { color: 'text-red-500', bg: 'bg-red-500/10', label: 'D', icon: '-' },
    renamed: { color: 'text-yellow-500', bg: 'bg-yellow-500/10', label: 'R', icon: '→' }
  }[change.status]

  const config = statusConfig || { color: 'text-gray-400', bg: 'bg-gray-400/10', label: '?', icon: '?' }

  return (
    <div className="border border-border/50 rounded-lg overflow-hidden bg-card">
      {/* File Header */}
      <button
        className="w-full px-3 py-2.5 flex items-center gap-3 hover:bg-accent/5 transition-colors text-left"
        onClick={() => setExpanded(!expanded)}
      >
        <span
          className={`w-5 h-5 rounded flex items-center justify-center text-[10px] font-bold ${config.bg} ${config.color}`}
        >
          {config.icon}
        </span>
        <div className="flex-1 min-w-0 flex items-center gap-2">
          <span className="text-sm font-medium truncate flex-1">{change.path}</span>
          <span className={`text-[10px] font-mono ${config.color} px-1.5 py-0.5 rounded shrink-0`}>{config.label}</span>
        </div>
        <div className="flex items-center gap-2 text-[10px] font-mono">
          <span className="text-green-500">+{change.additions}</span>
          <span className="text-red-500">-{change.deletions}</span>
        </div>
        <svg
          className={`w-4 h-4 text-muted-foreground transition-transform ${expanded ? 'rotate-90' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path d="M9 5l7 7-7 7" strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} />
        </svg>
      </button>

      {/* Diff Content */}
      {expanded && (
        <div className="border-t border-border/50 bg-muted/20 max-h-96 overflow-y-auto">
          {change.hunks.map((hunk, hunkIdx) => (
            <div className="p-3 border-b border-border/30 last:border-0" key={hunkIdx}>
              <div className="flex items-center gap-2 text-[10px] font-mono text-muted-foreground mb-2 px-2">
                <span>
                  @@ -{hunk.oldStart},{hunk.oldLines} +{hunk.newStart},{hunk.newLines} @@
                </span>
              </div>
              <div className="font-mono text-[11px] leading-relaxed">
                {hunk.lines.map((line, lineIdx) => (
                  <div
                    className={`flex gap-2 px-2 py-0.5 ${line.type === 'add' ? 'bg-green-500/10' : line.type === 'remove' ? 'bg-red-500/10' : ''}`}
                    key={lineIdx}
                  >
                    <span className="w-8 text-right text-muted-foreground/50 select-none">
                      {line.oldLineNum ?? '—'}
                    </span>
                    <span className="w-8 text-right text-muted-foreground/50 select-none">
                      {line.newLineNum ?? '—'}
                    </span>
                    <span
                      className={`flex-1 ${line.type === 'add' ? 'text-green-500' : line.type === 'remove' ? 'text-red-500' : 'text-foreground'}`}
                    >
                      {line.type === 'add' ? '+' : line.type === 'remove' ? '-' : ' '}
                      {line.content}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// Inline diff viewer for use in chat messages
export function InlineDiff({
  oldContent,
  newContent,
  filePath,
  maxLines = 50
}: {
  oldContent: string
  newContent: string
  filePath: string
  maxLines?: number
}) {
  // Simple diff algorithm - in production would use a proper diff library
  const oldLines = oldContent.split('\n')
  const newLines = newContent.split('\n')

  // This is a simplified version - real implementation would use Myers diff algorithm
  const maxDisplay = maxLines

  return (
    <div className="rounded-lg border border-border bg-card overflow-hidden">
      <div className="px-3 py-2 border-b border-border flex items-center justify-between">
        <span className="text-xs font-mono text-muted-foreground">{filePath}</span>
        <div className="flex items-center gap-2 text-[10px] font-mono">
          <span className="text-green-500">+{Math.max(0, newLines.length - oldLines.length)}</span>
          <span className="text-red-500">-{Math.max(0, oldLines.length - newLines.length)}</span>
        </div>
      </div>
      <div className="font-mono text-[11px] max-h-[300px] overflow-y-auto">
        {newLines.slice(0, maxDisplay).map((line, idx) => (
          <div className="px-3 py-0.5 flex gap-2 border-b border-border/30 last:border-0" key={idx}>
            <span className="w-8 text-right text-muted-foreground/50 select-none">{idx + 1}</span>
            <span className="flex-1">{line}</span>
          </div>
        ))}
        {newLines.length > maxDisplay && (
          <div className="px-3 py-2 text-center text-xs text-muted-foreground border-t border-border/30">
            ... {newLines.length - maxDisplay} more lines
          </div>
        )}
      </div>
    </div>
  )
}
