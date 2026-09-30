import { useStore } from '@nanostores/react'
import React from 'react'

import { $selectedStoredSessionId } from '@/store/session'
import { $subagentsBySession } from '@/store/subagents'

interface Worktree {
  id: string
  name: string
  path: string
  branch: string
  baseBranch: string
  status: 'active' | 'idle' | 'syncing' | 'error'
  agentId?: string
  lastSync: number
  changes: number
  description: string
}

export function WorktreePanel() {
  const sessionId = useStore($selectedStoredSessionId)
  const subagentsMap = useStore($subagentsBySession)
  const subagents = sessionId ? (subagentsMap[sessionId] ?? []) : []

  // Generate worktrees from active subagents
  const worktrees: Worktree[] = subagents
    .filter(a => a.status === 'running' || a.status === 'queued' || a.status === 'completed')
    .map((agent, index) => ({
      id: agent.id,
      name: `agent-${index + 1}-${agent.goal.split(' ').slice(0, 2).join('-').toLowerCase()}`,
      path: `.hermes/worktrees/${agent.id}`,
      branch: `hermes/${agent.id.slice(0, 8)}`,
      baseBranch: 'main',
      status: agent.status === 'running' ? 'active' : agent.status === 'completed' ? 'idle' : 'syncing',
      agentId: agent.id,
      lastSync: agent.updatedAt,
      changes: agent.filesWritten.length + agent.filesRead.length,
      description: agent.goal
    }))

  // Always show main worktree
  const allWorktrees = [
    {
      id: 'main',
      name: 'main',
      path: '.',
      branch: 'main',
      baseBranch: 'main',
      status: 'active' as const,
      lastSync: Date.now(),
      changes: 0,
      description: 'Primary workspace'
    },
    ...worktrees
  ]

  return (
    <div className="flex h-full flex-col bg-background">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border bg-card/50">
        <div className="flex items-center gap-3">
          <h2 className="text-sm font-bold uppercase tracking-wider opacity-60">Worktrees</h2>
          <span className="text-xs font-mono bg-accent/20 text-accent px-2 py-0.5 rounded-full">
            {allWorktrees.length} worktrees
          </span>
        </div>
        <button
          className="p-1.5 rounded-lg text-muted-foreground hover:text-foreground hover:bg-accent/10 transition-colors"
          title="Create worktree"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path d="M12 4v16m8-8H4" strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} />
          </svg>
        </button>
      </div>

      {/* Worktree List */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {allWorktrees.map(wt => (
          <WorktreeCard isMain={wt.id === 'main'} key={wt.id} worktree={wt} />
        ))}
      </div>

      {allWorktrees.length === 1 && (
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
            <h3 className="text-sm font-medium text-foreground">No parallel worktrees</h3>
            <p className="text-xs mt-1">Worktrees appear when agents are delegated tasks</p>
            <div className="mt-4 p-3 rounded-lg bg-muted/50 border border-border text-left text-[11px]">
              <p className="font-mono text-accent mb-1">Codex-style isolation:</p>
              <ul className="space-y-1 list-disc list-inside">
                <li>Each agent gets isolated git worktree</li>
                <li>No conflicts between parallel agents</li>
                <li>Review changes before merge</li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function WorktreeCard({ worktree, isMain }: { worktree: Worktree; isMain: boolean }) {
  const [expanded, setExpanded] = React.useState(false)

  const statusConfig = {
    active: { color: 'bg-blue-500', label: 'Active', pulse: true },
    idle: { color: 'bg-green-500', label: 'Ready', pulse: false },
    syncing: { color: 'bg-yellow-500', label: 'Syncing', pulse: true },
    error: { color: 'bg-red-500', label: 'Error', pulse: false }
  }[worktree.status]

  const config = statusConfig || { color: 'bg-gray-400', label: worktree.status, pulse: false }

  const formatTime = (timestamp: number) => {
    const diff = Date.now() - timestamp

    if (diff < 60000) {
      return 'just now'
    }

    if (diff < 3600000) {
      return `${Math.floor(diff / 60000)}m ago`
    }

    if (diff < 86400000) {
      return `${Math.floor(diff / 3600000)}h ago`
    }

    return `${Math.floor(diff / 86400000)}d ago`
  }

  return (
    <div
      className={`border border-border/50 rounded-lg overflow-hidden bg-card ${isMain ? 'ring-1 ring-accent/20' : ''}`}
    >
      {/* Worktree Header */}
      <div
        className="w-full px-3 py-2.5 flex items-center gap-3 hover:bg-accent/5 transition-colors text-left cursor-pointer"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-2">
          <div
            className={`w-8 h-8 rounded-lg flex items-center justify-center ${isMain ? 'bg-accent/20' : config.color + '/20'}`}
          >
            {isMain ? (
              <svg className="w-4 h-4 text-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                />
              </svg>
            ) : (
              <svg
                className="w-4 h-4"
                fill="none"
                stroke="currentColor"
                style={{ color: config.color.replace('bg-', '') }}
                viewBox="0 0 24 24"
              >
                <path d="M13 10V3L4 14h7v7l9-11h-7z" strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} />
              </svg>
            )}
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className={`text-sm font-medium truncate ${isMain ? 'text-accent' : ''}`}>{worktree.name}</span>
              {isMain && (
                <span className="text-[9px] font-mono bg-accent/20 text-accent px-1 py-0.5 rounded">primary</span>
              )}
            </div>
            <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
              <span className="font-mono truncate max-w-[150px]">{worktree.path}</span>
              <span className="text-muted-foreground/50">|</span>
              <span className="font-mono">{worktree.branch}</span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className={`w-1.5 h-1.5 rounded-full ${config.color} ${config.pulse ? 'animate-pulse' : ''}`} />
            <span className={`text-[10px] font-medium ${config.color.replace('bg-', 'text-')}`}>{config.label}</span>
            <svg
              className={`w-4 h-4 text-muted-foreground transition-transform ${expanded ? 'rotate-90' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path d="M9 5l7 7-7 7" strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} />
            </svg>
          </div>
        </div>
      </div>

      {/* Expanded Details - sibling, not child of header */}
      {expanded ? (
        <div className="border-t border-border/50 bg-muted/20 p-3 space-y-3">
          <div className="grid grid-cols-2 gap-3 text-[11px]">
            <div>
              <span className="text-muted-foreground block">Base Branch</span>
              <span className="font-mono">{worktree.baseBranch}</span>
            </div>
            <div>
              <span className="text-muted-foreground block">Last Sync</span>
              <span className="font-mono">{formatTime(worktree.lastSync)}</span>
            </div>
            <div>
              <span className="text-muted-foreground block">Changes</span>
              <span className="font-mono">{worktree.changes} files</span>
            </div>
            <div>
              <span className="text-muted-foreground block">Agent</span>
              <span className="font-mono truncate max-w-[100px]">{worktree.agentId?.slice(0, 12) || '—'}</span>
            </div>
          </div>

          <div className="pt-2 border-t border-border/30">
            <p className="text-xs text-muted-foreground mb-2">{worktree.description}</p>
            <div className="flex items-center gap-2">
              <button className="flex-1 py-1.5 px-2 text-xs font-medium rounded-lg bg-muted border border-border hover:bg-accent/10 transition-colors">
                Open in Terminal
              </button>
              <button className="flex-1 py-1.5 px-2 text-xs font-medium rounded-lg bg-accent/10 border border-accent/30 text-accent hover:bg-accent/20 transition-colors">
                Review Changes
              </button>
              {!isMain && (
                <button className="flex-1 py-1.5 px-2 text-xs font-medium rounded-lg bg-red-500/10 border border-red-500/30 text-red-500 hover:bg-red-500/20 transition-colors">
                  Remove
                </button>
              )}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  )
}

// Worktree status indicator for status bar
export function WorktreeStatusIndicator() {
  const sessionId = useStore($selectedStoredSessionId)
  const subagentsMap = useStore($subagentsBySession)
  const subagents = sessionId ? (subagentsMap[sessionId] ?? []) : []

  const activeWorktrees = subagents.filter(a => a.status === 'running' || a.status === 'queued').length

  if (activeWorktrees === 0) {
    return null
  }

  return (
    <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-lg bg-blue-500/10 border border-blue-500/20">
      <span className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse" />
      <span className="text-[10px] font-mono text-blue-500">
        {activeWorktrees} worktree{activeWorktrees > 1 ? 's' : ''} active
      </span>
    </div>
  )
}
