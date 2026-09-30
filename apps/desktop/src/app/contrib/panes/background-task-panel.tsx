import { useStore } from '@nanostores/react'
import React from 'react'

import { $terminalTakeover } from '@/app/right-sidebar/store'
import { $selectedStoredSessionId } from '@/store/session'
import { $subagentsBySession } from '@/store/subagents'

type TaskTab = 'tasks' | 'subagents' | 'terminals'

interface BackgroundTaskPanelProps {
  isOpen: boolean
  onClose: () => void
  onToggleTab: (tab: TaskTab) => void
  activeTab: TaskTab
  height: number
  onResize: (height: number) => void
}

const TAB_LABELS: Record<TaskTab, { label: string; icon: React.ReactNode }> = {
  tasks: {
    label: 'Tasks',
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path
          d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 000 4h12a2 2 0 000-4M9 5a2 2 0 012 2h6a2 2 0 012-2"
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
        />
      </svg>
    )
  },
  subagents: {
    label: 'Agents',
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path d="M13 10V3L4 14h7v7l9-11h-7z" strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} />
      </svg>
    )
  },
  terminals: {
    label: 'Terminals',
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path
          d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
        />
      </svg>
    )
  }
}

export function BackgroundTaskPanel({
  isOpen,
  onClose,
  onToggleTab,
  activeTab,
  height,
  onResize
}: BackgroundTaskPanelProps) {
  const sessionId = useStore($selectedStoredSessionId)
  const subagentsMap = useStore($subagentsBySession)
  const subagents = sessionId ? (subagentsMap[sessionId] ?? []) : []
  const terminalOpen = useStore($terminalTakeover)

  if (!isOpen) {
    return null
  }

  const activeSubagents = subagents.filter(a => a.status === 'running' || a.status === 'queued')

  const completedSubagents = subagents.filter(
    a => a.status === 'completed' || a.status === 'failed' || a.status === 'interrupted'
  )

  return (
    <div
      className="fixed bottom-0 left-0 right-0 bg-card border-t border-border shadow-[0_-4px_20px_rgba(0,0,0,0.15)] z-50 flex flex-col"
      style={{ height: `${height}px` }}
    >
      {/* Resize Handle */}
      <div
        className="h-1.5 w-full cursor-row-resize bg-transparent hover:bg-accent/20 transition-colors"
        onMouseDown={e => {
          e.preventDefault()
          const startY = e.clientY
          const startHeight = height

          const onMouseMove = (moveEvent: MouseEvent) => {
            const deltaY = startY - moveEvent.clientY
            const newHeight = Math.max(120, Math.min(600, startHeight + deltaY))
            onResize(newHeight)
          }

          const onMouseUp = () => {
            document.removeEventListener('mousemove', onMouseMove)
            document.removeEventListener('mouseup', onMouseUp)
          }

          document.addEventListener('mousemove', onMouseMove)
          document.addEventListener('mouseup', onMouseUp)
        }}
      >
        <div className="mx-auto mt-1 w-10 h-0.5 bg-border rounded-full" />
      </div>

      {/* Tab Bar */}
      <div className="flex px-3 py-1 border-b border-border">
        {(Object.keys(TAB_LABELS) as TaskTab[]).map(tab => {
          const count = tab === 'subagents' ? subagents.length : tab === 'terminals' ? (terminalOpen ? 1 : 0) : 0

          return (
            <button
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                activeTab === tab
                  ? 'bg-accent text-accent-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground hover:bg-accent/10'
              }`}
              key={tab}
              onClick={() => onToggleTab(tab)}
            >
              {TAB_LABELS[tab].icon}
              <span>{TAB_LABELS[tab].label}</span>
              {count > 0 && (
                <span
                  className={`text-[10px] font-mono px-1.5 py-0.5 rounded-full ${
                    activeTab === tab
                      ? 'bg-accent-foreground/20 text-accent-foreground'
                      : 'bg-muted text-muted-foreground'
                  }`}
                >
                  {count}
                </span>
              )}
            </button>
          )
        })}
        <div className="flex-1" />
        <button
          className="p-1.5 rounded-lg text-muted-foreground hover:text-foreground hover:bg-accent/10 transition-colors"
          onClick={onClose}
          title="Close panel"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path d="M6 18L18 6M6 6l12 12" strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} />
          </svg>
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        {activeTab === 'subagents' && <SubagentsPanel subagents={subagents} />}
        {activeTab === 'terminals' && <TerminalsPanel isOpen={terminalOpen} />}
        {activeTab === 'tasks' && <TasksPanel subagents={subagents} />}
      </div>
    </div>
  )
}

function SubagentsPanel({ subagents }: { subagents: any[] }) {
  if (subagents.length === 0) {
    return (
      <div className="flex h-full items-center justify-center p-8 text-center text-muted-foreground">
        <p className="text-sm">No subagents spawned yet</p>
        <p className="text-xs mt-1">Delegate work to see agents here</p>
      </div>
    )
  }

  return (
    <div className="h-full overflow-y-auto p-3 space-y-2">
      {subagents.map(agent => (
        <SubagentCard agent={agent} key={agent.id} />
      ))}
    </div>
  )
}

function SubagentCard({ agent }: { agent: any }) {
  const isRunning = agent.status === 'running' || agent.status === 'queued'
  const isFailed = agent.status === 'failed' || agent.status === 'interrupted'
  const isDone = agent.status === 'completed'

  type AgentStatus = 'running' | 'queued' | 'completed' | 'failed' | 'interrupted'

  const statusConfig: Record<AgentStatus, { color: string; label: string; pulse: boolean }> = {
    running: { color: 'bg-blue-500', label: 'Running', pulse: true },
    queued: { color: 'bg-yellow-500', label: 'Queued', pulse: true },
    completed: { color: 'bg-green-500', label: 'Done', pulse: false },
    failed: { color: 'bg-red-500', label: 'Failed', pulse: false },
    interrupted: { color: 'bg-orange-500', label: 'Stopped', pulse: false }
  }

  const config = statusConfig[agent.status as AgentStatus] || {
    color: 'bg-gray-500',
    label: agent.status,
    pulse: false
  }

  return (
    <div
      className={`p-3 rounded-lg border transition-all ${
        isRunning ? 'bg-accent/5 border-accent/30' : 'bg-card border-border'
      } ${isFailed ? 'border-red-500/30' : ''}`}
    >
      <div className="flex items-start gap-3">
        <div
          className={`mt-1 w-2.5 h-2.5 rounded-full shrink-0 ${config.color} ${config.pulse ? 'animate-pulse' : ''}`}
        />
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between mb-1">
            <span className="text-sm font-medium truncate pr-2">{agent.goal}</span>
            <span
              className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${config.color.replace('bg-', 'bg-')} text-white/90`}
            >
              {config.label}
            </span>
          </div>

          <div className="flex items-center gap-3 text-[11px] text-muted-foreground">
            {agent.currentTool && (
              <span className="font-mono text-accent flex items-center gap-1">
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                  />
                  <path
                    d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                  />
                </svg>
                {agent.currentTool}
              </span>
            )}
            {agent.durationSeconds && <span className="font-mono">{formatDuration(agent.durationSeconds)}</span>}
            {agent.costUsd && <span className="font-mono">${agent.costUsd.toFixed(4)}</span>}
          </div>

          {agent.summary && <p className="text-xs text-muted-foreground mt-2 line-clamp-2">{agent.summary}</p>}

          {/* Stream/Progress */}
          {agent.stream.length > 0 && (
            <div className="mt-2 pt-2 border-t border-border/50 space-y-1 max-h-24 overflow-y-auto">
              {agent.stream.slice(-5).map((entry: any, idx: number) => (
                <div className="flex items-start gap-1.5 text-[10px] font-mono" key={idx}>
                  <span
                    className={`shrink-0 ${entry.isError ? 'text-red-500' : entry.kind === 'thinking' ? 'text-yellow-500' : 'text-muted-foreground'}`}
                  >
                    {entry.kind === 'tool' ? '▸' : entry.kind === 'thinking' ? '◆' : '▹'}
                  </span>
                  <span className={`truncate ${entry.isError ? 'text-red-500/80' : 'text-muted-foreground'}`}>
                    {entry.text}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function TerminalsPanel({ isOpen }: { isOpen: boolean }) {
  if (!isOpen) {
    return (
      <div className="flex h-full items-center justify-center p-8 text-center text-muted-foreground">
        <p className="text-sm">No active terminal</p>
        <p className="text-xs mt-1">
          Press <kbd className="px-1.5 py-0.5 bg-muted rounded text-[10px] font-mono">⌃`</kbd> to open terminal
        </p>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col">
      <div className="p-3 border-b border-border">
        <div className="flex items-center gap-2 text-sm">
          <span className="w-2 h-2 rounded-full bg-green-500" />
          <span className="font-medium">Terminal</span>
          <span className="text-[10px] font-mono text-green-500">Active</span>
        </div>
      </div>
      <div className="flex-1 p-3 text-center text-muted-foreground">
        <p className="text-sm">Terminal running in background</p>
        <p className="text-xs mt-1">Click terminal tab to interact</p>
      </div>
    </div>
  )
}

function TasksPanel({ subagents }: { subagents: any[] }) {
  // Extract tasks from subagent goals and create a unified task view
  const tasks = subagents.map((agent, idx) => ({
    id: agent.id,
    title: agent.goal,
    status: agent.status,
    agentIndex: idx,
    subagent: agent
  }))

  if (tasks.length === 0) {
    return (
      <div className="flex h-full items-center justify-center p-8 text-center text-muted-foreground">
        <p className="text-sm">No tasks in current session</p>
        <p className="text-xs mt-1">Tasks appear when agent creates a plan</p>
      </div>
    )
  }

  return (
    <div className="h-full overflow-y-auto p-3 space-y-2">
      {tasks.map(task => (
        <TaskCard key={task.id} task={task} />
      ))}
    </div>
  )
}

function TaskCard({ task }: { task: any }) {
  const isRunning = task.status === 'running' || task.status === 'queued'
  const isFailed = task.status === 'failed' || task.status === 'interrupted'
  const isDone = task.status === 'completed'

  return (
    <div
      className={`p-3 rounded-lg border transition-all ${
        isRunning ? 'bg-accent/5 border-accent/30' : 'bg-card border-border'
      } ${isFailed ? 'border-red-500/30' : ''}`}
    >
      <div className="flex items-start gap-3">
        <div
          className={`mt-1 w-2.5 h-2.5 rounded-full shrink-0 ${
            isRunning ? 'bg-blue-500 animate-pulse' : isFailed ? 'bg-red-500' : isDone ? 'bg-green-500' : 'bg-gray-400'
          }`}
        />
        <div className="flex-1 min-w-0">
          <span className="text-sm font-medium truncate block">{task.title}</span>
          <div className="flex items-center gap-2 mt-1 text-[10px] text-muted-foreground">
            <span className="font-mono uppercase opacity-50">{task.status}</span>
            {task.subagent.currentTool && (
              <span className="font-mono text-accent">Tool: {task.subagent.currentTool}</span>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function formatDuration(seconds: number): string {
  if (seconds < 60) {
    return `${seconds}s`
  }

  const mins = Math.floor(seconds / 60)
  const secs = seconds % 60

  if (mins < 60) {
    return `${mins}m ${secs}s`
  }

  const hours = Math.floor(mins / 60)

  return `${hours}h ${mins % 60}m`
}
