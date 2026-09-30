import { useStore } from '@nanostores/react'
import React from 'react'

import { $selectedStoredSessionId } from '@/store/session'
import { $subagentsBySession } from '@/store/subagents'

interface PlanStep {
  id: string
  title: string
  description?: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'blocked'
  agentId?: string
  verifier?: {
    kind: 'test' | 'lint' | 'typecheck' | 'build' | 'custom'
    passed: boolean
    summary: string
    details?: string
  }
  budget?: {
    allocated: number
    used: number
    unit: 'tokens' | 'seconds' | 'turns'
  }
  outputContract?: string
  consumerDecision?: 'approve' | 'revise' | 'reject'
}

function parsePlanFromSubagents(subagents: any[]): PlanStep[] {
  return subagents.map((agent, index) => {
    const isRunning = agent.status === 'running' || agent.status === 'queued'
    const isFailed = agent.status === 'failed' || agent.status === 'interrupted'
    const isDone = agent.status === 'completed'

    // Extract structured info from summary/stream if available
    let verifier: PlanStep['verifier'] = undefined
    let budget: PlanStep['budget'] = undefined
    let outputContract = agent.goal

    if (agent.summary) {
      // Try to parse structured summary
      const lines = agent.summary.split('\n')

      for (const line of lines) {
        if (line.includes('verify:') || line.includes('test:') || line.includes('lint:')) {
          verifier = {
            kind: line.includes('test') ? 'test' : line.includes('lint') ? 'lint' : 'custom',
            passed: !line.toLowerCase().includes('fail'),
            summary: line.replace(/^(verify|test|lint):\s*/i, '')
          }
        }

        if (line.includes('budget:')) {
          const match = line.match(/budget:\s*(\d+)\s*(tokens?|seconds?|turns?)/i)

          if (match) {
            budget = {
              allocated: parseInt(match[1]),
              used: agent.inputTokens ? agent.inputTokens + (agent.outputTokens || 0) : 0,
              unit: match[2].includes('token') ? 'tokens' : match[2].includes('second') ? 'seconds' : 'turns'
            }
          }
        }
      }
    }

    // Default budget from tokens if available
    if (!budget && agent.inputTokens) {
      budget = {
        allocated: 50000, // default
        used: agent.inputTokens + (agent.outputTokens || 0),
        unit: 'tokens'
      }
    }

    return {
      id: agent.id,
      title: agent.goal.split('\n')[0],
      description:
        agent.goal.length > agent.goal.split('\n')[0].length
          ? agent.goal.substring(agent.goal.split('\n')[0].length).trim()
          : undefined,
      status: isRunning ? 'running' : isFailed ? 'failed' : isDone ? 'completed' : 'pending',
      agentId: agent.id,
      verifier,
      budget,
      outputContract,
      consumerDecision: isDone ? 'approve' : undefined
    }
  })
}

export function PlanPane() {
  const sessionId = useStore($selectedStoredSessionId)
  const subagentsMap = useStore($subagentsBySession)
  const subagents = sessionId ? (subagentsMap[sessionId] ?? []) : []

  const planSteps = parsePlanFromSubagents(subagents)
  const activeCount = planSteps.filter(s => s.status === 'running').length
  const completedCount = planSteps.filter(s => s.status === 'completed').length
  const failedCount = planSteps.filter(s => s.status === 'failed').length
  const totalBudget = planSteps.reduce((sum, s) => sum + (s.budget?.allocated || 0), 0)
  const usedBudget = planSteps.reduce((sum, s) => sum + (s.budget?.used || 0), 0)

  if (planSteps.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center p-8 text-center text-muted-foreground">
        <div className="mb-4 rounded-full bg-muted p-4">
          <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 012 2h6a2 2 0 012-2"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
            />
          </svg>
        </div>
        <h3 className="text-sm font-medium text-foreground">No active plan</h3>
        <p className="text-xs">Subagents will appear here when the agent delegates tasks.</p>
        <div className="mt-6 p-4 rounded-lg bg-muted/50 border border-border max-w-xs">
          <p className="text-xs font-mono text-accent mb-2">Parallel-first delegation</p>
          <p className="text-[11px]">The agent will fan out read-only subagents first, then consolidate writes.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-full flex-col p-4 bg-background text-(--ui-text-primary) overflow-y-auto">
      {/* Header with stats */}
      <div className="mb-6 space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-bold uppercase tracking-wider opacity-60">Execution Plan</h2>
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono bg-accent/20 text-accent px-2 py-0.5 rounded-full">
              {planSteps.length} steps
            </span>
            {activeCount > 0 && (
              <span className="text-xs font-mono bg-blue-500/20 text-blue-500 px-2 py-0.5 rounded-full animate-pulse">
                {activeCount} running
              </span>
            )}
          </div>
        </div>

        {/* Progress bar */}
        <div className="h-1.5 bg-muted rounded-full overflow-hidden">
          <div
            className="h-full bg-accent transition-all duration-300"
            style={{ width: `${planSteps.length > 0 ? (completedCount / planSteps.length) * 100 : 0}%` }}
          />
        </div>

        {/* Budget bar */}
        {totalBudget > 0 && (
          <div className="flex items-center justify-between text-[10px] font-mono text-muted-foreground">
            <span>
              Budget: {usedBudget.toLocaleString()} / {totalBudget.toLocaleString()} tokens
            </span>
            <div className="flex-1 mx-2 h-1 bg-muted rounded-full overflow-hidden" style={{ maxWidth: '100px' }}>
              <div
                className="h-full transition-all duration-300"
                style={{
                  width: `${Math.min(100, (usedBudget / totalBudget) * 100)}%`,
                  background:
                    usedBudget > totalBudget ? 'red' : usedBudget > totalBudget * 0.8 ? 'orange' : 'var(--accent)'
                }}
              />
            </div>
          </div>
        )}

        {/* Status badges */}
        <div className="flex flex-wrap gap-1.5">
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-green-500/20 text-green-500">
            ✓ {completedCount} done
          </span>
          {activeCount > 0 && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-500 animate-pulse">
              ⟳ {activeCount} running
            </span>
          )}
          {failedCount > 0 && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-red-500/20 text-red-500">✗ {failedCount} failed</span>
          )}
        </div>
      </div>

      {/* Plan Steps */}
      <div className="space-y-3">
        {planSteps.map((step, index) => (
          <PlanStepCard index={index} isActive={step.status === 'running'} key={step.id} step={step} />
        ))}
      </div>

      {/* Delegation Policy Reference */}
      <div className="mt-6 pt-4 border-t border-border/50">
        <details className="group">
          <summary className="flex items-center gap-2 text-xs font-medium text-muted-foreground cursor-pointer">
            <svg
              className="w-3 h-3 transition-transform group-open:rotate-90"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path d="M9 5l7 7-7 7" strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} />
            </svg>
            Delegation Policy (Codex-style)
          </summary>
          <div className="mt-3 p-3 rounded-lg bg-muted/30 border border-border/50 text-[11px] text-muted-foreground space-y-1">
            <p>
              <strong>Pre-spawn gates:</strong> Independent · Consumer decision · Bounded · Worth it
            </p>
            <p>
              <strong>Concurrency caps:</strong> 0 trivial · 1 evidence · 2–3 multi-axis · 4–6 broad audit
            </p>
            <p>
              <strong>Lane roles:</strong> read-only · validation/test · implementation · review
            </p>
            <p>
              <strong>Output contract:</strong> Required before spawn — defines verifier & consumer decision
            </p>
          </div>
        </details>
      </div>
    </div>
  )
}

function PlanStepCard({ step, index, isActive }: { step: PlanStep; index: number; isActive: boolean }) {
  const statusConfig = {
    running: { color: 'bg-blue-500', label: 'Running', border: 'border-blue-500/30', bg: 'bg-blue-500/5' },
    pending: { color: 'bg-gray-400', label: 'Pending', border: 'border-border', bg: 'bg-card' },
    completed: { color: 'bg-green-500', label: 'Completed', border: 'border-green-500/30', bg: 'bg-green-500/5' },
    failed: { color: 'bg-red-500', label: 'Failed', border: 'border-red-500/30', bg: 'bg-red-500/5' },
    blocked: { color: 'bg-orange-500', label: 'Blocked', border: 'border-orange-500/30', bg: 'bg-orange-500/5' }
  }[step.status]

  const config = statusConfig || { color: 'bg-gray-400', label: step.status, border: 'border-border', bg: 'bg-card' }

  return (
    <div
      className={`p-3 rounded-lg border transition-all ${config.bg} ${config.border} ${isActive ? 'shadow-sm' : ''}`}
    >
      <div className="flex items-start gap-3">
        {/* Step indicator */}
        <div className="flex flex-col items-center gap-1.5 min-w-[2rem]">
          <div className={`w-2.5 h-2.5 rounded-full shrink-0 ${config.color} ${isActive ? 'animate-pulse' : ''}`} />
          {index < 9 && <span className="text-[10px] font-mono text-muted-foreground">{index + 1}</span>}
        </div>

        {/* Main content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2 mb-1">
            <div className="flex items-center gap-2 flex-1 min-w-0">
              <span className="text-sm font-medium truncate">{step.title}</span>
              <span
                className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${config.color.replace('bg-', 'bg-')} text-white/90 shrink-0`}
              >
                {config.label}
              </span>
            </div>

            {/* Consumer decision for completed steps */}
            {step.consumerDecision && (
              <span
                className={`text-[10px] font-mono px-1.5 py-0.5 rounded shrink-0 ${
                  step.consumerDecision === 'approve'
                    ? 'bg-green-500/20 text-green-500'
                    : step.consumerDecision === 'revise'
                      ? 'bg-yellow-500/20 text-yellow-500'
                      : 'bg-red-500/20 text-red-500'
                }`}
              >
                {step.consumerDecision}
              </span>
            )}
          </div>

          {step.description && <p className="text-xs text-muted-foreground mb-2 line-clamp-1">{step.description}</p>}

          {/* Output Contract */}
          {step.outputContract && (
            <div className="mb-2 p-2 rounded bg-muted/50 border border-border/50">
              <div className="flex items-center gap-1.5 text-[10px] font-mono text-accent mb-1">
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                  />
                </svg>
                <span>Output Contract</span>
              </div>
              <p className="text-xs text-muted-foreground font-mono pl-5">{step.outputContract}</p>
            </div>
          )}

          {/* Verifier Result */}
          {step.verifier && (
            <div className="mb-2 p-2 rounded border text-[10px] font-mono">
              <div className="flex items-center gap-2 mb-1">
                <span
                  className={`px-1.5 py-0.5 rounded ${
                    step.verifier.passed ? 'bg-green-500/20 text-green-500' : 'bg-red-500/20 text-red-500'
                  }`}
                >
                  {step.verifier.passed ? '✓ PASS' : '✗ FAIL'}
                </span>
                <span className={`text-muted-foreground capitalize`}>{step.verifier.kind}</span>
              </div>
              <p className={`text-xs ${step.verifier.passed ? 'text-green-500/80' : 'text-red-500/80'}`}>
                {step.verifier.summary}
              </p>
              {step.verifier.details && (
                <p className="text-xs text-muted-foreground/70 mt-1">{step.verifier.details}</p>
              )}
            </div>
          )}

          {/* Budget */}
          {step.budget && (
            <div className="flex items-center gap-3 text-[10px] font-mono text-muted-foreground mb-2">
              <span className="flex items-center gap-1">
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                  />
                </svg>
                {step.budget.unit === 'tokens'
                  ? `${step.budget.used.toLocaleString()} / ${step.budget.allocated.toLocaleString()} tok`
                  : `${step.budget.used} / ${step.budget.allocated} ${step.budget.unit}`}
              </span>
              <div className="flex-1 h-1 bg-muted rounded-full overflow-hidden" style={{ maxWidth: '120px' }}>
                <div
                  className="h-full transition-all duration-300"
                  style={{
                    width: `${Math.min(100, (step.budget.used / step.budget.allocated) * 100)}%`,
                    background:
                      step.budget.used > step.budget.allocated
                        ? 'red'
                        : step.budget.used > step.budget.allocated * 0.8
                          ? 'orange'
                          : 'var(--accent)'
                  }}
                />
              </div>
            </div>
          )}

          {/* Stream preview for active */}
          {isActive && step.agentId && (
            <details className="group mt-2">
              <summary className="flex items-center gap-1.5 text-[10px] font-mono text-muted-foreground cursor-pointer">
                <svg
                  className="w-3 h-3 transition-transform group-open:rotate-90"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path d="M9 5l7 7-7 7" strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} />
                </svg>
                Live progress
              </summary>
              <div className="mt-2 pt-2 border-t border-border/50 space-y-1 max-h-32 overflow-y-auto text-[10px] font-mono">
                <p className="text-muted-foreground">Connecting to agent stream...</p>
              </div>
            </details>
          )}
        </div>
      </div>
    </div>
  )
}
