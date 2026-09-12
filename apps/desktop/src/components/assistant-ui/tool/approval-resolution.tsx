'use client'

import { type FC } from 'react'

import { cn } from '@/lib/utils'

export type ApprovalResolutionStatus = 'approved' | 'rejected' | 'unavailable'
export type ApprovalExecutionStatus = 'succeeded' | 'failed' | 'not_attempted'

export interface ApprovalResolution {
  status: ApprovalResolutionStatus
  choice?: 'once' | 'session' | 'always' | null
  actor?: 'authenticated_user' | 'system'
  resolved_at?: number
  request_id?: string
}

export interface ApprovalExecution {
  status: ApprovalExecutionStatus
}

export interface ApprovalResolutionRecord {
  approval: ApprovalResolution
  execution?: ApprovalExecution
}

function record(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : null
}

function parse(value: unknown): Record<string, unknown> | null {
  const direct = record(value)

  if (direct) {
    return direct
  }

  if (typeof value !== 'string' || !value.trim()) {
    return null
  }

  try {
    return record(JSON.parse(value))
  } catch {
    return null
  }
}

export function approvalResolutionFromResult(value: unknown): ApprovalResolutionRecord | null {
  const result = parse(value)
  const approval = record(result?.approval)
  const status = approval?.status

  if (!approval || (status !== 'approved' && status !== 'rejected' && status !== 'unavailable')) {
    return null
  }

  const execution = record(result?.execution)
  const executionStatus = execution?.status

  return {
    approval: {
      status,
      choice:
        approval.choice === 'once' || approval.choice === 'session' || approval.choice === 'always' || approval.choice === null
          ? approval.choice
          : undefined,
      actor: approval.actor === 'authenticated_user' || approval.actor === 'system' ? approval.actor : undefined,
      resolved_at: typeof approval.resolved_at === 'number' ? approval.resolved_at : undefined,
      request_id: typeof approval.request_id === 'string' ? approval.request_id : undefined
    },
    ...(executionStatus === 'succeeded' || executionStatus === 'failed' || executionStatus === 'not_attempted'
      ? { execution: { status: executionStatus } }
      : {})
  }
}

function resolutionLabel(resolution: ApprovalResolution): string {
  if (resolution.status === 'approved') {
    if (resolution.choice === 'session') {
      return 'Approved for this session'
    }

    return resolution.choice === 'always' ? 'Approved always' : 'Approved once'
  }

  if (resolution.status === 'rejected') {
    return 'Rejected'
  }

  return 'Approval unavailable'
}

function executionLabel(execution?: ApprovalExecution): string | null {
  if (!execution || execution.status === 'not_attempted') {
    return execution ? 'Not executed' : null
  }

  return execution.status === 'succeeded' ? 'Execution succeeded' : 'Execution failed'
}

export const ApprovalResolutionCard: FC<{ resolution: ApprovalResolutionRecord }> = ({ resolution }) => {
  const execution = executionLabel(resolution.execution)
  const isApproved = resolution.approval.status === 'approved'

  return (
    <div
      className={cn(
        'mt-1 flex items-center gap-2 rounded-md border px-2 py-1 text-xs',
        isApproved
          ? 'border-emerald-600/25 bg-emerald-600/5 text-emerald-700 dark:border-emerald-400/25 dark:bg-emerald-400/5 dark:text-emerald-300'
          : resolution.approval.status === 'rejected'
            ? 'border-amber-600/25 bg-amber-600/5 text-amber-700 dark:border-amber-400/25 dark:bg-amber-400/5 dark:text-amber-300'
            : 'border-destructive/25 bg-destructive/5 text-destructive'
      )}
      data-approval-resolution={resolution.approval.status}
    >
      <span className="font-medium">{resolutionLabel(resolution.approval)}</span>
      {resolution.approval.actor === 'authenticated_user' && <span className="opacity-75">by you</span>}
      {execution && <span className="opacity-75">· {execution}</span>}
    </div>
  )
}
