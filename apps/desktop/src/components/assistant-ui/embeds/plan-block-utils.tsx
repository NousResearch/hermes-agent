import type { ReactNode } from 'react'

import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'

import type { RichFenceProps } from './types'

export type JsonRecord = Record<string, unknown>

export function isRecord(value: unknown): value is JsonRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export function asText(value: unknown, fallback = ''): string {
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return String(value)
  }

  return fallback
}

export function asRecords(value: unknown): JsonRecord[] {
  return Array.isArray(value) ? value.filter(isRecord) : []
}

export function asStrings(value: unknown): string[] {
  return Array.isArray(value) ? value.map(item => asText(item)).filter(Boolean) : []
}

export function parseJsonRecord(code: string): JsonRecord | null {
  try {
    const parsed = JSON.parse(code)

    return isRecord(parsed) ? parsed : null
  } catch {
    return null
  }
}

export function InvalidPlanBlock({ code, kind }: RichFenceProps & { kind: string }) {
  return (
    <div className="my-2 rounded-lg border border-dashed border-(--ui-stroke-secondary) bg-muted/20 p-3">
      <div className="mb-2 text-xs font-semibold text-muted-foreground">Invalid {kind} block</div>
      <pre className="max-h-48 overflow-auto rounded-md bg-background/70 p-2 font-mono text-[0.7rem] leading-relaxed whitespace-pre-wrap text-muted-foreground">
        {code}
      </pre>
    </div>
  )
}

export function PlanBlockShell({ children, title }: { children: ReactNode; title: string }) {
  return (
    <div className="my-2 overflow-hidden rounded-xl border border-(--ui-stroke-secondary) bg-card text-card-foreground shadow-xs">
      <div className="border-b border-(--ui-stroke-tertiary) bg-muted/35 px-3 py-2 text-xs font-semibold tracking-wide text-muted-foreground uppercase">
        {title}
      </div>
      <div className="p-3">{children}</div>
    </div>
  )
}

export function MetaBadge({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <Badge className={cn('font-mono text-[0.62rem]', className)} variant="outline">
      {children}
    </Badge>
  )
}
