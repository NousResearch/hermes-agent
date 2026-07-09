import type { ReactNode } from 'react'

import { cn } from '@/lib/utils'

import type { ClientStatus, EngagementStatus, InvoiceStatus } from './types'

// This view is org-specific (not upstream), so its strings stay plain English
// instead of threading a new section through all four locale catalogs.

// Status dot colors shared by list rows and detail pills.
export const CLIENT_DOT: Record<ClientStatus, string> = {
  active: 'bg-emerald-500',
  churned: 'bg-neutral-400',
  lead: 'bg-sky-500',
  paused: 'bg-amber-500'
}

export const ENGAGEMENT_DOT: Record<EngagementStatus, string> = {
  active: 'bg-emerald-500',
  done: 'bg-neutral-400',
  'on-hold': 'bg-amber-500',
  proposal: 'bg-sky-500'
}

export const INVOICE_DOT: Record<InvoiceStatus, string> = {
  draft: 'bg-neutral-400',
  overdue: 'bg-red-500',
  paid: 'bg-emerald-500',
  sent: 'bg-sky-500'
}

/** Stacked label + control, the detail-form building block. */
export function Field({ children, label }: { children: ReactNode; label: string }) {
  return (
    <label className="block space-y-1">
      <span className="block text-[0.65rem] font-medium uppercase tracking-wider text-muted-foreground/60">
        {label}
      </span>
      {children}
    </label>
  )
}

/** Two fields side by side on wide detail columns. */
export function FieldRow({ children }: { children: ReactNode }) {
  return <div className="grid gap-3 sm:grid-cols-2">{children}</div>
}

/** Billing header stat (Outstanding / Overdue / Paid). */
export function StatCard({
  label,
  tone = 'default',
  value
}: {
  label: string
  tone?: 'bad' | 'default' | 'good'
  value: string
}) {
  return (
    <div className="min-w-0 rounded-md bg-(--ui-bg-quinary) px-3 py-2">
      <div className="truncate text-[0.62rem] font-medium uppercase tracking-wider text-muted-foreground/60">
        {label}
      </div>
      <div
        className={cn(
          'truncate text-base font-semibold tabular-nums',
          tone === 'good' && 'text-emerald-600 dark:text-emerald-400',
          tone === 'bad' && 'text-red-600 dark:text-red-400',
          tone === 'default' && 'text-foreground'
        )}
      >
        {value}
      </div>
    </div>
  )
}
