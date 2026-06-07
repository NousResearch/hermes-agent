import { useStore } from '@nanostores/react'

import { Codicon } from '@/components/ui/codicon'
import {
  Activity,
  Clock,
  FileText,
  Monitor,
  Package,
  Terminal as TerminalIcon,
  Wrench
} from '@/lib/icons'
import { $sessionChanges, $sessionChangeSummary, $toolCallSummary } from '@/store/session-changes'

const TOOL_ICONS: Record<string, typeof Monitor> = {
  edit_file: Wrench,
  patch: Wrench,
  terminal: TerminalIcon,
  write_file: FileText
}

export function SessionOverview() {
  const changes = useStore($sessionChanges)
  const summary = useStore($sessionChangeSummary)
  const toolSummary = useStore($toolCallSummary)

  return (
    <div className="flex flex-col gap-4 p-3 text-xs">
      {/* Files Changed */}
      <Section title={`Files Changed (${summary.files})`} icon={FileText}>
        {changes.length === 0 ? (
          <EmptyState message="No files changed yet" />
        ) : (
          <div className="flex flex-col gap-1">
            {changes.map((c, i) => (
              <div key={i} className="flex items-center gap-2 truncate py-0.5">
                <FileText size={12} className="shrink-0 text-muted-foreground" />
                <span className="truncate text-foreground">{shortPath(c.path)}</span>
                <span className="ml-auto shrink-0 text-emerald-500">+{c.additions}</span>
                <span className="shrink-0 text-rose-500">-{countDeletions(c.diff)}</span>
              </div>
            ))}
          </div>
        )}
      </Section>

      {/* Tool Activity */}
      <Section title="Tool Activity" icon={TerminalIcon}>
        {toolSummary.length === 0 ? (
          <EmptyState message="No tool calls yet" />
        ) : (
          <div className="flex flex-col gap-1">
            {toolSummary.map((t, i) => {
              const Icon = TOOL_ICONS[t.toolName] ?? Wrench
              return (
                <div key={i} className="flex items-center gap-2 py-0.5">
                  <Icon size={12} className="shrink-0 text-muted-foreground" />
                  <span className="text-foreground">{t.toolName}</span>
                  <span className="ml-auto text-muted-foreground">x{t.count}</span>
                </div>
              )
            })}
          </div>
        )}
      </Section>

      {/* Artifacts */}
      <Section title="Artifacts" icon={Package}>
        {changes.length === 0 ? (
          <EmptyState message="No artifacts created yet" />
        ) : (
          <div className="flex flex-col gap-1">
            {changes
              .filter(c => c.toolName === 'write_file')
              .map((c, i) => (
                <div key={i} className="flex items-center gap-2 py-0.5">
                  <Package size={12} className="shrink-0 text-muted-foreground" />
                  <span className="truncate text-foreground">{shortPath(c.path)}</span>
                  <span className="shrink-0 text-muted-foreground">(created)</span>
                </div>
              ))}
          </div>
        )}
      </Section>
    </div>
  )
}

function Section({
  children,
  icon: Icon,
  title
}: {
  children: React.ReactNode
  icon: typeof Monitor
  title: string
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center gap-1.5 text-[0.65rem] font-medium uppercase tracking-wider text-muted-foreground">
        <Icon size={12} />
        {title}
      </div>
      <div className="pl-0.5">{children}</div>
    </div>
  )
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center gap-2 py-0.5">
      <span className="text-muted-foreground">{label}:</span>
      <span className="truncate text-foreground">{value}</span>
    </div>
  )
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="py-2 text-center text-muted-foreground">
      <Codicon name="info" size={14} className="mb-1 block opacity-50" />
      {message}
    </div>
  )
}

function shortPath(path: string): string {
  const parts = path.replace(/\\/g, '/').split('/')
  return parts.length > 2 ? `.../${parts.slice(-2).join('/')}` : path
}

function countDeletions(diff: string): number {
  let count = 0
  for (const line of diff.split('\n')) {
    if (line.startsWith('-') && !line.startsWith('---')) count++
  }
  return count
}
