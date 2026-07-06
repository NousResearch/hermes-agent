import { useStore } from '@nanostores/react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { cn } from '@/lib/utils'
import { buildMissionControlBuckets, MISSION_CONTROL_BUCKET_META, MISSION_CONTROL_BUCKETS, type MissionControlEntry } from '@/store/mission-control'
import { $activeSessionId, $attentionSessionIds, $selectedStoredSessionId, $sessions, $workingSessionIds } from '@/store/session'
import { $subagentsBySession } from '@/store/subagents'
import { $todosBySession } from '@/store/todos'
import { $workstreamMetadata } from '@/store/workstream-metadata'

import { Panel, PanelEmpty, PanelHeader } from '../overlays/panel'

interface MissionControlViewProps {
  onClose: () => void
  onOpenSession: (sessionId: string) => void
}

function sessionTitle(entry: MissionControlEntry): string {
  return entry.session.title || entry.session.preview || entry.session.id
}

function sessionMeta(entry: MissionControlEntry): string {
  if (entry.session.cwd) {
    return entry.session.cwd
  }

  return `${entry.session.message_count} messages`
}

function WorkstreamRow({ entry, onOpenSession }: { entry: MissionControlEntry; onOpenSession: (sessionId: string) => void }) {
  return (
    <button
      className="group flex min-w-0 items-start gap-2 rounded-lg px-2 py-2 text-left transition-colors hover:bg-(--ui-row-hover-background)"
      onClick={() => onOpenSession(entry.session.id)}
      type="button"
    >
      <span className="mt-0.5 shrink-0 text-sm" role="img">
        {entry.icon}
      </span>
      <span className="min-w-0 flex-1">
        <span className="block truncate text-[0.78rem] font-medium text-foreground/90">{sessionTitle(entry)}</span>
        <span className="block truncate text-[0.66rem] text-muted-foreground/65">{sessionMeta(entry)}</span>
      </span>
      <span className="shrink-0 rounded-full bg-muted/45 px-1.5 py-0.5 text-[0.6rem] text-muted-foreground/75">
        {entry.stateLabel}
      </span>
    </button>
  )
}

function BucketCard({ entries, onOpenSession, bucket }: { bucket: (typeof MISSION_CONTROL_BUCKETS)[number]; entries: MissionControlEntry[]; onOpenSession: (sessionId: string) => void }) {
  const meta = MISSION_CONTROL_BUCKET_META[bucket]

  return (
    <section className="flex min-h-40 min-w-0 flex-col overflow-hidden rounded-xl border border-(--ui-stroke-secondary) bg-muted/10">
      <header className="flex items-center justify-between gap-2 border-b border-(--ui-stroke-tertiary) px-3 py-2">
        <div className="flex min-w-0 items-center gap-2">
          <span aria-hidden className="shrink-0 text-sm">
            {meta.icon}
          </span>
          <h3 className="truncate text-sm font-semibold text-foreground/90">{meta.label}</h3>
        </div>
        <span className="rounded-full bg-background/80 px-1.5 py-0.5 text-[0.62rem] tabular-nums text-muted-foreground">
          {entries.length}
        </span>
      </header>
      {entries.length > 0 ? (
        <div className="min-h-0 flex-1 overflow-y-auto p-1.5">
          {entries.map(entry => (
            <WorkstreamRow entry={entry} key={entry.session.id} onOpenSession={onOpenSession} />
          ))}
        </div>
      ) : (
        <div className="grid flex-1 place-items-center p-4 text-center text-xs text-muted-foreground/65">No workstreams</div>
      )}
    </section>
  )
}

export function MissionControlView({ onClose, onOpenSession }: MissionControlViewProps) {
  const sessions = useStore($sessions)
  const activeSessionId = useStore($activeSessionId)
  const selectedStoredSessionId = useStore($selectedStoredSessionId)
  const attentionSessionIds = useStore($attentionSessionIds)
  const workingSessionIds = useStore($workingSessionIds)
  const todosBySession = useStore($todosBySession)
  const subagentsBySession = useStore($subagentsBySession)
  const metadataBySession = useStore($workstreamMetadata)

  const buckets = buildMissionControlBuckets(sessions, {
    activeSessionId,
    attentionSessionIds,
    metadataBySession,
    selectedStoredSessionId,
    subagentsBySession,
    todosBySession,
    workingSessionIds
  })

  const total = MISSION_CONTROL_BUCKETS.reduce((count, bucket) => count + buckets[bucket].length, 0)

  return (
    <Panel onClose={onClose}>
      <PanelHeader
        actions={
          <Button className="gap-1.5" onClick={onClose} size="sm" variant="ghost">
            <Codicon name="close" size="0.875rem" />
            Close
          </Button>
        }
        subtitle={`${total} workstreams across active, blocked, review, restart, closed, and safe-delete buckets`}
        title="Mission Control"
      />
      {total === 0 ? (
        <PanelEmpty description="No sessions are currently visible to Mission Control." icon="dashboard" title="No workstreams" />
      ) : (
        <div
          className={cn(
            'grid min-h-0 flex-1 grid-cols-1 gap-3 overflow-y-auto pb-2 pr-1',
            'md:grid-cols-2 xl:grid-cols-3'
          )}
        >
          {MISSION_CONTROL_BUCKETS.map(bucket => (
            <BucketCard bucket={bucket} entries={buckets[bucket]} key={bucket} onOpenSession={onOpenSession} />
          ))}
        </div>
      )}
    </Panel>
  )
}
