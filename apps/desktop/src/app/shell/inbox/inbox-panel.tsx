import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router'

import {
  Panel,
  PanelBody,
  PanelDetail,
  PanelEmpty,
  PanelHeader,
  PanelList,
  PanelListRow,
  PanelMeta,
  PanelSectionLabel
} from '@/app/overlays/panel'
import { sessionRoute } from '@/app/routes'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { cn } from '@/lib/utils'
import { filterInboxItems, type InboxEntry, type InboxItem, refreshInbox } from '@/store/inbox'
import { $activeGatewayProfile } from '@/store/profile'
import { setSelectedStoredSessionId } from '@/store/session'

type Section = 'needs' | 'automation'

const SECTION_OPTIONS = [
  { id: 'needs' as const, label: 'Needs you' },
  { id: 'automation' as const, label: 'Automation' }
]

const LANE_LABEL: Record<string, string> = {
  needs_you: 'Needs you',
  running: 'Running',
  waiting: 'Waiting',
  scheduled: 'Scheduled'
}

function laneMatchesSection(item: InboxItem, section: Section): boolean {
  if (section === 'needs') {return item.lanes.includes('needs_you')}

  return item.lanes.includes('running') || item.lanes.includes('waiting') || item.lanes.includes('scheduled')
}

function laneDotClassName(item: InboxItem): string {
  if (item.lanes.includes('needs_you')) {return 'bg-destructive'}

  if (item.lanes.includes('running')) {return 'bg-primary/70'}

  if (item.lanes.includes('waiting')) {return 'bg-amber-500/70'}

  if (item.lanes.includes('scheduled')) {return 'bg-muted-foreground/40'}

  return 'bg-muted-foreground/30'
}

/**
 * Floating Agent Inbox panel. Read-only aggregation + navigation: opening or
 * closing it never resolves an approval or clarify prompt, and no inline button
 * claims to — acting on a request happens in its OWNING session (open session).
 * Declares its connection/profile coverage explicitly; an unsupported or partial
 * read state is never labelled all-clear.
 *
 * Receives `inbox` from the parent (InboxStatusbarChip) so polling is owned by
 * exactly one hook instance.
 */
export function InboxPanel({ inbox, onClose }: { inbox: InboxEntry; onClose: () => void }) {
  const navigate = useNavigate()
  const [section, setSection] = useState<Section>('needs')
  const [query, setQuery] = useState('')
  const [selectedKey, setSelectedKey] = useState<string | null>(null)

  const snapshot = inbox.snapshot
  const coverage = snapshot?.coverage
  const counts = snapshot?.counts

  const visible = useMemo(
    () => filterInboxItems(snapshot?.items ?? [], query).filter(item => laneMatchesSection(item, section)),
    [query, snapshot?.items, section]
  )

  const selected = useMemo(
    () => snapshot?.items.find(item => item.session_key === selectedKey) ?? null,
    [selectedKey, snapshot?.items]
  )

  const needsYouCount = counts?.needs_you ?? 0
  const automationCount = (counts?.running ?? 0) + (counts?.waiting ?? 0) + (counts?.scheduled ?? 0)
  const sectionCount = section === 'needs' ? needsYouCount : automationCount

  const coverageLine = coverage ? `Profile: ${coverage.profile} · This connection only` : 'loading connection…'
  const hasErrors = (coverage?.errors.length ?? 0) > 0
  const isPartial = coverage?.partial ?? false
  const isDisconnected = inbox.error !== null && inbox.capability !== 'unsupported'
  const isUnsupported = inbox.capability === 'unsupported'
  const isLoading = snapshot === null && !inbox.error && !isUnsupported

  const openSession = (item: InboxItem) => {
    setSelectedStoredSessionId(item.session_key)
    navigate(sessionRoute(item.session_key))
    onClose()
  }

  const handleRetry = () => {
    void refreshInbox($activeGatewayProfile.get() ?? '')
  }

  return (
    <Panel contentClassName={cn('flex h-full min-h-0 flex-col')} onClose={onClose}>
      <PanelHeader
        actions={
          <SegmentedControl
            onChange={setSection}
            options={SECTION_OPTIONS}
            value={section}
          />
        }
        subtitle={coverageLine}
        title="Agent Inbox"
      />

      <PanelBody>
        <PanelList
          className="min-[47.5rem]:w-72"
          onSearchChange={setQuery}
          searchPlaceholder="Filter sessions…"
          searchValue={query}
        >
          {isUnsupported ? (
            <PanelEmpty
              description="This connection or profile does not expose the inbox aggregation. Nothing is hidden as 'all clear'."
              icon="warning"
              title="Inbox not supported"
            />
          ) : isLoading ? (
            <PanelEmpty description="Reading the active profile…" icon="loading~spin" title="Loading inbox…" />
          ) : isDisconnected ? (
            <PanelEmpty
              action={
                <Button disabled={inbox.loading} onClick={handleRetry} size="xs" variant="secondary">
                  <Codicon name="refresh" size="0.75rem" />
                  Retry
                </Button>
              }
              description={inbox.error ?? 'Could not reach the gateway.'}
              icon="warning"
              title="Disconnected"
            />
          ) : hasErrors ? (
            <>
              <PanelEmpty
                action={
                  <Button disabled={inbox.loading} onClick={handleRetry} size="xs" variant="secondary">
                    <Codicon name="refresh" size="0.75rem" />
                    Refresh
                  </Button>
                }
                description={`${coverage!.errors.length} session snapshot${coverage!.errors.length === 1 ? '' : 's'} could not be read. Counts below are incomplete.`}
                icon="warning"
                title="Partial read"
              />
              {visible.length > 0 &&
                visible.map(item => (
                  <PanelListRow
                    active={selectedKey === item.session_key}
                    dotClassName={laneDotClassName(item)}
                    key={item.session_key}
                    menuItems={[{ icon: 'arrow-right', label: 'Open session', onSelect: () => openSession(item) }]}
                    menuLabel="Session actions"
                    meta={item.lanes.map(lane => LANE_LABEL[lane] ?? lane).join(' · ')}
                    onSelect={() => setSelectedKey(item.session_key)}
                    rowKey={item.session_key}
                    title={item.title || item.session_key}
                  />
                ))}
            </>
          ) : visible.length === 0 && query.trim() ? (
            <PanelEmpty
              description={`No sessions matching "${query.trim()}" in ${section === 'needs' ? 'needs you' : 'automation'}.`}
              icon="search"
              title="No results for filter"
            />
          ) : visible.length === 0 ? (
            <PanelEmpty
              action={
                isPartial ? (
                  <Button disabled={inbox.loading} onClick={handleRetry} size="xs" variant="secondary">
                    <Codicon name="refresh" size="0.75rem" />
                    Refresh
                  </Button>
                ) : undefined
              }
              description={
                isPartial
                  ? 'Some sessions could not be read. Counts may be incomplete.'
                  : `Nothing ${section === 'needs' ? 'needs you' : 'is running, waiting, or scheduled'} in ${coverage?.profile ?? 'this profile'}.`
              }
              icon={isPartial ? 'warning' : 'inbox'}
              title={isPartial ? 'Incomplete data' : 'All clear'}
            />
          ) : (
            visible.map(item => (
              <PanelListRow
                active={selectedKey === item.session_key}
                dotClassName={laneDotClassName(item)}
                key={item.session_key}
                menuItems={[{ icon: 'arrow-right', label: 'Open session', onSelect: () => openSession(item) }]}
                menuLabel="Session actions"
                meta={item.lanes.map(lane => LANE_LABEL[lane] ?? lane).join(' · ')}
                onSelect={() => setSelectedKey(item.session_key)}
                rowKey={item.session_key}
                title={item.title || item.session_key}
              />
            ))
          )}
        </PanelList>

        {visible.length > 0 || selected ? (
          <PanelDetail>
            {selected ? (
              <div className="flex flex-col gap-4">
                <div className="flex items-center justify-between gap-2">
                  <PanelSectionLabel>Session</PanelSectionLabel>
                </div>
                <PanelMeta
                  rows={[
                    { label: 'Title', value: selected.title || '—' },
                    { label: 'Session', value: selected.session_key },
                    { label: 'Lanes', value: selected.lanes.map(lane => LANE_LABEL[lane] ?? lane).join(', ') },
                    { label: 'Source', value: selected.source || '—' },
                    { label: 'Cwd', value: selected.cwd || '—' }
                  ]}
                />
                {selected.pending_approval ? (
                  <div>
                    <PanelSectionLabel>Needs you — approval</PanelSectionLabel>
                    <p className="mt-1 text-xs text-foreground/80">
                      {selected.pending_approval.description || 'A command is waiting for your approval.'}
                    </p>
                  </div>
                ) : null}
                {selected.pending_clarify ? (
                  <div>
                    <PanelSectionLabel>Needs you — question</PanelSectionLabel>
                    <p className="mt-1 text-xs text-foreground/80">
                      {selected.pending_clarify.count > 0
                        ? `${selected.pending_clarify.count} question${selected.pending_clarify.count === 1 ? '' : 's'} waiting in this chat`
                        : 'Question waiting in this chat'}
                    </p>
                  </div>
                ) : null}
                {selected.goal ? (
                  <div>
                    <PanelSectionLabel>Goal</PanelSectionLabel>
                    <PanelMeta
                      rows={[
                        { label: 'Title', value: String(selected.goal.title ?? '—') },
                        { label: 'Status', value: String(selected.goal.status ?? '—') }
                      ]}
                    />
                  </div>
                ) : null}
                {selected.loop ? (
                  <div>
                    <PanelSectionLabel>Loop</PanelSectionLabel>
                    <PanelMeta rows={[{ label: 'Status', value: String((selected.loop as Record<string, unknown>).status ?? '—') }]} />
                  </div>
                ) : null}
                {selected.heartbeat ? (
                  <div>
                    <PanelSectionLabel>Heartbeat</PanelSectionLabel>
                    <PanelMeta rows={[{ label: 'Status', value: String((selected.heartbeat as Record<string, unknown>).status ?? '—') }]} />
                  </div>
                ) : null}
                <div>
                  <Button onClick={() => openSession(selected)} size="xs" variant="secondary">
                    <Codicon name="arrow-right" size="0.75rem" />
                    Open session
                  </Button>
                  <p className="mt-1 text-[0.62rem] text-muted-foreground/60">
                    Approve, clarify, pause, or clear in the session's own UI — the inbox never resolves anything.
                  </p>
                </div>
              </div>
            ) : (
              <PanelEmpty
                description="Select a session to see its automation and pending requests, then open it to act."
                icon="inbox"
                title="Select a session"
              />
            )}
          </PanelDetail>
        ) : null}
      </PanelBody>
    </Panel>
  )
}
