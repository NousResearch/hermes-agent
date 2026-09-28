import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useRef, useState } from 'react'

import { usePaneVisible } from '@/components/pane-shell/pane-visibility'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ErrorState } from '@/components/ui/error-state'
import { Loader } from '@/components/ui/loader'
import { RowButton } from '@/components/ui/row-button'
import { SearchField } from '@/components/ui/search-field'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { type Translations, useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { type AgentRow, type Attention, overviewCache } from '@/store/agent-overview'
import { $overviewRows } from '@/store/agent-overview-status'
import type { SessionDotState } from '@/store/session-dot-state'

import { sessionDotClassName } from '../chat/session-status-dot'
import { PanelBody, PanelEmpty, PanelSectionLabel } from '../overlays/panel'

import { overviewActions } from './actions'
import { isRecentActivity, useActivityNow } from './activity'
import { SourceCoverage } from './source-coverage'

const GROUPS: Attention[] = ['needs-you', 'working', 'unread', 'idle']
const PAGE = 100

// The same dot vocabulary as the sidebar row, so a session never looks
// different here than in the sidebar. A stale row that was last seen active is
// "still open, nothing arriving" — exactly the hollow accent the sidebar uses.
const dotState = (row: AgentRow): SessionDotState =>
  row.stale
    ? row.lastKnownAttention
      ? 'stalled'
      : 'idle'
    : row.attention === 'needs-you'
      ? 'needs-input'
      : row.attention

// Coarse, sidebar-style age ("3m", "2h"). Ticks with the shared minute clock.
const fmtAge = (lastActive: number, nowMs: number, a: Translations['agents']): string => {
  if (lastActive <= 0) {
    return ''
  }

  const s = Math.max(0, Math.round((nowMs - lastActive * 1000) / 1000))

  if (s < 60) {
    return a.ageNow
  }

  const m = Math.floor(s / 60)

  if (m < 60) {
    return a.ageMinutes(m)
  }

  const h = Math.floor(m / 60)

  return h < 24 ? a.ageHours(h) : a.ageDays(Math.floor(h / 24))
}

// A minute-grained clock for the age labels; polling preserves row identity on
// unchanged data, so nothing else would ever re-render them.
function useMinuteClock(visible: boolean): number {
  const [now, setNow] = useState(Date.now)
  useEffect(() => {
    if (!visible) {
      return
    }

    setNow(Date.now())
    const timer = setInterval(() => setNow(Date.now()), 30_000)

    return () => clearInterval(timer)
  }, [visible])

  return now
}

export function SessionOverview({ onSummary }: { onSummary?: (summary: string) => void }) {
  const { t } = useI18n()
  const a = t.agents
  const { data, error, loading } = useStore(overviewCache.state)
  const paneVisible = usePaneVisible()
  const [documentVisible, setDocumentVisible] = useState(document.visibilityState !== 'hidden')
  const [search, setSearch] = useState('')
  const [source, setSource] = useState('all')
  const [profile, setProfile] = useState('all')
  const [provider, setProvider] = useState('all')
  const [activity, setActivity] = useState<'recent' | 'all'>('recent')
  const [selected, setSelected] = useState<string | null>(null)
  const [limit, setLimit] = useState(PAGE)
  useEffect(() => {
    const update = () => setDocumentVisible(document.visibilityState !== 'hidden')
    document.addEventListener('visibilitychange', update)

    return () => document.removeEventListener('visibilitychange', update)
  }, [])
  const visible = paneVisible && documentVisible
  useEffect(() => (visible ? overviewCache.watch() : undefined), [visible])
  const rows = useStore($overviewRows)
  const now = useActivityNow(rows, visible)
  const clock = useMinuteClock(visible)

  const filtered = useMemo(
    () =>
      rows
        .filter(
          row =>
            (activity === 'all' || isRecentActivity(row, now)) &&
            (source === 'all' || row.connectionId === source) &&
            (profile === 'all' || row.profile === profile) &&
            (provider === 'all' || row.provider === provider) &&
            [
              row.title,
              row.owner,
              row.description,
              row.profile,
              row.sourceLabel,
              row.provider,
              row.model,
              row.preview,
              row.id
            ]
              .join(' ')
              .toLowerCase()
              .includes(search.toLowerCase().trim())
        )
        .sort(
          (left, right) =>
            GROUPS.indexOf(left.attention) - GROUPS.indexOf(right.attention) || right.lastActive - left.lastActive
        ),
    [rows, source, profile, provider, search, now, activity]
  )

  const shown = filtered.slice(0, limit)
  const row = rows.find(row => row.key === selected)
  const sources = data?.sources ?? []
  const multiSource = sources.length > 1
  const profiles = useMemo(() => [...new Set(rows.map(row => row.profile))].sort(), [rows])
  const providers = useMemo(() => [...new Set(rows.map(row => row.provider).filter(Boolean))].sort(), [rows])

  const labels: Record<Attention, string> = {
    'needs-you': a.needsYou,
    working: a.working,
    unread: a.unread,
    idle: a.idle
  }

  // The header's one-line summary counts the live picture, not the filter.
  const counts = useMemo(() => {
    const result: Record<Attention, number> = { 'needs-you': 0, working: 0, unread: 0, idle: 0 }

    for (const row of rows) {
      if (row.stale) {
        continue
      }

      result[row.attention]++
    }

    return result
  }, [rows])

  const summary = useMemo(() => {
    const parts = [
      counts['needs-you'] > 0 ? a.needYouCount(counts['needs-you']) : '',
      counts.working > 0 ? a.workingCount(counts.working) : ''
    ].filter(Boolean)

    return parts.length > 0 ? parts.join(' · ') : a.allQuiet
  }, [a, counts])

  useEffect(() => onSummary?.(summary), [onSummary, summary])

  const reset = () => setLimit(PAGE)
  const quiet = activity === 'recent' && !search.trim() && source === 'all' && profile === 'all' && provider === 'all'

  return (
    <section className="flex min-h-0 flex-1 flex-col gap-2" data-testid="agent-overview">
      <div className="flex shrink-0 flex-wrap items-center gap-x-3 gap-y-1.5">
        <SearchField
          aria-label={a.searchSessions}
          containerClassName="min-w-40"
          onChange={value => {
            setSearch(value)
            reset()
          }}
          placeholder={a.searchSessions}
          value={search}
        />
        <div className="flex flex-1 flex-wrap items-center justify-end gap-1.5">
          {multiSource ? (
            <Filter
              label={a.allSources}
              onChange={value => {
                setSource(value)
                reset()
              }}
              options={sources.map(item => ({ value: item.connectionId, label: item.label }))}
              value={source}
            />
          ) : null}
          {profiles.length > 1 ? (
            <Filter
              label={a.allProfiles}
              onChange={value => {
                setProfile(value)
                reset()
              }}
              options={profiles.map(value => ({ value, label: value }))}
              value={profile}
            />
          ) : null}
          {providers.length > 1 ? (
            <Filter
              label={a.allProviders}
              onChange={value => {
                setProvider(value)
                reset()
              }}
              options={providers.map(value => ({ value, label: value }))}
              value={provider}
            />
          ) : null}
          <SegmentedControl
            onChange={value => {
              setActivity(value)
              reset()
            }}
            options={[
              { id: 'recent', label: a.recentActivity },
              { id: 'all', label: a.allSessions }
            ]}
            value={activity}
          />
        </div>
      </div>
      {loading ? <Loader label={a.sessionsTab} /> : null}
      {error ? (
        <ErrorState description={error} title={a.offline}>
          <Button onClick={() => void overviewCache.refresh({ force: true })} size="sm" variant="secondary">
            {a.retry}
          </Button>
        </ErrorState>
      ) : null}
      <PanelBody className="gap-3 min-[47.5rem]:gap-4">
        <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-y-auto overscroll-contain">
          {GROUPS.map(group => {
            const items = shown.filter(row => row.attention === group)

            return items.length > 0 ? (
              <section aria-label={labels[group]} className="mb-3" key={group}>
                <PanelSectionLabel className="mb-0.5 px-2">
                  {labels[group]} · {filtered.filter(row => row.attention === group).length}
                </PanelSectionLabel>
                {items.map(row => (
                  <AgentListRow
                    active={row.key === selected}
                    age={fmtAge(row.lastActive, clock, a)}
                    key={row.key}
                    multiSource={multiSource}
                    onOpen={() => void overviewActions.open(row, () => true)}
                    onSelect={() => setSelected(row.key)}
                    row={row}
                    staleLabel={row.stale ? a.stale : undefined}
                  />
                ))}
              </section>
            ) : null
          })}
          {!loading && filtered.length === 0 ? (
            <PanelEmpty
              description={quiet ? a.allQuietHint : undefined}
              icon={quiet ? 'hubot' : 'search'}
              title={quiet ? a.allQuiet : a.noSessions}
            />
          ) : null}
          {shown.length < filtered.length ? (
            <div className="flex items-center gap-2 px-2 py-1 text-xs text-muted-foreground">
              <span>{a.shown(shown.length, filtered.length)}</span>
              <Button onClick={() => setLimit(value => value + PAGE)} size="inline" variant="textStrong">
                {a.loadMore}
              </Button>
            </div>
          ) : null}
        </div>
        {row ? (
          <aside
            className="flex min-h-0 shrink-0 flex-col overflow-y-auto overflow-x-hidden overscroll-contain min-[47.5rem]:w-72 min-[47.5rem]:border-l min-[47.5rem]:border-(--ui-stroke-tertiary) min-[47.5rem]:pl-4"
            data-testid="agent-detail"
          >
            <Preview key={row.key} onClose={() => setSelected(null)} row={row} />
          </aside>
        ) : null}
      </PanelBody>
      <div
        className="flex shrink-0 flex-wrap items-center gap-x-4 gap-y-1 border-t border-(--ui-stroke-tertiary) pt-2 text-[0.68rem] text-muted-foreground/70"
        data-testid="agent-source-coverage"
      >
        {sources.map(item => (
          <SourceCoverage key={item.connectionId} source={item} />
        ))}
      </div>
    </section>
  )
}

function AgentListRow({
  active,
  age,
  multiSource,
  onOpen,
  onSelect,
  row,
  staleLabel
}: {
  active: boolean
  age: string
  multiSource: boolean
  onOpen: () => void
  onSelect: () => void
  row: AgentRow
  staleLabel?: string
}) {
  const detail = row.preview || row.description
  const who = multiSource ? `${row.owner} · ${row.sourceLabel}` : row.owner

  return (
    <RowButton
      aria-pressed={active}
      className={cn(
        'row-hover flex w-full items-start gap-2.5 rounded-md py-1.5 pl-2 pr-2.5 text-left',
        active ? 'bg-(--ui-row-active-background) text-foreground' : 'text-(--ui-text-secondary) hover:text-foreground'
      )}
      data-owner-key={row.key}
      data-testid="agent-row"
      onClick={onSelect}
      onDoubleClick={onOpen}
    >
      <span className="flex h-5 shrink-0 items-center">
        <span aria-hidden="true" className={sessionDotClassName(dotState(row))} />
      </span>
      <span className="flex min-w-0 flex-1 flex-col">
        <span className="flex items-baseline gap-3">
          <span className="min-w-0 flex-1 truncate text-[0.8125rem] font-medium text-foreground/90">
            {row.title || row.id}
          </span>
          <span className="shrink-0 truncate text-[0.68rem] tabular-nums text-muted-foreground/60">
            {staleLabel ? `${staleLabel} · ` : ''}
            {who}
            {age ? ` · ${age}` : ''}
          </span>
        </span>
        {detail ? <span className="truncate text-xs text-muted-foreground/75">{detail}</span> : null}
      </span>
    </RowButton>
  )
}

function Preview({ onClose, row }: { onClose: () => void; row: AgentRow }) {
  const { t } = useI18n()
  const a = t.agents
  const [error, setError] = useState<string | null>(null)
  const [pending, setPending] = useState(false)
  const [reply, setReply] = useState('')
  const mounted = useRef(true)
  // Lifecycle-only guard, not mirrored reactive state; reset for Strict Mode replay.
  // eslint-disable-next-line no-restricted-syntax
  useEffect(() => {
    mounted.current = true

    return () => {
      mounted.current = false
    }
  }, [])

  const run = async (action: 'open' | 'reply' | 'stop') => {
    setPending(true)
    setError(null)

    try {
      if (action === 'open') {
        await overviewActions.open(row, () => mounted.current)
      }

      if (action === 'reply') {
        await overviewActions.reply(row, reply)

        if (mounted.current) {
          setReply('')
        }
      }

      if (action === 'stop') {
        await overviewActions.stop(row)
      }

      if (action !== 'open') {
        await overviewCache.refresh()
      }
    } catch (error) {
      if (mounted.current) {
        setError(String(error instanceof Error ? error.message : error).slice(0, 240))
      }
    } finally {
      if (mounted.current) {
        setPending(false)
      }
    }
  }

  const canStop = Boolean(row.runtimeId) && !row.stale && (row.attention === 'working' || row.attention === 'needs-you')
  const runtime = [row.model, row.provider].filter(Boolean).join(' · ')

  return (
    <div className="flex flex-col gap-3 pb-4">
      <div className="flex items-start gap-2">
        <div className="min-w-0 flex-1">
          <h3 className="truncate text-sm font-semibold">{row.title || row.id}</h3>
          <p className="truncate text-xs text-muted-foreground">
            {row.owner} · {row.sourceLabel} / {row.profile}
          </p>
          {runtime ? <p className="truncate text-xs text-muted-foreground/70">{runtime}</p> : null}
        </div>
        <Button
          aria-label={a.closePreview}
          className="-mt-1 text-muted-foreground"
          onClick={onClose}
          size="icon-xs"
          variant="ghost"
        >
          <Codicon name="close" size="0.8rem" />
        </Button>
      </div>
      {row.preview ? (
        <p
          className="whitespace-pre-wrap break-words text-xs leading-relaxed text-foreground/80"
          data-selectable-text="true"
        >
          {row.preview}
        </p>
      ) : null}
      {row.attention === 'needs-you' ? (
        <p className="text-xs text-amber-600 dark:text-amber-300">{a.promptHint}</p>
      ) : null}
      <div className="flex flex-wrap gap-1.5">
        <Button disabled={pending} onClick={() => void run('open')} size="sm">
          {a.openConversation}
        </Button>
        {canStop ? (
          <Button disabled={pending} onClick={() => void run('stop')} size="sm" variant="secondary">
            {a.stop}
          </Button>
        ) : null}
      </div>
      {row.attention !== 'needs-you' ? (
        <form
          className="grid gap-1.5"
          onSubmit={event => {
            event.preventDefault()
            void run('reply')
          }}
        >
          <Textarea
            aria-label={a.reply}
            className="min-h-12"
            disabled={pending}
            onChange={event => setReply(event.target.value)}
            onKeyDown={event => {
              if (event.key === 'Enter' && !event.shiftKey && reply.trim()) {
                event.preventDefault()
                void run('reply')
              }
            }}
            placeholder={a.replyPlaceholder}
            size="sm"
            value={reply}
          />
          <div>
            <Button disabled={pending || !reply.trim()} size="sm" type="submit" variant="secondary">
              {a.reply}
            </Button>
          </div>
        </form>
      ) : null}
      {error ? <ErrorState description={error} title={a.failed} /> : null}
    </div>
  )
}

function Filter({
  label,
  value,
  onChange,
  options
}: {
  label: string
  value: string
  onChange: (value: string) => void
  options: Array<{ value: string; label: string }>
}) {
  return (
    <Select onValueChange={onChange} value={value}>
      <span className="inline-flex max-w-44">
        <SelectTrigger aria-label={label} size="sm">
          <SelectValue />
        </SelectTrigger>
      </span>
      <SelectContent>
        <SelectItem value="all">{label}</SelectItem>
        {options.map(option => (
          <SelectItem key={option.value} value={option.value}>
            {option.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}
