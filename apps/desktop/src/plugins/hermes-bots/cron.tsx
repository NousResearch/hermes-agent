/**
 * Routines: the Hermes cron jobs scoped to the bot you're chatting with — the
 * list query and its owner resolution, the schedule picker, the create and
 * detail dialogs, and the pane the right tile renders.
 */

import {
  atom,
  Button,
  cn,
  Codicon,
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  GlyphSpinner,
  host,
  Input,
  PanelEmpty,
  queryClient,
  relativeTime,
  RowButton,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Switch,
  Textarea,
  Tip,
  useI18n,
  useQuery,
  useValue
} from '@hermes/plugin-sdk'
import { useState } from 'react'

import { avatarColor, botAppearance, BotFace } from './avatar'
import { $focusedBotOwner, $selectedBot, focusedRosterOwner } from './bot-state'
import { $botMeta, $lastRoster, botHandle, botRosterKey, botSelectionKey, isActiveRosterBot } from './data'
import { labeled } from './dialog-parts'
import { displayName } from './labels'
import { botConnectionRoute, botRosterMeta, requestForBot } from './routing'
import { ID } from './shared'
import type { BotMeta, RosterRow, RoutineJob } from './types'

const ROUTINES_KEY = [ID, 'routines']

/** Last good cron list, same idea as the roster snapshot. */
const $lastJobs = atom<RoutineJob[]>([])

function showsHandle(name: string, meta: BotMeta | null | undefined, bot?: RosterRow) {
  const display = displayName(
    {
      name
    },
    meta
  )

  return Boolean(name && display.toLowerCase() !== botHandle(name, bot).toLowerCase())
}

// ── routines (cron) ──────────────────────────────────────────────────────────
//
// Jobs are namespaced "[bot:<name>] <routine>". A job running in the active
// bot profile uses the plain instruction; a different profile keeps the
// hermes -p <bot> chat delegation wrapper so the run reaches that bot's
// history. The tile follows the bot you're chatting with (gateway profile).
const BOT_TAG_RE = /^\[bot:([a-z0-9][a-z0-9_-]*)\]\s*/i
const SAFE_ROUTINE_MARKER = '[bot-mode:routine:v2] '
const LEGACY_DELEGATED_ROUTINE_PREFIX = 'You are running the scheduled routine "'

/** A routine's owner: a roster row, a bare profile name, or nothing resolved yet. */
type RoutineOwner = RosterRow | string | null | undefined

/** `cron.manage {action: 'list'}` reply. `scoped` carries the profile the
 *  gateway scoped the store to; gateways that ignore `profile` omit it. */
interface RoutineListResult {
  jobs?: RoutineJob[]
  scoped?: string
}

function routineBot(job: RoutineJob | null | undefined): null | string {
  const match = BOT_TAG_RE.exec(job?.name || '')

  return match ? match[1].toLowerCase() : null
}

function routineTitle(job: RoutineJob | null | undefined): string {
  return (job?.name || '').replace(BOT_TAG_RE, '') || 'Untitled job'
}

export function isLegacyDelegatedRoutine(job: RoutineJob | null | undefined): boolean {
  const preview = typeof job?.prompt_preview === 'string' ? job.prompt_preview : job?.prompt

  return Boolean(routineBot(job) && typeof preview === 'string' && preview.startsWith(LEGACY_DELEGATED_ROUTINE_PREFIX))
}

export async function loadRoutines(owner: RoutineOwner): Promise<RoutineListResult> {
  const bot =
    typeof owner === 'string'
      ? {
          name: owner
        }
      : owner

  const profile = String(bot?.name || '').trim()

  // profile scopes cron.manage to that bot's own cron store (core RPC gained an
  // optional `profile` param). Older gateways ignore the unknown param and
  // return the launch-profile store — the [bot:] tag filter in selectRoutineJobs
  // remains the graceful fallback there.
  const scope = profile
    ? {
        profile
      }
    : {}

  const data = (await requestForBot(bot, 'cron.manage', {
    action: 'list',
    include_disabled: true,
    ...scope
  })) as RoutineListResult

  const jobs = Array.isArray(data?.jobs) ? data.jobs : []

  const activeLegacyJobs = jobs.filter(
    job => isLegacyDelegatedRoutine(job) && job.enabled !== false && job.state !== 'paused'
  )

  // A pause failing must not fail the LIST — the pane would report "could
  // not load cronjobs" over data that loaded fine, and the 20s poll would
  // re-attempt the failing pause inside a failing query forever. Each pause
  // swallows its own error; the overlay only claims jobs the gateway
  // actually paused, and the next poll retries the rest.
  const pauses = await Promise.all(
    activeLegacyJobs.map(job =>
      requestForBot(bot, 'cron.manage', {
        action: 'pause',
        name: job.job_id,
        ...scope
      })
        .then(() => true)
        .catch(() => false)
    )
  )

  if (!activeLegacyJobs.length) {
    return data
  }

  const pausedIds = new Set(activeLegacyJobs.filter((job, index) => pauses[index]).map(job => job.job_id))

  return {
    ...data,
    jobs: jobs.map(job =>
      pausedIds.has(job.job_id)
        ? {
            ...job,
            enabled: false,
            state: 'paused'
          }
        : job
    )
  }
}

function useRoutines(owner: RoutineOwner) {
  const bot =
    typeof owner === 'string'
      ? {
          name: owner
        }
      : owner

  const route = botConnectionRoute(bot)
  const key = route ? botRosterKey(bot) : bot?.name || ''

  return useQuery({
    queryKey: [...ROUTINES_KEY, key],
    queryFn: () => loadRoutines(bot),
    enabled: Boolean(bot?.name),
    refetchInterval: 20000,
    staleTime: 8000
  })
}

/** `activeBot` is always a non-empty profile name, so the result is never
 *  nullish even when no create-owner has been captured yet. */
export function routineCreateTarget(owner: RoutineOwner, activeBot: string): RosterRow | string {
  return owner || activeBot
}

export async function invalidateRoutineOwner(owner: RoutineOwner) {
  const bot =
    typeof owner === 'string'
      ? {
          name: owner
        }
      : owner

  const route = botConnectionRoute(bot)
  const key = route ? botRosterKey(bot) : bot?.name || ''
  await queryClient.invalidateQueries({
    queryKey: [...ROUTINES_KEY, key],
    exact: true
  })
}

/** Pick which cron jobs to show. A failed refresh keeps the last good list. */
export function selectRoutineJobs(
  data: RoutineListResult | undefined,
  error: unknown,
  lastJobs: RoutineJob[],
  bot: string
) {
  const live = Array.isArray(data?.jobs) ? data.jobs : null
  const all = live ?? (error ? lastJobs : [])
  const scopedToBot = normalizedProfileName(data?.scoped) === normalizedProfileName(bot)

  return {
    live,
    all,
    jobs: scopedToBot ? all : all.filter(job => (routineBot(job) || 'default') === bot)
  }
}

/**
 * Why the Routines pane can be empty while the bot's cron store has jobs.
 *
 * On older gateways the pane only shows jobs namespaced `[bot:<name>]` for the
 * active bot (plus untagged legacy jobs on the default bot). When jobs exist in
 * the store but none surface for this bot, the user is left staring at the
 * generic empty state with no hint that cronjobs are present but hidden.
 * Return a short explanation string in that case, or null when the store is
 * genuinely empty (or the active bot's jobs are already shown).
 */
export function routineFilterHint(all: RoutineJob[], jobs: RoutineJob[]): null | string {
  if (jobs.length !== 0 || !Array.isArray(all) || all.length === 0) {
    return null
  }

  return (
    'Scheduled jobs exist in this profile but none are tagged for this bot. ' +
    'Name a job "[bot:<name>] …" to show it here, or see them in Cron below.'
  )
}

export function normalizedProfileName(profile: unknown): string {
  return typeof profile === 'string' ? profile.trim().toLowerCase() : ''
}

function shellQuote(value: unknown): string {
  return `'${String(value).replaceAll("'", "'\"'\"'")}'`
}

export function routineInputError(title: string, instruction: string): null | string {
  if (String(title).includes('\0')) {
    return 'Job name cannot contain NUL (U+0000).'
  }

  if (String(instruction).includes('\0')) {
    return 'Job instruction cannot contain NUL (U+0000).'
  }

  return null
}

export function routinePrompt(
  bot: string | undefined,
  title: string,
  instruction: string,
  activeProfile: string
): string {
  if (normalizedProfileName(bot) && normalizedProfileName(bot) === normalizedProfileName(activeProfile)) {
    return instruction
  }

  return (
    `${SAFE_ROUTINE_MARKER}You are running the scheduled routine "${title}" for agent '${bot}'. ` +
    `Execute it AS that agent so the run lands in its own history: run this in the terminal and relay the output:\n\n` +
    `hermes -p ${shellQuote(bot)} chat -c ${shellQuote(`Routine: ${title}`)} -q ${shellQuote(`[Scheduled routine] ${instruction}`)}\n\n` +
    `If the command fails, report the error instead.`
  )
}

function scheduleLabel(schedule: string | undefined): string {
  const once = /^once in (.+)$/.exec(schedule || '')

  if (once) {
    return `Once (${once[1]})`
  }

  const bare = /^(\d+)([mhd])$/.exec(schedule || '')

  if (bare) {
    return `Once (${bare[1]}${bare[2]})`
  }

  const match = /^every (\d+)m$/.exec(schedule || '')

  if (match) {
    const minutes = Number(match[1])

    if (minutes % 1440 === 0) {
      const d = minutes / 1440

      return d === 1 ? 'Daily' : `Every ${d} days`
    }

    if (minutes % 60 === 0) {
      const h = minutes / 60

      return h === 1 ? 'Hourly' : `Every ${h}h`
    }

    return `Every ${minutes}m`
  }

  return schedule || ''
}

/** Absolute + relative rendering of a cron timestamp, or null when the job
 *  has never carried one (a job that has not run yet has no `last_run_at`). */
function routineTimestamp(value: string | undefined): null | string {
  const ms = value ? new Date(value).getTime() : Number.NaN

  return Number.isFinite(ms) ? `${relativeTime(ms)} · ${new Date(ms).toLocaleString()}` : null
}

/** The facts `cron.manage list` already sends with every job, as label/value
 *  rows. Pure so the detail contract is testable without a renderer, and so
 *  the dialog cannot invent a field the gateway never sent: an absent value
 *  drops its row instead of rendering "undefined". */
export function routineDetailRows(job: RoutineJob | null | undefined): Array<{ label: string; value: string }> {
  const paused = job?.enabled === false || job?.state === 'paused'
  const label = scheduleLabel(job?.schedule)
  const raw = String(job?.schedule || '').trim()

  // Cells are `number | string | null | undefined` until the filter below
  // drops the non-strings; a destructured, non-predicate callback can't carry
  // that narrowing into the map, so the rows are typed as filtered.
  return (
    [
      ['Status', paused ? 'Paused' : 'Active'],
      ['Schedule', label],
      // `scheduleLabel` humanizes "every 1440m" and cron expressions; keep the
      // raw string when it says something the label dropped.
      ['Schedule (raw)', raw && raw !== label ? raw : null],
      ['Repeat', job?.repeat],
      ['Next run', paused ? null : routineTimestamp(job?.next_run_at)],
      ['Last run', routineTimestamp(job?.last_run_at)],
      ['Last result', job?.last_status],
      ['Delivers to', job?.deliver],
      ['Model', job?.model],
      ['Working directory', job?.workdir]
    ] as Array<[string, string]>
  )
    .filter(([, value]) => typeof value === 'string' && value.trim())
    .map(([name, value]) => ({
      label: name,
      value: value.trim()
    }))
}

/** Why a job is not doing what the user expects. The row only ever showed
 *  "paused"; the scheduler's own reason and the last fire/delivery failures
 *  had no surface in Bot Mode at all. */
export function routineDetailIssue(job: RoutineJob | null | undefined): null | string {
  const reasons = [job?.last_fire_error, job?.last_delivery_error, job?.paused_reason]
  const first = reasons.find(value => typeof value === 'string' && value.trim())

  return first ? first.trim() : null
}

interface RoutineDetailDialogProps {
  job: RoutineJob | null
  onClose: () => void
  open: boolean
}

/** Read-only inspector for one cronjob, rendered from the list payload the
 *  pane already holds — no extra RPC, and no second mutation path beside the
 *  row's own switch and delete. */
export function RoutineDetailDialog({ job, onClose, open }: RoutineDetailDialogProps) {
  const { t } = useI18n()
  const rows = job ? routineDetailRows(job) : []
  const issue = job ? routineDetailIssue(job) : null
  const instruction = String(job?.prompt_preview || '').trim()

  return (
    <Dialog
      onOpenChange={value => {
        if (!value) {
          onClose()
        }
      }}
      open={Boolean(open && job)}
    >
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle className="truncate">{routineTitle(job)}</DialogTitle>
          <DialogDescription>What this job runs, and when it runs next.</DialogDescription>
        </DialogHeader>
        <div className="grid gap-3.5">
          {issue ? (
            <div className="rounded-md border border-(--ui-stroke-secondary) px-3 py-2 text-xs leading-5 text-(--ui-accent)">
              {issue}
            </div>
          ) : null}
          <div className="grid gap-1.5">
            {rows.map(row => (
              <div className="flex items-baseline justify-between gap-3 text-xs" key={row.label}>
                <span className="shrink-0 text-(--ui-text-tertiary)">{row.label}</span>
                <span className="min-w-0 truncate text-right">{row.value}</span>
              </div>
            ))}
          </div>
          {instruction
            ? labeled(
                'Instruction',
                <div className="max-h-48 overflow-y-auto whitespace-pre-wrap break-words rounded-md border border-(--ui-stroke-secondary) px-3 py-2 text-xs leading-5 text-(--ui-text-secondary)">
                  {instruction}
                </div>
              )
            : null}
        </div>
        <DialogFooter>
          <Button onClick={onClose} variant="secondary">
            {t.common.close}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

interface RoutineRowProps {
  job: RoutineJob
  onOpen?: (job: RoutineJob) => void
  /** The pane only renders rows once it has resolved an exact roster row, so
   *  this never sees the bare-name arm of RoutineOwner. */
  owner: RosterRow
}

export function RoutineRow({ job, onOpen, owner }: RoutineRowProps) {
  const { t } = useI18n()
  const c = t.cron
  const profile = typeof owner === 'string' ? owner : owner?.name
  const [busy, setBusy] = useState(false)
  // Optimistic overlay: null = trust server state. Set immediately on
  // toggle so the switch responds even before the refetch lands.
  const [pendingActive, setPendingActive] = useState<boolean | null>(null)
  const legacyUnsafe = isLegacyDelegatedRoutine(job)
  const serverActive = !legacyUnsafe && job.enabled !== false && job.state !== 'paused'
  const active = pendingActive === null ? serverActive : pendingActive

  if (pendingActive !== null && pendingActive === serverActive) {
    setPendingActive(null) // server caught up
  }

  const act = async (action: 'pause' | 'remove' | 'resume') => {
    if (busy) {
      return
    }

    setBusy(true)

    if (action === 'pause' || action === 'resume') {
      setPendingActive(action === 'resume')
    }

    try {
      await requestForBot(owner, 'cron.manage', {
        action,
        name: job.job_id,
        ...(profile
          ? {
              profile
            }
          : {})
      })
      await invalidateRoutineOwner(owner)
    } catch (err) {
      setPendingActive(null)
      host.notifyError(err, c.failedUpdate)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div
      className={cn(
        'group grid gap-1.5 rounded-lg border border-(--ui-stroke-secondary) p-2.5 transition-colors',
        'hover:border-(--ui-stroke-primary, var(--ui-stroke-secondary))'
      )}
    >
      <div className="flex items-center gap-2">
        {/* The row's own button, not a click handler on the card: the switch */
        /* and delete control are siblings, so opening the details can never */
        /* swallow a toggle (and a nested button would be invalid markup). */}
        <RowButton
          className="flex min-w-0 flex-1 items-center gap-2 text-left transition-colors hover:text-foreground"
          onClick={() => onOpen?.(job)}
          title={c.manage}
        >
          {/* `--ui-success` rather than a literal emerald: the token is rotated
              toward the accent, so a column of active dots can't fight the
              theme — the same reason the session status dots use it. */}
          <span
            aria-hidden
            className={cn('size-1.5 shrink-0 rounded-full', active ? 'bg-(--ui-success)' : 'bg-(--ui-text-quaternary)')}
          />
          <span className={cn('min-w-0 flex-1 truncate text-xs font-medium', !active && 'text-(--ui-text-tertiary)')}>
            {routineTitle(job)}
          </span>
        </RowButton>
        <Switch
          checked={active}
          disabled={busy || legacyUnsafe}
          onCheckedChange={value => act(value ? 'resume' : 'pause')}
        />
        <Tip label={t.common.delete}>
          <Button
            aria-label={t.common.delete}
            className="opacity-0 transition-opacity group-hover:opacity-100"
            disabled={busy}
            onClick={() => act('remove')}
            size="icon-xs"
            variant="ghost"
          >
            <Codicon name="trash" />
          </Button>
        </Tip>
      </div>
      <div className="flex items-center justify-between gap-2 pl-3.5">
        <span className="inline-flex items-center gap-1 rounded-full border border-(--ui-stroke-secondary) px-1.5 py-0.5 text-[0.65rem] text-(--ui-text-tertiary)">
          <Codicon className="text-[0.7rem]" name="calendar" />
          {scheduleLabel(job.schedule)}
        </span>
        <span className="truncate text-[0.65rem] text-(--ui-text-quaternary)">
          {active && job.next_run_at
            ? `${c.next} ${relativeTime(new Date(job.next_run_at).getTime())}`
            : c.states.paused}
        </span>
      </div>
      {legacyUnsafe ? (
        <div className="rounded-md border border-(--ui-stroke-secondary) px-2 py-1.5 text-[0.65rem] leading-4 text-(--ui-accent)">
          Paused for security: delete and recreate this legacy job before running it again.
        </div>
      ) : null}
    </div>
  )
}

// Structured schedule picker: frequency first, then only the detail that
// frequency needs (time of day, weekday, day of month, interval). Emits a
// Hermes-native schedule string; Advanced exposes it raw.
type ScheduleFreq = 'once' | 'hourly' | 'daily' | 'weekdays' | 'weekly' | 'monthly' | 'interval' | 'advanced'

/** Picker form state. Every detail field stays a string: they are edited as
 *  raw `Input` text and only coerced when the schedule string is composed. */
interface ScheduleState {
  freq: ScheduleFreq
  intervalN: string
  intervalUnit: string
  monthday: string
  onceN: string
  onceUnit: string
  raw: string
  repeatN: string
  time: string
  weekday: string
}

const FREQUENCIES: Array<{ id: ScheduleFreq; label: string }> = [
  {
    id: 'once',
    label: 'Once, in\u2026'
  },
  {
    id: 'hourly',
    label: 'Every hour'
  },
  {
    id: 'daily',
    label: 'Every day'
  },
  {
    id: 'weekdays',
    label: 'Weekdays'
  },
  {
    id: 'weekly',
    label: 'Every week'
  },
  {
    id: 'monthly',
    label: 'Every month'
  },
  {
    id: 'interval',
    label: 'Interval'
  },
  {
    id: 'advanced',
    label: 'Advanced\u2026'
  }
]

const WEEKDAYS = [
  {
    id: '1',
    label: 'Monday'
  },
  {
    id: '2',
    label: 'Tuesday'
  },
  {
    id: '3',
    label: 'Wednesday'
  },
  {
    id: '4',
    label: 'Thursday'
  },
  {
    id: '5',
    label: 'Friday'
  },
  {
    id: '6',
    label: 'Saturday'
  },
  {
    id: '0',
    label: 'Sunday'
  }
]

const TIMES = (() => {
  const out = []

  for (let h = 0; h < 24; h++) {
    for (const m of [0, 30]) {
      const ampm = h < 12 ? 'AM' : 'PM'
      const h12 = h % 12 === 0 ? 12 : h % 12
      out.push({
        id: `${h}:${m}`,
        label: `${h12}:${String(m).padStart(2, '0')} ${ampm}`,
        h,
        m
      })
    }
  }

  return out
})()

/** Compose the Hermes schedule string from picker state. */
function composeSchedule(state: ScheduleState): string {
  const [h, m] = (state.time || '9:0').split(':').map(Number)

  switch (state.freq) {
    case 'once': {
      const n = Math.max(1, parseInt(state.onceN, 10) || 1)

      return `${n}${state.onceUnit || 'h'}`
    }

    case 'hourly':
      return 'every 1h'

    case 'daily':
      return `${m} ${h} * * *`

    case 'weekdays':
      return `${m} ${h} * * 1-5`

    case 'weekly':
      return `${m} ${h} * * ${state.weekday || '1'}`

    case 'monthly':
      return `${m} ${h} ${state.monthday || '1'} * *`
    case 'interval': {
      const n = Math.max(1, parseInt(state.intervalN, 10) || 1)

      return `every ${n}${state.intervalUnit || 'h'}`
    }

    default:
      return state.raw || ''
  }
}

function scheduleSummary(state: ScheduleState): string {
  const t = TIMES.find(x => x.id === state.time)
  const tl = t ? t.label : '9:00 AM'
  const unitWord = (u: string) => (u === 'm' ? 'minute(s)' : u === 'd' ? 'day(s)' : 'hour(s)')

  const cap =
    state.freq !== 'once' && String(state.repeatN || '').trim()
      ? `, ${Math.max(1, parseInt(state.repeatN, 10) || 1)} time(s) total`
      : ''

  switch (state.freq) {
    case 'once':
      return `Runs once, ${Math.max(1, parseInt(state.onceN, 10) || 1)} ${unitWord(state.onceUnit)} from now`

    case 'hourly':
      return 'Runs at the top of every hour' + cap

    case 'daily':
      return `Runs every day at ${tl}` + cap

    case 'weekdays':
      return `Runs Monday\u2013Friday at ${tl}` + cap

    case 'weekly':
      return `Runs every ${(WEEKDAYS.find(w => w.id === state.weekday) || WEEKDAYS[0]).label} at ${tl}` + cap

    case 'monthly':
      return `Runs on day ${state.monthday || '1'} of each month at ${tl}` + cap

    case 'interval':
      return `Runs every ${Math.max(1, parseInt(state.intervalN, 10) || 1)} ${unitWord(state.intervalUnit)}` + cap

    default:
      return 'Raw schedule \u2014 every Nm/Nh/Nd or 5-field cron'
  }
}

function pickerSelect<T extends string>(
  value: T,
  onChange: (value: T) => void,
  options: Array<{ id: T; label: string }>
) {
  return (
    <Select onValueChange={onChange} value={value}>
      <SelectTrigger className="h-8 rounded-md">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {options.map(o => (
          <SelectItem key={o.id} value={o.id}>
            {o.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

interface SchedulePickerProps {
  setState: React.Dispatch<React.SetStateAction<ScheduleState>>
  state: ScheduleState
}

function SchedulePicker({ state, setState }: SchedulePickerProps) {
  const upd = (patch: Partial<ScheduleState>) =>
    setState(prev => ({
      ...prev,
      ...patch
    }))

  const needsTime = ['daily', 'weekdays', 'weekly', 'monthly'].includes(state.freq)

  return (
    <div className="grid gap-2">
      <div className={cn('grid gap-2', needsTime ? 'grid-cols-2' : 'grid-cols-1')}>
        {pickerSelect(
          state.freq,
          v =>
            upd({
              freq: v
            }),
          FREQUENCIES
        )}
        {needsTime
          ? pickerSelect(
              state.time,
              v =>
                upd({
                  time: v
                }),
              TIMES
            )
          : null}
      </div>
      {state.freq === 'once' ? (
        <div className="grid grid-cols-2 gap-2">
          <Input
            className="h-8"
            onChange={event =>
              upd({
                onceN: event.target.value.replace(/[^0-9]/g, '').slice(0, 4)
              })
            }
            placeholder="30"
            value={state.onceN}
          />
          {pickerSelect(
            state.onceUnit,
            v =>
              upd({
                onceUnit: v
              }),
            [
              {
                id: 'm',
                label: 'minutes from now'
              },
              {
                id: 'h',
                label: 'hours from now'
              },
              {
                id: 'd',
                label: 'days from now'
              }
            ]
          )}
        </div>
      ) : null}
      {state.freq === 'weekly'
        ? pickerSelect(
            state.weekday,
            v =>
              upd({
                weekday: v
              }),
            WEEKDAYS
          )
        : null}
      {state.freq === 'monthly'
        ? labeled(
            'Day of month',
            <Input
              className="h-8"
              onChange={event =>
                upd({
                  monthday: event.target.value.replace(/[^0-9]/g, '').slice(0, 2)
                })
              }
              placeholder="1"
              value={state.monthday}
            />
          )
        : null}
      {state.freq === 'interval' ? (
        <div className="grid grid-cols-2 gap-2">
          <Input
            className="h-8"
            onChange={event =>
              upd({
                intervalN: event.target.value.replace(/[^0-9]/g, '').slice(0, 4)
              })
            }
            placeholder="2"
            value={state.intervalN}
          />
          {pickerSelect(
            state.intervalUnit,
            v =>
              upd({
                intervalUnit: v
              }),
            [
              {
                id: 'm',
                label: 'minutes'
              },
              {
                id: 'h',
                label: 'hours'
              },
              {
                id: 'd',
                label: 'days'
              }
            ]
          )}
        </div>
      ) : null}
      {state.freq === 'advanced' ? (
        <Input
          className="h-8 font-mono text-xs"
          onChange={event =>
            upd({
              raw: event.target.value
            })
          }
          placeholder="every 1d · every 2h · 0 9 * * * (cron)"
          value={state.raw}
        />
      ) : null}
      {state.freq !== 'once' && state.freq !== 'advanced' ? (
        <div className="flex items-center gap-2">
          <span className="text-xs text-(--ui-text-tertiary)">Stop after</span>
          <Input
            className="h-7 w-16 text-xs"
            onChange={event =>
              upd({
                repeatN: event.target.value.replace(/[^0-9]/g, '').slice(0, 4)
              })
            }
            placeholder="∞"
            value={state.repeatN}
          />
          <span className="text-xs text-(--ui-text-tertiary)">runs (blank = forever)</span>
        </div>
      ) : null}
      <div className="text-[0.65rem] text-(--ui-text-quaternary)">{`${scheduleSummary(state)} \u00b7 ${composeSchedule(state) || '\u2014'}`}</div>
    </div>
  )
}

function defaultScheduleState(): ScheduleState {
  return {
    freq: 'daily',
    time: '9:0',
    weekday: '1',
    monthday: '1',
    intervalN: '2',
    intervalUnit: 'h',
    onceN: '30',
    onceUnit: 'm',
    repeatN: '',
    raw: ''
  }
}

interface CreateRoutineDialogProps {
  /** Never nullish: the pane early-returns its empty state before it renders
   *  this dialog, so `createTarget` has already fallen back to the active
   *  profile name — which is the bare-name arm, normalized to a row on entry. */
  bot: RosterRow | string
  onClose: () => void
  open: boolean
}

export function CreateRoutineDialog({ bot, open, onClose }: CreateRoutineDialogProps) {
  const { t } = useI18n()
  const c = t.cron
  const [name, setName] = useState('')
  const [instruction, setInstruction] = useState('')
  const [sched, setSched] = useState(defaultScheduleState())
  const [continuity, setContinuity] = useState(false)
  // Where the run's output lands: 'history' = the run session only (Run
  // history / cron page, today's behavior); 'bot-chat' = inject into this
  // bot's canonical Bot Chat as a real message — the bot reads it, acts on
  // it, and responds there (costs the bot one agent turn per run).
  const [target, setTarget] = useState('history')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<null | string>(null)
  const activeProfile = useValue(host.state.profile)
  // Normalize the bare-name arm to a row once, the way loadRoutines /
  // useRoutines / invalidateRoutineOwner already do. Everything below reads
  // `owner.name` — on a raw string those reads are undefined, which silently
  // costs the dialog its title and costs requestForBot its route.
  const owner: RosterRow = typeof bot === 'string' ? { name: bot } : bot
  const profile = owner.name
  const schedule = composeSchedule(sched)

  const reset = () => {
    setName('')
    setInstruction('')
    setSched(defaultScheduleState())
    setContinuity(false)
    setTarget('history')
    setBusy(false)
    setError(null)
  }

  const submit = async () => {
    const title = name.trim()
    const task = instruction.trim()
    const inputError = routineInputError(title, task)

    if (inputError) {
      setError(inputError)

      return
    }

    if (!title || !task || !schedule.trim() || busy) {
      return
    }

    setBusy(true)
    setError(null)

    try {
      const repeatN =
        sched.freq !== 'once' && sched.freq !== 'advanced' && String(sched.repeatN || '').trim()
          ? Math.max(1, parseInt(sched.repeatN, 10) || 1)
          : null

      await requestForBot(owner, 'cron.manage', {
        action: 'add',
        name: `[bot:${profile}] ${title}`,
        schedule: schedule.trim(),
        prompt: routinePrompt(profile, title, task, activeProfile),
        ...(profile
          ? {
              profile
            }
          : {}),
        ...(repeatN
          ? {
              repeat: repeatN
            }
          : {}),
        ...(continuity
          ? {
              continuity: true
            }
          : {}),
        // 'bot-chat' (bare, no name): the job is created IN the bot's own
        // cron store (profile scoping above), so the scheduler resolves the
        // token to that profile — no cross-gateway name ambiguity possible.
        ...(target === 'bot-chat'
          ? {
              deliver: 'bot-chat'
            }
          : {})
      })
      await invalidateRoutineOwner(owner)
      host.notify({
        kind: 'success',
        message: `${c.created}: ${title}`
      })
      reset()
      onClose()
    } catch (err) {
      setBusy(false)
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  const ownerLabel = displayName(owner, botRosterMeta(owner, $botMeta.get()))

  return (
    <Dialog
      onOpenChange={value => {
        if (!value && !busy) {
          reset()
          onClose()
        }
      }}
      open={open}
    >
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{c.createTitle}</DialogTitle>
          <DialogDescription>{`A recurring task ${ownerLabel} runs on a schedule. Runs land in its own chat history.`}</DialogDescription>
        </DialogHeader>
        <div className="grid gap-3.5">
          {labeled(
            c.nameLabel,
            <Input
              autoFocus
              onChange={event => setName(event.target.value)}
              placeholder={c.namePlaceholder}
              value={name}
            />
          )}
          {labeled(
            c.promptLabel,
            <Textarea
              className="min-h-20"
              onChange={event => setInstruction(event.target.value)}
              placeholder={c.promptPlaceholder}
              value={instruction}
            />
          )}
          {labeled('When to run', <SchedulePicker setState={setSched} state={sched} />)}
          {labeled(
            'Send results to',
            pickerSelect(target, setTarget, [
              {
                id: 'history',
                label: 'Run history only'
              },
              {
                id: 'bot-chat',
                label: `${ownerLabel}\u2019s chat (bot responds)`
              }
            ])
          )}
          <label className="flex items-center gap-2 text-xs text-(--ui-text-tertiary) cursor-pointer select-none">
            <input
              checked={continuity}
              className="accent-(--ui-accent)"
              onChange={event => setContinuity(event.target.checked)}
              type="checkbox"
            />
            Continuity: each run sees the previous run’s output (dedupe, continue where it left off)
          </label>
          {error ? (
            <div className="rounded-md border border-(--ui-stroke-secondary) px-3 py-2 text-xs text-(--ui-accent)">
              {error}
            </div>
          ) : null}
        </div>
        <DialogFooter>
          <Button
            disabled={busy}
            onClick={() => {
              reset()
              onClose()
            }}
            variant="ghost"
          >
            {t.common.cancel}
          </Button>
          <Button disabled={busy || !name.trim() || !instruction.trim() || !schedule.trim()} onClick={submit}>
            {busy ? t.common.saving : c.createAction}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

interface ProfileSyncOwnerRow {
  connectionId?: string
  profile?: string
}

/** `$focusedBotOwner`: the SDK atom on newer desktops, the synthesized
 *  profile-only fallback on older ones. */
interface ProfileSyncStore {
  get?: () => ProfileSyncOwnerRow | null | string | undefined
  listen: (listener: (owner: ProfileSyncOwnerRow | null | string | undefined) => void) => () => void
}

/** Keeps $selectedBot in sync with the focused chat's owner profile.
 *  nanostores' `.listen()` never replays the current value the way
 *  `.subscribe()` does, so a disable → profile switch → re-enable sequence
 *  would otherwise leave $selectedBot pointed at whichever bot was active
 *  before the plugin was disabled — reseeding here on every register() call
 *  closes that gap. Connection qualification keeps same-named owners isolated.
 *  Returns the unbind function for ctx.onDispose. */
export function bindProfileSync(ownerStore: ProfileSyncStore) {
  const sync = (owner: ProfileSyncOwnerRow | null | string | undefined) => {
    const profile = typeof owner === 'string' ? owner : owner?.profile

    if (!profile || typeof profile !== 'string') {
      return
    }

    const connectionId = String(typeof owner === 'object' ? owner?.connectionId || '' : '').trim()
    $selectedBot.set(connectionId ? `${connectionId}::${profile}` : profile)
  }

  sync(ownerStore.get?.())

  return ownerStore.listen(sync)
}

export function resolveRoutineOwner(
  roster: RosterRow[],
  focusedOwner: { authoritative?: boolean; connectionId?: string; name: string } | null,
  selected: string
): RosterRow | null {
  // A null focused owner is NOT a failure: the SDK fails closed to null
  // whenever the focused session has no unique bot owner (a normal chat,
  // ambiguous owner hints) — the common case while the user browses the
  // Bots pane. Fall through to the roster-clicked bot (the previously
  // working scope) instead of dead-ending the pane on the unavailable
  // placeholder for every agent (#94516).
  const selectedBot = roster.find(bot => botSelectionKey(bot) === selected)
  const focusedBot = focusedOwner ? roster.find(bot => isActiveRosterBot(bot, focusedOwner)) : null

  if (focusedOwner?.authoritative) {
    // An authoritative focused owner wins, but only through its exact roster
    // row. If that row is absent, fail closed instead of routing cron
    // reads/mutations through a stale selection or an unscoped profile name.
    return focusedBot || null
  }

  return (
    focusedBot ||
    selectedBot ||
    (focusedOwner
      ? {
          name: focusedOwner.name
        }
      : null)
  )
}

export function RoutinesPane() {
  const selected = useValue($selectedBot)
  const focusedOwner = focusedRosterOwner(useValue($focusedBotOwner))
  // Subscribe instead of a bare read: BotsPane owns the roster fetch and
  // can hydrate (or replace) rows after this pane mounted, so a .get()
  // snapshot captured while the roster was still empty pinned the pane on
  // "unavailable" until some unrelated atom happened to re-render it (#94483).
  // A complete focused owner is still authoritative. If its exact roster row
  // is absent, fail closed rather than routing cron reads/mutations through a
  // stale selection or an unscoped profile name.
  const owner = resolveRoutineOwner(useValue($lastRoster), focusedOwner, selected)
  const bot = String(owner?.name || focusedOwner?.name || 'default').trim() || 'default'
  const allMeta = useValue($botMeta)
  const meta = owner ? botRosterMeta(owner, allMeta) : null
  const { shape, color, image } = botAppearance(bot, meta)
  const { data, error, isLoading, refetch } = useRoutines(owner)
  const { t } = useI18n()
  const c = t.cron
  const [createOpen, setCreateOpen] = useState(false)
  const [createOwner, setCreateOwner] = useState<RosterRow | null>(null)
  // Hold the id, not the record: the 20s poll replaces every job object, and
  // an open inspector must follow the live row (next run, pause, last error)
  // instead of freezing the snapshot that was on screen when it opened.
  const [detailJobId, setDetailJobId] = useState<null | string>(null)
  const createTarget = owner ? routineCreateTarget(createOwner, bot) : null

  const openCreate = () => {
    if (!owner) {
      return
    }

    setCreateOwner(owner)
    setCreateOpen(true)
  }

  if (!owner) {
    return <PanelEmpty description="This agent has to appear in the roster first." icon="hubot" title={c.title} />
  }

  const view = selectRoutineJobs(data, error, $lastJobs.get(), bot)

  if (view.live) {
    $lastJobs.set(view.live)
  }

  const jobs = view.jobs
  const detailJob = detailJobId ? jobs.find(job => job.job_id === detailJobId) || null : null

  const staleNotice =
    error && !view.live && view.all.length ? 'Could not refresh scheduled jobs. Showing the last list we had.' : null

  const filterHint = routineFilterHint(view.all, jobs)

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2 px-3 pt-3 pb-2">
        <BotFace color={avatarColor(color, bot)} image={image} name={bot} shape={shape} size={22} />
        <div className="min-w-0 flex-1">
          <div className="flex min-w-0 items-baseline gap-1.5 truncate">
            <div className="truncate text-xs font-semibold">
              {displayName(
                {
                  name: bot
                },
                meta
              )}
            </div>
            {showsHandle(bot, meta) ? (
              <span className="shrink-0 font-mono text-[0.65rem] text-(--ui-text-quaternary)">{`@${botHandle(bot)}`}</span>
            ) : null}
          </div>
          <div className="text-[0.65rem] uppercase tracking-wider text-(--ui-text-quaternary)">{c.title}</div>
        </div>
        <Tip label={c.newCron}>
          <Button aria-label={c.newCron} onClick={openCreate} size="icon-xs" variant="ghost">
            <Codicon name="add" />
          </Button>
        </Tip>
      </div>
      <div className="mx-3 border-t border-(--ui-stroke-secondary)" />
      {staleNotice ? (
        <div className="mx-3 mt-2 rounded-md bg-(--chrome-action-hover) px-2 py-1.5 text-[0.6875rem] text-(--ui-text-tertiary)">
          {staleNotice}
        </div>
      ) : null}
      {isLoading && !view.all.length ? (
        <div className="flex flex-1 items-center justify-center">
          <GlyphSpinner className="text-(--ui-text-tertiary)" spinner="breathe" />
        </div>
      ) : error && !view.all.length ? (
        <PanelEmpty
          action={
            <Button onClick={() => void refetch()} size="sm" variant="secondary">
              {t.common.retry}
            </Button>
          }
          description="The list may still be there — this was a read failure, not a delete."
          icon="warning"
          title={c.failedLoad}
        />
      ) : jobs.length === 0 ? (
        // `filterHint` is the informative case (jobs exist on the profile but
        // none are tagged for this bot), so it wins the description slot.
        <PanelEmpty
          action={
            <Button onClick={openCreate} size="sm">
              {c.newCron}
            </Button>
          }
          description={filterHint || c.emptyDescNew}
          icon="watch"
          title={c.emptyTitleNew}
        />
      ) : (
        <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain">
          <div className="grid gap-1.5 px-2.5 py-2">
            {jobs.map(job => (
              <RoutineRow job={job} key={job.job_id} onOpen={opened => setDetailJobId(opened.job_id)} owner={owner} />
            ))}
          </div>
        </div>
      )}
      <RoutineDetailDialog job={detailJob} onClose={() => setDetailJobId(null)} open={Boolean(detailJob)} />
      <CreateRoutineDialog
        // Non-null past the `!owner` early return above: `routineCreateTarget`
        // falls back to the active profile name.
        bot={createTarget!}
        // TODO(bot-mode-types): `createTarget` is a roster row whenever a create
        // owner is set, so this key stringifies to "[object Object]" instead of
        // identifying the target bot. Cast to keep the as-written behavior.
        key={createTarget as string}
        onClose={() => {
          setCreateOpen(false)
          setCreateOwner(null)
        }}
        open={createOpen}
      />
    </div>
  )
}
