/**
 * Presentational parts both room views share (gateway-owned `canonical-group-workspace.tsx`
 * and the Desktop-driven `group-chat-view.tsx`), so the two cannot drift apart. They read as
 * the app's own session chat: transcript rows with avatar + name + time, the main composer's
 * surface and icon buttons, and ghost/icon header actions with the app's tooltips.
 */

import {
  Button,
  cn,
  Codicon,
  composerInputBacking,
  composerSurfaceRing,
  GHOST_ICON_BTN,
  GlyphSpinner,
  PRIMARY_ICON_BTN,
  relativeTime,
  Tip
} from '@hermes/plugin-sdk'
import type { ComponentProps, FormEvent, ReactNode } from 'react'

import { formatElapsed, type MemberStatus, type RoomCycle } from './group-room-status'

/** Plugin copy uses `{name}` style slots so one string table serves every locale. */
export const fill = (template: string, values: Record<string, number | string>) =>
  template.replace(/\{(\w+)\}/g, (_, key: string) => String(values[key] ?? ''))

export interface RoomStatusLabels {
  statusWorking: string
  statusStalled: string
  memberQueued: string
  statusReplied: string
  statusPassed: string
  statusFailed: string
  statusCancelled: string
  statusUnavailable: string
  statusIdle: string
  roundOf: string
  round: string
  cycleWorking: string
  cycleSettled: string
  cycleBounded: string
  cycleMessageCap: string
  cycleStopped: string
  typing: string
  typingTool: string
}

const STATUS_TONE: Record<MemberStatus['kind'], string> = {
  idle: 'text-(--ui-text-quaternary)',
  queued: 'text-(--ui-text-tertiary)',
  working: 'text-(--ui-accent)',
  stalled: 'text-(--ui-warning,var(--ui-text-secondary))',
  replied: 'text-(--ui-text-tertiary)',
  passed: 'text-(--ui-text-quaternary)',
  failed: 'text-destructive',
  cancelled: 'text-(--ui-text-tertiary)',
  unavailable: 'text-destructive'
}

const STATUS_WORD: Record<MemberStatus['kind'], keyof RoomStatusLabels> = {
  idle: 'statusIdle',
  queued: 'memberQueued',
  working: 'statusWorking',
  stalled: 'statusStalled',
  replied: 'statusReplied',
  passed: 'statusPassed',
  failed: 'statusFailed',
  cancelled: 'statusCancelled',
  unavailable: 'statusUnavailable'
}

/** `terminal · 1m20s`, `no progress for 2m05s`, `failed · rate_limited`, `passed`. */
export function memberStatusText(status: MemberStatus, labels: RoomStatusLabels): string {
  if (status.kind === 'working' || status.kind === 'stalled') {
    const parts = [status.tool, status.elapsedS !== undefined ? formatElapsed(status.elapsedS) : undefined].filter(Boolean)
    const lead = status.kind === 'stalled' ? fill(labels.statusStalled, { time: formatElapsed(status.quietS ?? 0) }) : ''

    return [lead, ...parts].filter(Boolean).join(' · ') || labels.statusWorking
  }

  const word = labels[STATUS_WORD[status.kind]]
  const reason = 'reason' in status ? status.reason : undefined

  return reason ? `${word} · ${reason}` : word
}

export function MemberStatusChip({ avatar, badge, labels, name, phase, status }: {
  avatar: ReactNode; badge?: string; labels: RoomStatusLabels; name: string; phase?: string; status: MemberStatus
}) {
  const busy = status.kind === 'working' || status.kind === 'stalled'
  const text = memberStatusText(status, labels)
  const detail = 'detail' in status && status.detail ? status.detail : undefined

  return <Tip label={detail ? `${name} · ${detail}` : `${name} · ${text}`}>
    <li className="inline-flex min-w-0 items-center gap-1.5 rounded-full bg-(--chrome-action-hover) py-0.5 pl-0.5 pr-2 text-[0.6875rem]"
      data-member-status={status.kind} data-phase={phase} tabIndex={0}>
      <span className="shrink-0">{avatar}</span>
      <span className="shrink-0 font-medium text-foreground">{name}</span>
      {badge && <span className="shrink-0 rounded-sm bg-(--ui-accent)/15 px-1 text-[0.625rem] font-medium text-(--ui-accent)" data-member-badge="">{badge}</span>}
      {busy && <span aria-hidden="true" className={STATUS_TONE[status.kind]}><GlyphSpinner /></span>}
      <span className={cn('min-w-0 truncate', STATUS_TONE[status.kind])}>{text}</span>
    </li>
  </Tip>
}

/** One transcript row: avatar, name and time over the markdown body, like the main thread.
 *  The user's own rows wear the main chat's user-bubble fill. */
export function RoomMessage({ actions, author, avatar, children, isUser = false, timestampMs }: {
  actions?: ReactNode; author?: ReactNode; avatar?: ReactNode; children: ReactNode; isUser?: boolean; timestampMs?: number
}) {
  return <div className={cn('group/room-message flex items-start gap-2.5 px-3 py-2', isUser && 'rounded-xl bg-(--dt-user-bubble)')}
    data-room-message={isUser ? 'user' : 'member'}>
    {avatar && <div className="mt-0.5 shrink-0">{avatar}</div>}
    <div className="min-w-0 flex-1">
      <div className="flex min-h-5 items-center gap-2">
        {author && <span className="truncate text-xs font-semibold text-foreground">{author}</span>}
        {typeof timestampMs === 'number' && Number.isFinite(timestampMs) &&
          <time className="shrink-0 text-[0.6875rem] text-(--ui-text-quaternary)" dateTime={new Date(timestampMs).toISOString()}>{relativeTime(timestampMs)}</time>}
        {actions && <div className="ml-auto flex shrink-0 items-center gap-0.5 opacity-0 transition-opacity group-hover/room-message:opacity-100 focus-within:opacity-100">{actions}</div>}
      </div>
      <div className="wrap-anywhere min-w-0 text-pretty text-[length:var(--conversation-text-font-size)] leading-(--dt-line-height) text-foreground [&_pre]:whitespace-pre-wrap" data-selectable-text="true">
        {children}
      </div>
    </div>
  </div>
}

/** A compact centred activity row between messages (handoffs, notices). */
export function RoomActivityRow({ children, icon = 'arrow-small-right', tone }: { children: ReactNode; icon?: string; tone?: string }) {
  return <div className={cn('flex items-center justify-center gap-1.5 px-3 py-0.5 text-[0.6875rem] text-(--ui-text-tertiary)', tone)} data-room-activity="">
    <Codicon className="shrink-0" name={icon} size="0.8125rem" />
    <span className="wrap-anywhere">{children}</span>
  </div>
}

/** The typing-style placeholder a working member gets at the end of the transcript. */
export function RoomTypingRow({ avatar, labels, name, status }: {
  avatar: ReactNode; labels: RoomStatusLabels; name: string; status: Extract<MemberStatus, { kind: 'stalled' | 'working' }>
}) {
  const line = status.tool ? fill(labels.typingTool, { name, tool: status.tool }) : fill(labels.typing, { name })

  const tail = status.kind === 'stalled'
    ? fill(labels.statusStalled, { time: formatElapsed(status.quietS ?? 0) })
    : status.elapsedS !== undefined ? formatElapsed(status.elapsedS) : ''

  return <div className="flex items-start gap-2.5 px-3 py-2" data-room-typing={status.kind}>
    <div className="mt-0.5 shrink-0">{avatar}</div>
    <div className="min-w-0">
      {/* The line already names the member; a separate name label would repeat it. */}
      <div className="inline-flex items-center gap-2 rounded-2xl bg-(--chrome-action-hover) px-3 py-1.5">
        <span aria-hidden="true" className="inline-flex gap-1">
          {[0, 160, 320].map(delay => <span className="size-1.5 animate-pulse rounded-full bg-(--ui-text-tertiary)" key={delay} style={{ animationDelay: `${delay}ms` }} />)}
        </span>
        <span className="text-xs text-(--ui-text-secondary)">{line}</span>
        {tail && <span className={cn('text-[0.6875rem]', status.kind === 'stalled' ? STATUS_TONE.stalled : 'text-(--ui-text-quaternary)')}>{tail}</span>}
      </div>
    </div>
  </div>
}

/** `Round 2 of 3 · Working: builder (terminal, 1m20s)` while members work, else how it ended. */
export function roomCycleText(cycle: RoomCycle, labels: RoomStatusLabels, working: string[]): string {
  if (cycle.kind !== 'working') {
    return labels[({ settled: 'cycleSettled', bounded: 'cycleBounded', messageCap: 'cycleMessageCap', stopped: 'cycleStopped' } as const)[cycle.kind]]
  }

  const round = cycle.maxRounds ? fill(labels.roundOf, { n: cycle.round, max: cycle.maxRounds }) : fill(labels.round, { n: cycle.round })

  return working.length ? `${round} · ${fill(labels.cycleWorking, { names: working.join(', ') })}` : round
}

export function RoomCycleFooter({ cycle, labels, text, working }: {
  cycle: null | RoomCycle; labels: RoomStatusLabels; text?: string; working: string[]
}) {
  if (!cycle) {return null}

  return <div className="flex items-center justify-center gap-1.5 px-3 py-1 text-[0.6875rem] text-(--ui-text-quaternary)" data-room-cycle={cycle.kind} role="status">
    {cycle.kind === 'working' && <span aria-hidden="true"><GlyphSpinner /></span>}
    <span>{text ?? roomCycleText(cycle, labels, working)}</span>
  </div>
}

/** The room header: title (opens details), member avatar stack + status chips, Stop while
 *  the room works, and the overflow/details trigger. */
export function RoomHeader({ backLabel, children, onBack, onStop, onTitle, stopLabel, title, titleExpanded, titleLabel, trailing, working }: {
  backLabel?: string; children?: ReactNode; onBack?: () => void; onStop?: () => void; onTitle?: () => void
  stopLabel: string; title: ReactNode; titleExpanded?: boolean; titleLabel?: string; trailing?: ReactNode; working: boolean
}) {
  return <header className="flex min-w-0 flex-col gap-1.5 border-b border-(--ui-stroke-tertiary) px-3 py-2">
    <div className="flex min-w-0 items-center gap-1">
      {onBack && <Tip label={backLabel}>
        <Button aria-label={backLabel} className="text-(--ui-text-tertiary)" onClick={onBack} size="icon-sm" variant="ghost"><Codicon name="arrow-left" /></Button>
      </Tip>}
      {onTitle
        ? <Tip label={titleLabel}>
          <Button aria-expanded={titleExpanded} aria-label={titleLabel} className="min-w-0 gap-1 text-sm font-semibold text-foreground" onClick={onTitle} size="inline" variant="text">
            <span className="truncate">{title}</span><Codicon className="shrink-0 text-(--ui-text-tertiary)" name={titleExpanded ? 'chevron-left' : 'chevron-right'} size="0.75rem" />
          </Button>
        </Tip>
        : <h2 className="min-w-0 truncate text-sm font-semibold text-foreground">{title}</h2>}
      <div className="ml-auto flex shrink-0 items-center gap-0.5">
        {working && onStop && <Tip label={stopLabel}>
          <Button aria-label={stopLabel} className="text-(--ui-accent)" onClick={onStop} size="icon-sm" variant="ghost"><Codicon name="debug-stop" /></Button>
        </Tip>}
        {trailing}
      </div>
    </div>
    {children}
  </header>
}

/** Up to `max` overlapping avatars, the rest as `+N`. */
export function AvatarStack({ avatars, max = 4 }: { avatars: { key: string; node: ReactNode }[]; max?: number }) {
  const shown = avatars.slice(0, max)

  return <span className="inline-flex shrink-0 items-center -space-x-1.5" data-avatar-stack="">
    {shown.map(({ key, node }) => <span className="rounded-full ring-2 ring-(--ui-bg-primary,var(--dt-background))" key={key}>{node}</span>)}
    {avatars.length > max && <span className="pl-2.5 text-[0.625rem] text-(--ui-text-tertiary)">+{avatars.length - max}</span>}
  </span>
}

/** The main composer's surface around a plugin input: ring, fill, ghost attach on the left,
 *  the round primary send on the right. Enter sends (the input owns that). `attach` renders
 *  straight into the surface grid: its parts pick `ROOM_COMPOSER_AREAS` (the attach button,
 *  the pending-file chips). */
export function RoomComposer({ above, attach, canSend, children, onSubmit, sendLabel, ...form }: {
  above?: ReactNode; attach?: ReactNode; canSend: boolean; children: ReactNode; onSubmit: () => void; sendLabel: string
} & Omit<ComponentProps<'form'>, 'onSubmit'>) {
  return <form {...form} className={cn('px-3 pb-3 pt-1', form.className)} onSubmit={(event: FormEvent) => { event.preventDefault(); onSubmit() }}>
    <div className={cn('relative isolate rounded-2xl', composerSurfaceRing)} data-slot="room-composer-surface">
      <div aria-hidden className={composerInputBacking} />
      <div className="grid grid-cols-[auto_minmax(0,1fr)_auto] items-end gap-x-(--composer-control-gap) gap-y-1 px-(--composer-surface-pad-x) py-(--composer-surface-pad-y) [grid-template-areas:'reply_reply_reply''files_files_files''attach_input_send']">
        {above && <div className="min-w-0 [grid-area:reply]">{above}</div>}
        {attach}
        <div className="min-w-0 [grid-area:input]">{children}</div>
        <Tip label={sendLabel}>
          <Button aria-label={sendLabel} className={cn(PRIMARY_ICON_BTN, 'mb-1 [grid-area:send]')} disabled={!canSend} type="submit">
            <Codicon name="arrow-up" size="0.875rem" />
          </Button>
        </Tip>
      </div>
    </div>
  </form>
}

/** Where an attachment control sits inside a `RoomComposer` surface. */
export const ROOM_COMPOSER_AREAS = { button: 'mb-1 [grid-area:attach]', files: 'min-w-0 [grid-area:files]' } as const

/** The ghost `+` the main composer uses for attachments. */
export function RoomAttachButton({ disabled, label, onClick }: { disabled?: boolean; label: string; onClick: () => void }) {
  return <Tip label={label}>
    <Button aria-label={label} className={GHOST_ICON_BTN} disabled={disabled} onClick={onClick} size="icon" type="button" variant="ghost">
      <Codicon name="add" size="0.875rem" />
    </Button>
  </Tip>
}

/** `Replying to Ada: first line…` above the composer, dismissible. */
export function RoomReplyChip({ cancelLabel, label, onCancel, quote }: { cancelLabel: string; label: string; onCancel: () => void; quote?: string }) {
  return <div className="flex min-w-0 items-center gap-1.5 rounded-lg bg-(--chrome-action-hover) px-2 py-1 text-[0.6875rem]" data-room-reply="">
    <Codicon className="shrink-0 text-(--ui-text-tertiary)" name="reply" size="0.75rem" />
    <span className="shrink-0 font-medium text-foreground">{label}</span>
    {quote && <span className="min-w-0 truncate text-(--ui-text-tertiary)">{quote}</span>}
    <Button aria-label={cancelLabel} className="ml-auto" onClick={onCancel} size="icon-xs" type="button" variant="ghost"><Codicon name="close" /></Button>
  </div>
}

/** Borderless input inside a `RoomComposer` surface (the surface owns the ring). */
export const ROOM_COMPOSER_INPUT_CLASS =
  '!border-transparent !bg-transparent !shadow-none min-h-(--composer-input-min-height) px-1 py-1 text-[length:var(--conversation-text-font-size)] leading-(--dt-line-height) placeholder:text-(--ui-text-tertiary)'

