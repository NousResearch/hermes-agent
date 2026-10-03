import { Button, cn, Codicon, ConfirmDialog, DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger,
  GlyphSpinner, useI18n } from '@hermes/plugin-sdk'
import { useEffect, useRef, useState } from 'react'
import type { ReactNode } from 'react'

import { type CanonicalGroupEvent, CanonicalGroupHistory } from './canonical-group-history'
import { useCanonicalGroupLabels } from './canonical-group-labels'
import { offeredTargets, readSeparateEvents, successionFailure } from './canonical-group-succession'
import type { DesktopComputer, SuccessionBackup, SuccessionComputer, SuccessionPreview, SuccessionStatus } from './canonical-group-succession'
import type { SuccessionController, SuccessionMoveFailure } from './canonical-group-succession-state'
import type { CanonicalGroupBinding, CanonicalGroupRoute, CanonicalRoomMember } from './canonical-groups'
import { useBots } from './i18n'

type Words = ReturnType<typeof useBots>['succession']
type Tone = 'warning' | 'error' | 'info'

const TONE_ICON: Record<Tone, string> = { warning: 'text-primary', error: 'text-destructive', info: 'text-(--ui-text-secondary)' }

/** A room-level strip under the header, like the classic room's hold status: flat, tokenized, one line of title. */
function Strip({ tone, icon, title, children }: { tone: Tone; icon: string; title: ReactNode; children?: ReactNode }) {
  return <div className="shrink-0 border-b border-(--ui-stroke-secondary) bg-(--ui-bg-tertiary)" data-slot="group-succession-banner"
    data-tone={tone} role={tone === 'error' ? 'alert' : 'status'}>
    <div className="mx-auto flex w-full max-w-3xl items-start gap-2.5 px-4 py-2.5">
      <Codicon aria-hidden className={cn('mt-0.5 shrink-0', TONE_ICON[tone])} name={icon} />
      <div className="grid min-w-0 flex-1 gap-1.5 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-secondary)">
        <p className="text-sm font-medium text-(--ui-text-primary)">{title}</p>
        {children}
      </div>
    </div>
  </div>
}

/** Display label from the gateway, else the name Desktop gives its own connection to that computer. */
export function computerName(controller: SuccessionController, computer: SuccessionComputer | null | undefined) {
  return computer?.name ?? controller.computerFor(computer?.install_id)?.label ?? null
}

export function readinessItem(words: Words, backup: SuccessionBackup | undefined, name: string | null) {
  switch (backup?.readiness) {
    case 'caught_up': return words.menuUpToDate(name)

    case 'behind': return words.menuBehind(name, backup.behind_by)

    case 'offline': return words.menuOffline(name)

    case 'needs_reauthorization': return words.needsReauthorization(name)

    default: return words.menuUnconfirmed(name)
  }
}

export function failureText(words: Words, failure: SuccessionMoveFailure, status: SuccessionStatus | null, controller: SuccessionController) {
  const target = computerName(controller, failure.target), host = computerName(controller, status?.host)

  switch (failure.reason) {
    case 'not_owner': return words.onlyOwnerCanContinue(status?.owner.name ?? null)

    case 'host_reachable': return words.errorHostReachable(host)

    case 'room_authority_promised': return words.errorPromised(computerName(controller, failure.other))

    case 'preview_stale': return words.errorPreviewStale

    case 'target_not_ready': return words.errorTargetNotReady(target)

    case 'target_not_local': return words.connectToContinue(target)

    case 'unreachable': return words.errorUnreachable(target)

    default: return words.errorGeneric
  }
}

interface Preparation { target: DesktopComputer; route: CanonicalGroupRoute; preview: SuccessionPreview }

/** The confirmation: what continuing changes, from `prepare` on the target computer. */
function ContinueDialog({ controller, preparation, members, onClose, onPrepared }: {
  controller: SuccessionController; preparation: Preparation | null; members: CanonicalRoomMember[]
  onClose: () => void; onPrepared: (next: Preparation) => void
}) {
  const words = useBots().succession
  const labels = useCanonicalGroupLabels()
  const { locale } = useI18n()
  const status = controller.status
  const preview = preparation?.preview
  const target = preview ? preview.target.name ?? preparation?.target.label ?? null : null
  const host = computerName(controller, status?.host)
  const list = (names: string[]) => new Intl.ListFormat(locale, { type: 'conjunction' }).format(names)
  const bots = preview?.unavailable_bots.map(bot => bot.name ?? members.find(member => member.member_id === bot.member_id)?.display_name ?? labels.unknownBot) ?? []
  const work = preview?.work
  const owner = preview?.owner.name ?? status?.owner.name ?? null
  const operator = preview?.target.operator_name ?? null
  const cautions = preview?.cautions ?? []
  const fencing = cautions.find(caution => caution.code === 'participant_not_fenced')
  const inFlight = useRef(false)

  // One promote at a time: the dialog stays open and busy until it answers, and a second click does nothing.
  return <ConfirmDialog cancelLabel={labels.cancel} confirmLabel={words.confirmContinue(target)} description={words.becomesHost(target)}
    onClose={onClose} onConfirm={async () => {
      if (!preparation || inFlight.current) {return}
      inFlight.current = true

      try {
        await controller.promote(preparation.target, preparation.route, preparation.preview)
      } catch (error) {
        if (successionFailure(error)?.reason === 'preview_stale') {
          onPrepared(await controller.prepare(preparation.target.installId))
          throw new Error(words.errorPreviewStale)
        }

        controller.recordFailure(preparation.preview.target, error)
        throw new Error(words.continueFailed(target, failureText(words, { target: preparation.preview.target,
          reason: successionFailure(error)?.reason ?? 'unreachable', other: successionFailure(error)?.other ?? null }, status, controller)))
      } finally {inFlight.current = false}
    }} open={!!preparation} title={words.confirmTitle(target)}>
    <div className="grid gap-2 text-sm text-(--ui-text-secondary)" data-slot="continue-summary">
      {!!bots.length && <p>{words.botsUnavailable(bots.length, host, list(bots))}</p>}
      {work && (work.completed || work.elsewhere || work.unknown) > 0 && <p>{words.workInProgress(work.completed, work.elsewhere, work.unknown)}</p>}
      {!!preview?.behind_by && <p>{words.targetBehind(target, preview.behind_by, host)}</p>}
      <p>{words.rejoinsAsMember(host)}</p>
      {operator && owner && operator !== owner && <p>{words.managedBy(operator, target)}</p>}
      {fencing && <p>{fencing.count > fencing.names.length ? words.notFencedCount(fencing.count, host)
        : words.notFenced(list(fencing.names), fencing.names.length, host)}</p>}
      {(!preview || !cautions.length || cautions.some(caution => caution.code === 'host_may_be_running')) &&
        <p className="flex items-start gap-2 rounded-md bg-(--ui-bg-tertiary) px-3 py-2 text-(--ui-text-primary)" data-slot="continue-caution">
          <Codicon aria-hidden className="mt-0.5 shrink-0 text-primary" name="warning" />
          <span>{words.hostMayBeRunning(host)}</span>
        </p>}
    </div>
  </ConfirmDialog>
}

function Separate({ binding, branchId, members }: { binding: CanonicalGroupBinding; branchId: string; members: CanonicalRoomMember[] }) {
  const words = useBots().succession
  const labels = useCanonicalGroupLabels()
  const [events, setEvents] = useState<CanonicalGroupEvent[] | null>(null)
  const [failed, setFailed] = useState(false)
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    let current = true
    setFailed(false)
    void (async () => {
      const read: CanonicalGroupEvent[] = []

      for (let after = 0, page = 0; page < 20; page++) {
        const result = await readSeparateEvents(binding, binding.roomId, branchId, after)
        const batch = Array.isArray(result?.events) ? result.events as CanonicalGroupEvent[] : []
        read.push(...batch)
        const next = batch.at(-1)?.seq

        if (result?.has_more !== true || !next || next <= after) {break}
        after = next
      }

      if (current) {setEvents(read)}
    })().catch(() => {if (current) {setFailed(true)}})

    return () => {current = false}
  }, [binding, branchId, attempt])

  return <section aria-label={words.separateEvents} className="shrink-0 border-b border-(--ui-stroke-secondary)" data-slot="group-separate-events">
    <div className="mx-auto max-h-64 w-full max-w-3xl overflow-y-auto px-2 py-1">
      {failed ? <div className="flex items-center gap-2 px-3 py-2 text-xs text-(--ui-text-secondary)" role="alert">
        <span>{labels.invalidLogCursor}</span>
        <Button onClick={() => setAttempt(value => value + 1)} size="inline" variant="text">{labels.retry}</Button>
      </div> : events ? <CanonicalGroupHistory binding={binding} disabled events={events} members={members} />
        : <p className="flex items-center gap-1.5 px-3 py-2 text-xs text-(--ui-text-tertiary)"><GlyphSpinner />{labels.loadingGroup}</p>}
    </div>
  </section>
}

/** The banner for a host that is offline or restarting, a move, a group continued on two computers, and the old host after
 * a move. Nothing shows while the group is normal. */
export function CanonicalGroupSuccessionBanner({ controller, binding, members }: {
  controller: SuccessionController; binding: CanonicalGroupBinding; members: CanonicalRoomMember[]
}) {
  const words = useBots().succession
  const [preparation, setPreparation] = useState<Preparation | null>(null)
  const [preparing, setPreparing] = useState<string | null>(null)
  const [keeping, setKeeping] = useState<SuccessionComputer | null>(null)
  const [showSeparate, setShowSeparate] = useState(false)
  const status = controller.status

  if (!status) {return null}
  const host = computerName(controller, status.host)
  const owner = status.owner.name

  const choose = async (target: SuccessionComputer) => {
    if (preparing) {return}
    setPreparing(target.install_id)
    controller.clearFailure()

    try {
      setPreparation(await controller.prepare(target.install_id))
    } catch (error) {
      controller.recordFailure(target, error)
    } finally {setPreparing(null)}
  }

  const failure = controller.failure ?? (status.last_attempt && { target: status.last_attempt.to, reason: status.last_attempt.error, other: null })

  const failed = failure && <div className="flex flex-wrap items-center gap-2 text-destructive" role="alert">
    <span>{words.continueFailed(computerName(controller, failure.target), failureText(words, failure, status, controller))}</span>
    {controller.computerFor(failure.target.install_id) && failure.reason !== 'not_owner' &&
      <Button disabled={!!preparing} onClick={() => void choose(failure.target)} size="inline" variant="textStrong">{words.tryAgain}</Button>}
  </div>

  const dialog = <ContinueDialog controller={controller} members={members} onClose={() => setPreparation(null)}
    onPrepared={setPreparation} preparation={preparation} />

  if (controller.moving || status.state === 'moving') {
    const target = controller.moving?.label ?? computerName(controller, status.moving?.to)
    const step = status.moving?.step

    const label = step === 'fencing' ? words.stepFencing(host) : step === 'catching_up' ? words.stepCatchingUp
      : step === 'reconciling' ? words.stepReconciling : step === 'finishing' ? words.stepFinishing : null

    return <Strip icon="sync" title={words.continuingOn(target)} tone="info">
      {label && <p className="flex items-center gap-1.5" data-step={step}><GlyphSpinner />{label}</p>}
    </Strip>
  }

  if (status.state === 'host_restarting') {return <Strip icon="debug-restart" title={words.hostRestarting(host)} tone="info" />}

  if (status.state === 'host_unreachable') {
    // The gateway lists targets best first; the first one this Desktop can route to is the primary action.
    const targets = offeredTargets(status, 'continue')
    const computer = (id: string) => status.backups.find(entry => entry.install_id === id) ?? { install_id: id, name: null }
    const best = targets.find(id => controller.computerFor(id))
    const others = targets.filter(id => id !== (best ?? targets[0]))

    const reason = status.unavailable_reason === 'not_owner' ? words.onlyOwnerCanContinue(owner)
      : status.unavailable_reason === 'successor_behind_offline' ? words.successorsOffline(host) : words.noFullCopy(host)

    return <>
      <Strip icon="warning" title={words.hostOffline(host)} tone="warning">
        <p>{words.paused(host)}</p>
        {targets.length ? <div className="flex flex-wrap items-center gap-2 pt-0.5">
          {best ? <Button disabled={!!preparing} loading={preparing === best} onClick={() => void choose(computer(best))} size="sm">
            {words.confirmContinue(computerName(controller, computer(best)))}</Button>
            : <span>{words.connectToContinue(computerName(controller, computer(targets[0])))}</span>}
          {!!others.length && <DropdownMenu>
            <DropdownMenuTrigger asChild><Button disabled={!!preparing} size="sm" variant="secondary">{words.otherComputers}</Button></DropdownMenuTrigger>
            <DropdownMenuContent align="start">
              {others.map(id => {
                const backup = status.backups.find(entry => entry.install_id === id)
                const name = computerName(controller, computer(id))
                const item = readinessItem(words, backup, name)

                return controller.computerFor(id)
                  ? <DropdownMenuItem key={id} onSelect={() => void choose(computer(id))}>{item}</DropdownMenuItem>
                  : <DropdownMenuItem disabled key={id}><span className="grid gap-0.5">
                    <span>{item}</span><span className="text-xs text-(--ui-text-tertiary)">{words.connectToContinue(name)}</span>
                  </span></DropdownMenuItem>
              })}
            </DropdownMenuContent>
          </DropdownMenu>}
        </div> : <p>{reason}</p>}
        {failed}
      </Strip>
      {dialog}
    </>
  }

  if (status.state === 'continued_on_two') {
    const keep = offeredTargets(status, 'keep')
    const named = status.conflict.map((computer, index) => ({ computer, name: computerName(controller, computer) ?? words.computerNumber(index + 1) }))
    const other = named.find(entry => entry.computer.install_id !== keeping?.install_id)

    return <Strip icon="error" title={words.continuedOnTwoTitle} tone="error">
      <p>{words.continuedOnTwoBody(named[0]?.name ?? words.computerNumber(1), named[1]?.name ?? words.computerNumber(2))}</p>
      {keep.length ? <div className="flex flex-wrap items-center gap-2 pt-0.5">
        {named.filter(entry => keep.includes(entry.computer.install_id)).map(entry =>
          <Button key={entry.computer.install_id} onClick={() => setKeeping(entry.computer)} size="sm" variant="secondary">{words.keep(entry.name)}</Button>)}
      </div> : <p>{words.waitingForOwnerChoice(owner)}</p>}
      <ConfirmDialog confirmLabel={keeping ? words.keep(named.find(entry => entry.computer.install_id === keeping.install_id)?.name ?? '') : ''}
        description={other ? words.keepBody(other.name) : undefined} onClose={() => setKeeping(null)}
        onConfirm={async () => {if (keeping) {await controller.keep(keeping.install_id)}}} open={!!keeping}
        title={keeping ? words.keepTitle(named.find(entry => entry.computer.install_id === keeping.install_id)?.name ?? '') : ''} />
    </Strip>
  }

  if (status.state === 'moved_away' && status.moved) {
    const branch = status.moved.branch_id

    return <>
      <Strip icon="info" title={words.movedTo(computerName(controller, status.moved.to))} tone="info">
        <p>{words.movedWhileOffline(computerName(controller, status.this_install), status.moved.separate_events)}</p>
        {!!status.moved.separate_events && branch && <div><Button aria-expanded={showSeparate} onClick={() => setShowSeparate(value => !value)}
          size="inline" variant="textStrong">{showSeparate ? words.hideThem : words.showThem}</Button></div>}
      </Strip>
      {showSeparate && branch && <Separate binding={binding} branchId={branch} members={members} />}
    </>
  }

  // A copy or a backup answered: another computer hosts the room, and Desktop has no connection to it to follow.
  if (status.state === 'ok' && !controller.answeredByHost) {
    return <Strip icon="info" title={words.hostedOn(host)} tone="info"><p>{words.connectToContinue(host)}</p></Strip>
  }

  return null
}

/** Moved away: the old host keeps a copy; chatting continues on the new host. */
export function CanonicalGroupMovedAway({ controller }: { controller: SuccessionController }) {
  const words = useBots().succession
  const moved = controller.status?.moved
  const target = moved ? computerName(controller, moved.to) : null
  const openable = moved && offeredTargets(controller.status, 'open_on').includes(moved.to.install_id) && controller.computerFor(moved.to.install_id)

  return <div className="grid justify-items-start gap-2 rounded-2xl border border-(--ui-stroke-tertiary) p-3 text-sm text-(--ui-text-secondary)"
    data-slot="moved-away-composer">
    <p>{words.movedAwayComposer(target)}</p>
    {openable && moved && <Button onClick={() => controller.openOn(moved.to.install_id)} size="sm">{words.openOn(target)}</Button>}
  </div>
}
