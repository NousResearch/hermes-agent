import { Button, ConfirmDialog, Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, gatewayActivationEpoch, Popover,
  PopoverContent, PopoverTrigger, Switch, useI18n } from '@hermes/plugin-sdk'
import { useState } from 'react'

import { AutomaticSection, automaticSublabel } from './canonical-group-automatic'
import { useCanonicalGroupLabels } from './canonical-group-labels'
import { CONSENT_CONFIRM_MS, offeredTargets, offers } from './canonical-group-succession'
import type { DesktopComputer, SuccessionBackup } from './canonical-group-succession'
import type { SuccessionController } from './canonical-group-succession-state'
import { computerName } from './canonical-group-succession-view'
import { ownership } from './canonical-group-successor-offer'
import { readGroupExecutionMode } from './canonical-groups'
import { useBots } from './i18n'

type Words = ReturnType<typeof useBots>['succession']

function readinessLine(words: Words, backup: SuccessionBackup, name: string | null, locale: string | undefined) {
  switch (backup.readiness) {
    case 'caught_up': return words.backupUpToDate(name)

    case 'behind': return words.backupBehind(name, backup.behind_by)

    case 'offline': return words.backupOffline(name, backup.last_seen
      ? new Intl.DateTimeFormat(locale, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }).format(new Date(backup.last_seen * 1000)) : null)

    case 'unsupported': return words.backupUnsupported(name)

    case 'needs_reauthorization': return words.needsReauthorization(name)

    default: return words.backupUnconfirmed(name)
  }
}

function BackupRow({ controller, backup, onRemove }: { controller: SuccessionController; backup: SuccessionBackup; onRemove: () => void }) {
  const words = useBots().succession
  const { locale } = useI18n()
  const status = controller.status
  const name = computerName(controller, backup)
  const host = computerName(controller, status?.host)
  const pending = controller.switches[backup.install_id]
  const designate = offeredTargets(status, 'designate').includes(backup.install_id)
  const removable = backup.kind === 'backup' && offeredTargets(status, 'remove_backup').includes(backup.install_id)
  // Your own computer can record its operator's consent as well; someone else's computer has to allow it first.
  const blocked = !backup.allowed && !controller.computerFor(backup.install_id)

  const hint = pending?.error ? words.changeFailed : pending ? Date.now() - pending.since >= CONSENT_CONFIRM_MS ? words.waitingToConfirm(host) : words.updating
    : blocked && designate ? words.notAllowed(backup.operator_name, name) : null

  // Someone else's computer, when both names are known and differ: it keeps the whole history. The row already says it
  // keeps a full copy, so a computer whose owner isn't known gets no extra line.
  const guest = ownership(status?.owner.name, backup.operator_name) === 'guest' ? backup.operator_name : null

  return <li className="grid gap-1.5" data-install-id={backup.install_id} data-slot="backup-copy">
    <span className="text-xs text-(--ui-text-primary)"><bdi>{readinessLine(words, backup, name, locale)}</bdi></span>
    {designate && backup.readiness !== 'unsupported' && <label className="flex items-center gap-2 text-xs text-(--ui-text-secondary)">
      <Switch aria-label={words.canContinue} checked={pending && !pending.error ? pending.on : backup.successor}
        disabled={blocked || !!pending && !pending.error} onCheckedChange={on => {
          controller.dismissSwitchError(backup.install_id)
          void controller.setSuccessor(backup.install_id, on).catch(() => undefined)
        }} size="xs" />
      {words.canContinue}
    </label>}
    {hint && <span className={pending?.error ? 'text-xs text-destructive' : 'text-xs text-(--ui-text-tertiary)'} role={pending?.error ? 'alert' : undefined}>{hint}</span>}
    {!hint && automaticSublabel(words, status, backup.always_on, pending && !pending.error ? pending.on : backup.successor) &&
      <span className="text-xs text-(--ui-text-tertiary)">{automaticSublabel(words, status, backup.always_on, pending && !pending.error ? pending.on : backup.successor)}</span>}
    {guest && <span className="text-xs text-(--ui-text-tertiary)" data-slot="guest-history">{words.guestHistory(guest)}</span>}
    {removable && <div><Button onClick={onRemove} size="inline" variant="text">{words.stopKeepingCopy(name)}</Button></div>}
  </li>
}

function AddBackupDialog({ controller, open, onClose }: { controller: SuccessionController; open: boolean; onClose: () => void }) {
  const words = useBots().succession
  const status = controller.status
  const [adding, setAdding] = useState<string | null>(null)
  const [failed, setFailed] = useState<{ id: string; text: string } | null>(null)
  // Someone else's computer keeps the whole history too: said before it's added, and a second Add confirms.
  const [guest, setGuest] = useState<{ id: string; person: string } | null>(null)
  const inGroup = new Set([status?.host.install_id, ...status?.backups.map(backup => backup.install_id) ?? []])
  const candidates = controller.computers.filter(computer => !inGroup.has(computer.installId))
  const host = computerName(controller, status?.host)

  const add = async (computer: DesktopComputer) => {
    if (adding) {return}
    setAdding(computer.connectionId)
    setFailed(null)

    try {
      if (guest?.id !== computer.connectionId) {
        const operator = (await readGroupExecutionMode({ connectionId: computer.connectionId, profile: 'default' }, gatewayActivationEpoch())).operatorName

        if (ownership(controller.operatorName, operator) === 'guest') {return setGuest({ id: computer.connectionId, person: operator! })}
      }

      await controller.addBackup(computer)
      onClose()
    } catch (error) {
      const reason = (error as { roomSetupReason?: string } | null)?.roomSetupReason
      setFailed({ id: computer.connectionId, text: reason === 'backup_gateway_unsupported' ? words.backupUnsupported(computer.label) : words.addBackupFailed(computer.label) })
    } finally {setAdding(null)}
  }

  return <Dialog onOpenChange={value => {if (!value && !adding) {setFailed(null); setGuest(null); onClose()}}} open={open}>
    <DialogContent className="max-w-sm">
      <DialogHeader>
        <DialogTitle>{words.addBackupTitle}</DialogTitle>
        <DialogDescription>{words.addBackupDescription(host)}</DialogDescription>
      </DialogHeader>
      {candidates.length ? <ul className="grid gap-1" data-slot="backup-candidates">
        {candidates.map(computer => <li className="grid gap-1" key={computer.connectionId}>
          <div className="flex items-center gap-3 py-1">
            <span className="min-w-0 flex-1 truncate text-sm"><bdi>{computer.label}</bdi></span>
            <Button aria-label={`${words.addBackupAction}: ${computer.label}`} disabled={!!adding} loading={adding === computer.connectionId}
              onClick={() => void add(computer)} size="sm" variant="secondary">{words.addBackupAction}</Button>
          </div>
          {guest?.id === computer.connectionId && <p className="text-xs text-(--ui-text-secondary)" data-slot="guest-history">{words.guestHistory(guest.person)}</p>}
          {failed?.id === computer.connectionId && <p className="text-xs text-destructive" role="alert">{failed.text}</p>}
        </li>)}
      </ul> : <p className="text-sm text-(--ui-text-secondary)">{words.addBackupEmpty}</p>}
    </DialogContent>
  </Dialog>
}

/** Group info, Backup copies: where the group runs, which computers hold a full copy, and which can continue it.
 * Owner controls come only from the host's `actions`. */
export function CanonicalGroupBackups({ controller, group }: { controller: SuccessionController; group: string }) {
  const words = useBots().succession
  const labels = useCanonicalGroupLabels()
  const { locale } = useI18n()
  const [adding, setAdding] = useState(false)
  const [removing, setRemoving] = useState<SuccessionBackup | null>(null)
  const status = controller.status

  if (!status) {return null}
  const host = computerName(controller, status.host)

  const eligible = status.backups.filter(backup => controller.switches[backup.install_id]?.on ?? backup.successor)
    .map(backup => computerName(controller, backup) ?? words.computerNumber(status.backups.indexOf(backup) + 1))

  const managing = offers(status, 'designate') || offers(status, 'add_backup')
  const removingName = removing ? computerName(controller, removing) : null

  return <>
    <Popover>
      <PopoverTrigger asChild><Button data-slot="group-host" size="inline" variant="text">{words.hostedOn(host)}</Button></PopoverTrigger>
      <PopoverContent align="start" className="max-h-[min(70vh,28rem)] w-80 overflow-y-auto" variant="menu">
        <p className="mb-2 text-xs font-medium text-(--ui-text-secondary)">{words.backupCopies}</p>
        <div className="grid gap-3 text-xs text-(--ui-text-secondary)" data-slot="backup-copies">
          <p className="text-(--ui-text-primary)">{words.hostedOn(host)}</p>
          {!!status.backups.length && <ul aria-label={words.backupCopies} className="grid gap-3">
            {status.backups.map(backup => <BackupRow backup={backup} controller={controller} key={backup.install_id} onRemove={() => setRemoving(backup)} />)}
          </ul>}
          <p>{eligible.length ? words.continueOnList(host, new Intl.ListFormat(locale, { type: 'conjunction' }).format(eligible))
            : managing ? words.nothingCanContinue(host) : words.pausesUntilBack(host)}</p>
          {status.at_risk > 0 && <p>{words.atRisk(status.at_risk, host)}</p>}
          <AutomaticSection controller={controller} group={group}
            onAddBackup={window.hermesDesktop?.roomSetup?.addBackup ? () => setAdding(true) : undefined} />
          {offers(status, 'add_backup') && !!window.hermesDesktop?.roomSetup?.addBackup && <div>
            <Button onClick={() => setAdding(true)} size="sm" variant="secondary">{words.addBackup}</Button>
          </div>}
        </div>
      </PopoverContent>
    </Popover>
    <AddBackupDialog controller={controller} onClose={() => setAdding(false)} open={adding} />
    <ConfirmDialog cancelLabel={labels.cancel} confirmLabel={words.stopKeepingCopyConfirm} description={words.copyWillBeDeleted} destructive
      onClose={() => setRemoving(null)} onConfirm={async () => {
        if (!removing) {return}

        try {await controller.removeBackup(removing.install_id)} catch {throw new Error(words.changeFailed)}
      }} open={!!removing} title={words.stopKeepingCopyTitle(removingName)} />
  </>
}
