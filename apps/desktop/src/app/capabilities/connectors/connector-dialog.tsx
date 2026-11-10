import { type ReactNode, type RefObject, useRef } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ConnectorLogo } from '@/components/ui/connector-logo'
import { Dialog, DialogClose, DialogContent, DialogDescription, DialogTitle } from '@/components/ui/dialog'
import { Switch } from '@/components/ui/switch'
import { useI18n } from '@/i18n'
import { connectorIconUrl } from '@/lib/connector-tools'
import { X } from '@/lib/icons'
import { cn } from '@/lib/utils'

import './connectors.css'

import { CatalogMark } from './catalog-mark'
import { connectorKindWord, showsCatalogMark } from './connector-kind'
import { type InstallField, LocalInstall } from './local-server-control'
import type { ConnectorCardModel, ConnectorState, ConnectorVerb, ConnectorWayHosted, ConnectorWayLocal } from './types'
import { type WayChoice, WaysSection } from './ways-section'

type BadgeVariant = 'default' | 'destructive' | 'muted' | 'success' | 'warn'

const STATE_BADGE = {
  available: 'muted',
  broken: 'destructive',
  connected: 'success',
  connecting: 'warn',
  expired: 'warn',
  off: 'muted',
  unknown: 'muted'
} satisfies Record<ConnectorState, BadgeVariant>

const RULEABLE = {
  available: false,
  broken: true,
  connected: true,
  connecting: false,
  expired: true,
  off: true,
  unknown: true
} satisfies Record<ConnectorState, boolean>

const ruleable = (way: ConnectorWayHosted | null): boolean => way !== null && way.connected && RULEABLE[way.state]

export interface ConnectorDialogProps {
  advanced?: ReactNode
  card: ConnectorCardModel
  connectElement?: ReactNode
  cost?: { tokensPerCall?: string; usesPerMonth?: string }
  installFields?: readonly InstallField[]
  installing?: boolean
  menu?: ReactNode
  onAuthenticate?: () => void
  onConnect?: () => void
  onDisconnect?: () => void
  onInstall?: (env: Record<string, string>) => void
  onOpenAdmin?: () => void
  onOpenChange: (open: boolean) => void
  onReconnect?: () => void
  onServerToggle?: (next: boolean) => void
  onToggleForMe?: (next: boolean) => void
  onVerb?: () => void
  onWayChange?: (way: WayChoice) => void
  open: boolean
  orgDisabledCount?: number
  rulesReadOnly?: boolean
  togglePending?: boolean
  tools: ReactNode
  way?: WayChoice
}

const localTarget = (card: ConnectorCardModel): string | undefined => card.ways.local?.target

export function ConnectorDialog({ card, onOpenChange, open, tools, ...rest }: ConnectorDialogProps) {
  const { t } = useI18n()
  const local = card.residency === 'local'
  const titleRef = useRef<HTMLHeadingElement>(null)

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent
        bodyClassName="gap-0 overflow-hidden p-0"
        className="max-h-[85vh] max-w-xl"
        data-connector-dialog=""
        onOpenAutoFocus={event => {
          event.preventDefault()
          titleRef.current?.focus()
        }}
        showCloseButton={false}
      >
        <Header card={card} titleRef={titleRef} {...rest} />

        <div className="flex min-h-0 flex-1 flex-col bg-[#f7f9fc] dark:bg-transparent">
          {card.ways.hosted ? <HostedLead card={card} {...rest} /> : <LocalLead card={card} {...rest} />}

          {tools}

          {local ? <LocalFoot card={card} {...rest} /> : <HostedFoot card={card} {...rest} />}
        </div>

        <span className="sr-only">{t.connectorsPage.title}</span>
      </DialogContent>
    </Dialog>
  )
}

type PartProps = Omit<ConnectorDialogProps, 'onOpenChange' | 'open' | 'tools'>

function Header({ card, titleRef, ...rest }: PartProps & { titleRef: RefObject<HTMLHeadingElement | null> }) {
  const { t } = useI18n()
  const copy = t.connectorsPage.card
  const local = card.residency === 'local'

  return (
    <header
      className="flex shrink-0 items-center gap-3 px-5 py-4"
      data-connector-dialog-header=""
    >
      <ConnectorLogo
        className="size-11 shrink-0 rounded-[12px] border border-[#e2e8f0] shadow-[0_4px_12px_rgba(38,64,140,0.08)]"
        connector={{ iconUrl: local ? undefined : connectorIconUrl(card.slug), name: card.slug, title: card.name }}
      />

      <div className="grid min-w-0 flex-1 gap-1">
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          <DialogTitle
            className="truncate text-[18px] font-bold leading-none text-[#0f172a] outline-none dark:text-foreground"
            ref={titleRef}
            tabIndex={-1}
          >
            {card.name}
          </DialogTitle>

          <span className="shrink-0 text-[12px] font-medium text-[#64748b]">{connectorKindWord(card, copy)}</span>

          {showsCatalogMark(card) ? <CatalogMark /> : null}

          <Badge
            className={cn(
              'shrink-0 rounded-lg px-2 py-[3px] text-[11px] font-semibold',
              card.state === 'broken'
                ? 'border border-[#fecaca] bg-[#fef2f2] text-[#dc2626]'
                : card.state === 'connected'
                  ? 'border border-[#9db7f0] bg-[#e3ecfd] text-[#2f5bea]'
                  : undefined
            )}
            size="xs"
            variant={STATE_BADGE[card.state]}
          >
            {copy.state[card.stateWord]}
          </Badge>
        </div>

        <DialogDescription className="truncate text-[13px] text-[#475569] dark:text-(--ui-text-secondary)">
          {card.description ?? localTarget(card) ?? copy.state[card.stateWord]}
        </DialogDescription>
      </div>

      <HeaderActions card={card} {...rest} />
    </header>
  )
}

function HeaderActions({
  card,
  connectElement,
  installFields,
  installing,
  menu,
  onAuthenticate,
  onInstall,
  onServerToggle,
  onToggleForMe,
  onVerb,
  rulesReadOnly = false,
  togglePending = false
}: PartProps) {
  const { t } = useI18n()
  const copy = t.connectorsPage.card
  const local = card.residency === 'local'
  const hosted = card.ways.hosted
  const localWay = card.ways.local
  const appSwitch = ruleable(hosted) && onToggleForMe !== undefined

  const action = hosted
    ? localWay
      ? undefined
      : leadVerb({ appSwitch, card, hasElement: connectElement !== undefined, onVerb })
    : localAction({ installFields, installing, onAuthenticate, onInstall, way: localWay })

  return (
    <div className="flex shrink-0 items-center gap-2">
      {action ? (
        <Button disabled={action.busy} onClick={action.run} size="xs">
          {copy.verb[action.verb]}
        </Button>
      ) : null}

      {local && !hosted && localWay?.installed === true && card.plugin === undefined && onServerToggle ? (
        <Switch
          aria-label={localWay.serverEnabled === true ? copy.turnServerOff(card.name) : copy.turnServerOn(card.name)}
          checked={localWay.serverEnabled ?? false}
          className="h-[22px] w-10 data-[state=checked]:border-transparent data-[state=checked]:bg-[#7c3aed] [&_[data-slot=switch-thumb]]:size-[18px] data-[state=checked]:[&_[data-slot=switch-thumb]]:translate-x-[18px]"
          onCheckedChange={onServerToggle}
          size="default"
        />
      ) : null}

      {appSwitch && hosted && onToggleForMe ? (
        <Switch
          aria-label={t.connectorsPage.dialog.appSwitch(card.name)}
          checked={hosted.state !== 'off'}
          className="h-[22px] w-10 data-[state=checked]:border-transparent data-[state=checked]:bg-[#7c3aed] [&_[data-slot=switch-thumb]]:size-[18px] data-[state=checked]:[&_[data-slot=switch-thumb]]:translate-x-[18px]"
          disabled={card.offBy === 'org' || togglePending || rulesReadOnly}
          onCheckedChange={onToggleForMe}
          size="default"
        />
      ) : null}

      {menu}

      <DialogClose asChild>
        <Button aria-label={t.common.close} size="icon-xs" variant="ghost">
          <X className="size-3" />
        </Button>
      </DialogClose>
    </div>
  )
}

interface LeadVerb {
  busy?: boolean
  run: () => void
  verb: ConnectorVerb
}

function localAction({
  installFields,
  installing = false,
  onAuthenticate,
  onInstall,
  way
}: {
  installFields?: readonly InstallField[]
  installing?: boolean
  onAuthenticate?: () => void
  onInstall?: (env: Record<string, string>) => void
  way: ConnectorWayLocal | null
}): LeadVerb | undefined {
  if (!way) {
    return undefined
  }

  if (way.installed === false) {
    return onInstall && (installFields ?? []).length === 0
      ? { busy: installing, run: () => onInstall({}), verb: 'install' }
      : undefined
  }

  return way.verb === 'authenticate' && onAuthenticate ? { run: onAuthenticate, verb: 'authenticate' } : undefined
}

function leadVerb({
  appSwitch,
  card,
  hasElement,
  onVerb
}: {
  appSwitch: boolean
  card: ConnectorCardModel
  hasElement: boolean
  onVerb?: () => void
}): LeadVerb | undefined {
  const verb = card.verb

  if (
    hasElement ||
    verb === undefined ||
    verb === 'stopWaiting' ||
    onVerb === undefined ||
    (verb === 'turnBackOn' && appSwitch)
  ) {
    return undefined
  }

  return { run: onVerb, verb }
}

function HostedLead({
  card,
  connectElement,
  installFields,
  installing,
  onAuthenticate,
  onConnect,
  onInstall,
  onOpenAdmin,
  onReconnect,
  onServerToggle,
  onToggleForMe,
  onVerb,
  onWayChange,
  orgDisabledCount = 0,
  way = 'hosted'
}: PartProps) {
  const { t } = useI18n()
  const waiting = card.verb === 'stopWaiting' && connectElement === undefined ? onVerb : undefined
  const reason = card.reason ? (card.reason.text ?? t.connectorsPage.card.reason[card.reason.key]) : undefined
  const paired = card.ways.local !== null
  const showsReason = reason !== undefined && connectElement === undefined && (!paired || waiting !== undefined)

  if (!paired && connectElement === undefined && !showsReason && orgDisabledCount <= 0) {
    return null
  }

  return (
    <div className="grid shrink-0 gap-3 border-b border-[#e2e8f0] bg-white px-5 py-3 dark:border-(--ui-stroke-tertiary) dark:bg-transparent">
      {connectElement}

      {showsReason ? (
        <div className="flex items-center gap-3 px-3 py-2" data-connector-alert="">
          <p className="min-w-0 flex-1 text-[13px] text-[#475569]">{reason}</p>
          {waiting ? (
            <Button onClick={waiting} size="inline" variant="textStrong">
              {t.connectorsPage.card.verb.stopWaiting}
            </Button>
          ) : null}
        </div>
      ) : null}

      <OrgNote count={orgDisabledCount} onOpenAdmin={onOpenAdmin} />

      {onWayChange ? (
        <WaysSection
          card={card}
          installFields={installFields}
          installing={installing}
          onAuthenticate={onAuthenticate}
          onChange={onWayChange}
          onConnect={onConnect}
          onInstall={onInstall}
          onReconnect={onReconnect}
          onServerToggle={card.plugin === undefined ? onServerToggle : undefined}
          onToggleForMe={onToggleForMe}
          value={way}
        />
      ) : null}
    </div>
  )
}

function HostedFoot({ card }: PartProps) {
  const { t } = useI18n()

  return card.ways.hosted ? <FootLine>{t.connectorsPage.dialog.nousLine}</FootLine> : null
}

function LocalLead({ card, installFields = [], installing, onInstall }: PartProps) {
  const local = card.ways.local
  const install = local?.installed === false && installFields.length > 0 ? onInstall : undefined

  if (!install) {
    return null
  }

  return (
    <div className="shrink-0 border-b border-[#e2e8f0] bg-white px-5 py-3 dark:border-(--ui-stroke-tertiary) dark:bg-transparent">
      <LocalInstall installFields={installFields} installing={installing} onInstall={install} />
    </div>
  )
}

function LocalFoot({ advanced, cost }: PartProps) {
  const { t } = useI18n()
  const copy = t.connectorsPage.dialog
  const metrics = cost && (cost.tokensPerCall || cost.usesPerMonth)

  if (!metrics && !advanced) {
    return null
  }

  return (
    <div className="grid shrink-0 gap-3 border-t border-[#e2e8f0] bg-white px-5 py-4 dark:border-(--ui-stroke-tertiary) dark:bg-transparent">
      {metrics ? (
        <p className="text-[12px] font-medium text-[#64748b]">
          {[
            cost?.usesPerMonth ? `${cost.usesPerMonth} ${copy.usesPerMonth}` : null,
            cost?.tokensPerCall ? `${cost.tokensPerCall} ${copy.tokensPerCall}` : null
          ]
            .filter(Boolean)
            .join(' · ')}
        </p>
      ) : null}

      {advanced ? (
        <details className="group grid gap-3">
          <summary className="flex cursor-pointer list-none items-center gap-2 text-[13px] font-semibold text-[#0f172a] dark:text-foreground">
            <Codicon
              className={cn('shrink-0 text-[#64748b] transition-transform duration-100 group-open:rotate-90')}
              name="chevron-right"
              size="0.8rem"
            />
            <span className="shrink-0">{copy.advanced}</span>
            <span className="min-w-0 truncate font-medium text-[#94a3b8]">{copy.advancedHint}</span>
          </summary>
          <div className="pt-1">{advanced}</div>
        </details>
      ) : null}
    </div>
  )
}

function FootLine({ children }: { children: string }) {
  return (
    <p className="shrink-0 border-t border-[#e2e8f0] bg-white px-5 py-3 text-[12px] text-[#64748b] dark:border-(--ui-stroke-tertiary) dark:bg-transparent">
      {children}
    </p>
  )
}

function OrgNote({ count, onOpenAdmin }: { count: number; onOpenAdmin?: () => void }) {
  const { t } = useI18n()
  const copy = t.connectorsPage.dialog

  if (count <= 0) {
    return null
  }

  return (
    <div className="grid gap-1.5 rounded-xl border border-[#fde68a] bg-[#fffbeb] px-3 py-2.5 dark:border-transparent dark:bg-(--ui-orange)/8">
      <p className="text-[13px] text-[#92400e] dark:text-(--ui-text-secondary)">{copy.orgNote(count)}</p>
      {onOpenAdmin ? (
        <Button className="justify-self-start text-[13px] font-semibold text-[#2f5bea]" onClick={onOpenAdmin} size="inline" variant="textStrong">
          {copy.orgLink}
        </Button>
      ) : null}
    </div>
  )
}
