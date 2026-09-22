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

import { CatalogMark } from './catalog-mark'
import { connectorKindWord, showsCatalogMark } from './connector-kind'
import { type InstallField, LocalInstall, LocalServerControl } from './local-server-control'
import type { ConnectorCardModel, ConnectorState, ConnectorVerb, ConnectorWayHosted } from './types'
import { WaysSection } from './ways-section'

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
  broken: false,
  connected: true,
  connecting: false,
  expired: false,
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
  open: boolean
  orgDisabledCount?: number
  rulesReadOnly?: boolean
  togglePending?: boolean
  tools: ReactNode
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
        className="max-h-[85vh] max-w-2xl"
        onOpenAutoFocus={event => {
          event.preventDefault()
          titleRef.current?.focus()
        }}
        showCloseButton={false}
      >
        <Header card={card} titleRef={titleRef} {...rest} />

        <div className="flex min-h-0 flex-1 flex-col">
          {local ? <LocalLead card={card} {...rest} /> : <HostedLead card={card} {...rest} />}

          {tools}

          {local ? <LocalFoot card={card} {...rest} /> : <HostedFoot card={card} {...rest} />}
        </div>

        <span className="sr-only">{t.connectorsPage.title}</span>
      </DialogContent>
    </Dialog>
  )
}

type PartProps = Omit<ConnectorDialogProps, 'onOpenChange' | 'open' | 'tools'>

function Header({
  card,
  menu,
  onToggleForMe,
  rulesReadOnly = false,
  titleRef,
  togglePending = false
}: PartProps & { titleRef: RefObject<HTMLHeadingElement | null> }) {
  const { t } = useI18n()
  const copy = t.connectorsPage.card
  const local = card.residency === 'local'
  const hosted = card.ways.hosted
  const appSwitch = ruleable(hosted) && onToggleForMe !== undefined

  return (
    <header className="flex shrink-0 items-center gap-2.5 border-b border-(--ui-stroke-tertiary) px-5 py-3">
      <ConnectorLogo
        className="size-9 shrink-0 rounded-[9px]"
        connector={{ iconUrl: local ? undefined : connectorIconUrl(card.slug), name: card.slug, title: card.name }}
      />

      <div className="grid min-w-0 flex-1 gap-0.5">
        <div className="flex min-w-0 items-center gap-2">
          <DialogTitle className="truncate text-base font-semibold outline-none" ref={titleRef} tabIndex={-1}>
            {card.name}
          </DialogTitle>

          <span className="shrink-0 text-[0.6875rem] text-(--ui-text-tertiary)">{connectorKindWord(card, copy)}</span>

          {showsCatalogMark(card) ? <CatalogMark /> : null}

          <Badge className="shrink-0" size="xs" variant={STATE_BADGE[card.state]}>
            {copy.state[card.stateWord]}
          </Badge>
        </div>

        <DialogDescription className="truncate text-[0.72rem] text-(--ui-text-secondary)">
          {card.description ?? localTarget(card) ?? copy.state[card.stateWord]}
        </DialogDescription>
      </div>

      <div className="flex shrink-0 items-center gap-2">
        {appSwitch && hosted && onToggleForMe ? (
          <Switch
            aria-label={t.connectorsPage.dialog.appSwitch(card.name)}
            checked={hosted.state !== 'off'}
            disabled={card.offBy === 'org' || togglePending || rulesReadOnly}
            onCheckedChange={onToggleForMe}
            size="xs"
          />
        ) : null}

        {menu}

        <DialogClose asChild>
          <Button aria-label={t.common.close} size="icon-xs" variant="ghost">
            <X className="size-3" />
          </Button>
        </DialogClose>
      </div>
    </header>
  )
}

interface LeadVerb {
  run: () => void
  verb: ConnectorVerb
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

  if (hasElement || verb === undefined || onVerb === undefined || (verb === 'turnBackOn' && appSwitch)) {
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
  onDisconnect,
  onInstall,
  onOpenAdmin,
  onServerToggle,
  onToggleForMe,
  onVerb,
  orgDisabledCount = 0
}: PartProps) {
  const { t } = useI18n()
  const hosted = card.ways.hosted
  const reason = card.reason ? (card.reason.text ?? t.connectorsPage.card.reason[card.reason.key]) : undefined
  const appSwitch = ruleable(hosted) && onToggleForMe !== undefined
  const offerVerb = leadVerb({ appSwitch, card, hasElement: connectElement !== undefined, onVerb })
  const showsReason = reason !== undefined && connectElement === undefined
  const paired = card.ways.local !== null

  if (!paired && connectElement === undefined && offerVerb === undefined && !showsReason && orgDisabledCount <= 0) {
    return null
  }

  return (
    <div className="grid shrink-0 gap-3 border-b border-(--ui-stroke-tertiary) px-3.5 py-3">
      {connectElement}

      {showsReason ? <p className="text-[0.72rem] text-(--ui-text-secondary)">{reason}</p> : null}

      {offerVerb === undefined ? null : (
        <Button className="self-start" onClick={offerVerb.run} size="sm">
          {t.connectorsPage.card.verb[offerVerb.verb]}
        </Button>
      )}

      <OrgNote count={orgDisabledCount} onOpenAdmin={onOpenAdmin} />

      <WaysSection
        card={card}
        hostedVerb={false}
        installFields={installFields}
        installing={installing}
        onAuthenticate={onAuthenticate}
        onDisconnect={onDisconnect}
        onInstall={onInstall}
        onServerToggle={onServerToggle}
      />
    </div>
  )
}

function HostedFoot({ card }: PartProps) {
  const { t } = useI18n()

  return card.ways.hosted ? <FootLine>{t.connectorsPage.dialog.nousLine}</FootLine> : null
}

function LocalLead({
  card,
  installFields,
  installing,
  onAuthenticate,
  onConnect,
  onInstall,
  onReconnect,
  onServerToggle
}: PartProps) {
  const local = card.ways.local

  if (!local || (card.plugin !== undefined && !card.ways.hosted)) {
    return null
  }

  return (
    <div className="shrink-0 border-b border-(--ui-stroke-tertiary) px-3.5 py-2.5">
      {card.ways.hosted ? (
        <WaysSection
          card={card}
          installFields={installFields}
          installing={installing}
          onAuthenticate={onAuthenticate}
          onConnect={onConnect}
          onInstall={onInstall}
          onReconnect={onReconnect}
          onServerToggle={card.plugin === undefined ? onServerToggle : undefined}
        />
      ) : local.installed === false && onInstall ? (
        <LocalInstall installFields={installFields} installing={installing} onInstall={onInstall} />
      ) : (
        <LocalServerControl
          name={card.name}
          onAuthenticate={onAuthenticate}
          onServerToggle={card.plugin === undefined ? onServerToggle : undefined}
          way={local}
        />
      )}
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
    <div className="grid shrink-0 gap-3 border-t border-(--ui-stroke-tertiary) px-3.5 py-2.5">
      {metrics ? (
        <div className="flex gap-6">
          <Metric label={copy.tokensPerCall} value={cost?.tokensPerCall} />
          <Metric label={copy.usesPerMonth} value={cost?.usesPerMonth} />
        </div>
      ) : null}

      {advanced ? (
        <details className="group grid gap-2">
          <summary className="flex cursor-pointer list-none items-center gap-1.5 text-xs font-medium text-(--ui-text-primary)">
            <Codicon
              className={cn('shrink-0 transition-transform duration-100 group-open:rotate-90')}
              name="chevron-right"
              size="0.75rem"
            />
            <span className="shrink-0">{copy.advanced}</span>
            <span className="min-w-0 truncate font-normal text-(--ui-text-quaternary)">{copy.advancedHint}</span>
          </summary>
          <div className="pt-2">{advanced}</div>
        </details>
      ) : null}
    </div>
  )
}

function FootLine({ children }: { children: string }) {
  return (
    <p className="shrink-0 border-t border-(--ui-stroke-tertiary) px-3.5 py-2 text-[0.7rem] text-(--ui-text-tertiary)">
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
    <div className="grid gap-1 rounded-md bg-(--ui-orange)/8 p-2.5">
      <p className="text-[0.7rem] text-(--ui-text-secondary)">{copy.orgNote(count)}</p>
      {onOpenAdmin ? (
        <Button className="justify-self-start" onClick={onOpenAdmin} size="inline" variant="textStrong">
          {copy.orgLink}
        </Button>
      ) : null}
    </div>
  )
}

function Metric({ label, value }: { label: string; value?: string }) {
  if (!value) {
    return null
  }

  return (
    <div className="grid gap-0.5">
      <span className="text-sm font-semibold tabular-nums text-(--ui-text-primary)">{value}</span>
      <span className="text-[0.65rem] text-(--ui-text-quaternary)">{label}</span>
    </div>
  )
}
