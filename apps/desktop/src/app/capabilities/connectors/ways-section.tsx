import { type ReactNode } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'

import { bothWaysOn, hostedStateWord } from './derive'
import { type InstallField, LocalInstall, LocalServerControl } from './local-server-control'
import type { ConnectorCardModel, ConnectorWayHosted, ConnectorWayLocal } from './types'

export interface WaysSectionProps {
  card: ConnectorCardModel
  hostedVerb?: boolean
  installFields?: readonly InstallField[]
  installing?: boolean
  onAuthenticate?: () => void
  onConnect?: () => void
  onDisconnect?: () => void
  onInstall?: (env: Record<string, string>) => void
  onReconnect?: () => void
  onServerToggle?: (next: boolean) => void
}

export function WaysSection({ card, ...rest }: WaysSectionProps) {
  const { t } = useI18n()
  const copy = t.connectorsPage.dialog
  const { hosted, local } = card.ways

  if (!hosted || !local) {
    return null
  }

  return (
    <section className="grid gap-2.5">
      <h3 className="text-xs font-medium text-(--ui-text-primary)">{copy.waysTitle(card.name)}</h3>

      <HostedWay
        onConnect={rest.hostedVerb === false ? undefined : rest.onConnect}
        onDisconnect={rest.hostedVerb === false ? undefined : rest.onDisconnect}
        onReconnect={rest.hostedVerb === false ? undefined : rest.onReconnect}
        quiet={local.installed === true && local.verb === 'authenticate'}
        way={hosted}
      />

      <LocalWay
        installFields={rest.installFields}
        installing={rest.installing}
        name={card.name}
        onAuthenticate={rest.onAuthenticate}
        onInstall={rest.onInstall}
        onServerToggle={rest.onServerToggle}
        way={local}
      />

      {bothWaysOn(card.ways) && rest.onServerToggle ? (
        <div className="grid justify-items-start gap-1">
          <p className="text-[0.7rem] text-(--ui-text-secondary)">{copy.bothOn(card.name)}</p>
          <Button onClick={() => rest.onServerToggle?.(false)} size="inline" variant="textStrong">
            {copy.turnOffLocal}
          </Button>
        </div>
      ) : null}
    </section>
  )
}

function WayRow({ children, title }: { children: ReactNode; title: string }) {
  return (
    <div className="grid gap-1.5 border-t border-(--ui-stroke-tertiary) pt-2">
      <span className="text-[0.72rem] font-medium text-(--ui-text-primary)">{title}</span>
      {children}
    </div>
  )
}

function HostedWay({
  onConnect,
  onDisconnect,
  onReconnect,
  quiet = false,
  way
}: {
  onConnect?: () => void
  onDisconnect?: () => void
  onReconnect?: () => void
  quiet?: boolean
  way: ConnectorWayHosted
}) {
  const { t } = useI18n()
  const copy = t.connectorsPage

  return (
    <WayRow title={copy.dialog.wayHosted}>
      <div className="flex flex-wrap items-center gap-2">
        <span className="min-w-0 flex-1 truncate text-[0.7rem] text-(--ui-text-tertiary)">
          {copy.card.state[hostedStateWord(way)]}
        </span>

        {way.state === 'available' && onConnect ? (
          <Button onClick={onConnect} size="xs" variant={quiet ? 'outline' : undefined}>
            {copy.card.verb.connect}
          </Button>
        ) : null}

        {(way.state === 'expired' || way.state === 'broken') && onReconnect ? (
          <Button onClick={onReconnect} size="xs" variant="secondary">
            {copy.card.verb.reconnect}
          </Button>
        ) : null}

        {way.state === 'connected' && onDisconnect ? (
          <Button className="text-destructive hover:text-destructive" onClick={onDisconnect} size="xs" variant="text">
            {copy.dialog.disconnect}
          </Button>
        ) : null}
      </div>
    </WayRow>
  )
}

function LocalWay({
  installFields = [],
  installing = false,
  name,
  onAuthenticate,
  onInstall,
  onServerToggle,
  way
}: {
  installFields?: readonly InstallField[]
  installing?: boolean
  name: string
  onAuthenticate?: () => void
  onInstall?: (env: Record<string, string>) => void
  onServerToggle?: (next: boolean) => void
  way: ConnectorWayLocal
}) {
  const { t } = useI18n()

  return (
    <WayRow title={t.connectorsPage.residencyLocal}>
      {way.installed === true || onInstall === undefined ? (
        <LocalServerControl name={name} onAuthenticate={onAuthenticate} onServerToggle={onServerToggle} way={way} />
      ) : (
        <LocalInstall installFields={installFields} installing={installing} onInstall={onInstall} />
      )}
    </WayRow>
  )
}
