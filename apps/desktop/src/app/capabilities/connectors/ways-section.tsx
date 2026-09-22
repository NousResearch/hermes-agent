import { Button } from '@/components/ui/button'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { useI18n } from '@/i18n'

import { bothWaysOn, hostedStateWord } from './derive'
import { type InstallField, LocalInstall, LocalServerControl } from './local-server-control'
import type { ConnectorCardModel, ConnectorWayHosted, ConnectorWays } from './types'

export type WayChoice = 'hosted' | 'local'

export function wayInUse({ hosted, local }: ConnectorWays): null | WayChoice {
  if (hosted !== null && hosted.connected && hosted.state !== 'off' && hosted.state !== 'available') {
    return 'hosted'
  }

  return local !== null && local.installed === true && local.serverEnabled === true ? 'local' : null
}

export interface WaysSectionProps {
  card: ConnectorCardModel
  installFields?: readonly InstallField[]
  installing?: boolean
  onAuthenticate?: () => void
  onChange: (way: WayChoice) => void
  onConnect?: () => void
  onInstall?: (env: Record<string, string>) => void
  onReconnect?: () => void
  onServerToggle?: (next: boolean) => void
  onToggleForMe?: (next: boolean) => void
  value: WayChoice
}

export function WaysSection({ card, onChange, value, ...rest }: WaysSectionProps) {
  const { t } = useI18n()
  const copy = t.connectorsPage
  const { hosted, local } = card.ways

  if (!hosted || !local) {
    return null
  }

  const choose = (way: WayChoice) => {
    onChange(way)

    if (way === 'hosted') {
      if (local.installed === true && local.serverEnabled === true) {
        rest.onServerToggle?.(false)
      }

      if (hosted.connected && hosted.offBy === 'me') {
        rest.onToggleForMe?.(true)
      }

      return
    }

    if (hosted.connected && hosted.state !== 'off') {
      rest.onToggleForMe?.(false)
    }

    if (local.installed === true && local.serverEnabled !== true) {
      rest.onServerToggle?.(true)
    }
  }

  return (
    <section className="grid gap-2.5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-xs font-medium text-(--ui-text-primary)">{copy.dialog.waysTitle(card.name)}</h3>
        <SegmentedControl
          onChange={choose}
          options={[
            { id: 'hosted', label: copy.dialog.wayHosted },
            { id: 'local', label: copy.residencyLocal }
          ]}
          value={value}
        />
      </div>

      {value === 'hosted' ? (
        <HostedWay
          name={card.name}
          onConnect={rest.onConnect}
          onReconnect={rest.onReconnect}
          onToggleForMe={rest.onToggleForMe}
          way={hosted}
        />
      ) : local.installed === true || rest.onInstall === undefined ? (
        <LocalServerControl
          name={card.name}
          onAuthenticate={rest.onAuthenticate}
          onServerToggle={rest.onServerToggle}
          way={local}
        />
      ) : (
        <LocalInstall installFields={rest.installFields} installing={rest.installing} onInstall={rest.onInstall} />
      )}

      {bothWaysOn(card.ways) && rest.onServerToggle ? (
        <div className="flex flex-wrap items-center gap-2">
          <p className="min-w-0 flex-1 text-[0.7rem] text-(--ui-text-secondary)">{copy.dialog.bothOn(card.name)}</p>
          <Button onClick={() => rest.onServerToggle?.(false)} size="inline" variant="textStrong">
            {copy.dialog.turnOffLocal}
          </Button>
        </div>
      ) : null}
    </section>
  )
}

function HostedWay({
  name,
  onConnect,
  onReconnect,
  onToggleForMe,
  way
}: {
  name: string
  onConnect?: () => void
  onReconnect?: () => void
  onToggleForMe?: (next: boolean) => void
  way: ConnectorWayHosted
}) {
  const { t } = useI18n()
  const copy = t.connectorsPage
  const broken = way.state === 'expired' || way.state === 'broken'
  const reason = broken ? (way.reason?.text ?? (way.reason ? copy.card.reason[way.reason.key] : undefined)) : undefined

  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="min-w-0 flex-1 text-[0.7rem] text-(--ui-text-tertiary)">
        {way.state === 'available' ? copy.dialog.wayNotConnected(name) : (reason ?? copy.card.state[hostedStateWord(way)])}
      </span>

      {way.state === 'available' && onConnect ? (
        <Button onClick={onConnect} size="xs">
          {copy.card.verb.connect}
        </Button>
      ) : null}

      {broken && onReconnect ? (
        <Button onClick={onReconnect} size="xs" variant="secondary">
          {copy.card.verb.reconnect}
        </Button>
      ) : null}

      {way.state === 'off' && way.offBy === 'me' && onToggleForMe ? (
        <Button onClick={() => onToggleForMe(true)} size="xs" variant="secondary">
          {copy.card.verb.turnBackOn}
        </Button>
      ) : null}
    </div>
  )
}
