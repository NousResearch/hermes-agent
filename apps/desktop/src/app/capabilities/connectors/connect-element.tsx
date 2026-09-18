// The connect element, in the dialog.
//
// This is the SHIPPED element (`components/ui/connector-card.tsx`) driven by the
// SHIPPED phase table (`CONNECTOR_CARD_PHASES`) — the same two pieces the chat
// card uses. Nothing about a connect is redrawn here, so the page and a chat can
// never say different things about the same operation.
//
// The only difference is the owner: a chat's operation belongs to a session,
// this one belongs to the account, so the frames arrive on the broadcast path
// and the store they land in is `data/account-operations.ts`.

import { useEffect } from 'react'

import { CONNECTOR_CARD_PHASES, MARK_LABEL } from '@/components/assistant-ui/connector-tool'
import { Button } from '@/components/ui/button'
import { ConnectorCard, ConnectorRow, ConnectorSummary } from '@/components/ui/connector-card'
import { useI18n } from '@/i18n'
import { connectorIconUrl, connectorTitle } from '@/lib/connector-tools'
import type { ConnectionTarget } from '@/store/connection-request'

import { type AccountOperation, syncAccountOperation } from './data/account-operations'

export interface ConnectElementProps {
  /** True while a `connectors.connect` for this app is in flight. */
  busy: boolean
  operation: AccountOperation
  /** Try again: re-mint the link and open it. */
  onReissue: (name: string) => void
  /** Stop waiting: end the operation now. */
  onStopWaiting: () => void
}

export function ConnectElement({ busy, operation, onReissue, onStopWaiting }: ConnectElementProps) {
  const { t } = useI18n()
  const copy = t.connectors

  // A waiting target with no link is a link this window does not hold: another
  // window started the attempt, or the backend re-minted one and the broadcast
  // that told us so could not carry it. Ask for the operation again.
  const missingLink = operation.targets.some(
    target => CONNECTOR_CARD_PHASES[target.state].verb === 'open' && target.connectUrl === null
  )

  useEffect(() => {
    if (missingLink && !operation.settled) {
      void syncAccountOperation(operation.opId)
    }
  }, [missingLink, operation.opId, operation.settled])

  // A settled operation is a static per-target summary: no controls, nothing live.
  if (operation.settled) {
    return (
      <div className="grid min-w-0 gap-1" data-connector-offer>
        {operation.targets.map(target => {
          const { meta, tone } = CONNECTOR_CARD_PHASES[target.state].settled(copy)

          return <ConnectorSummary connector={subject(target.name)} key={target.name} meta={meta} tone={tone} />
        })}
      </div>
    )
  }

  const unresolved = operation.targets.some(target => !CONNECTOR_CARD_PHASES[target.state].resolved)

  return (
    <div className="grid min-w-0 gap-1" data-connector-offer>
      <ConnectorCard title={copy.title}>
        {operation.targets.map(target => (
          <ConnectorRow
            action={actionFor(target)}
            connector={subject(target.name)}
            cue={CONNECTOR_CARD_PHASES[target.state].mark === 'waiting' ? copy.waiting : undefined}
            key={target.name}
            mark={CONNECTOR_CARD_PHASES[target.state].mark}
            markLabel={MARK_LABEL[CONNECTOR_CARD_PHASES[target.state].mark](copy)}
          />
        ))}
      </ConnectorCard>

      {unresolved ? (
        <div className="px-3.5">
          <Button onClick={onStopWaiting} size="xs" variant="textStrong">
            {t.connectorsPage.card.verb.stopWaiting}
          </Button>
        </div>
      ) : null}
    </div>
  )

  function actionFor(target: ConnectionTarget) {
    const phase = CONNECTOR_CARD_PHASES[target.state]

    if (phase.verb === 'none') {
      return undefined
    }

    return {
      busy,
      // A waiting row with no link yet has nothing to open; one concurrent
      // sign-in tab at a time.
      disabled: busy || (phase.verb === 'open' && target.connectUrl === null),
      label: phase.verb === 'open' ? copy.connect : copy.retry,
      onClick: () => {
        if (phase.verb === 'open' && target.connectUrl) {
          void window.hermesDesktop?.openExternal?.(target.connectUrl)

          return
        }

        if (phase.verb === 'reissue') {
          onReissue(target.name)
        }
      }
    }
  }
}

const subject = (name: string) => ({ iconUrl: connectorIconUrl(name), name, title: connectorTitle(name) })
