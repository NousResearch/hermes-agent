import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { requestComposerSubmit } from '@/app/chat/composer/focus'
import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { connectorTitle } from '@/lib/connector-tools'
import {
  $firstBuildConnections,
  type FirstBuildConnectorPart,
  openFirstBuildLinks
} from '@/store/first-build-connectors'

interface FirstBuildConnectorOfferProps {
  part: FirstBuildConnectorPart
  storedId: string
  target: string
}

export function FirstBuildConnectorOffer({ part, storedId, target }: FirstBuildConnectorOfferProps) {
  const connections = useStore($firstBuildConnections, { keys: [storedId] })
  const { t } = useI18n()

  useEffect(() => {
    void openFirstBuildLinks(storedId, part, {
      open: window.hermesDesktop?.openExternal ? url => window.hermesDesktop.openExternal(url) : undefined,
      submit: text => void requestComposerSubmit(text, { displayKind: 'hidden', target })
    })
  }, [storedId, part, target])

  return (
    <div className="my-2 grid min-w-0 max-w-lg gap-3" data-connector-offer>
      {connections[storedId]?.rows.map(row => (
        <div className="flex items-center justify-between gap-3 text-sm" key={row.connector}>
          <span>{connectorTitle(row.connector)}</span>
          <span className="text-xs text-muted-foreground">
            {row.phase === 'connected' ? `✓ ${t.connectors.connected}` : 'Waiting for you to finish signing in…'}
          </span>
          {row.connectUrl && row.phase !== 'connected' && !window.hermesDesktop?.openExternal ? (
            <Button asChild size="xs" variant="link">
              <a href={row.connectUrl} rel="noreferrer" target="_blank">
                {t.connectors.connect}
              </a>
            </Button>
          ) : null}
        </div>
      ))}
    </div>
  )
}
