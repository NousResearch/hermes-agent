import { Button } from '@/components/ui/button'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { useI18n } from '@/i18n'

import { McpJsonEditor, McpLogPane } from '../mcp/mcp-editor'
import type { McpServersController } from '../mcp/use-mcp-servers'

import { localServerName } from './derive'
import type { ConnectorCardModel } from './types'

export interface LocalAdvancedProps {
  controller: McpServersController
  name: string
  onRemove: () => void
}

export function LocalAdvanced({ controller, name, onRemove }: LocalAdvancedProps) {
  const { t } = useI18n()
  const m = t.settings.mcp

  return (
    <div className="grid min-h-0 gap-3">
      <div className="h-56 shadow-[0_4px_16px_rgba(38,64,140,0.06)]" data-connector-panel="">
        <McpJsonEditor controller={controller} highlightServer={name} />
      </div>

      <div className="h-40 shadow-[0_4px_16px_rgba(38,64,140,0.06)]" data-connector-panel="">
        <McpLogPane server={name} />
      </div>

      <Button
        className="justify-self-start px-0 text-[13px] font-semibold text-[#dc2626] hover:text-[#b91c1c]"
        onClick={onRemove}
        size="xs"
        variant="text"
      >
        {m.remove}
      </Button>
    </div>
  )
}

export interface RemoveServerConfirmProps {
  card: ConnectorCardModel | null
  controller: McpServersController
  onClose: () => void
  onRemoved: () => void
}

export function RemoveServerConfirm({ card, controller, onClose, onRemoved }: RemoveServerConfirmProps) {
  const { t } = useI18n()
  const m = t.settings.mcp
  const name = card ? localServerName(card) : null

  return (
    <ConfirmDialog
      confirmLabel={m.remove}
      description={t.connectorsPage.dialog.removeServerBody}
      destructive
      onClose={onClose}
      onConfirm={async () => {
        if (!name) {
          return
        }

        if (!(await controller.removeServer(name))) {
          throw new Error(m.removeFailed)
        }

        onRemoved()
      }}
      open={card !== null}
      title={t.connectorsPage.dialog.removeServerTitle(card?.name ?? '')}
    />
  )
}
