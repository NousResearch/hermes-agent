import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import type { HermesGateway } from '@/hermes'
import { useI18n } from '@/i18n'

type ReloadResult = { status?: string }

export function ManualMcpReload({ gateway, sessionId }: { gateway: HermesGateway | null; sessionId: null | string }) {
  const { t } = useI18n()
  const copy = t.connectorsPage.manualReload
  const [open, setOpen] = useState(false)
  const [reloaded, setReloaded] = useState(false)

  const reload = async () => {
    if (!gateway || !sessionId) {
      throw new Error(copy.noSession)
    }

    const result = (await gateway.request('reload.mcp', {
      confirm: true,
      session_id: sessionId
    })) as ReloadResult

    if (result.status !== 'reloaded') {
      throw new Error(copy.failed)
    }

    setReloaded(true)
  }

  return (
    <>
      <div className="flex shrink-0 items-center gap-2">
        <Button disabled={!gateway || !sessionId} onClick={() => { setReloaded(false); setOpen(true) }} size="xs" variant="outline">
          {copy.action}
        </Button>
        {reloaded ? <span className="text-xs text-(--ui-text-secondary)" role="status">{copy.success}</span> : null}
      </div>
      <ConfirmDialog
        confirmLabel={copy.confirm}
        description={copy.warning}
        onClose={() => setOpen(false)}
        onConfirm={reload}
        open={open}
        title={copy.title}
      />
    </>
  )
}
