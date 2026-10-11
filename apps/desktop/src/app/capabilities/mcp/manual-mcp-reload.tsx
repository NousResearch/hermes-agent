import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import type { HermesGateway, ProfileScope } from '@/hermes'
import { useI18n } from '@/i18n'
import { requestMcpReload } from '@/lib/mcp-reload'
import { notify } from '@/store/notifications'
import { $activeSessionId } from '@/store/session'

export interface ManualMcpReloadProps {
  gateway: HermesGateway | null
  profile: ProfileScope
}

export function ManualMcpReload({ gateway, profile }: ManualMcpReloadProps) {
  const { t } = useI18n()
  const copy = t.connectorsPage.reload
  const activeSessionId = useStore($activeSessionId)
  const [open, setOpen] = useState(false)

  const reload = async () => {
    const result = await requestMcpReload(gateway, { confirm: true, profile, sessionId: activeSessionId })

    if (result.status !== 'reloaded') {
      throw new Error(result.message || copy.error)
    }

    if (result.failed.length) {
      notify({ kind: 'warning', title: copy.action, message: copy.failed(result.failed.join(', ')) })
    } else {
      notify({ kind: 'success', title: copy.action, message: copy.success })
    }
  }

  return (
    <>
      <div className="flex shrink-0 items-center justify-end">
        <Button disabled={!gateway} onClick={() => setOpen(true)} size="xs" variant="outline">
          {copy.action}
        </Button>
      </div>
      <ConfirmDialog
        confirmLabel={copy.confirm}
        description={copy.description}
        onClose={() => setOpen(false)}
        onConfirm={reload}
        open={open}
        title={copy.title}
      />
    </>
  )
}
