import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { capabilityScoped } from '@/api/client'
import { Button } from '@/components/ui/button'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import type { HermesGateway, ProfileScope } from '@/hermes'
import { useI18n } from '@/i18n'
import { reloadMcp } from '@/lib/mcp-reload'
import { notify } from '@/store/notifications'
import { $activeSessionId, sessionOfProfile } from '@/store/session'

export interface ManualMcpReloadProps {
  gateway: HermesGateway | null
  profile: ProfileScope
}

export function ManualMcpReload({ gateway, profile }: ManualMcpReloadProps) {
  const { t } = useI18n()
  const copy = t.connectorsPage.reload
  const activeSessionId = useStore($activeSessionId)
  const [open, setOpen] = useState(false)
  const profileName = capabilityScoped(profile).profile

  const reload = async () => {
    if (!gateway) {
      throw new Error(copy.error)
    }

    // A chat of another profile is not the one this pane's reload may change.
    const sessionId = sessionOfProfile(activeSessionId, profileName)
    const outcome = await reloadMcp(gateway, sessionId, profileName)

    if (outcome.deferred) {
      notify({ kind: 'info', title: copy.action, message: copy.deferred })
    }

    if (outcome.unavailable.length > 0) {
      notify({ kind: 'warning', title: copy.action, message: copy.failed(outcome.unavailable.join(', ')) })
    }

    if (!outcome.deferred && outcome.unavailable.length === 0) {
      notify({
        kind: 'success',
        title: copy.action,
        message: outcome.session === 'unchanged' ? copy.unchanged : copy.success
      })
    }
  }

  return (
    <>
      <Button disabled={!gateway} onClick={() => setOpen(true)} size="xs" variant="outline">
        {copy.action}
      </Button>
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
