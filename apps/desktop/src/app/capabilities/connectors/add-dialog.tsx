import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogDescription, DialogTitle } from '@/components/ui/dialog'
import type { ProfileScope } from '@/hermes'
import { useI18n } from '@/i18n'
import { notifyError } from '@/store/notifications'

import { McpJsonEditor } from '../mcp/mcp-editor'
import type { McpServersController } from '../mcp/use-mcp-servers'

import { type AddServerDraft, EMPTY_ADD_DRAFT, entryOfDraft, isDraftComplete } from './add-server-draft'
import { AddServerForm } from './add-server-form'
import { setMcpBearerToken } from './data/rpc'

export interface AddServerDialogProps {
  controller: McpServersController
  onOpenChange: (open: boolean) => void
  open: boolean
  profile: ProfileScope
}

export function AddServerDialog({ controller, onOpenChange, open, profile }: AddServerDialogProps) {
  const { t } = useI18n()
  const copy = t.connectorsPage.add
  const [draft, setDraft] = useState<AddServerDraft>(EMPTY_ADD_DRAFT)
  const [raw, setRaw] = useState(false)
  const [saving, setSaving] = useState(false)

  const name = draft.name.trim()
  const nameTaken = name !== '' && name in controller.servers

  const discardRawDraft = () => {
    if (controller.dirty) {
      controller.resetDraft(controller.servers)
    }
  }

  const close = () => {
    discardRawDraft()
    setDraft(EMPTY_ADD_DRAFT)
    setRaw(false)
    onOpenChange(false)
  }

  const openRaw = () => {
    controller.addServer()
    setRaw(true)
  }

  const leaveRaw = () => {
    discardRawDraft()
    setRaw(false)
  }

  const save = async () => {
    setSaving(true)

    try {
      if (!(await controller.addServerEntry(name, entryOfDraft(draft)))) {
        return
      }

      if (draft.transport === 'http' && draft.auth === 'bearer' && draft.bearer.trim() !== '') {
        await setMcpBearerToken(profile, name, draft.bearer.trim())
        controller.refetchConfig()
      }

      close()
    } catch (err) {
      notifyError(err, copy.saveFailed)
    } finally {
      setSaving(false)
    }
  }

  return (
    <Dialog onOpenChange={next => (next ? onOpenChange(true) : close())} open={open}>
      <DialogContent
        bodyClassName="gap-0 overflow-hidden p-0"
        className={
          raw ? 'h-[min(40rem,80vh)] min-w-[min(48rem,90vw)]' : 'max-h-[min(44rem,85vh)] min-w-[min(34rem,90vw)]'
        }
        fitContent
      >
        <header className="flex shrink-0 items-center border-b border-(--ui-stroke-tertiary) px-5 py-3">
          <DialogTitle className="text-base font-semibold">{copy.title}</DialogTitle>
        </header>
        <DialogDescription className="sr-only">{copy.hint}</DialogDescription>

        {raw ? (
          <div className="flex min-h-0 flex-1 flex-col">
            <McpJsonEditor controller={controller} />
          </div>
        ) : (
          <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">
            <AddServerForm draft={draft} nameTaken={nameTaken} onChange={setDraft} />
          </div>
        )}

        <footer className="flex shrink-0 items-center gap-2 border-t border-(--ui-stroke-tertiary) px-5 py-3">
          {raw ? (
            <Button onClick={leaveRaw} size="sm" variant="outline">
              {t.common.back}
            </Button>
          ) : (
            <>
              <Button onClick={openRaw} size="sm" variant="outline">
                {copy.editJson}
              </Button>
              <Button
                className="ms-auto"
                disabled={saving || nameTaken || !isDraftComplete(draft) || controller.profilePending}
                loading={saving}
                onClick={() => void save()}
                size="sm"
              >
                {saving ? t.common.saving : t.common.save}
              </Button>
            </>
          )}
        </footer>
      </DialogContent>
    </Dialog>
  )
}
