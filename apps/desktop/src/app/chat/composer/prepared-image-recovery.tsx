import { useEffect, useState } from 'react'

import { listPreparedImageDrafts } from '@/app/session/hooks/use-prompt-actions/prepared-submissions'
import { captureSubmissionDestination } from '@/app/session/hooks/use-prompt-actions/submission-destination'
import type { GatewayRequest } from '@/app/session/hooks/use-prompt-actions/utils'
import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import type { ComposerAttachment } from '@/store/composer'
import { notifyError } from '@/store/notifications'

interface Props {
  sessionKey: string | null
  request: GatewayRequest
  occupied: boolean
  onRestore: (text: string, attachments: ComposerAttachment[]) => void
}

export function PreparedImageRecovery({ sessionKey, request, occupied, onRestore }: Props) {
  const { t } = useI18n()
  const [drafts, setDrafts] = useState<Awaited<ReturnType<typeof listPreparedImageDrafts>>>([])
  const scopeKey = captureSubmissionDestination(sessionKey, request).scopeKey

  useEffect(() => {
    let cancelled = false
    setDrafts([])

    if (sessionKey) {
      void listPreparedImageDrafts(sessionKey, scopeKey).then(entries => {
        if (!cancelled) { setDrafts(entries) }
      }).catch(error => {
        if (!cancelled) { notifyError(error, t.composer.restoreImageDraft) }
      })
    }

    return () => { cancelled = true }
  }, [sessionKey, scopeKey, t.composer.restoreImageDraft])

  return drafts.map(draft => (
    <div className="flex items-center gap-2 rounded-md border bg-background px-3 py-2 text-xs" key={draft.key}>
      <span className="min-w-0 flex-1 truncate">{draft.text || draft.attachments.map(attachment => attachment.label).join(', ')}</span>
      <Button disabled={occupied} onClick={() => {
        onRestore(draft.text, draft.attachments)
        setDrafts(current => current.filter(entry => entry.key !== draft.key))
      }} size="sm" type="button" variant="outline">{t.composer.restoreImageDraft}</Button>
    </div>
  ))
}
