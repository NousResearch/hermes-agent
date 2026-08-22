import { Codicon, host } from '@hermes/plugin-sdk'

import { attachmentDataUrl } from './api'
import { useKanban } from './i18n'
import type { KanbanAttachment } from './types'

export function AttachmentList({ attachments }: { attachments: KanbanAttachment[] }) {
  const k = useKanban()

  const preview = async (attachment: KanbanAttachment, storedPath: string) => {
    const { filename } = attachment
    let opened = false

    try {
      // `storedPath` is an absolute path on the BACKEND host. It resolves
      // directly when that host is this machine; when it isn't (remote
      // gateway / Hermes Cloud) the host falls back to this loader, which
      // fetches the bytes over the plugin's own REST transport — the one
      // door that reaches the backend wherever it runs.
      opened = await host.previewFile(storedPath, filename, async () => {
        const { contentType, dataUrl } = await attachmentDataUrl(attachment.id)

        return dataUrl ? { contentType, dataUrl } : null
      })
    } catch {
      opened = false
    }

    // Still nothing to show: an unreadable blob, an oversize file the preview
    // cap refuses, or a type the rail can't render from bytes.
    if (!opened) {
      host.notify({ kind: 'error', message: k.previewUnavailable(filename) })
    }
  }

  return (
    <ul className="flex flex-col gap-1">
      {attachments.map(attachment => {
        const storedPath = attachment.stored_path

        return (
          <li className="flex items-center gap-1.5 text-[0.75rem] text-(--ui-text-tertiary)" key={attachment.id}>
            <Codicon name="file" size="0.75rem" />
            {storedPath ? (
              <button
                aria-label={k.previewAttachment(attachment.filename)}
                className="min-w-0 truncate text-left underline-offset-2 hover:text-foreground hover:underline"
                onClick={() => void preview(attachment, storedPath)}
                title={attachment.filename}
                type="button"
              >
                {attachment.filename}
              </button>
            ) : (
              <span className="min-w-0 truncate" title={attachment.filename}>
                {attachment.filename}
              </span>
            )}
          </li>
        )
      })}
    </ul>
  )
}
