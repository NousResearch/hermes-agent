import { type ReactNode, useState } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { resolveSessionRpcOwner } from '@/app/contrib/wiring-routing'
import { useI18n } from '@/i18n'
import { folderPathFromMarkdownHref } from '@/lib/folder-links'
import { $connection, getSessionOwnerHint, knownSessionOwner, ownerLookupSessionRows } from '@/store/session'
import { sessionTileOwnerRoute, storedSessionIdForRuntimeId } from '@/store/session-states'

export function FolderLink({ href, children }: { href: string; children: ReactNode }) {
  const view = useSessionView()
  const { t } = useI18n()

  const [error, setError] = useState<
    { kind: 'localConnectionRequired' | 'unavailable' } | { kind: 'failed'; detail?: string } | null
  >(null)

  const path = folderPathFromMarkdownHref(href)

  const open = async () => {
    setError(null)

    try {
      // Read this transcript's owner, never the focused tile or ambient profile.
      const runtimeId = view.$runtimeId.get()
      const storedId = view.$storedId.get() ?? (runtimeId ? storedSessionIdForRuntimeId(runtimeId) : null)

      const owner = resolveSessionRpcOwner({
        routingSessionId: storedId,
        sessionOwnerHint: getSessionOwnerHint,
        sessionRowOwner: id => knownSessionOwner(ownerLookupSessionRows(), id),
        tileOwnerRoute: sessionTileOwnerRoute
      })

      const connection = $connection.get()

      if (
        !owner ||
        typeof owner === 'string' ||
        owner.mode === 'remote' ||
        !owner.connectionId ||
        connection?.mode !== 'local' ||
        connection.connectionId !== owner.connectionId ||
        (connection.profile ?? 'default') !== owner.profile
      ) {
        setError({ kind: 'localConnectionRequired' })

        return
      }

      const capability = window.hermesDesktop?.openExistingDirectory

      if (!capability) {
        setError({ kind: 'unavailable' })

        return
      }

      if (!path) {
        return
      }

      const result = await capability({ path, owner: { connectionId: owner.connectionId, profile: owner.profile } })

      if (!result.ok) {
        setError({ kind: 'failed', detail: result.error })
      }
    } catch (cause) {
      setError({ kind: 'failed', detail: cause instanceof Error ? cause.message : String(cause) })
    }
  }

  // Reserved malformed links must never escape to the browser or file preview.
  if (!path) {
    return <span>{children}</span>
  }

  return (
    <span>
      <button className="ref wrap-anywhere" onClick={() => void open()} title={path} type="button">
        {children}
      </button>
      {error && (
        <span className="ml-2 text-xs text-muted-foreground" role="alert">
          {error.kind === 'failed'
            ? error.detail
              ? t.folderLinks.openFailedWithMessage(error.detail)
              : t.folderLinks.openFailed
            : t.folderLinks[error.kind]}
        </span>
      )}
    </span>
  )
}
