import { useQuery } from '@tanstack/react-query'
import { useEffect } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { PLUGIN_FOLDER_GRANTS_KEY } from '@/contrib/plugin-files'
import type { PluginRecord } from '@/contrib/plugins-store'
import { useI18n } from '@/i18n'
import { queryClient } from '@/lib/query-client'
import { notify, notifyError } from '@/store/notifications'

interface PluginFolderAccessProps {
  /** The package's desktop half; a package without one has no grants. */
  plugin: null | PluginRecord
  name: string
}

const grantsKey = (pluginId: string) => [...PLUGIN_FOLDER_GRANTS_KEY, pluginId]

/** The folders the user granted a desktop plugin through `ctx.files`, each
 *  with a Revoke that takes effect on the plugin's next call. Renders nothing
 *  when the plugin holds no grant (or the shell predates the door), and a
 *  retry when the grants could not be read. */
export function PluginFolderAccess({ plugin, name }: PluginFolderAccessProps) {
  const { t } = useI18n()
  const copy = t.skills.plugins.folderAccess
  const door = window.hermesDesktop?.pluginFiles
  const pluginId = plugin?.id ?? ''

  const {
    data: grants = [],
    isError,
    refetch
  } = useQuery({
    enabled: Boolean(door && pluginId),
    queryFn: () => door!.grants(pluginId),
    queryKey: grantsKey(pluginId),
    retry: false
  })

  // A grant made or revoked in another window refreshes this list too.
  useEffect(
    () =>
      door?.onChanged?.(change => {
        if (change.pluginId === pluginId) {
          void queryClient.invalidateQueries({ queryKey: grantsKey(pluginId) })
        }
      }),
    [door, pluginId]
  )

  // A record the app could not read is not "no access": say so, offer a retry.
  if (isError) {
    return (
      <div className="flex flex-col gap-2" data-testid={`plugin-folder-access-${pluginId}`}>
        <div className="text-xs font-medium text-(--ui-text-secondary)">{copy.title}</div>
        <div className="flex min-w-0 items-center gap-2">
          <span className="flex-1 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
            {copy.loadFailed(name)}
          </span>
          <Button onClick={() => void refetch()} size="xs" variant="text">
            {t.common.retry}
          </Button>
        </div>
      </div>
    )
  }

  if (!grants.length) {
    return null
  }

  const revoke = async (folderId: string, folderPath: string) => {
    try {
      await door!.revoke(pluginId, folderId)
      notify({ kind: 'success', message: copy.revoked(name, folderPath) })
    } catch (error) {
      notifyError(error, copy.revokeFailed(name))
    } finally {
      void queryClient.invalidateQueries({ queryKey: PLUGIN_FOLDER_GRANTS_KEY })
    }
  }

  return (
    <div className="flex flex-col gap-2" data-testid={`plugin-folder-access-${pluginId}`}>
      <div className="text-xs font-medium text-(--ui-text-secondary)">{copy.title}</div>
      <div className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
        {copy.hint(name)}
      </div>
      {grants.map(grant => (
        <div className="flex min-w-0 items-center gap-2" key={grant.id}>
          <Codicon name="folder" size="0.85rem" />
          <span className="flex min-w-0 flex-1 flex-col" title={grant.path}>
            <span className="truncate text-xs text-(--ui-text-secondary)">{grant.name}</span>
            <span className="truncate font-mono text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
              {grant.path}
            </span>
          </span>
          <Button
            aria-label={copy.revokeLabel(grant.path)}
            className="text-destructive hover:text-destructive"
            onClick={() => void revoke(grant.id, grant.path)}
            size="xs"
            variant="text"
          >
            {copy.revoke}
          </Button>
        </div>
      ))}
    </div>
  )
}
