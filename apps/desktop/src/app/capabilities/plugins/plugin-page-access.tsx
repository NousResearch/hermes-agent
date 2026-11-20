import { useStore } from '@nanostores/react'

import { Codicon } from '@/components/ui/codicon'
import { PREVIEW_SCRIPT_CAPABILITY } from '@/contrib/plugin-preview'
import type { PluginRecord } from '@/contrib/plugins-store'
import { useI18n } from '@/i18n'
import { notify } from '@/store/notifications'
import { $previewScriptGrants, revokePreviewScript } from '@/store/plugin-preview-grants'

import { Pill } from '../../settings/primitives'

interface PluginPageAccessProps {
  /** The Desktop half of the package; absent for agent-only packages. */
  plugin?: null | Pick<PluginRecord, 'capabilities' | 'id'>
  pluginName: string
}

/**
 * The revocation side of `ctx.preview.executeJavaScript` consent, shown on a
 * Desktop plugin's row: every page origin the user allowed it to script, each
 * with its own revoke. A revoked origin asks again on the plugin's next call.
 * Rendered only for plugins that declared the capability.
 */
export function PluginPageAccess({ plugin, pluginName }: PluginPageAccessProps) {
  const { t } = useI18n()
  const copy = t.skills.plugins
  const grants = useStore($previewScriptGrants)

  if (!plugin?.capabilities?.includes(PREVIEW_SCRIPT_CAPABILITY)) {
    return null
  }

  const pluginId = plugin.id
  const origins = grants[pluginId] ?? []

  return (
    <div
      className="mt-1 flex flex-wrap items-center gap-1.5 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)"
      data-testid={`plugin-page-access-${pluginId}`}
    >
      <span className="font-medium text-(--ui-text-secondary)">{copy.previewScriptAccessLabel}</span>
      {origins.length === 0 ? (
        <span>{copy.previewScriptAccessNone}</span>
      ) : (
        origins.map(origin => (
          <Pill className="gap-1 pr-0.5" key={origin}>
            <span>{origin}</span>
            <button
              aria-label={copy.previewScriptRevoke(origin)}
              className="inline-flex size-4 items-center justify-center rounded-sm hover:bg-(--ui-control-hover-background) hover:text-foreground"
              onClick={() => {
                if (revokePreviewScript(pluginId, origin)) {
                  notify({ kind: 'info', message: copy.previewScriptRevoked(pluginName, origin) })
                } else {
                  notify({ kind: 'error', message: copy.previewScriptRevokeUnsaved(pluginName, origin) })
                }
              }}
              type="button"
            >
              <Codicon name="close" size="0.7rem" />
            </button>
          </Pill>
        ))
      )}
    </div>
  )
}
