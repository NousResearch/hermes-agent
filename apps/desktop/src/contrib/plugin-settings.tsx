import { type ComponentType, createElement, useMemo } from 'react'

import { ContribBoundary } from './react/boundary'
import { useContributions } from './react/use-contributions'

export const PLUGIN_SETTINGS_AREA = 'capabilities.pluginSettings'

/** Concrete Capabilities selection, never the focused chat or an implicit
 * legacy profile route. The host passes a frozen copy for this mount. */
export interface PluginSettingsScope {
  readonly connectionId: string
  readonly profile: string
}

export interface PluginSettingsContributionProps {
  readonly scope: PluginSettingsScope
}

/** Optional content beneath the plugin's own package description. Register
 * through ctx.register({ area: PLUGIN_SETTINGS_AREA, data: { render } }).
 * Scope changes remount the contribution: keep dialogs component-owned and
 * cancel/ignore pending work in cleanup. A remount cannot undo dispatched writes. */
export interface PluginSettingsContribution {
  render: ComponentType<PluginSettingsContributionProps>
}

export function PluginSettingsSlot({ pluginId, scope }: { pluginId: string; scope: PluginSettingsScope }) {
  const contributions = useContributions(PLUGIN_SETTINGS_AREA)

  const pinnedScope = useMemo(
    () => Object.freeze({ connectionId: scope.connectionId, profile: scope.profile }),
    [scope.connectionId, scope.profile]
  )

  return contributions
    .filter(c => c.source === `plugin:${pluginId}`)
    .map(c => {
      const render = (c.data as PluginSettingsContribution | undefined)?.render

      return render ? (
        <ContribBoundary id={c.id} key={JSON.stringify([c.source, c.id, pinnedScope])} variant="chip">
          {createElement(render, { scope: pinnedScope })}
        </ContribBoundary>
      ) : null
    })
}
