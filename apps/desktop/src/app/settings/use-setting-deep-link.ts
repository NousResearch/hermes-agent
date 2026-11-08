import { useCallback } from 'react'

import { settingDefinition, settingElementId } from './settings-manifest'
import type { SettingsView } from './types'
import { useDeepLinkHighlight } from './use-deep-link-highlight'

/**
 * Land a palette hit (`?setting=<id>`) on a manifest row of this view: once
 * the row's subpage is showing, scroll to it and flash it. Pages with no
 * subpages pass no `show`; pages with subpages pass the same predicate that
 * gates their rows so the highlight waits for the right child page.
 */
export function useSettingDeepLink(view: SettingsView, show: (subpage: string | undefined) => boolean = () => true) {
  const ready = useCallback(
    (id: string) => {
      const setting = settingDefinition(view, id)

      return setting !== undefined && show(setting.subpage)
    },
    [show, view]
  )

  return useDeepLinkHighlight({ elementId: settingElementId, param: 'setting', ready })
}
