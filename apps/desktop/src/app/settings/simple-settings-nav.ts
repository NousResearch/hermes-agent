import type { Translations } from '@/i18n'

import type { OverlayNavGroup } from '../overlays/overlay-split-layout'

/** Navigation topics only: each page keeps its existing profile/app scope controls. */
export function groupSimpleSettings(
  groups: OverlayNavGroup[],
  copy: Translations['settings']['page']
): OverlayNavGroup[] {
  const sections = [
    {
      label: copy.agent,
      ids: [
        'config:model',
        'config:chat',
        'capabilities',
        'config:safety',
        'config:memory',
        'config:voice',
        'config:browser',
        'vault',
        'config:advanced'
      ]
    },
    { label: copy.connections, ids: ['providers', 'gateway', 'keys', 'billing'] },
    {
      label: copy.app,
      ids: ['config:appearance', 'config:workspace', 'notifications', 'keybinds', 'sessions', 'about']
    }
  ]

  const result: OverlayNavGroup[] = []
  const included = new Set<string>()

  for (const section of sections) {
    const entries = section.ids.flatMap(id => {
      const group = groups.find(candidate => candidate.id === id)

      if (!group) {
        return []
      }

      included.add(id)

      return [group]
    })

    entries.forEach((group, index) =>
      result.push({ ...group, gapBefore: false, heading: index === 0 ? section.label : undefined })
    )
  }

  // New/optional settings remain reachable even without a category assignment.
  return [...result, ...groups.filter(group => !included.has(group.id))]
}
