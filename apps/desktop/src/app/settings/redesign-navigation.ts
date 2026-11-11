import type { OverlayNavGroup, OverlayNavLink } from '../overlays/overlay-split-layout'

import type { SettingsView } from './types'

function flattenSubpages(links: OverlayNavLink[]): OverlayNavLink[] {
  return links.flatMap(link => [
    { ...link, children: undefined },
    ...flattenSubpages(link.children ?? []).map(child => ({ ...child, label: `${link.label} › ${child.label}` }))
  ])
}

/** Names the redesign gives existing destinations; routes and owners stay untouched. */
export type RedesignNavigationLabels = Partial<Record<string, string>>

/** Rearrange destinations only: callbacks retain their existing route and scope. The order is the
 *  Figma rail (856:1834); everything not drawn there (provider accounts and keys, per-agent config
 *  sections, plugins) stays reachable under Advanced. */
export function redesignSettingsNavigation(
  groups: OverlayNavGroup[],
  labels: RedesignNavigationLabels = {}
): OverlayNavGroup[] {
  const remaining = new Map(groups.map(group => [group.id, group]))
  const result: OverlayNavGroup[] = []

  function append(id: string, related: string[] = []) {
    const primary = remaining.get(id)

    if (!primary) {
      return
    }

    remaining.delete(id)
    const children: OverlayNavLink[] = [...(primary.children ?? [])]

    for (const key of related) {
      const group = remaining.get(key)

      if (group) {
        children.push(group)
        remaining.delete(key)
      }
    }

    result.push({
      ...primary,
      active: primary.active || children.some(child => child.active),
      children: children.length ? children : undefined,
      gapBefore: false,
      label: labels[id] ?? primary.label
    })
  }

  append('billing')
  append('config:appearance')
  append('notifications')
  append('keybinds')
  append('gateway', ['web-access'])
  append('sessions')
  append('about')

  const advanced = remaining.get('config:advanced')
  remaining.delete('config:advanced')

  if (advanced) {
    const children = [
      ...(advanced.children ?? []),
      ...[...remaining.values()].map(group => ({
        ...group,
        children: flattenSubpages(group.children ?? [])
      }))
    ]

    result.push({
      ...advanced,
      active: advanced.active || children.some(child => child.active),
      children,
      gapBefore: true
    })
  } else {
    result.push(...remaining.values())
  }

  return result
}

const REDESIGN_LANDING_PAGES: Partial<Record<SettingsView, string>> = {
  'config:appearance': 'general',
  billing: 'overview',
  notifications: 'alerts',
  keybinds: 'shortcuts',
  sessions: 'archived',
  about: 'updates',
  gateway: 'connection'
}

/** Pages the redesign draws as one destination: the landing renders every subpage's content, so the
 *  rail lists none of them (related pages filed under the group, like Web access, stay). */
const REDESIGN_SINGLE_PAGES: ReadonlySet<string> = new Set([
  'notifications',
  'keybinds',
  'sessions',
  'about',
  'gateway'
])

function withoutOwnSubpages(group: OverlayNavGroup): OverlayNavGroup {
  const children = group.children?.filter(child => !child.id.startsWith(`${group.id}:`))

  return { ...group, children: children?.length ? children : undefined }
}

export function settingsNavigationPresentation(
  groups: OverlayNavGroup[],
  redesign: boolean,
  view: SettingsView,
  subpage: string | undefined,
  labels: RedesignNavigationLabels = {}
) {
  const activeGroup = groups.find(group => group.active)
  const activeChild = activeGroup?.children?.find(child => child.active)

  if (!redesign) {
    return { groups, headerGroup: activeGroup, headerChild: activeChild }
  }

  const appearanceLanding = view === 'config:appearance' && subpage === 'general'
  // Billing tracks its sub-view in `bview`, not `page`: its overview is the Account & Plan landing.
  const overviewLanding = REDESIGN_LANDING_PAGES[view] === subpage || activeChild?.id === 'bview:overview'

  const redesigned = redesignSettingsNavigation(groups, labels).map(group =>
    (group.id === 'config:appearance' && appearanceLanding) || (group.id === 'billing' && group.children?.length === 1)
      ? { ...group, children: undefined }
      : REDESIGN_SINGLE_PAGES.has(group.id)
        ? withoutOwnSubpages(group)
        : group
  )

  const headerGroup = redesigned.find(group => group.id === activeGroup?.id) ?? activeGroup

  return {
    groups: redesigned,
    headerGroup,
    headerChild: overviewLanding ? undefined : activeChild
  }
}
