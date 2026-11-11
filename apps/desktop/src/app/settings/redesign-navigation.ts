import type { OverlayNavGroup, OverlayNavLink } from '../overlays/overlay-split-layout'

import type { SettingsView } from './types'

function flattenSubpages(links: OverlayNavLink[]): OverlayNavLink[] {
  return links.flatMap(link => [
    { ...link, children: undefined },
    ...flattenSubpages(link.children ?? []).map(child => ({ ...child, label: `${link.label} › ${child.label}` }))
  ])
}

/** Rearrange destinations only: callbacks retain their existing route and scope. */
export function redesignSettingsNavigation(groups: OverlayNavGroup[]): OverlayNavGroup[] {
  const remaining = new Map(groups.map(group => [group.id, group]))
  const result: OverlayNavGroup[] = []
  const providers = remaining.get('providers')
  const account = providers?.children?.find(child => child.id === 'pview:accounts')

  if (providers && account) {
    const billing = remaining.get('billing')
    result.push({
      ...account,
      active: account.active || Boolean(billing?.active),
      children: billing ? [billing] : undefined
    })
    remaining.delete('billing')
    remaining.set('providers', {
      ...providers,
      active: providers.active && !account.active,
      children: providers.children?.filter(child => child !== account),
      onSelect: providers.children?.find(child => child.id === 'pview:keys')?.onSelect ?? providers.onSelect
    })
  }

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
      gapBefore: false
    })
  }

  // Billing remains a top-level destination only when there is no account page.
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
  about: 'updates'
}

export function settingsNavigationPresentation(
  groups: OverlayNavGroup[],
  redesign: boolean,
  view: SettingsView,
  subpage: string | undefined
) {
  const activeGroup = groups.find(group => group.active)
  const activeChild = activeGroup?.children?.find(child => child.active)
  if (!redesign) {
    return { groups, headerGroup: activeGroup, headerChild: activeChild }
  }

  const appearanceLanding = view === 'config:appearance' && subpage === 'general'
  const overviewLanding = REDESIGN_LANDING_PAGES[view] === subpage
  const accountLanding = activeChild?.id === 'pview:accounts'
  return {
    groups: redesignSettingsNavigation(groups).map(group =>
      group.id === 'config:appearance' && appearanceLanding ? { ...group, children: undefined } : group
    ),
    headerGroup: accountLanding ? activeChild : activeGroup,
    headerChild: overviewLanding || accountLanding ? undefined : activeChild
  }
}
