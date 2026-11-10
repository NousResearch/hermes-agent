import type { OverlayNavGroup, OverlayNavLink } from '../overlays/overlay-split-layout'

/** Rearrange destinations only: callbacks retain their existing route and scope. */
export function redesignSettingsNavigation(groups: OverlayNavGroup[]): OverlayNavGroup[] {
  const remaining = new Map(groups.map(group => [group.id, group]))
  const result: OverlayNavGroup[] = []
  const providers = remaining.get('providers')
  const account = providers?.children?.find(child => child.id === 'pview:accounts')

  if (providers && account) {
    result.push({ ...account })
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

  append('billing')
  append('config:appearance')
  append('notifications')
  append('keybinds')
  append('gateway', ['web-access'])
  append('sessions')
  append('providers', ['keys', 'config:model'])
  append('config:voice')
  append('config:safety', ['vault'])
  // Chat and memory deserve direct destinations even where the mock is silent.
  append('config:chat')
  append('config:memory')
  append('config:advanced', ['config:workspace', 'config:browser'])
  append('plugins')
  const about = remaining.get('about')
  remaining.delete('about')

  return [...result, ...remaining.values(), ...(about ? [{ ...about, gapBefore: true }] : [])]
}
