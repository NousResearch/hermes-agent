import { Codecs, persistentAtom } from '@/lib/persisted'

export interface AgentSidebarOwner {
  connectionId: string | null
  profile: string
}

/** Presentation identity uses the backend target profile, never its display name. */
export function agentSidebarKey(owner: AgentSidebarOwner): string {
  const connection = owner.connectionId?.trim()

  return JSON.stringify([!connection || connection === 'local' ? null : connection, owner.profile.trim() || 'default'])
}

export const $hiddenSidebarAgents = persistentAtom('hermes.desktop.hiddenSidebarAgents.v1', [], Codecs.stringArray)
export const $sidebarAgentExpanded = persistentAtom<Record<string, boolean>>(
  'hermes.desktop.sidebarAgentExpanded.v1',
  {},
  Codecs.json(value =>
    value && typeof value === 'object' && !Array.isArray(value)
      ? Object.fromEntries(Object.entries(value).filter(([, expanded]) => typeof expanded === 'boolean'))
      : {}
  )
)
export const $sidebarAgentScroll = persistentAtom(
  'hermes.desktop.sidebarAgentScroll.v1',
  0,
  Codecs.json(value => (typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : 0))
)
export const $hiddenSidebarAgentsExpanded = persistentAtom(
  'hermes.desktop.hiddenSidebarAgentsExpanded.v1',
  false,
  Codecs.bool
)

/** Redesign-only Agents section disclosure. Global: one window presentation preference, never shared
 *  with classic's section atoms, so collapsing in one shell leaves the other untouched. */
export const $portalAgentsSectionOpen = persistentAtom(
  'hermes.desktop.portalSidebar.agentsOpen.global.v1',
  true,
  Codecs.bool
)

/** Redesign-only Pinned section disclosure, scoped like `$portalAgentsSectionOpen`: classic's
 *  Pinned section keeps its own (session-only) `$sidebarPinsOpen`. */
export const $portalPinnedSectionOpen = persistentAtom(
  'hermes.desktop.portalSidebar.pinnedOpen.global.v1',
  true,
  Codecs.bool
)

export function setSidebarAgentHidden(owner: AgentSidebarOwner, hidden: boolean): void {
  const key = agentSidebarKey(owner)
  const current = $hiddenSidebarAgents.get()

  if (current.includes(key) !== hidden) {
    $hiddenSidebarAgents.set(hidden ? [...current, key] : current.filter(item => item !== key))
  }
}

export function dropAgentSidebarPreferences(owner: AgentSidebarOwner): void {
  setSidebarAgentHidden(owner, false)
  const key = agentSidebarKey(owner)
  const expanded = $sidebarAgentExpanded.get()

  if (Object.hasOwn(expanded, key)) {
    const { [key]: _removed, ...rest } = expanded
    $sidebarAgentExpanded.set(rest)
  }
}

export function migrateAgentSidebarPreferences(owner: AgentSidebarOwner, profile: string): void {
  const from = agentSidebarKey(owner)
  const to = agentSidebarKey({ ...owner, profile })

  if (from === to) {
    return
  }

  const hidden = $hiddenSidebarAgents.get()

  if (hidden.includes(from)) {
    $hiddenSidebarAgents.set([...new Set(hidden.map(key => (key === from ? to : key)))])
  }

  const expanded = $sidebarAgentExpanded.get()

  if (Object.hasOwn(expanded, from)) {
    const { [from]: value, ...rest } = expanded
    $sidebarAgentExpanded.set({ ...rest, [to]: value })
  }
}
