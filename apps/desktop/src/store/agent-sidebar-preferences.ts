import type { WritableAtom } from 'nanostores'

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

const booleanRecord = Codecs.json<Record<string, boolean>>(value =>
  value && typeof value === 'object' && !Array.isArray(value)
    ? Object.fromEntries(Object.entries(value).filter(([, flag]) => typeof flag === 'boolean'))
    : {}
)

export const $sidebarAgentExpanded = persistentAtom<Record<string, boolean>>(
  'hermes.desktop.sidebarAgentExpanded.v1',
  {},
  booleanRecord
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

/** A hand-made order for one agent's chats: durable session ids as arranged, and when. */
export interface AgentChatOrder {
  at: number
  ids: string[]
}

const isAgentChatOrder = (value: unknown): value is AgentChatOrder =>
  Boolean(value) &&
  typeof value === 'object' &&
  typeof (value as AgentChatOrder).at === 'number' &&
  Array.isArray((value as AgentChatOrder).ids) &&
  (value as AgentChatOrder).ids.every(id => typeof id === 'string')

/** Redesign-only: each agent's dragged chat order, keyed by `agentSidebarKey` (machine + profile). */
export const $portalAgentChatOrder = persistentAtom<Record<string, AgentChatOrder>>(
  'hermes.desktop.portalSidebar.agentChatOrder.v1',
  {},
  Codecs.json(value =>
    value && typeof value === 'object' && !Array.isArray(value)
      ? Object.fromEntries(Object.entries(value).filter(([, order]) => isAgentChatOrder(order)))
      : {}
  )
)

export function setAgentChatOrder(owner: AgentSidebarOwner, order: AgentChatOrder): void {
  $portalAgentChatOrder.set({ ...$portalAgentChatOrder.get(), [agentSidebarKey(owner)]: order })
}

/** Redesign-only: the agents whose card lists every chat past its cap, keyed by `agentSidebarKey`. */
export const $portalAgentShowAll = persistentAtom<Record<string, boolean>>(
  'hermes.desktop.portalSidebar.agentShowAll.v1',
  {},
  booleanRecord
)

export function setAgentShowAll(owner: AgentSidebarOwner, showAll: boolean): void {
  $portalAgentShowAll.set({ ...$portalAgentShowAll.get(), [agentSidebarKey(owner)]: showAll })
}

/** Move one agent's entry of a keyed preference to `to`, or drop it when `to` is absent. */
function rekeyAgentEntry<T>($record: WritableAtom<Record<string, T>>, from: string, to?: string): void {
  const record = $record.get()

  if (!Object.hasOwn(record, from)) {
    return
  }

  const { [from]: value, ...rest } = record
  $record.set(to === undefined ? rest : { ...rest, [to]: value })
}

// Every per-agent record: a rename carries each one, a deletion drops each one.
function rekeyAgentEntries(from: string, to?: string): void {
  rekeyAgentEntry($sidebarAgentExpanded, from, to)
  rekeyAgentEntry($portalAgentChatOrder, from, to)
  rekeyAgentEntry($portalAgentShowAll, from, to)
}

export function setSidebarAgentHidden(owner: AgentSidebarOwner, hidden: boolean): void {
  const key = agentSidebarKey(owner)
  const current = $hiddenSidebarAgents.get()

  if (current.includes(key) !== hidden) {
    $hiddenSidebarAgents.set(hidden ? [...current, key] : current.filter(item => item !== key))
  }
}

export function dropAgentSidebarPreferences(owner: AgentSidebarOwner): void {
  setSidebarAgentHidden(owner, false)
  rekeyAgentEntries(agentSidebarKey(owner))
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

  rekeyAgentEntries(from, to)
}
