import { Codecs, persistentAtom } from '@/lib/persisted'

import { mergeVisibleReorder } from './order'

const PREFIX = 'hermes.desktop.sidebar.gatewayGroups.v1'
export const $gatewayGroupAliases = persistentAtom(`${PREFIX}.aliases`, {}, Codecs.stringRecord)
export const $gatewayGroupOrder = persistentAtom(`${PREFIX}.order`, [], Codecs.stringArray)
export const $gatewayGroupCollapsed = persistentAtom(`${PREFIX}.collapsed`, [], Codecs.stringArray)
// Gateways the user took out of the rail, keyed by the REGISTRY connection id
// (not a session-group id): this is a presentation choice about which gateways
// the fleet strip shows, deliberately scoped to connection identity so a
// profile rename or a group re-key cannot silently unhide or hide one.
//
// Hiding is never reachability. The connection stays registered, keeps its
// roster entry, and stays switchable from the statusbar and Settings →
// Gateways; only the rail group goes away. That is what keeps it safe for the
// app-managed local gateway on a remote-only install: the way back to a local
// backend survives the hide (#96532).
export const $gatewayGroupHidden = persistentAtom(`${PREFIX}.hidden`, [], Codecs.stringArray)

export function renameGatewayGroup(id: string, alias: string) {
  const aliases = { ...$gatewayGroupAliases.get() }
  const name = alias.trim()

  if (name) {
    aliases[id] = name
  } else {
    delete aliases[id]
  }

  $gatewayGroupAliases.set(aliases)
}

export function reorderGatewayGroups(ids: string[]) {
  // A filtered-out or temporarily offline section keeps its place.
  const order = $gatewayGroupOrder.get()
  const allIds = [...order, ...ids.filter(id => !order.includes(id))]
  $gatewayGroupOrder.set(mergeVisibleReorder(allIds, ids))
}

export function toggleGatewayGroup(id: string) {
  const collapsed = $gatewayGroupCollapsed.get()
  $gatewayGroupCollapsed.set(collapsed.includes(id) ? collapsed.filter(key => key !== id) : [...collapsed, id])
}

/**
 * Show/hide one gateway in the fleet rail. Callers pass the registry
 * connection id. Hiding is idempotent per id, so the same preference can be
 * written from Settings → Gateways without the caller tracking current state.
 */
export function setGatewayGroupHidden(connectionId: string, hidden: boolean) {
  const current = $gatewayGroupHidden.get()
  const already = current.includes(connectionId)

  if (already === hidden) {
    return
  }

  $gatewayGroupHidden.set(hidden ? [...current, connectionId] : current.filter(id => id !== connectionId))
}
