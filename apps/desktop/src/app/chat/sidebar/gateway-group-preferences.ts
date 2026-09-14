import { Codecs, persistentAtom } from '@/lib/persisted'

import { mergeVisibleReorder } from './order'

const PREFIX = 'hermes.desktop.sidebar.gatewayGroups.v1'
export const $gatewayGroupAliases = persistentAtom(`${PREFIX}.aliases`, {}, Codecs.stringRecord)
export const $gatewayGroupOrder = persistentAtom(`${PREFIX}.order`, [], Codecs.stringArray)
export const $gatewayGroupCollapsed = persistentAtom(`${PREFIX}.collapsed`, [], Codecs.stringArray)

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
