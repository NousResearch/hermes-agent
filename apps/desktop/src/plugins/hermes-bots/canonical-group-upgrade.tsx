import { Button, host, useI18n, useValue } from '@hermes/plugin-sdk'
import { useRef, useState } from 'react'

import { groupCreationEngine, groupNeedsDesktopCourier, groupSeatsPeers, revalidateGroupCreation, useGroupCapability } from './canonical-group-capabilities'
import { createGroupAcrossConnections, type GroupCreationIntent, usePeerTargets } from './canonical-group-peers'
import { registerCanonicalGroup } from './canonical-group-registry'
import { canonicalGroupCreateError } from './canonical-groups'
import { useBots } from './i18n'
import type { GroupMember } from './types'

/** A separate gateway conversation; the Desktop room and its history remain usable. */
export function GatewayGroupUpgrade({ group, members, visible, onCreated }: {
  visible: boolean; group: string; members: GroupMember[]; onCreated: (key: string) => void
}) {
  const { t } = useI18n()
  const b = useBots()
  const [peerCheck, setPeerCheck] = useState(0)
  const activeConnection = useValue(host.state.connectionId)
  const { capability, connectionId, mode, observedRoute, source } = useGroupCapability(visible && groupNeedsDesktopCourier(members, activeConnection), peerCheck)
  const targets = usePeerTargets(members, connectionId, groupSeatsPeers(capability?.snapshot), peerCheck)
  const plan = groupCreationEngine(mode, members, connectionId, capability?.snapshot, targets)
  const intent = useRef<GroupCreationIntent>({})
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  if (plan.engine === 'unavailable' && plan.reason === 'unreachable') {
    return <div className="grid gap-1 text-xs">
      <p role="status">{b.crossGateway.peerCheckFailed(plan.connection || '', plan.connectionId === 'local')}</p>
      <Button onClick={() => setPeerCheck(value => value + 1)}>{t.common.retry}</Button>
    </div>
  }

  if (plan.engine !== 'gateway-peer') {return null}

  const create = async () => {
    if (!observedRoute || busy) {return}
    setBusy(true)
    setError('')

    try {
      const fresh = await revalidateGroupCreation(observedRoute, capability, source, members, 'gateway-peer', b.canonical.driverUnavailable).catch(error => {
        setPeerCheck(value => value + 1)
        throw error
      })

      if (!fresh.sourceCurrent()) {throw new Error(b.canonical.driverUnavailable)}
      const created = await createGroupAcrossConnections(observedRoute, group, members, fresh.targets, intent.current)

      if (!fresh.sourceCurrent()) {return}
      intent.current = {}

      if (fresh.sourceCurrent()) {onCreated(registerCanonicalGroup(observedRoute, created.room, { created: true }))}
    } catch (error) {
      setError(canonicalGroupCreateError(error, b.canonical))
    } finally {setBusy(false)}
  }

  return <div className="grid gap-1 text-xs">
    <p>{b.crossGateway.upgradeHistory}</p>
    <Button disabled={busy} onClick={() => void create()}>{b.crossGateway.startGatewayGroup}</Button>
    {error && <p role="alert">{error}</p>}
  </div>
}
