import { useStore } from '@nanostores/react'
import { useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { Monitor } from '@/lib/icons'
import { $activeConnectionId } from '@/store/connections'
import { activeGateway, gatewayActivationEpoch } from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'
import { $activeSessionId, $gatewayState } from '@/store/session'

import { ListRow, SettingsSection } from './primitives'
import { deviceCommand, revokeRemoteAccess, type DeviceAction, type RemoteDevice } from './remote-control-actions'

export function RemoteControlSection() {
  const { t } = useI18n()
  const copy = t.settings.remoteControl
  const sessionId = useStore($activeSessionId)
  const gatewayState = useStore($gatewayState)
  const connectionId = useStore($activeConnectionId)
  const profile = useStore($activeGatewayProfile)
  const [origins, setOrigins] = useState<{ id: string; name: string }[] | null>(null)
  const [devices, setDevices] = useState<RemoteDevice[]>([])
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const generation = useRef(0)

  async function run(action: DeviceAction) {
    const requestGeneration = ++generation.current
    const epoch = gatewayActivationEpoch()
    const socket = activeGateway()
    const native = window.hermesDesktop?.desktop
    const current = () => generation.current === requestGeneration && epoch === gatewayActivationEpoch()
    setBusy(true)
    setError('')
    setMessage('')
    try {
      if (!native?.remote) throw new Error('This Desktop build does not support cross-device control.')
      const command = (next: DeviceAction) => {
        if (!socket || socket.connectionState !== 'open') throw new Error('The gateway is disconnected.')
        return deviceCommand(
          (method, params, timeout) => socket.request(method, params, timeout),
          sessionId ?? '',
          next
        )
      }
      if (action === 'revoke') {
        const result = await revokeRemoteAccess(native, sessionId ?? '', () => command('revoke'))
        if (!current()) return
        setOrigins([])
        setDevices([])
        setMessage(result.gatewayConfirmed ? copy.revoked : copy.cleanupPending)
        return
      }
      if (action !== 'list') await command(action)
      if (!current()) return
      const local = await native.remote({ sessionId: sessionId ?? '', action: 'status', arguments: {} })
      if (!current()) return
      setOrigins((local.origins ?? []) as { id: string; name: string }[])
      const result = await command('list')
      if (current()) setDevices(result.devices ?? [])
    } catch (failure) {
      if (current()) setError(String(failure))
    } finally {
      if (current()) setBusy(false)
    }
  }

  useEffect(() => {
    ++generation.current
    setOrigins(null)
    setDevices([])
    setBusy(false)
    setError('')
    setMessage('')
    // Explicit refresh keeps this view read-only until the user chooses an action.
    return () => {
      ++generation.current
    }
  }, [sessionId, connectionId, profile, gatewayState])

  const canEnroll = !!sessionId && gatewayState === 'open'
  return (
    <SettingsSection icon={Monitor} title={copy.title}>
      <ListRow
        title={copy.authorized}
        description={copy.description}
        below={
          <div className="text-sm text-(--ui-text-secondary)">
            {origins?.length ? origins.map(origin => origin.name).join(', ') : origins ? copy.off : copy.refresh}
          </div>
        }
        action={
          <>
            <Button disabled={busy || !canEnroll} onClick={() => void run('target')}>
              {copy.authorize}
            </Button>
            <Button variant="destructive" disabled={busy} onClick={() => void run('revoke')}>
              {copy.revoke}
            </Button>
          </>
        }
      />
      <ListRow
        title={copy.origin}
        description={copy.originDescription}
        action={
          <Button variant="secondary" disabled={busy || !canEnroll} onClick={() => void run('origin')}>
            {copy.origin}
          </Button>
        }
      />
      <ListRow
        title={copy.devices}
        action={
          <Button variant="secondary" disabled={busy || !canEnroll} onClick={() => void run('list')}>
            {copy.refresh}
          </Button>
        }
        below={
          <ul className="text-sm text-(--ui-text-secondary)">
            {devices.map(device => (
              <li key={device.id}>
                {device.name} · {device.platform} · {device.role === 'origin' ? copy.approves : copy.receives}
                {device.online ? '' : ` · ${copy.offline}`}
              </li>
            ))}
          </ul>
        }
      />
      {!canEnroll && <p className="text-sm text-(--ui-text-tertiary)">{copy.openChat}</p>}
      {message && (
        <p role="status" className="text-sm text-(--ui-text-secondary)">
          {message}
        </p>
      )}
      {error && (
        <p role="alert" className="text-sm text-(--ui-text-secondary)">
          {error}
        </p>
      )}
    </SettingsSection>
  )
}
