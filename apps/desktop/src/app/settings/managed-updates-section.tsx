import { useStore } from '@nanostores/react'

import { Button } from '@/components/ui/button'
import type { DesktopRegistryConnection } from '@/global'
import { useI18n } from '@/i18n'
import { Download, Loader2 } from '@/lib/icons'
import { $connectionsRegistry } from '@/store/connections'
import {
  $managedUpdates,
  managedUpdatesSupported,
  type ManagedUpdateState,
  runManagedUpdate
} from '@/store/managed-updates'

import { ListRow, Pill, SectionHeading } from './primitives'

function stateTone(state: ManagedUpdateState | undefined): 'muted' | 'primary' | 'warn' {
  if (!state || state.status === 'idle') {
    return 'muted'
  }

  if (state.status === 'updating' || state.status === 'updated') {
    return 'primary'
  }

  return 'warn'
}

function sshTarget(connection: DesktopRegistryConnection): string | null {
  if (!connection.host) {
    return null
  }

  return connection.user ? `${connection.user}@${connection.host}` : connection.host
}

/** Per-connection driver for #95942's transactional SSH update engine: one
 * Update button per registered Desktop-managed SSH install, a single honest
 * in-flight state (the engine exposes no streaming progress channel), and the
 * correlated receipt once it lands. */
export function ManagedUpdatesSection() {
  const { t } = useI18n()
  const m = t.settings.managedUpdates
  const registry = useStore($connectionsRegistry)
  const states = useStore($managedUpdates)
  const sshConnections = (registry?.connections ?? []).filter(connection => connection.kind === 'ssh')

  // Fail closed: no section on an Electron main without the transactional
  // bridge, and nothing to drive when no SSH install is registered.
  if (!managedUpdatesSupported() || sshConnections.length === 0) {
    return null
  }

  const statusLabel = (state: ManagedUpdateState | undefined): string | null => {
    switch (state?.status) {
      case 'failed':
        return m.failed

      case 'partial':
        return m.partial

      case 'refused':
        return state.alreadyRunning ? m.alreadyRunning : m.refused

      case 'updated':
        return m.updated

      case 'updating':
        return m.updating

      default:
        return null
    }
  }

  const receiptLine = (state: ManagedUpdateState): string | null => {
    if (!state.receipt) {
      return null
    }

    const parts = [m.receipt(state.receipt.correlationId.slice(0, 8), state.receipt.outcome)]

    if (state.receipt.preVersion && state.receipt.postVersion) {
      parts.push(m.receiptVersions(state.receipt.preVersion, state.receipt.postVersion))
    }

    if (state.receipt.stopReason) {
      parts.push(state.receipt.stopReason)
    }

    return parts.join(' · ')
  }

  return (
    <section className="mt-8">
      <SectionHeading icon={Download} title={m.title} />
      <p className="mb-1 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
        {m.intro}
      </p>

      <div className="grid gap-1">
        {sshConnections.map(connection => {
          const state = states[connection.id]
          const updating = state?.status === 'updating'
          const label = statusLabel(state)
          const receipt = state ? receiptLine(state) : null
          const restored = state?.scopes.filter(scope => scope.restored).map(scope => scope.profile) ?? []
          const unrestored = state?.scopes.filter(scope => !scope.restored) ?? []

          return (
            <ListRow
              action={
                updating ? (
                  <Button disabled size="sm" variant="secondary">
                    <Loader2 className="animate-spin" /> {m.updating}
                  </Button>
                ) : (
                  <Button onClick={() => void runManagedUpdate(connection.id)} size="sm">
                    <Download /> {m.update}
                  </Button>
                )
              }
              below={
                state && (updating || state.message || receipt || state.scopes.length > 0) ? (
                  <div className="mt-1 grid gap-1 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                    {updating ? <p>{m.progress}</p> : null}
                    {!updating && state.message ? <p>{state.message}</p> : null}
                    {receipt ? <p className="font-mono text-[0.68rem]">{receipt}</p> : null}
                    {!updating && restored.length > 0 ? <p>{m.scopesRestored(restored.join(', '))}</p> : null}
                    {!updating &&
                      unrestored.map(scope => (
                        <p key={scope.profile}>{m.scopeNotRestored(scope.profile, scope.error ?? m.failed)}</p>
                      ))}
                  </div>
                ) : null
              }
              description={sshTarget(connection) ?? m.sshConnection}
              key={connection.id}
              title={
                <span className="flex items-center gap-2">
                  <span>{connection.label}</span>
                  {label ? <Pill tone={stateTone(state)}>{label}</Pill> : null}
                </span>
              }
            />
          )
        })}
      </div>
    </section>
  )
}
