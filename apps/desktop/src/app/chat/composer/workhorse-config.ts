import { useStore } from '@nanostores/react'

import { hermesConfigKey, useHermesConfigRecord } from '@/app/hooks/use-config-record'
import { retainConfigReadOrigin, saveHermesConfig } from '@/hermes'
import { useI18n } from '@/i18n'
import { queryClient } from '@/lib/query-client'
import { $activeConnectionId } from '@/store/connections'
import { requestGatewayForAgent } from '@/store/gateway'
import { notifyError } from '@/store/notifications'
import { $activeGatewayProfile } from '@/store/profile'
import type { HermesConfigRecord } from '@/types/hermes'

import { useComposerScope } from './scope'

export const workhorseRouteReset = { base_url: '', api_key: '', api_mode: '' }

function delegationRecord(config: HermesConfigRecord | undefined): Record<string, unknown> {
  const raw = config?.delegation

  return raw && typeof raw === 'object' ? (raw as Record<string, unknown>) : {}
}

type WorkhorseField = 'model' | 'effort'
interface WorkhorseWrites {
  tail: Promise<void>
  pending: number
  revisions: Partial<Record<WorkhorseField, number>>
  confirmed: Partial<Record<WorkhorseField, Record<string, unknown>>>
}

// Model/effort edits (and multiple tiles for one owner) share this queue. Serialize the
// whole delegation section because the backend saves it by read/merge/write;
// generations are per field so an effort change cannot obsolete a model pick.
const writes = new Map<string, WorkhorseWrites>()

export function useWorkhorseConfig() {
  const scope = useComposerScope()
  const activeProfile = useStore($activeGatewayProfile)
  const activeConnection = useStore($activeConnectionId)
  const profile = scope.profile ?? activeProfile
  const connectionId = scope.connectionId ?? activeConnection
  const owner = { connectionId, profile }
  const { data: config, writeScope } = useHermesConfigRecord(owner)
  const { t } = useI18n()
  const failureTitle = t.shell.modelOptions.updateFailed
  const delegation = delegationRecord(config)

  const commit = (field: WorkhorseField, patch: Record<string, unknown>) => {
    if (!config) {
      return
    }

    // Capture BOTH destinations before any await; neither an active-connection
    // switch nor a later GET may move the tail of this operation to another owner.
    const destination = writeScope ?? owner
    const key = hermesConfigKey(owner, connectionId)
    const id = JSON.stringify([destination.connectionId, destination.profile])
    let queue = writes.get(id)

    if (!queue) {
      queue = { tail: Promise.resolve(), pending: 0, revisions: {}, confirmed: {} }
      writes.set(id, queue)
    }

    const state = queue
    const revision = (state.revisions[field] ?? 0) + 1
    state.revisions[field] = revision
    state.pending++

    const publish = (confirmed: Record<string, unknown>) => {
      queryClient.setQueryData<HermesConfigRecord>(key, previous =>
        retainConfigReadOrigin(
          {
            ...(previous ?? config),
            delegation: { ...delegationRecord(previous ?? config), ...confirmed }
          },
          previous ?? config
        )
      )
    }

    state.tail = state.tail.then(async () => {
      try {
        await queryClient.cancelQueries({ queryKey: key, exact: true })
        const result = await saveHermesConfig({ delegation: patch }, destination)

        if (!result.ok) {
          throw new Error(failureTitle)
        }

        state.confirmed[field] = patch
      } catch (error) {
        if (state.revisions[field] === revision) {
          notifyError(error, failureTitle)
        }
      } finally {
        // Another consumer may have started a refetch during the PUT. Its
        // pre-write snapshot must not replace the confirmed value afterward.
        await queryClient.cancelQueries({ queryKey: key, exact: true })

        // An older success must not paint over a queued newer choice. If the
        // latest fails, display the last confirmed server value, not a phantom.
        if (state.revisions[field] === revision && state.confirmed[field]) {
          publish(state.confirmed[field])
        }

        if (--state.pending === 0) {
          writes.delete(id)
        }
      }
    })
  }

  return {
    profile,
    connectionId,
    delegation,
    ready: Boolean(config),
    commit,
    request: <T>(method: string, params?: Record<string, unknown>) =>
      requestGatewayForAgent<T>(connectionId, profile, method, params)
  }
}
