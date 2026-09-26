import type { ConnectionAnswer, ConnectorPolicyGetResult } from '@hermes/shared'
import { useCallback, useRef, useState } from 'react'

import type { ProfileScope } from '@/hermes'
import { translateNow } from '@/i18n'
import { queryClient } from '@/lib/query-client'
import { notifyError } from '@/store/notifications'

import type { SaveResult } from '../use-tools-editor'

import {
  $accountOperations,
  abandonConnect,
  type AccountOperation,
  clearAccountOperation,
  startAccountOperation
} from './account-operations'
import { memberDisabledTools, memberRevision, readPolicy } from './join'
import { connectorsPolicyQueryKey, invalidateConnectors } from './keys'
import {
  asConnectorError,
  connectAccountConnectors,
  connectorPolicy,
  type ConnectorRpcError,
  isConnectorReason,
  removeConnectorAccount,
  respondToAccountOperation,
  setConnectorPolicy
} from './rpc'

export type WriteOutcome = { error: ConnectorRpcError; ok: false } | { ok: true }

const failed = (error: ConnectorRpcError): WriteOutcome => ({ error, ok: false })

async function attempt<T>(run: () => Promise<T>): Promise<WriteOutcome> {
  try {
    await run()

    return { ok: true }
  } catch (error) {
    return failed(asConnectorError(error))
  }
}

interface WrittenRevision {
  after: string | undefined
  revision: string
}

const tokenFor = (written: null | WrittenRevision, seen: string | undefined): string | undefined =>
  written && written.after === seen ? written.revision : seen

const remember = (seen: string | undefined, revision: string | undefined): null | WrittenRevision =>
  revision === undefined ? null : { after: seen, revision }

function refusedSave(error: ConnectorRpcError): SaveResult {
  notifyError(error, translateNow('connectorsPage.tools.saveFailed'))

  return 'failed'
}

export interface ConnectorToolsSaver {
  onSave: (disabled: string[]) => Promise<SaveResult>
  reload: () => void
  theirs: string[] | null
}

export function useConnectorToolsSave(
  scope: ProfileScope,
  connector: string,
  seenRevision: string | undefined
): ConnectorToolsSaver {
  const [theirs, setTheirs] = useState<string[] | null>(null)
  const written = useRef<null | WrittenRevision>(null)

  const reload = useCallback(() => {
    setTheirs(null)
    invalidateConnectors(scope, 'policy')
  }, [scope])

  const onSave = useCallback(
    async (disabled: string[]): Promise<SaveResult> => {
      const expected = tokenFor(written.current, seenRevision)

      if (expected === undefined) {
        return refusedSave(asConnectorError(new Error(translateNow('connectorsPage.dialog.rulesReadOnly'))))
      }

      try {
        const result = await setConnectorPolicy(scope, { connector, disabled_tools: disabled, type: 'tools' }, expected)

        written.current = remember(seenRevision, result.revision)
      } catch (error) {
        if (!isConnectorReason(error, 'POLICY_CONFLICT')) {
          return refusedSave(asConnectorError(error))
        }

        try {
          const policy = readPolicy(await connectorPolicy(scope))

          written.current = remember(seenRevision, memberRevision(policy))
          setTheirs([...memberDisabledTools(policy, connector)])

          return 'conflict'
        } catch (reread) {
          return refusedSave(asConnectorError(reread))
        }
      }

      setTheirs(null)
      invalidateConnectors(scope, 'policy', 'list')

      return 'saved'
    },
    [connector, scope, seenRevision]
  )

  return { onSave, reload, theirs }
}

export interface ConnectorSwitch {
  pending: null | string
  setEnabled: (connector: string, enabled: boolean) => Promise<WriteOutcome>
}

export function useConnectorSwitch(scope: ProfileScope): ConnectorSwitch {
  const [pending, setPending] = useState<null | string>(null)
  const written = useRef<null | WrittenRevision>(null)

  const setEnabled = useCallback(
    async (connector: string, enabled: boolean): Promise<WriteOutcome> => {
      setPending(connector)

      try {
        return await attempt(async () => {
          const cached = queryClient.getQueryData<ConnectorPolicyGetResult>(connectorsPolicyQueryKey(scope))
          const seen = memberRevision(readPolicy(cached ?? (await connectorPolicy(scope))))
          const expected = tokenFor(written.current, seen)

          if (expected === undefined) {
            throw new Error(translateNow('connectorsPage.dialog.rulesReadOnly'))
          }

          const result = await setConnectorPolicy(scope, { connector, enabled, type: 'connector' }, expected)

          written.current = remember(seen, result.revision)
          invalidateConnectors(scope, 'policy', 'list')
        })
      } finally {
        setPending(null)
      }
    },
    [scope]
  )

  return { pending, setEnabled }
}

export interface AccountDisconnect {
  disconnect: (connectionId: string) => Promise<WriteOutcome>
  pending: boolean
}

export function useDisconnectAccount(scope: ProfileScope): AccountDisconnect {
  const [pending, setPending] = useState(false)

  const disconnect = useCallback(
    async (connectionId: string): Promise<WriteOutcome> => {
      setPending(true)

      try {
        return await attempt(async () => {
          await removeConnectorAccount(scope, connectionId)
          invalidateConnectors(scope, 'list', 'accounts')
        })
      } finally {
        setPending(false)
      }
    },
    [scope]
  )

  return { disconnect, pending }
}

export type ConnectOutcome = { error: ConnectorRpcError; ok: false } | { ok: true; operation: AccountOperation }

export interface ConnectorConnect {
  connect: (slug: string, options?: { reconnect?: boolean }) => Promise<ConnectOutcome>
  giveUp: (opId: string) => Promise<WriteOutcome>
  pending: null | string
}

export function useConnectConnector(scope: ProfileScope): ConnectorConnect {
  const [pending, setPending] = useState<null | string>(null)

  const connect = useCallback(
    async (slug: string, options?: { reconnect?: boolean }): Promise<ConnectOutcome> => {
      setPending(slug)

      try {
        const snapshot = await connectAccountConnectors(scope, [slug], options?.reconnect ?? false)

        return { ok: true, operation: startAccountOperation(scope, [slug], snapshot) }
      } catch (error) {
        return { error: asConnectorError(error), ok: false }
      } finally {
        setPending(null)
      }
    },
    [scope]
  )

  const respond = useCallback(
    (opId: string, answer: ConnectionAnswer): Promise<WriteOutcome> =>
      attempt(() => respondToAccountOperation(scope, opId, answer)),
    [scope]
  )

  const giveUp = useCallback(
    async (opId: string): Promise<WriteOutcome> => {
      const operation = $accountOperations.get()[opId]

      abandonConnect(operation?.connectors ?? [])
      clearAccountOperation(opId)

      const outcome = await respond(opId, { settled_by: 'continue' })

      for (const target of operation?.targets ?? []) {
        if (target.state !== 'connected' && target.connectionId) {
          await attempt(() => removeConnectorAccount(scope, target.connectionId))
        }
      }

      invalidateConnectors(scope, 'list', 'accounts')

      return outcome
    },
    [respond, scope]
  )

  return { connect, giveUp, pending }
}
