import { getApiRequestConnection, getApiRequestProfile, type ProfileScope } from '@/hermes'
import { translateNow } from '@/i18n'

import { confirm } from './confirm'
import { $connectionsRegistry } from './connection-registry-state'
import { HubInstallBlockedError, installHubSkill, notifyHubActionFailed } from './hub-actions'
import { notify } from './notifications'

/** The URL supplies only an identifier, never a destination profile or scan override. */
export async function requestSkillInstallFromDeepLink(identifier: string): Promise<void> {
  const connectionId = getApiRequestConnection()
  const profile = getApiRequestProfile()
  const scope: ProfileScope = { connectionId, profile }
  const name = identifier.split('/').filter(Boolean).at(-1) || identifier

  const connectionLabel =
    !connectionId || connectionId === 'local'
      ? translateNow('catalog.thisComputer')
      : $connectionsRegistry.get()?.connections.find(connection => connection.id === connectionId)?.label ||
        connectionId

  const destination = `${connectionLabel} · ${profile || 'default'}`

  const assertDestination = () => {
    if (connectionId !== getApiRequestConnection() || profile !== getApiRequestProfile()) {
      throw new Error(translateNow('catalog.destinationChanged'))
    }
  }

  await confirm({
    title: translateNow('catalog.installTitle', name),
    description: translateNow('catalog.installDescription'),
    details: [
      { label: translateNow('catalog.source'), value: identifier },
      { label: translateNow('catalog.installTo'), value: destination }
    ],
    confirmLabel: translateNow('skills.hub.install'),
    busyLabel: translateNow('catalog.installing'),
    doneLabel: translateNow('catalog.installed'),
    onConfirm: async () => {
      // Recheck on retries too: a link must never follow a changed destination.
      assertDestination()

      try {
        await installHubSkill(identifier, scope)
      } catch (error) {
        if (error instanceof HubInstallBlockedError) {
          notifyHubActionFailed(error, translateNow('skills.hub.actionFailed'), name, scope)
        }

        throw error
      }

      // The hub abandons polling when the active profile changes. That is not
      // proof of success, so do not show an Installed state in that case.
      assertDestination()
      notify({
        kind: 'success',
        title: translateNow('catalog.installComplete', name),
        message: translateNow('skills.changesApplyNewSessions')
      })
    }
  })
}
