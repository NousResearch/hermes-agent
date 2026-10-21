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
      ? translateNow('skillDeepLink.thisComputer')
      : $connectionsRegistry.get()?.connections.find(connection => connection.id === connectionId)?.label ||
        connectionId

  const destination = `${connectionLabel} · ${profile || 'default'}`

  const assertDestination = () => {
    if (connectionId !== getApiRequestConnection() || profile !== getApiRequestProfile()) {
      throw new Error(translateNow('skillDeepLink.destinationChanged'))
    }
  }

  await confirm({
    title: translateNow('skillDeepLink.installTitle', name),
    description: translateNow('skillDeepLink.installDescription'),
    details: [
      { label: translateNow('skillDeepLink.source'), value: identifier },
      { label: translateNow('skillDeepLink.installTo'), value: destination }
    ],
    confirmLabel: translateNow('skills.hub.install'),
    busyLabel: translateNow('skillDeepLink.installing'),
    doneLabel: translateNow('skillDeepLink.installed'),
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
        title: translateNow('skillDeepLink.installComplete', name),
        message: translateNow('skills.changesApplyNewSessions')
      })
    }
  })
}
