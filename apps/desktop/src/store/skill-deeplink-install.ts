import { getApiRequestConnection, getApiRequestProfile, type ProfileScope } from '@/hermes'
import { translateNow } from '@/i18n'

import { confirm } from './confirm'
import { installHubSkill, notifyHubActionFailed } from './hub-actions'
import { notify } from './notifications'

/** The URL supplies only an identifier, never a destination profile or scan override. */
export async function requestSkillInstallFromDeepLink(identifier: string): Promise<void> {
  const connectionId = getApiRequestConnection()
  const profile = getApiRequestProfile()
  const scope: ProfileScope = { connectionId, profile }
  const destination = [connectionId, profile || 'default'].filter(Boolean).join(' / ')
  const confirmed = await confirm({
    title: translateNow('skills.hub.policyAsk'),
    description: `${identifier}
${translateNow('skills.configuringProfile')} ${destination}
${translateNow('skills.changesApplyNewSessions')}`,
    confirmLabel: translateNow('skills.hub.install')
  })

  // A pending website request cannot follow the user to a different agent.
  if (!confirmed || connectionId !== getApiRequestConnection() || profile !== getApiRequestProfile()) {
    return
  }

  notify({
    kind: 'success',
    title: translateNow('skills.hub.installStarted', identifier),
    message: translateNow('skills.hub.actionLog')
  })
  await installHubSkill(identifier, scope).catch(err =>
    notifyHubActionFailed(err, translateNow('skills.hub.actionFailed'), identifier, scope)
  )
}
