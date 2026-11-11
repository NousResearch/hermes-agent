import { LOCAL_CONNECTION_ID } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { useI18n } from '@/i18n'
import { hostHas } from '@/lib/host-capabilities'
import { $defaultProfileRoute, setDefaultProfile } from '@/store/default-profile'
import { notify, notifyError } from '@/store/notifications'
import type { ProfileInfo } from '@/types/hermes'

import { settingsScopeLabel } from '../settings/profile-scope'

import type { AgentSettingsOwner } from './data'

/** The owner's default-agent state and the one write that changes it. There is no "unset": another
 *  agent becoming the default is the only way off, so callers render a set-only control. */
export function useDefaultAgent(owner: AgentSettingsOwner, profile?: ProfileInfo) {
  const { t } = useI18n()
  const defaultRoute = useStore($defaultProfileRoute)
  const [saving, setSaving] = useState(false)
  const route = { connectionId: owner.connectionId ?? null, profile: owner.profile }
  const isDefault = defaultRoute?.connectionId === route.connectionId && defaultRoute.profile === route.profile

  const canSetDefault =
    Boolean(window.hermesDesktop?.profile?.setDefault) &&
    (!hostHas('servingOriginConnection') || !route.connectionId || route.connectionId === LOCAL_CONNECTION_ID)

  async function makeDefault() {
    setSaving(true)

    try {
      await setDefaultProfile(route)
      notify({
        kind: 'success',
        title: t.profiles.defaultSet(profile ? settingsScopeLabel(profile) : owner.profile),
        message: t.profiles.defaultDescription
      })
    } catch (error) {
      notifyError(error, t.profiles.failedSetDefault)
    } finally {
      setSaving(false)
    }
  }

  return { canSetDefault, isDefault, makeDefault, saving }
}
