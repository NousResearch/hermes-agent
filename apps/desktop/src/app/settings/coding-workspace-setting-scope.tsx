import { useStore } from '@nanostores/react'

import { profileScopeKey } from '@/hermes'
import { $connection } from '@/store/session'
import { $settingsScopeProfile } from '@/store/settings-scope'

import { CodingWorkspaceSetting } from './coding-workspace-setting'

export function ScopedCodingWorkspaceSetting() {
  const connection = useStore($connection)
  const profile = useStore($settingsScopeProfile)
  const scope = { connectionId: connection?.connectionId, profile }

  return <CodingWorkspaceSetting key={profileScopeKey(scope)} profile={scope} />
}
