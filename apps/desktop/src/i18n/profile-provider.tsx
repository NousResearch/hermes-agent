import { useStore } from '@nanostores/react'
import type { ReactNode } from 'react'

import { $activeGatewayProfile } from '@/store/profile'
import { $connection } from '@/store/session'

import { I18nProvider } from './context'

/** Keep gateway/store imports outside the shared i18n context's import graph. */
export function ProfileI18nProvider({ children }: { children: ReactNode }) {
  const profile = useStore($activeGatewayProfile)
  const connection = useStore($connection)

  // The API captures the real read origin (including legacy profile overrides).
  // This is an invalidation key, not an explicit local/registry request pin.
  const scopeKey = JSON.stringify([
    connection?.connectionId ?? null,
    connection?.registryScoped ?? false,
    connection?.baseUrl,
    connection?.profile,
    profile
  ])

  return <I18nProvider scopeKey={scopeKey}>{children}</I18nProvider>
}
