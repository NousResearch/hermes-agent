import type { QueryClient } from '@tanstack/react-query'
import { useEffect } from 'react'

import { getCronJobs, type HermesConfigRecord, type SkillInfo } from '@/hermes'
import {
  broadcastActiveDesktopProfile,
  onDesktopStateSync,
  parseCronSyncValue,
  parseNamedEnabledSyncValue
} from '@/lib/desktop-state-sync'
import { setCronJobs, updateCronJobs } from '@/store/cron'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import { isSecondaryWindow } from '@/store/windows'

interface MenuBarCompanionSyncOptions {
  activeGatewayProfile: string
  queryClient: QueryClient
  refreshCurrentModel: (force?: boolean) => Promise<void>
  refreshHermesConfig: () => Promise<void>
}

export function useMenuBarCompanionSync({
  activeGatewayProfile,
  queryClient,
  refreshCurrentModel,
  refreshHermesConfig
}: MenuBarCompanionSyncOptions) {
  useEffect(() => {
    if (isSecondaryWindow()) {
      return
    }

    broadcastActiveDesktopProfile(activeGatewayProfile)

    return onDesktopStateSync(message => {
      if (message.type === 'active-profile-request') {
        broadcastActiveDesktopProfile($activeGatewayProfile.get())

        return
      }

      if (message.type !== 'changed') {
        return
      }

      if (
        message.profile &&
        normalizeProfileKey(message.profile) !== normalizeProfileKey($activeGatewayProfile.get())
      ) {
        return
      }

      if (message.domain === 'model') {
        void refreshCurrentModel(true)
        void queryClient.invalidateQueries({ queryKey: ['model-options'] })

        return
      }

      if (message.domain === 'config') {
        void refreshHermesConfig()
        void queryClient.invalidateQueries({ queryKey: ['hermes-config-record'] })

        return
      }

      if (message.domain === 'cron') {
        const changed = parseCronSyncValue(message.value)

        if (changed) {
          updateCronJobs(jobs => jobs.map(job => (job.id === changed.id ? { ...job, ...changed } : job)))
        }

        void getCronJobs()
          .then(setCronJobs)
          .catch(() => undefined)

        return
      }

      if (message.domain === 'skills') {
        const changed = parseNamedEnabledSyncValue(message.value)

        if (changed) {
          queryClient.setQueryData<SkillInfo[]>(['skills-list'], current =>
            current?.map(skill => (skill.name === changed.name ? { ...skill, enabled: changed.enabled } : skill))
          )
        }

        void queryClient.invalidateQueries({ queryKey: ['skills-list'] })

        return
      }

      if (message.domain === 'mcp') {
        const changed = parseNamedEnabledSyncValue(message.value)

        if (changed) {
          queryClient.setQueryData<HermesConfigRecord>(['hermes-config-record'], current => {
            const servers = current?.mcp_servers

            if (!current || !servers || typeof servers !== 'object' || Array.isArray(servers)) {
              return current
            }

            const server = (servers as Record<string, unknown>)[changed.name]

            if (!server || typeof server !== 'object' || Array.isArray(server)) {
              return current
            }

            return {
              ...current,
              mcp_servers: {
                ...(servers as Record<string, unknown>),
                [changed.name]: { ...(server as Record<string, unknown>), enabled: changed.enabled }
              }
            }
          })
        }

        void queryClient.invalidateQueries({ queryKey: ['hermes-config-record'] })
      }
    })
  }, [activeGatewayProfile, queryClient, refreshCurrentModel, refreshHermesConfig])
}
