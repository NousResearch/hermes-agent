import { useMemo } from 'react'

import type { EnvVarInfo } from '@/types/hermes'

import { CredentialKeyCard, credentialPlaceholder, credentialRowLabel } from './credential-key-ui'
import { useEnvCredentials } from './env-credentials'
import { asText } from './helpers'
import { LoadingState, SettingsContent } from './primitives'

// Sub-views surfaced as sidebar subnav under Tools & Keys (see settings/index.tsx).
export const KEYS_VIEWS = ['tools', 'settings'] as const

export type KeysView = (typeof KEYS_VIEWS)[number]

// Providers live on their own page; messaging-platform credentials live on the
// dedicated Messaging page (and are hidden here via `channel_managed`). This
// view covers tool API keys plus server/setting env vars (API server, webhook,
// gateway), which fold into the Settings subnav.

// Backend categories that surface under each subnav. Platform credentials use the
// `messaging` category but are flagged ``channel_managed`` and configured on
// the Messaging page; only gateway-wide ``messaging`` rows (e.g. GATEWAY_PROXY)
// appear here alongside ``setting``.
const VIEW_CATEGORIES: Record<KeysView, readonly string[]> = {
  settings: ['setting', 'messaging'],
  tools: ['tool']
}

export function KeysSettings({ view }: KeysSettingsProps) {
  const { rowProps, vars } = useEnvCredentials()

  const groups = useMemo(() => {
    if (!vars) {
      return []
    }

    return KEYS_VIEWS.flatMap(v => {
      const cats = VIEW_CATEGORIES[v]

      const entries = Object.entries(vars)
        .filter(([, info]) => !info.channel_managed && cats.includes(asText(info.category)))
        .sort(([a], [b]) => a.localeCompare(b))

      return entries.length === 0 ? [] : [{ category: v, entries }]
    })
  }, [vars])

  if (!vars) {
    return <LoadingState label="Loading API keys and credentials..." />
  }

  const visible = groups.filter(g => g.category === view)

  return (
    <SettingsContent>
      {visible.map(group => (
        <div className="grid gap-1" key={group.category}>
          {group.entries.map(([key, info]: [string, EnvVarInfo]) => {
            const label = credentialRowLabel(key, info)

            return (
              <CredentialKeyCard
                info={info}
                key={key}
                label={label}
                placeholder={credentialPlaceholder(key, info, label)}
                rowProps={rowProps}
                varKey={key}
              />
            )
          })}
        </div>
      ))}

      {visible.length === 0 && (
        <div className="rounded-lg border border-dashed border-(--ui-stroke-tertiary) px-4 py-8 text-center text-[length:var(--conversation-caption-font-size)] text-muted-foreground">
          Nothing configured in this category yet.
        </div>
      )}
    </SettingsContent>
  )
}

interface KeysSettingsProps {
  view: KeysView
}
