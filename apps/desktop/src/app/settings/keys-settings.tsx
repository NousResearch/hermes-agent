import { useEffect, useMemo, useState } from 'react'

import { useI18n } from '@/i18n'
import type { EnvVarInfo } from '@/types/hermes'

import { CredentialKeyCard, credentialPlaceholder, credentialRowLabel } from './credential-key-ui'
import { useEnvCredentials } from './env-credentials'
import { asText } from './helpers'
import { LoadingState, SettingsContent } from './primitives'
import { useDeepLinkHighlight } from './use-deep-link-highlight'

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
  const { t } = useI18n()
  const { rowProps, vars } = useEnvCredentials()
  const [openKey, setOpenKey] = useState<null | string>(null)

  useEffect(() => {
    setOpenKey(null)
  }, [view])

  // Deep link from Capabilities env-var rows (?tab=keys&key=<ENV_KEY>): scroll
  // the credential card into view, flash it, and expand it. Same mechanism the
  // command palette uses for config fields / archived sessions.
  useDeepLinkHighlight({
    elementId: key => `credential-key-${key}`,
    onResolve: key => setOpenKey(key),
    param: 'key',
    ready: key => Boolean(vars && key in vars)
  })

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
    return <LoadingState label={t.settings.keys.loading} />
  }

  const visible = groups.filter(g => g.category === view)

  return (
    <SettingsContent>
      {visible.map(group => (
        <div className="grid gap-2" key={group.category}>
          {group.entries.map(([key, info]: [string, EnvVarInfo]) => {
            const label = credentialRowLabel(key, info)

            return (
              <div className="scroll-mt-6 rounded-[6px]" id={`credential-key-${key}`} key={key}>
                <CredentialKeyCard
                  expanded={openKey === key}
                  info={info}
                  label={label}
                  onExpand={() => setOpenKey(key)}
                  onToggle={() => setOpenKey(prev => (prev === key ? null : key))}
                  placeholder={credentialPlaceholder(key, info, label)}
                  rowProps={rowProps}
                  varKey={key}
                />
              </div>
            )
          })}
        </div>
      ))}

      {visible.length === 0 && (
        <div className="rounded-lg border border-dashed border-(--ui-stroke-tertiary) px-4 py-8 text-center text-[length:var(--conversation-caption-font-size)] text-muted-foreground">
          {t.settings.keys.empty}
        </div>
      )}
    </SettingsContent>
  )
}

interface KeysSettingsProps {
  view: KeysView
}
