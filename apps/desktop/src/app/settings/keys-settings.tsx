import { useCallback, useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'

import { ScopedCommandSearch, type ScopedCommandSearchItem } from '@/components/ui/scoped-command-search'
import { useI18n } from '@/i18n'

import { CredentialKeyCard, credentialPlaceholder, credentialRowLabel } from './credential-key-ui'
import { useEnvCredentials } from './env-credentials'
import { asText } from './helpers'
import { SettingsContent, SettingsSkeleton } from './primitives'
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

const credentialElementId = (key: string) => `credential-key-${key}`

export function KeysSettings({ view }: KeysSettingsProps) {
  const { t } = useI18n()
  const { rowProps, vars } = useEnvCredentials()
  const [, setSearchParams] = useSearchParams()
  const [openKey, setOpenKey] = useState<null | string>(null)
  const [query, setQuery] = useState('')

  useEffect(() => {
    setOpenKey(null)
    setQuery('')
  }, [view])

  const entries = useMemo(() => {
    if (!vars) {
      return []
    }

    const cats = VIEW_CATEGORIES[view]

    return Object.entries(vars)
      .filter(([, info]) => !info.channel_managed && cats.includes(asText(info.category)))
      .sort(([a], [b]) => a.localeCompare(b))
  }, [vars, view])

  const searchItems = useMemo<ScopedCommandSearchItem[]>(
    () =>
      entries.map(([key, info]) => ({
        description: info.description?.trim() || key,
        id: key,
        keywords: [
          key,
          asText(info.description),
          asText(info.url),
          ...(Array.isArray(info.tools) ? info.tools.map(asText) : [])
        ].filter(Boolean),
        label: credentialRowLabel(key, info)
      })),
    [entries]
  )

  const renderableKeys = useMemo(() => new Set(entries.map(([key]) => key)), [entries])

  const resolveDeepLink = useCallback((key: string) => {
    setQuery('')
    setOpenKey(key)
  }, [])

  const deepLinkReady = useCallback((key: string) => renderableKeys.has(key), [renderableKeys])

  const selectSearchResult = useCallback(
    (item: ScopedCommandSearchItem) => {
      setQuery('')
      setSearchParams(
        previous => {
          const next = new URLSearchParams(previous)
          next.set('key', item.id)

          return next
        },
        { replace: true }
      )
    },
    [setSearchParams]
  )

  // Deep link from Capabilities env-var rows (?tab=keys&key=<ENV_KEY>): scroll
  // the credential card into view, flash it, and expand it. Only consume keys
  // rendered by this sub-view so a stale Tools/Settings pairing cannot keep
  // clearing the user's search while the target remains impossible to mount.
  useDeepLinkHighlight({
    elementId: credentialElementId,
    onResolve: resolveDeepLink,
    param: 'key',
    ready: deepLinkReady
  })

  if (!vars) {
    return <SettingsSkeleton search sections={[{ rows: 5 }]} />
  }

  const searchLabel = view === 'tools' ? t.settings.keys.searchTools : t.settings.keys.searchSettings

  return (
    <SettingsContent>
      {entries.length > 0 ? (
        <>
          <div className="pb-3">
            <ScopedCommandSearch
              emptyMessage={t.settings.keys.noMatch}
              items={searchItems}
              onSelect={selectSearchResult}
              onValueChange={setQuery}
              placeholder={searchLabel}
              value={query}
            />
          </div>

          <div className="grid gap-2">
            {entries.map(([key, info]) => {
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
        </>
      ) : (
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
