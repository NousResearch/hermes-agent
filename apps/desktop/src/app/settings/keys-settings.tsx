import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Field, FieldHint } from '@/components/ui/field'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import { KeyRound, Plus } from '@/lib/icons'
import { $settingsRequestProfile } from '@/store/settings-scope'

import { CREDENTIAL_CONTROL_CLASS, CredentialKeyCard, credentialPlaceholder, credentialRowLabel } from './credential-key-ui'
import { useEnvCredentials } from './env-credentials'
import { asText } from './helpers'
import { SectionHeading, SettingsContent, SettingsSkeleton } from './primitives'
import { SettingsProfileScope } from './profile-scope'
import { useDeepLinkHighlight } from './use-deep-link-highlight'

// Sub-views surfaced as sidebar subnav under Tools & Keys (see settings/index.tsx).
export const KEYS_VIEWS = ['tools', 'settings', 'custom'] as const

export type KeysView = (typeof KEYS_VIEWS)[number]

// Providers live on their own page; messaging-platform credentials live on the
// dedicated Messaging page (and are hidden here via `channel_managed`). This
// view covers tool API keys plus server/setting env vars (API server, webhook,
// gateway), which fold into the Settings subnav.

// Backend categories that surface under each subnav. Platform credentials use the
// `messaging` category but are flagged ``channel_managed`` and configured on
// the Messaging page; only gateway-wide ``messaging`` rows (e.g. GATEWAY_PROXY)
// appear here alongside ``setting``. ``custom`` rows are arbitrary on-disk .env
// keys the backend synthesises (GET /api/env emits them with no catalog entry).
const VIEW_CATEGORIES: Record<KeysView, readonly string[]> = {
  custom: ['custom'],
  settings: ['setting', 'messaging'],
  tools: ['tool']
}

const credentialElementId = (key: string) => `credential-key-${key}`

const customKeyInputId = 'settings-custom-key-name'
const customKeyFeedbackId = 'settings-custom-key-feedback'

// Mirror of the backend env-name guard (hermes_cli/config.py `_ENV_VAR_NAME_RE`).
// The server also keeps a name denylist; the client validates the shape only —
// a denylisted name simply surfaces the server's rejection on save.
const ENV_VAR_NAME_RE = /^[A-Za-z_][A-Za-z0-9_]*$/

// Add-a-key row for the Custom sub-view: register an arbitrary env-var name,
// which drops a pending row into the list (via addKey) ready for its value to
// be typed and saved through the normal credential save path. It renders even
// with zero custom entries — it is the only way to create the first one.
function CustomKeyAddForm({ addKey, existing }: { addKey: (key: string) => void; existing: Set<string> }) {
  const { t } = useI18n()
  const [newKey, setNewKey] = useState('')
  const trimmed = newKey.trim().toUpperCase()
  const nameValid = ENV_VAR_NAME_RE.test(trimmed)
  const showInvalid = trimmed.length > 0 && !nameValid
  const alreadyEditing = existing.has(trimmed)
  // A name that already exists (in this or any other category) can't be added.
  // Without a reason on screen the disabled Add button is a dead end, so this
  // state gets its own message. Mutually exclusive with `showInvalid`.
  const showConflict = trimmed.length > 0 && nameValid && alreadyEditing
  const blocked = showInvalid || showConflict

  const feedback = showInvalid
    ? t.settings.keys.invalidKeyName
    : showConflict
      ? t.settings.keys.customKeyExists
      : null

  const handleAdd = () => {
    if (!nameValid || alreadyEditing) {
      return
    }

    addKey(trimmed)
    setNewKey('')
  }

  return (
    <div className="mt-3 rounded-lg border border-dashed border-(--ui-stroke-tertiary) p-4">
      <p className="mb-2 text-[length:var(--conversation-caption-font-size)] text-muted-foreground">
        {t.settings.keys.addCustomKey}
      </p>
      {/* The label names the FIELD, and is the accessible name too — the action
          lives in the caption above, so visible text and accessible name agree
          (WCAG 2.5.3 Label in Name) instead of an aria-label overriding it. */}
      <Field htmlFor={customKeyInputId} label={t.settings.keys.customKeyName}>
        <div className="flex items-start gap-2">
          <div className="min-w-0 flex-1">
            <Input
              aria-describedby={blocked ? customKeyFeedbackId : undefined}
              aria-invalid={blocked || undefined}
              className={CREDENTIAL_CONTROL_CLASS}
              id={customKeyInputId}
              onChange={e => setNewKey(e.target.value)}
              onKeyDown={e => {
                // Ignore the Enter that confirms an IME composition candidate.
                if (e.key === 'Enter' && !e.nativeEvent.isComposing) {
                  e.preventDefault()
                  handleAdd()
                }
              }}
              placeholder={t.settings.keys.customKeyNamePlaceholder}
              value={newKey}
            />
            {feedback && (
              <div id={customKeyFeedbackId}>
                <FieldHint error>{feedback}</FieldHint>
              </div>
            )}
          </div>
          <Button className="h-8" disabled={!nameValid || alreadyEditing} onClick={handleAdd} size="sm">
            <Plus />
            {t.settings.keys.add}
          </Button>
        </div>
      </Field>
    </div>
  )
}

export function KeysSettings({ view }: KeysSettingsProps) {
  const { t } = useI18n()
  // Shared settings "Applies to" scope: fetch + edit the selected profile's
  // env store instead of the active one (undefined → active, the default
  // path — request-shaped so the API helpers never see a primary-targeting
  // null).
  const scopeProfile = useStore($settingsRequestProfile)
  const { addKey, rowProps, vars } = useEnvCredentials(scopeProfile)
  const [openKey, setOpenKey] = useState<null | string>(null)

  useEffect(() => {
    setOpenKey(null)
  }, [scopeProfile, view])

  const entries = useMemo(() => {
    if (!vars) {
      return []
    }

    const cats = VIEW_CATEGORIES[view]

    return Object.entries(vars)
      .filter(([, info]) => !info.channel_managed && cats.includes(asText(info.category)))
      .sort(([a], [b]) => a.localeCompare(b))
  }, [vars, view])

  const renderableKeys = useMemo(() => new Set(entries.map(([key]) => key)), [entries])

  // Names the add-key form must refuse: anything currently in the list,
  // including pending rows added this session (they live in `vars`). Deliberately
  // NOT `edits` — edits survive a profile-scope reload that clears `vars`, so a
  // name typed but never saved in another profile would keep Add disabled here
  // with no visible row to explain it.
  const existingKeys = useMemo(() => new Set(Object.keys(vars ?? {})), [vars])

  const resolveDeepLink = useCallback((key: string) => {
    setOpenKey(key)
  }, [])

  const deepLinkReady = useCallback((key: string) => renderableKeys.has(key), [renderableKeys])

  // Deep link from ⌘K / Capabilities env-var rows (?tab=keys&key=<ENV_KEY>):
  // scroll the credential card into view, flash it, and expand it. Only
  // consume keys rendered by this sub-view so a stale Tools/Settings pairing
  // cannot keep trying to mount a target this pane never shows.
  useDeepLinkHighlight({
    elementId: credentialElementId,
    onResolve: resolveDeepLink,
    param: 'key',
    ready: deepLinkReady
  })

  if (!vars) {
    return <SettingsSkeleton sections={[{ rows: 5 }]} />
  }

  return (
    <SettingsContent>
      <SettingsProfileScope className="mb-5" />
      {view === 'custom' ? (
        <>
          <SectionHeading icon={KeyRound} title={t.settings.keys.customTitle} />
          <p className="-mt-1 mb-3 text-[length:var(--conversation-caption-font-size)] text-muted-foreground">
            {t.settings.keys.customHint}
          </p>
        </>
      ) : null}
      {entries.length > 0 ? (
        <div className="grid gap-2">
          {entries.map(([key, info]) => {
            const label = credentialRowLabel(key, info)

            return (
              <div className="scroll-mt-6 rounded-[6px]" id={credentialElementId(key)} key={key}>
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
      ) : (
        <div className="rounded-lg border border-dashed border-(--ui-stroke-tertiary) px-4 py-8 text-center text-[length:var(--conversation-caption-font-size)] text-muted-foreground">
          {t.settings.keys.empty}
        </div>
      )}
      {view === 'custom' ? <CustomKeyAddForm addKey={addKey} existing={existingKeys} /> : null}
    </SettingsContent>
  )
}

interface KeysSettingsProps {
  view: KeysView
}
