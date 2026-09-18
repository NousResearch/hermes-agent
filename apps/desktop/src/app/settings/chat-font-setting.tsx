import { useStore } from '@nanostores/react'
import { useEffect, useRef, useState } from 'react'

import { profileScopeKey } from '@/api/client'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { saveHermesConfig } from '@/hermes'
import { useI18n } from '@/i18n'
import { notifyError } from '@/store/notifications'
import { $settingsOwner } from '@/store/settings-scope'
import { CHAT_FONT_SUGGESTIONS, normalizeChatFontFamily, setChatFontFamilyFromConfig } from '@/themes/chat-font'
import type { HermesConfigRecord } from '@/types/hermes'

import { hermesConfigCacheWriter, useHermesConfigRecord } from '../hooks/use-config-record'

import { getNested, setNested } from './helpers'
import { ListRow } from './primitives'

const AUTOSAVE_DELAY_MS = 550
const CONFIG_PATH = 'desktop.font_family'

function fontFamilyFromConfig(config: HermesConfigRecord): string {
  return normalizeChatFontFamily(getNested(config, CONFIG_PATH))
}

/**
 * Chat / UI face picker. Same seed → autosave → rollback contract as
 * `TerminalFontSetting`; the live value publishes through `$chatFontFamily`
 * so the theme paint (`--dt-font-sans`) follows every keystroke.
 */
export function ChatFontSetting() {
  const settingsOwner = useStore($settingsOwner)

  // ponytail: a draft lives exactly as long as its full connection/profile owner.
  return (
    <ChatFontSettingInner
      key={settingsOwner ? profileScopeKey(settingsOwner) : 'unavailable'}
      settingsOwner={settingsOwner}
    />
  )
}

function ChatFontSettingInner({ settingsOwner }: { settingsOwner: ReturnType<typeof $settingsOwner.get> }) {
  const { t } = useI18n()
  const copy = t.settings.appearance
  const { data: loadedConfig } = useHermesConfigRecord(settingsOwner ?? undefined, Boolean(settingsOwner))
  const [draft, setDraft] = useState<string | null>(null)
  const [saveVersion, setSaveVersion] = useState(0)
  const saveVersionRef = useRef(0)

  const cancelPendingSave = () => {
    saveVersionRef.current = 0
  }

  useEffect(() => {
    if (!settingsOwner || !loadedConfig) {
      setChatFontFamilyFromConfig('')

      return
    }

    if (draft !== null) {
      return
    }

    const value = fontFamilyFromConfig(loadedConfig)
    setDraft(value)
    setChatFontFamilyFromConfig(value)
  }, [draft, loadedConfig, settingsOwner])

  useEffect(() => {
    if (!settingsOwner || draft === null || saveVersion === 0 || !loadedConfig) {
      return
    }

    let cancelled = false
    const version = saveVersion
    const value = normalizeChatFontFamily(draft)

    if (value === fontFamilyFromConfig(loadedConfig)) {
      return
    }

    const rollback = fontFamilyFromConfig(loadedConfig)

    const timeout = window.setTimeout(() => {
      if ($settingsOwner.get() !== settingsOwner || saveVersionRef.current !== version) {
        return
      }

      const next = setNested(loadedConfig, CONFIG_PATH, value)

      // Sparse patch: PUT /api/config deep-merges; echoing the cached snapshot
      // would overwrite keys other surfaces changed since it loaded.
      void saveHermesConfig(setNested({}, CONFIG_PATH, value), settingsOwner ?? undefined)
        .then(result => {
          if (!result.ok) {
            throw new Error(t.settings.config.autosaveFailed)
          }

          if (cancelled || $settingsOwner.get() !== settingsOwner || saveVersionRef.current !== version) {
            return
          }

          hermesConfigCacheWriter(settingsOwner ?? undefined)(next)
        })
        .catch(error => {
          if (cancelled || $settingsOwner.get() !== settingsOwner || saveVersionRef.current !== version) {
            return
          }

          cancelPendingSave()
          setSaveVersion(0)
          setDraft(rollback)
          setChatFontFamilyFromConfig(rollback)
          notifyError(error, t.settings.config.autosaveFailed)
        })
    }, AUTOSAVE_DELAY_MS)

    return () => {
      cancelled = true
      window.clearTimeout(timeout)
    }
  }, [draft, loadedConfig, saveVersion, settingsOwner, t.settings.config.autosaveFailed])

  const update = (value: string) => {
    saveVersionRef.current += 1
    setDraft(value)
    setSaveVersion(saveVersionRef.current)
    setChatFontFamilyFromConfig(value)
  }

  const value = draft ?? ''

  return (
    <ListRow
      below={
        <div className="mt-3 space-y-2">
          <div className="flex items-center gap-3">
            <Input
              aria-label={copy.chatFontTitle}
              className="flex-1"
              disabled={draft === null}
              list="hermes-chat-font-families"
              onChange={event => update(event.target.value)}
              placeholder={copy.chatFontPlaceholder}
              value={value}
            />
            <Button disabled={!value || draft === null} onClick={() => update('')} size="inline" variant="text">
              {copy.chatFontReset}
            </Button>
          </div>
          <datalist id="hermes-chat-font-families">
            {CHAT_FONT_SUGGESTIONS.map(font => (
              <option key={font} value={font} />
            ))}
          </datalist>
          {/* Inherits --dt-font-sans, so it IS the live result, not a simulation. */}
          <div
            aria-label={copy.chatFontPreview}
            className="overflow-hidden px-1 py-2 text-sm text-(--ui-text-secondary)"
          >
            <span className="mr-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
              {copy.chatFontPreview}
            </span>
            <span>{copy.chatFontSample}</span>
          </div>
        </div>
      }
      description={copy.chatFontDesc}
      title={copy.chatFontTitle}
      wide
    />
  )
}
