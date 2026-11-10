import { useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { saveHermesConfig } from '@/hermes'
import { useI18n } from '@/i18n'
import { notifyError } from '@/store/notifications'
import {
  CHAT_FONT_SUGGESTIONS,
  normalizeChatFontFamily,
  resolveChatFontFamily,
  setChatFontFamilyFromConfig
} from '@/themes/chat-font'
import type { HermesConfigRecord } from '@/types/hermes'

import { setHermesConfigCache, useHermesConfigRecord } from '../hooks/use-config-record'
import { useOnProfileSwitch } from '../hooks/use-on-profile-switch'
import { useProfileSwitchLatch } from '../hooks/use-profile-switch-latch'

import { ComboboxInput } from './combobox-input'
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
  const { t } = useI18n()
  const copy = t.settings.appearance
  const { data: loadedConfig, dataUpdatedAt, writeScope } = useHermesConfigRecord()
  const [draft, setDraft] = useState<string | null>(null)
  // The seed effect refuses to reseed while the query still carries the
  // previous profile's stamp. A structurally-shared refetch keeps the object
  // reference but bumps the stamp.
  const { arm: armProfileLatch, pending: profilePending } = useProfileSwitchLatch({ dataUpdatedAt })
  const [saveVersion, setSaveVersion] = useState(0)
  const saveVersionRef = useRef(0)

  const cancelPendingSave = () => {
    saveVersionRef.current = 0
  }

  useEffect(() => {
    if (!loadedConfig || draft !== null || profilePending) {
      return
    }

    const value = fontFamilyFromConfig(loadedConfig)
    setDraft(value)
    setChatFontFamilyFromConfig(value)
  }, [draft, loadedConfig, profilePending])

  useOnProfileSwitch(() => {
    saveVersionRef.current += 1
    setDraft(null)
    armProfileLatch()
    setSaveVersion(0)
    setChatFontFamilyFromConfig('')
  })

  useEffect(() => {
    if (draft === null || saveVersion === 0 || !loadedConfig) {
      return
    }

    const version = saveVersion
    const value = normalizeChatFontFamily(draft)

    if (value === fontFamilyFromConfig(loadedConfig)) {
      return
    }

    const rollback = fontFamilyFromConfig(loadedConfig)

    const timeout = window.setTimeout(() => {
      const next = setNested(loadedConfig, CONFIG_PATH, value)

      // Sparse patch: PUT /api/config deep-merges; echoing the cached snapshot
      // would overwrite keys other surfaces changed since it loaded.
      void saveHermesConfig(setNested({}, CONFIG_PATH, value), writeScope)
        .then(result => {
          if (!result.ok) {
            throw new Error(t.settings.config.autosaveFailed)
          }

          if (saveVersionRef.current !== version) {
            return
          }

          setHermesConfigCache(next)
        })
        .catch(error => {
          if (saveVersionRef.current !== version) {
            return
          }

          cancelPendingSave()
          setSaveVersion(0)
          setDraft(rollback)
          setChatFontFamilyFromConfig(rollback)
          notifyError(error, t.settings.config.autosaveFailed)
        })
    }, AUTOSAVE_DELAY_MS)

    return () => window.clearTimeout(timeout)
  }, [draft, loadedConfig, saveVersion, t.settings.config.autosaveFailed, writeScope])

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
            <ComboboxInput
              aria-label={copy.chatFontTitle}
              className="flex-1"
              disabled={draft === null}
              onChange={update}
              options={CHAT_FONT_SUGGESTIONS}
              placeholder={copy.chatFontPlaceholder}
              renderOption={font => (
                <span style={{ fontFamily: resolveChatFontFamily(font, 'var(--dt-font-sans)') }}>{font}</span>
              )}
              value={value}
            />
            <Button
              className="h-10 shrink-0 rounded-xl border border-[#e2e8f0] bg-white px-3.5 text-[13px] font-semibold text-[#475569] shadow-none hover:bg-[#f8fafc] dark:border-(--ui-stroke-tertiary) dark:bg-(--ui-bg-quinary) dark:text-(--ui-text-secondary)"
              disabled={!value || draft === null}
              onClick={() => update('')}
              size="inline"
              variant="outline"
            >
              {copy.chatFontReset}
            </Button>
          </div>
          {/* Inherits --dt-font-sans, so it IS the live result, not a simulation. */}
          <div
            aria-label={copy.chatFontPreview}
            className="overflow-hidden rounded-xl border border-[#e2e8f0] bg-[#f7f9fc] px-3.5 py-3 dark:border-(--ui-stroke-tertiary) dark:bg-(--ui-bg-quinary)"
            data-appearance-font-preview="chat"
          >
            <div className="text-[12px] font-medium text-[#94a3b8] dark:text-(--ui-text-tertiary)">
              {copy.chatFontPreview}
            </div>
            <div className="mt-1 text-sm text-[#0f172a] dark:text-foreground">{copy.chatFontSample}</div>
          </div>
        </div>
      }
      description={copy.chatFontDesc}
      title={copy.chatFontTitle}
      wide
    />
  )
}
