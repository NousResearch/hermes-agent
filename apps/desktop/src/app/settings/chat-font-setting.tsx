import { useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { saveHermesConfig } from '@/hermes'
import { useI18n } from '@/i18n'
import { notifyError } from '@/store/notifications'
import { CHAT_FONT_SUGGESTIONS, normalizeChatFontFamily, setChatFontFamilyFromConfig } from '@/themes/chat-font'
import type { HermesConfigRecord } from '@/types/hermes'

import { setHermesConfigCache, useHermesConfigRecord } from '../hooks/use-config-record'
import { useOnProfileSwitch } from '../hooks/use-on-profile-switch'

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
  const { data: loadedConfig, dataUpdatedAt } = useHermesConfigRecord()
  const [draft, setDraft] = useState<string | null>(null)
  // dataUpdatedAt of the record the previous profile was looking at; the seed
  // effect refuses to reseed while the query still carries that stamp. A
  // structurally-shared refetch keeps the object reference but bumps the stamp.
  const [staleStamp, setStaleStamp] = useState<null | number>(null)
  const [saveVersion, setSaveVersion] = useState(0)
  const saveVersionRef = useRef(0)

  const cancelPendingSave = () => {
    saveVersionRef.current = 0
  }

  useEffect(() => {
    if (!loadedConfig || draft !== null || dataUpdatedAt === staleStamp) {
      return
    }

    const value = fontFamilyFromConfig(loadedConfig)
    setDraft(value)
    setChatFontFamilyFromConfig(value)
  }, [dataUpdatedAt, draft, loadedConfig, staleStamp])

  useOnProfileSwitch(() => {
    saveVersionRef.current += 1
    setDraft(null)
    setStaleStamp(dataUpdatedAt)
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
      void saveHermesConfig(setNested({}, CONFIG_PATH, value))
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
  }, [draft, loadedConfig, saveVersion, t.settings.config.autosaveFailed])

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
          <div aria-label={copy.chatFontPreview} className="overflow-hidden px-1 py-2 text-sm text-(--ui-text-secondary)">
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
