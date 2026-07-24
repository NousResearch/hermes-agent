import { useEffect, useRef, useState } from 'react'

import {
  normalizeTerminalFontFamily,
  resolveTerminalFontFamily,
  setTerminalFontFamilyFromConfig,
  TERMINAL_FONT_SUGGESTIONS
} from '@/app/right-sidebar/terminal/terminal-font'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { saveHermesConfig } from '@/hermes'
import { useI18n } from '@/i18n'
import { notifyError } from '@/store/notifications'
import type { HermesConfigRecord } from '@/types/hermes'

import { setHermesConfigCache, useHermesConfigRecord } from '../hooks/use-config-record'
import { useOnProfileSwitch } from '../hooks/use-on-profile-switch'

import { getNested, setNested } from './helpers'
import { ListRow } from './primitives'

const AUTOSAVE_DELAY_MS = 550

function fontFamilyFromConfig(config: HermesConfigRecord): string {
  return normalizeTerminalFontFamily(getNested(config, 'terminal.font_family'))
}

export function TerminalFontSetting() {
  const { t } = useI18n()
  const copy = t.settings.appearance
  const { data: loadedConfig } = useHermesConfigRecord()
  const [draft, setDraft] = useState<string | null>(null)
  const [saveVersion, setSaveVersion] = useState(0)
  const seededRef = useRef(false)
  const latestConfigRef = useRef<HermesConfigRecord | null>(null)
  const lastSavedRef = useRef('')
  const saveVersionRef = useRef(0)

  useEffect(() => {
    if (!loadedConfig) {
      return
    }

    latestConfigRef.current = loadedConfig

    if (seededRef.current) {
      return
    }

    const value = fontFamilyFromConfig(loadedConfig)
    seededRef.current = true
    lastSavedRef.current = value
    setDraft(value)
    setTerminalFontFamilyFromConfig(value)
  }, [loadedConfig])

  useOnProfileSwitch(() => {
    saveVersionRef.current += 1
    seededRef.current = false
    latestConfigRef.current = null
    lastSavedRef.current = ''
    setDraft(null)
    setSaveVersion(0)
    // Do not show the previous profile's font while the new profile loads.
    setTerminalFontFamilyFromConfig('')
  })

  useEffect(() => {
    if (draft === null || saveVersion === 0) {
      return
    }

    const version = saveVersion
    const value = normalizeTerminalFontFamily(draft)

    const timeout = window.setTimeout(() => {
      const base = latestConfigRef.current

      if (!base) {
        return
      }

      const next = setNested(base, 'terminal.font_family', value)

      void saveHermesConfig(next)
        .then(result => {
          if (!result.ok) {
            throw new Error(t.settings.config.autosaveFailed)
          }

          if (saveVersionRef.current !== version) {
            return
          }

          latestConfigRef.current = next
          lastSavedRef.current = value
          setHermesConfigCache(next)
        })
        .catch(error => {
          if (saveVersionRef.current !== version) {
            return
          }

          const rollback = lastSavedRef.current
          saveVersionRef.current = 0
          setSaveVersion(0)
          setDraft(rollback)
          setTerminalFontFamilyFromConfig(rollback)
          notifyError(error, t.settings.config.autosaveFailed)
        })
    }, AUTOSAVE_DELAY_MS)

    return () => window.clearTimeout(timeout)
  }, [draft, saveVersion, t.settings.config.autosaveFailed])

  const update = (value: string) => {
    saveVersionRef.current += 1
    setDraft(value)
    setSaveVersion(saveVersionRef.current)
    setTerminalFontFamilyFromConfig(value)
  }

  const value = draft ?? ''
  const previewFontFamily = resolveTerminalFontFamily(value)

  return (
    <ListRow
      below={
        <div className="mt-3 space-y-2">
          <div className="flex items-center gap-3">
            <Input
              aria-label={copy.terminalFontTitle}
              className="flex-1"
              disabled={draft === null}
              list="hermes-terminal-font-families"
              onChange={event => update(event.target.value)}
              placeholder={copy.terminalFontPlaceholder}
              value={value}
            />
            <Button disabled={!value || draft === null} onClick={() => update('')} size="inline" variant="text">
              {copy.terminalFontReset}
            </Button>
          </div>
          <datalist id="hermes-terminal-font-families">
            {TERMINAL_FONT_SUGGESTIONS.map(font => (
              <option key={font} value={font} />
            ))}
          </datalist>
          <div
            aria-label={copy.terminalFontPreview}
            className="overflow-hidden px-1 py-2 text-sm text-(--ui-text-secondary)"
            style={{ fontFamily: previewFontFamily }}
          >
            <span className="mr-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
              {copy.terminalFontPreview}
            </span>
            <span>  ~/project  git:main  ❯</span>
          </div>
        </div>
      }
      description={copy.terminalFontDesc}
      title={copy.terminalFontTitle}
      wide
    />
  )
}
