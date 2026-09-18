import { useStore } from '@nanostores/react'
import { useEffect, useRef, useState } from 'react'

import { profileScopeKey } from '@/api/client'
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
import { $settingsOwner } from '@/store/settings-scope'
import type { HermesConfigRecord } from '@/types/hermes'

import { hermesConfigCacheWriter, useHermesConfigRecord } from '../hooks/use-config-record'

import { getNested, setNested } from './helpers'
import { ListRow } from './primitives'

const AUTOSAVE_DELAY_MS = 550

function fontFamilyFromConfig(config: HermesConfigRecord): string {
  return normalizeTerminalFontFamily(getNested(config, 'terminal.font_family'))
}

export function TerminalFontSetting() {
  const settingsOwner = useStore($settingsOwner)

  // ponytail: a draft lives exactly as long as its full connection/profile owner.
  return (
    <TerminalFontSettingInner
      key={settingsOwner ? profileScopeKey(settingsOwner) : 'unavailable'}
      settingsOwner={settingsOwner}
    />
  )
}

function TerminalFontSettingInner({ settingsOwner }: { settingsOwner: ReturnType<typeof $settingsOwner.get> }) {
  const { t } = useI18n()
  const copy = t.settings.appearance
  const { data: loadedConfig } = useHermesConfigRecord(settingsOwner ?? undefined, Boolean(settingsOwner))
  const [draft, setDraft] = useState<string | null>(null)
  const [saveVersion, setSaveVersion] = useState(0)
  const saveVersionRef = useRef(0)

  // Lexically outside every useEffect so async save callbacks can cancel the
  // in-flight version without assigning to a ref inside an effect body.
  const cancelPendingSave = () => {
    saveVersionRef.current = 0
  }

  useEffect(() => {
    if (!settingsOwner || !loadedConfig) {
      setTerminalFontFamilyFromConfig('')

      return
    }

    if (draft !== null) {
      return
    }

    const value = fontFamilyFromConfig(loadedConfig)
    setDraft(value)
    setTerminalFontFamilyFromConfig(value)
  }, [draft, loadedConfig, settingsOwner])

  useEffect(() => {
    if (!settingsOwner || draft === null || saveVersion === 0 || !loadedConfig) {
      return
    }

    let cancelled = false
    const version = saveVersion
    const value = normalizeTerminalFontFamily(draft)

    // Already persisted (or a cache refresh confirmed it) — nothing to save.
    // This also terminates the effect re-run after a successful save updates
    // the shared config cache.
    if (value === fontFamilyFromConfig(loadedConfig)) {
      return
    }

    // The last successfully saved value IS what the shared config cache
    // holds — successful saves write it back via setHermesConfigCache, so
    // rollback re-derives from there instead of mirroring into a ref.
    const rollback = fontFamilyFromConfig(loadedConfig)

    const timeout = window.setTimeout(() => {
      if ($settingsOwner.get() !== settingsOwner || saveVersionRef.current !== version) {
        return
      }

      const next = setNested(loadedConfig, 'terminal.font_family', value)

      // Sparse patch: PUT /api/config deep-merges, and echoing the cached
      // snapshot would overwrite keys other surfaces changed since it loaded.
      void saveHermesConfig(setNested({}, 'terminal.font_family', value), settingsOwner ?? undefined)
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
          setTerminalFontFamilyFromConfig(rollback)
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
            <span> ~/project git:main ❯</span>
          </div>
        </div>
      }
      description={copy.terminalFontDesc}
      title={copy.terminalFontTitle}
      wide
    />
  )
}
