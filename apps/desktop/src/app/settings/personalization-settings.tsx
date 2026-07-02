import { useEffect, useState } from 'react'

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { getHermesConfigRecord, saveHermesConfig } from '@/hermes'
import { useI18n } from '@/i18n'
import { Sparkles } from '@/lib/icons'
import { notifyError } from '@/store/notifications'
import type { HermesConfigRecord } from '@/types/hermes'

import { AboutSettingsBody } from './about-settings'
import { CONTROL_TEXT, EMPTY_SELECT_VALUE, FIELD_DESCRIPTIONS, FIELD_LABELS } from './constants'
import { fieldCopyForSchemaKey } from './field-copy'
import { enumOptionsFor, getNested, prettyName, setNested } from './helpers'
import { ListRow, SectionHeading, SettingsContent } from './primitives'

const PERSONALITY_KEY = 'display.personality'

// 个性化 — the consumer landing section. Top: the 人格 (personality) picker,
// moved here from the former 对话 section (same config key, same option
// source, saved through the same config API). Below: the former 关于 content
// (version, app updates, engine updates, uninstall) via AboutSettingsBody.
export function PersonalizationSettings({ onConfigSaved }: { onConfigSaved?: () => void }) {
  const { t } = useI18n()
  const copy = t.settings.personalization

  const [config, setConfig] = useState<HermesConfigRecord | null>(null)

  useEffect(() => {
    let cancelled = false

    getHermesConfigRecord()
      .then(cfg => {
        if (!cancelled) {
          setConfig(cfg)
        }
      })
      .catch(err => notifyError(err, t.settings.config.failedLoad))

    return () => void (cancelled = true)
  }, [])

  const value = config ? String(getNested(config, PERSONALITY_KEY) ?? '') : ''
  const options = config ? (enumOptionsFor(PERSONALITY_KEY, value, config) ?? ['']) : ['']

  const label =
    fieldCopyForSchemaKey(t.settings.fieldLabels, PERSONALITY_KEY) ??
    fieldCopyForSchemaKey(FIELD_LABELS, PERSONALITY_KEY) ??
    prettyName(PERSONALITY_KEY)

  const description =
    fieldCopyForSchemaKey(t.settings.fieldDescriptions, PERSONALITY_KEY) ??
    fieldCopyForSchemaKey(FIELD_DESCRIPTIONS, PERSONALITY_KEY)

  const handleChange = (next: string) => {
    if (!config) {
      return
    }

    const updated = setNested(config, PERSONALITY_KEY, next === EMPTY_SELECT_VALUE ? '' : next)

    setConfig(updated)
    saveHermesConfig(updated)
      .then(() => onConfigSaved?.())
      .catch(err => notifyError(err, t.settings.config.autosaveFailed))
  }

  return (
    <SettingsContent>
      <div>
        <SectionHeading icon={Sparkles} title={copy.personalityTitle} />
        <p className="p5-section-intro">{copy.personalityIntro}</p>

        <div className="p5-card p5-rows mt-3.5">
          <ListRow
            action={
              <Select disabled={!config} onValueChange={handleChange} value={value || EMPTY_SELECT_VALUE}>
                <SelectTrigger className={CONTROL_TEXT}>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {options.map(option => (
                    <SelectItem key={option || EMPTY_SELECT_VALUE} value={option || EMPTY_SELECT_VALUE}>
                      {option ? prettyName(option) : t.settings.config.none}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            }
            description={description}
            title={label}
          />
        </div>
      </div>

      {/* 关于 content lives here now (item: About merged into 个性化); it keeps
          its own headings (updates / engine / uninstall). */}
      <div className="mt-6">
        <AboutSettingsBody />
      </div>
    </SettingsContent>
  )
}
