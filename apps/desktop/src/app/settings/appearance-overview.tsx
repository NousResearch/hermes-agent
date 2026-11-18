import './appearance-overview.css'

import { useStore } from '@nanostores/react'
import { type ReactNode, useEffect, useId, useRef } from 'react'

import { LanguageSwitcher } from '@/components/language-switcher'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { $chatTextScale, setChatTextScale } from '@/store/chat-text-scale'
import { $spentTipCount, $tipsEnabled, resetTips, setTipsEnabled } from '@/store/tips'
import { $toursEnabled, setToursEnabled } from '@/store/tours'
import { getBaseColors, useTheme } from '@/themes/context'

import { appearanceOverviewCopy } from './appearance-overview-copy'
import { ListRow, RowFootnoteAction, SettingsContent, ToggleRow } from './primitives'
import { SettingsCard, SettingsDisclosure } from './settings-card'
import { SETTING_IDS, settingElementId } from './settings-manifest'
import { forHost, useSettingsPagesCopy } from './settings-pages-copy'
import { useSettingDeepLink } from './use-setting-deep-link'

const modes = ['light', 'dark', 'system'] as const
const ids = SETTING_IDS.appearance

interface AppearanceOverviewProps {
  advanced: ReactNode
}

export function AppearanceOverview({ advanced }: AppearanceOverviewProps) {
  const { t, locale, isSavingLocale } = useI18n()
  const copy = appearanceOverviewCopy(locale)
  const pages = useSettingsPagesCopy()
  const { themeName, mode, setMode } = useTheme()
  const chatTextScale = useStore($chatTextScale)
  const tipsEnabled = useStore($tipsEnabled)
  const toursEnabled = useStore($toursEnabled)
  const spentTips = useStore($spentTipCount)
  const groupName = useId()
  const advancedRef = useRef<HTMLDetailsElement>(null)
  const pendingSetting = useSettingDeepLink('config:appearance')
  const light = getBaseColors(themeName, 'light').background
  const dark = getBaseColors(themeName, 'dark').background

  // Saved general-setting links must reveal their target before highlighting it.
  useEffect(() => {
    if (
      pendingSetting &&
      pendingSetting !== ids.language &&
      pendingSetting !== ids.chatTextScale &&
      advancedRef.current
    ) {
      advancedRef.current.open = true
    }
  }, [pendingSetting])

  return (
    <SettingsContent>
      <p className="appearance-overview-intro" data-slot="page-description">
        {copy.intro}
      </p>
      <div className="appearance-overview" data-slot="appearance-overview">
        <fieldset className="appearance-mode-section">
          <legend>{t.settings.appearance.themeTitle}</legend>
          <p className="appearance-overview-description">{t.settings.appearance.colorModeDesc}</p>
          <div className="appearance-mode-grid">
            {modes.map(value => (
              <label className="appearance-mode-card" key={value}>
                <input
                  aria-label={t.settings.modeOptions[value].label}
                  checked={mode === value}
                  className="sr-only"
                  name={groupName}
                  onChange={() => {
                    triggerHaptic('crisp')
                    setMode(value)
                  }}
                  type="radio"
                  value={value}
                />
                <span
                  aria-hidden="true"
                  className="appearance-mode-preview"
                  style={{
                    background:
                      value === 'system'
                        ? `linear-gradient(to right, ${light} 50%, ${dark} 50%)`
                        : value === 'light'
                          ? light
                          : dark
                  }}
                />
                <span className="appearance-mode-label">
                  {t.settings.modeOptions[value].label}
                  {value === 'system' && <span className="appearance-mode-default">{copy.default}</span>}
                </span>
                <span className="appearance-overview-description">{t.settings.modeOptions[value].description}</span>
              </label>
            ))}
          </div>
        </fieldset>

        <section aria-label={copy.reading} className="global-settings-section">
          <div className="global-settings-section-head">
            <h3>{copy.reading}</h3>
            <span>{forHost(pages.savedHere)}</span>
          </div>
          <SettingsCard>
            <ListRow
              action={
                <SegmentedControl
                  onChange={value => {
                    triggerHaptic('selection')
                    setChatTextScale(Number(value))
                  }}
                  options={[
                    { id: '100', label: copy.small },
                    { id: '110', label: copy.default },
                    { id: '125', label: copy.large }
                  ]}
                  value={String(chatTextScale)}
                />
              }
              description={copy.sizeDescription}
              id={settingElementId(ids.chatTextScale)}
              title={copy.size}
            />
            <ListRow
              action={<LanguageSwitcher />}
              description={isSavingLocale ? t.language.saving : copy.languageDescription}
              id={settingElementId(ids.language)}
              title={t.language.label}
            />
            {/* The one tips switch (Advanced repeats neither half): it answers both settings'
                deep links and keeps the spent-tip reset. */}
            <div id={settingElementId(ids.tours)}>
              <ToggleRow
                below={
                  spentTips > 0 && (
                    <RowFootnoteAction onClick={resetTips}>
                      {t.settings.appearance.tipsReset(spentTips)}
                    </RowFootnoteAction>
                  )
                }
                checked={tipsEnabled || toursEnabled}
                description={copy.tipsDescription}
                id={settingElementId(ids.tips)}
                label={copy.tips}
                onChange={enabled => {
                  setTipsEnabled(enabled)
                  setToursEnabled(enabled)
                }}
              />
            </div>
          </SettingsCard>
        </section>

        <SettingsDisclosure
          description={copy.advancedDescription}
          ref={advancedRef}
          title={t.settings.sections.advanced}
        >
          {advanced}
        </SettingsDisclosure>
      </div>
    </SettingsContent>
  )
}
