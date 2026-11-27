import '../discovery/discovery.css'
import './appearance-overview.css'

import { useStore } from '@nanostores/react'
import { type ReactNode, useEffect, useRef } from 'react'

import { LanguageSwitcher } from '@/components/language-switcher'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { $chatTextScale, setChatTextScale } from '@/store/chat-text-scale'
import { $spentTipCount, $tipsEnabled, resetTips, setTipsEnabled } from '@/store/tips'
import { $toursEnabled, setToursEnabled } from '@/store/tours'

import { MakeItYours } from '../discovery/make-it-yours'
import { $artworkChoice, NO_ARTWORK, setArtworkShown } from '../redesign/artwork'

import { appearanceOverviewCopy } from './appearance-overview-copy'
import { ListRow, RowFootnoteAction, SettingsContent, ToggleRow } from './primitives'
import { SettingsCard, SettingsDisclosure } from './settings-card'
import { SETTING_IDS, settingElementId } from './settings-manifest'
import { forHost, useSettingsPagesCopy } from './settings-pages-copy'
import { useSettingDeepLink } from './use-setting-deep-link'

const ids = SETTING_IDS.appearance

interface AppearanceOverviewProps {
  advanced: ReactNode
}

export function AppearanceOverview({ advanced }: AppearanceOverviewProps) {
  const { t, locale, isSavingLocale } = useI18n()
  const copy = appearanceOverviewCopy(locale)
  const pages = useSettingsPagesCopy()
  const chatTextScale = useStore($chatTextScale)
  const tipsEnabled = useStore($tipsEnabled)
  const toursEnabled = useStore($toursEnabled)
  const spentTips = useStore($spentTipCount)
  const artworkChoice = useStore($artworkChoice)
  const advancedRef = useRef<HTMLDetailsElement>(null)
  const pendingSetting = useSettingDeepLink('config:appearance')

  // Saved general-setting links must reveal their target before highlighting it.
  useEffect(() => {
    if (
      pendingSetting &&
      pendingSetting !== ids.theme &&
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
        <section aria-label={t.settings.appearance.themeTitle} className="appearance-mode-section appearance-make-it-yours" id={settingElementId(ids.theme)}>
          <div className="global-settings-section-head">
            <h3>{t.settings.appearance.themeTitle}</h3>
            <span>{forHost(pages.savedHere)}</span>
          </div>
          <MakeItYours />
        </section>

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
            <ToggleRow
              checked={artworkChoice !== NO_ARTWORK}
              description={copy.artworkDescription}
              label={copy.artwork}
              onChange={setArtworkShown}
            />
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
