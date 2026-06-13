import { LanguageSwitcher } from '@/components/language-switcher'
import { useI18n } from '@/i18n'
import { Globe } from '@/lib/icons'

import { ListRow, SectionHeading, SettingsContent } from './primitives'

export function LanguageSettings() {
  const { t, isSavingLocale } = useI18n()
  const a = t.language

  return (
    <SettingsContent>
      <div>
        <SectionHeading icon={Globe} title={a.label} />
        <p className="max-w-2xl text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
          {a.description}
        </p>

        <div className="mt-2 divide-y divide-(--ui-stroke-tertiary)">
          <ListRow
            action={<LanguageSwitcher />}
            description={isSavingLocale ? a.saving : a.description}
            title={a.label}
          />
        </div>
      </div>
    </SettingsContent>
  )
}
