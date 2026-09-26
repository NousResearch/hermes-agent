import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { useI18n } from '@/i18n'

import { type CzesiekPersonality, normalizePersonality, personalityPrompt } from './personality'
import { setupCopy } from './setup-copy'

export function PersonalityStep({
  value,
  onChange
}: {
  value?: CzesiekPersonality
  onChange: (value: CzesiekPersonality) => void
}) {
  const { locale } = useI18n()
  const copy = setupCopy[locale]
  const soul = value ?? normalizePersonality(undefined)

  return (
    <div className="grid gap-5">
      <p className="text-sm text-(--ui-text-secondary)">{copy.soulIntro}</p>
      <label className="grid gap-2 text-sm">
        {copy.name}
        <Input maxLength={120} onChange={event => onChange({ ...soul, name: event.target.value })} value={soul.name} />
      </label>
      <label className="grid gap-2 text-sm">
        {copy.purpose}
        <Textarea
          maxLength={1000}
          onChange={event => onChange({ ...soul, purpose: event.target.value })}
          rows={3}
          value={soul.purpose}
        />
      </label>
      <label className="grid gap-2 text-sm">
        {copy.tone}
        <select
          className="min-h-11 rounded-md bg-(--ui-bg-input) px-3"
          onChange={event => onChange({ ...soul, tone: event.target.value as CzesiekPersonality['tone'] })}
          value={soul.tone}
        >
          {(['friendly', 'direct', 'professional'] as const).map(tone => (
            <option key={tone} value={tone}>
              {copy[tone]}
            </option>
          ))}
        </select>
      </label>
      <label className="grid gap-2 text-sm">
        {copy.detail}
        <select
          className="min-h-11 rounded-md bg-(--ui-bg-input) px-3"
          onChange={event => onChange({ ...soul, detail: event.target.value as CzesiekPersonality['detail'] })}
          value={soul.detail}
        >
          {(['brief', 'balanced', 'thorough'] as const).map(detail => (
            <option key={detail} value={detail}>
              {copy[detail]}
            </option>
          ))}
        </select>
      </label>
      <details className="text-sm">
        <summary className="cursor-pointer font-medium">{copy.preview}</summary>
        <p className="mt-3 whitespace-pre-wrap leading-6">{personalityPrompt(soul)}</p>
        <p className="mt-3 text-(--ui-text-tertiary)">{copy.savedLater}</p>
      </details>
    </div>
  )
}
