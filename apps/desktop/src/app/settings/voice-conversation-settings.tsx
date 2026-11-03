import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useI18n } from '@/i18n'
import { isExtensionVoiceChatMode, type VoiceChatMode } from '@/lib/voice-live'

import { CONTROL_TEXT, ENUM_OPTIONS, FIELD_LABELS } from './constants'
import { ListRow } from './primitives'
import { useVoiceChatModeChoices } from './voice-engine-settings'

/** Two UI choices, one existing persisted engine id. No migration or extra config authority. */
export function VoiceConversationSettings({
  mode,
  onChange
}: {
  mode: unknown
  onChange: (value: VoiceChatMode) => void
}) {
  const { t } = useI18n()
  const c = t.composer
  const choices = useVoiceChatModeChoices(mode)
  const duplex = mode === 'gpt-live' || isExtensionVoiceChatMode(mode)
  const label = t.settings.fieldLabels['voice.voiceChatMode'] ?? FIELD_LABELS['voice.voiceChatMode']

  return (
    <>
      <ListRow
        action={
          <Select
            onValueChange={value => onChange(value === 'chained' ? 'chained' : 'gpt-live')}
            value={duplex ? 'duplex' : 'chained'}
          >
            <SelectTrigger aria-label={label} className={CONTROL_TEXT}>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="chained">{c.voiceEngineChained}</SelectItem>
              <SelectItem value="duplex">{c.voiceModeDuplex}</SelectItem>
            </SelectContent>
          </Select>
        }
        data-tour="field-voice.voice_chat_mode"
        title={label}
      />
      {duplex ? (
        <ListRow
          action={
            <Select onValueChange={value => onChange(value as VoiceChatMode)} value={String(mode)}>
              <SelectTrigger aria-label={c.voiceEngine} className={CONTROL_TEXT}>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {choices.options
                  .filter(value => value !== 'chained')
                  .map(value => (
                    <SelectItem key={value} value={value}>
                      {value === 'gpt-live' ? 'OpenAI' : choices.labels[value]}
                    </SelectItem>
                  ))}
              </SelectContent>
            </Select>
          }
          title={c.voiceEngine}
        />
      ) : null}
    </>
  )
}

export function liveLanguageLabels(locale: string, autoLabel: string): Record<string, string> {
  const names = new Intl.DisplayNames([locale], { type: 'language' })

  return Object.fromEntries(
    ENUM_OPTIONS['voice.gpt_live.language'].map(code => [
      code,
      code === 'auto' ? autoLabel : `${names.of(code)} (${code})`
    ])
  )
}
