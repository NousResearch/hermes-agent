import {
  Button,
  haptic,
  host,
  ListRow,
  SegmentedControl,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Slider,
  Textarea,
  ToggleRow,
  usePluginI18n,
  useValue
} from '@hermes/plugin-sdk'

import { PLUGIN_ID } from './i18n'
import { MAX_CUSTOM_STYLE_CHARS, type SettingsStore } from './settings'
import {
  REFINEMENT_DEPTHS,
  REFINEMENT_STYLES,
  type RefinementMode,
  type RefinementModelSource,
  type RefinementStyle
} from './types'

const MODES: readonly RefinementMode[] = ['off', 'manual', 'auto']
const SOURCES: readonly RefinementModelSource[] = ['dedicated', 'chat']
const MIN_WORD_CHOICES = ['1', '4', '8'] as const

/** The aux row for this plugin's slot, scrolled into view and flashed. */
const MODEL_SETTINGS_PATH = '/settings?tab=config:model&aux=prompt_refinement'

export function createSettingsPage(store: SettingsStore) {
  return function PromptRefinementSettings() {
    const t = usePluginI18n(PLUGIN_ID)
    const settings = useValue(store.$settings)

    const minWords = MIN_WORD_CHOICES.includes(String(settings.minWords) as (typeof MIN_WORD_CHOICES)[number])
      ? String(settings.minWords)
      : '4'

    return (
      <div className="grid gap-1">
        <ListRow
          action={
            <SegmentedControl
              onChange={mode => store.update({ mode })}
              options={MODES.map(id => ({ id, label: t(`settings.modes.${id}`) }))}
              sizing="content"
              value={settings.mode}
            />
          }
          description={t(`settings.modeHints.${settings.mode}`)}
          title={t('settings.modeTitle')}
        />

        <ToggleRow
          checked={settings.autoDepth}
          description={t('settings.autoDepthHint')}
          disabled={settings.mode === 'off'}
          label={t('settings.autoDepthTitle')}
          onChange={autoDepth => store.update({ autoDepth })}
        />

        <ListRow
          action={
            <div className="flex items-center gap-3">
              <Slider
                aria-label={t('settings.depthTitle')}
                aria-valuetext={t(`settings.depths.${settings.depth}`)}
                disabled={settings.mode === 'off' || settings.autoDepth}
                max={REFINEMENT_DEPTHS.length - 1}
                min={0}
                onChange={event => {
                  haptic('selection')
                  store.update({ depth: REFINEMENT_DEPTHS[Number(event.target.value)] })
                }}
                step={1}
                value={REFINEMENT_DEPTHS.indexOf(settings.depth)}
              />
              <span className="w-14 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                {t(`settings.depths.${settings.depth}`)}
              </span>
            </div>
          }
          description={t(`settings.depthHints.${settings.depth}`)}
          title={t('settings.depthTitle')}
        />

        <ListRow
          action={
            <Select
              disabled={settings.mode === 'off'}
              onValueChange={value => store.update({ style: value as RefinementStyle })}
              value={settings.style}
            >
              <SelectTrigger aria-label={t('settings.styleTitle')} className="min-w-44 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {REFINEMENT_STYLES.map(id => (
                  <SelectItem key={id} value={id}>
                    {t(`settings.styles.${id}`)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          }
          below={
            settings.style === 'custom' ? (
              <div className="mt-2 grid gap-1">
                <Textarea
                  aria-label={t('settings.customStyleTitle')}
                  className="min-h-20 resize-y text-xs"
                  disabled={settings.mode === 'off'}
                  maxLength={MAX_CUSTOM_STYLE_CHARS}
                  onChange={event => store.update({ customStyle: event.currentTarget.value })}
                  placeholder={t('settings.customStylePlaceholder')}
                  value={settings.customStyle}
                />
                <span className="text-xs text-(--ui-text-tertiary)">
                  {t('settings.customStyleHint', MAX_CUSTOM_STYLE_CHARS)}
                </span>
              </div>
            ) : undefined
          }
          description={t(`settings.styleHints.${settings.style}`)}
          title={t('settings.styleTitle')}
        />

        <ToggleRow
          checked={settings.useChatContext}
          description={t('settings.contextHint')}
          disabled={settings.mode === 'off'}
          label={t('settings.contextTitle')}
          onChange={useChatContext => store.update({ useChatContext })}
        />

        <ListRow
          action={
            <SegmentedControl
              disabled={settings.mode === 'off'}
              onChange={modelSource => store.update({ modelSource })}
              options={SOURCES.map(id => ({
                id,
                label: t(id === 'dedicated' ? 'settings.modelDedicated' : 'settings.modelChat')
              }))}
              sizing="content"
              value={settings.modelSource}
            />
          }
          below={
            settings.modelSource === 'dedicated' ? (
              <div className="mt-2">
                <Button onClick={() => host.navigate(MODEL_SETTINGS_PATH)} size="sm" variant="secondary">
                  {t('settings.openModelSettings')}
                </Button>
              </div>
            ) : undefined
          }
          description={`${t(settings.modelSource === 'dedicated' ? 'settings.modelDedicatedHint' : 'settings.modelChatHint')} ${t('settings.modelSpeedHint')}`}
          title={t('settings.modelTitle')}
        />

        <ListRow
          action={
            <SegmentedControl
              disabled={settings.mode === 'off'}
              onChange={value => store.update({ minWords: Number(value) })}
              options={MIN_WORD_CHOICES.map(id => ({ id, label: id }))}
              value={minWords}
            />
          }
          description={t('settings.minWordsHint', settings.minWords)}
          title={t('settings.minWordsTitle')}
        />

        <p className="pt-3 text-xs text-(--ui-text-tertiary)">{t('settings.privacy')}</p>
      </div>
    )
  }
}
