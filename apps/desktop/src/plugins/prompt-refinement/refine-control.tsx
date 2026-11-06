import {
  Button,
  cn,
  Codicon,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  host,
  Tip,
  usePluginI18n,
  useValue
} from '@hermes/plugin-sdk'
import type { ReactNode } from 'react'

import { PLUGIN_ID } from './i18n'
import type { SettingsStore } from './settings'
import { REFINEMENT_DEPTHS, REFINEMENT_STYLES, type RefinementMode } from './types'

const MODES: readonly RefinementMode[] = ['off', 'manual', 'auto']

/** The composer's own pill skin (model and reasoning pills). */
const PILL = cn(
  'h-(--composer-control-size) shrink-0 gap-1 rounded-md px-2 text-xs font-normal',
  'text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground'
)

export const SETTINGS_PATH = '/settings?tab=plugins&plugin=prompt-refinement'

function Section({ children, title }: { children: ReactNode; title: string }) {
  return (
    <div aria-label={title} role="group">
      <div className="px-2 pt-1.5 pb-1 text-[0.6875rem] text-(--ui-text-tertiary)">{title}</div>
      {children}
    </div>
  )
}

function Choice({ label, onSelect, selected }: { label: string; onSelect: () => void; selected: boolean }) {
  return (
    <DropdownMenuItem aria-checked={selected} onSelect={onSelect} role="menuitemradio">
      <span className="flex-1">{label}</span>
      {selected && <Codicon className="text-(--ui-text-secondary)" name="check" size="0.8rem" />}
    </DropdownMenuItem>
  )
}

/**
 * Prompt Refinement in the composer: a "Refine" pill with the same
 * pill-and-menu as the model and reasoning pills. Mode, depth and style are
 * one click away; Enter does the refining (the send middleware), so there is
 * no separate button.
 */
export function createRefineControl(store: SettingsStore) {
  return function RefineControl() {
    const t = usePluginI18n(PLUGIN_ID)
    const settings = useValue(store.$settings)
    const pillLabel = t(`control.pill.${settings.mode}`)

    return (
      <div className="flex items-center" data-slot="prompt-refinement-control">
        <DropdownMenu>
          <Tip label={t('control.tip')} side="top">
            <DropdownMenuTrigger asChild>
              <Button
                aria-label={t('control.label', pillLabel)}
                className={cn(PILL, settings.mode === 'auto' && 'text-primary hover:text-primary')}
                data-slot="prompt-refinement-pill"
                type="button"
                variant="ghost"
              >
                <span>{pillLabel}</span>
                <Codicon className="opacity-50" name="chevron-down" size="0.625rem" />
              </Button>
            </DropdownMenuTrigger>
          </Tip>

          <DropdownMenuContent align="end" className="w-56" side="top" sideOffset={8}>
            <Section title={t('settings.modeTitle')}>
              {MODES.map(mode => (
                <Choice
                  key={mode}
                  label={t(`settings.modes.${mode}`)}
                  onSelect={() => store.update({ mode })}
                  selected={settings.mode === mode}
                />
              ))}
            </Section>
            <DropdownMenuSeparator />
            <Section title={t('settings.depthTitle')}>
              <Choice
                label={t('settings.depths.auto')}
                onSelect={() => store.update({ autoDepth: true })}
                selected={settings.autoDepth}
              />
              {REFINEMENT_DEPTHS.map(depth => (
                <Choice
                  key={depth}
                  label={t(`settings.depths.${depth}`)}
                  onSelect={() => store.update({ autoDepth: false, depth })}
                  selected={!settings.autoDepth && settings.depth === depth}
                />
              ))}
            </Section>
            <DropdownMenuSeparator />
            <Section title={t('settings.styleTitle')}>
              {REFINEMENT_STYLES.map(style => (
                <Choice
                  key={style}
                  label={t(`settings.styles.${style}`)}
                  onSelect={() => {
                    store.update({ style })

                    // A custom style needs its words: take the user to write them.
                    if (style === 'custom' && !store.get().customStyle.trim()) {
                      host.navigate(SETTINGS_PATH)
                    }
                  }}
                  selected={settings.style === style}
                />
              ))}
            </Section>
            <DropdownMenuSeparator />
            <DropdownMenuItem onSelect={() => host.navigate(SETTINGS_PATH)}>{t('control.more')}</DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    )
  }
}
