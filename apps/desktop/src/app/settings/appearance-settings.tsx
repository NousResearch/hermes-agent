import { useStore } from '@nanostores/react'
import type { ReactNode } from 'react'

import { triggerHaptic } from '@/lib/haptics'
import { Check, type IconComponent } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $toolViewMode, setToolViewMode } from '@/store/tool-view'
import { useTheme } from '@/themes/context'
import { BUILTIN_THEMES } from '@/themes/presets'

import { MODE_OPTIONS } from './constants'
import { prettyName } from './helpers'
import { Pill, SettingsContent } from './primitives'

function ThemePreview({ name }: { name: string }) {
  const t = BUILTIN_THEMES[name]

  if (!t) {
    return null
  }

  const c = t.colors

  return (
    <div
      className="h-20 overflow-hidden rounded-xl border shadow-xs"
      style={{ backgroundColor: c.background, borderColor: c.border }}
    >
      <div className="flex h-full">
        <div
          className="w-12 border-r"
          style={{
            backgroundColor: c.sidebarBackground ?? c.muted,
            borderColor: c.sidebarBorder ?? c.border
          }}
        />
        <div className="flex flex-1 flex-col gap-2 p-3">
          <div className="h-2.5 w-16 rounded-full" style={{ backgroundColor: c.foreground }} />
          <div className="h-2 w-24 rounded-full" style={{ backgroundColor: c.mutedForeground }} />
          <div className="mt-auto flex justify-end">
            <div
              className="h-5 w-16 rounded-full border"
              style={{
                backgroundColor: c.userBubble ?? c.muted,
                borderColor: c.userBubbleBorder ?? c.border
              }}
            />
          </div>
        </div>
      </div>
    </div>
  )
}

function SectionHead({ title, description, pill }: { title: string; description: string; pill?: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <div>
        <div className="text-[length:var(--conversation-text-font-size)] font-medium">{title}</div>
        <div className="mt-1 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
          {description}
        </div>
      </div>
      {pill}
    </div>
  )
}

function OptionRow({
  icon: Icon,
  label,
  description,
  active,
  onClick
}: {
  icon?: IconComponent
  label: string
  description: string
  active: boolean
  onClick: () => void
}) {
  return (
    <button
      className={cn(
        'group flex items-center gap-3 rounded-md px-3 py-2 text-left transition hover:bg-(--chrome-action-hover)',
        active && 'bg-(--ui-bg-tertiary)'
      )}
      onClick={onClick}
      type="button"
    >
      {Icon && <Icon className={cn('size-4 shrink-0', active ? 'text-foreground' : 'text-muted-foreground')} />}
      <div className="min-w-0 flex-1">
        <div className="text-[length:var(--conversation-text-font-size)] font-medium">{label}</div>
        <div className="mt-0.5 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
          {description}
        </div>
      </div>
      <Check className={cn('size-4 shrink-0 text-primary transition-opacity', active ? 'opacity-100' : 'opacity-0')} />
    </button>
  )
}

export function AppearanceSettings() {
  const { themeName, mode, availableThemes, setTheme, setMode } = useTheme()
  const toolViewMode = useStore($toolViewMode)
  const activeTheme = availableThemes.find(t => t.name === themeName)

  return (
    <SettingsContent>
      <div className="grid gap-8">
        <p className="max-w-2xl text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
          These are desktop-only display preferences. Mode controls brightness; theme controls the accent palette and
          chat surface styling.
        </p>

        <section className="grid gap-3">
          <SectionHead
            description="Pick a fixed mode or let Hermes follow your system setting."
            pill={<Pill>{prettyName(mode)}</Pill>}
            title="Color Mode"
          />
          <div className="grid gap-0.5">
            {MODE_OPTIONS.map(({ id, label, description, icon }) => (
              <OptionRow
                active={mode === id}
                description={description}
                icon={icon}
                key={id}
                label={label}
                onClick={() => {
                  triggerHaptic('crisp')
                  setMode(id)
                }}
              />
            ))}
          </div>
        </section>

        <section className="grid gap-3">
          <SectionHead
            description="Product hides raw tool payloads; Technical shows full input/output."
            pill={<Pill>{toolViewMode === 'technical' ? 'Technical' : 'Product'}</Pill>}
            title="Tool Call Display"
          />
          <div className="grid gap-0.5">
            {(
              [
                {
                  id: 'product',
                  label: 'Product',
                  description: 'Human-friendly tool activity with concise summaries.'
                },
                {
                  id: 'technical',
                  label: 'Technical',
                  description: 'Include raw tool args/results and low-level details.'
                }
              ] as const
            ).map(option => (
              <OptionRow
                active={toolViewMode === option.id}
                description={option.description}
                key={option.id}
                label={option.label}
                onClick={() => {
                  triggerHaptic('selection')
                  setToolViewMode(option.id)
                }}
              />
            ))}
          </div>
        </section>

        <section className="grid gap-3">
          <SectionHead
            description="Desktop palettes only. The selected mode is applied on top."
            pill={activeTheme ? <Pill>{activeTheme.label}</Pill> : undefined}
            title="Theme"
          />
          <div className="grid gap-x-4 gap-y-5 sm:grid-cols-2 xl:grid-cols-3">
            {availableThemes.map(theme => {
              const active = themeName === theme.name

              return (
                <button
                  className="group text-left"
                  key={theme.name}
                  onClick={() => {
                    triggerHaptic('crisp')
                    setTheme(theme.name)
                  }}
                  type="button"
                >
                  <div
                    className={cn(
                      'rounded-xl transition',
                      active
                        ? 'ring-2 ring-primary ring-offset-2 ring-offset-background'
                        : 'opacity-90 group-hover:opacity-100'
                    )}
                  >
                    <ThemePreview name={theme.name} />
                  </div>
                  <div className="mt-2.5 flex items-start justify-between gap-2 px-0.5">
                    <div className="min-w-0">
                      <div className="truncate text-[length:var(--conversation-text-font-size)] font-medium">
                        {theme.label}
                      </div>
                      <div className="mt-0.5 line-clamp-2 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
                        {theme.description}
                      </div>
                    </div>
                    {active && <Check className="mt-0.5 size-4 shrink-0 text-primary" />}
                  </div>
                </button>
              )
            })}
          </div>
        </section>
      </div>
    </SettingsContent>
  )
}
