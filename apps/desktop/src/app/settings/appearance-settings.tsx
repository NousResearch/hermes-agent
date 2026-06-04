import { useStore } from '@nanostores/react'
import type { ReactNode } from 'react'

import { SegmentedControl } from '@/components/ui/segmented-control'
import { triggerHaptic } from '@/lib/haptics'
import { Check } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $toolViewMode, setToolViewMode } from '@/store/tool-view'
import { useTheme } from '@/themes/context'
import { BUILTIN_THEMES } from '@/themes/presets'

import { MODE_OPTIONS } from './constants'
import { SettingsContent } from './primitives'

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

function SectionHead({ title, description, control }: { title: string; description: string; control?: ReactNode }) {
  return (
    <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between sm:gap-4">
      <div className="min-w-0">
        <div className="text-[length:var(--conversation-text-font-size)] font-medium">{title}</div>
        <div className="mt-1 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
          {description}
        </div>
      </div>
      {control && <div className="shrink-0">{control}</div>}
    </div>
  )
}

export function AppearanceSettings() {
  const { themeName, mode, availableThemes, setTheme, setMode } = useTheme()
  const toolViewMode = useStore($toolViewMode)

  return (
    <SettingsContent>
      <div className="grid gap-8">
        <p className="max-w-2xl text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
          These are desktop-only display preferences. Mode controls brightness; theme controls the accent palette and
          chat surface styling.
        </p>

        <section>
          <SectionHead
            control={
              <SegmentedControl
                onChange={id => {
                  triggerHaptic('crisp')
                  setMode(id)
                }}
                options={MODE_OPTIONS}
                value={mode}
              />
            }
            description="Pick a fixed mode or let Hermes follow your system setting."
            title="Color Mode"
          />
        </section>

        <section>
          <SectionHead
            control={
              <SegmentedControl
                onChange={id => {
                  triggerHaptic('selection')
                  setToolViewMode(id)
                }}
                options={
                  [
                    { id: 'product', label: 'Product' },
                    { id: 'technical', label: 'Technical' }
                  ] as const
                }
                value={toolViewMode}
              />
            }
            description="Product hides raw tool payloads; Technical shows full input/output."
            title="Tool Call Display"
          />
        </section>

        <section className="grid gap-3">
          <SectionHead description="Desktop palettes only. The selected mode is applied on top." title="Theme" />
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
