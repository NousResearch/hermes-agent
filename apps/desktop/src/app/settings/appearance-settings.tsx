import { useStore } from '@nanostores/react'

import { triggerHaptic } from '@/lib/haptics'
import { dt } from '@/lib/i18n'
import { Check, Globe, Palette } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $desktopLanguage, setDesktopLanguage, type DesktopLanguage } from '@/store/language'
import { $toolViewMode, setToolViewMode } from '@/store/tool-view'
import { useTheme } from '@/themes/context'
import { BUILTIN_THEMES } from '@/themes/presets'

import { MODE_OPTIONS } from './constants'
import { prettyName } from './helpers'
import { Pill, SectionHeading, SettingsContent } from './primitives'

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

function modeNameZh(mode: string) {
  if (mode === 'light') return '浅色'
  if (mode === 'dark') return '深色'
  if (mode === 'system') return '跟随系统'
  return prettyName(mode)
}

function modeDescriptionZh(mode: string) {
  if (mode === 'light') return '明亮的桌面界面'
  if (mode === 'dark') return '低眩光工作区'
  if (mode === 'system') return '跟随 macOS 外观'
  return ''
}

export function AppearanceSettings() {
  const { themeName, mode, availableThemes, setTheme, setMode } = useTheme()
  const desktopLanguage = useStore($desktopLanguage)
  const toolViewMode = useStore($toolViewMode)
  const activeTheme = availableThemes.find(t => t.name === themeName)

  return (
    <SettingsContent>
      <div className="space-y-7">
        <div>
          <SectionHeading icon={Palette} title={dt(desktopLanguage, 'appearance', 'Appearance')} />
          <p className="max-w-2xl text-sm leading-6 text-muted-foreground">
            {desktopLanguage === 'zh'
              ? '这些是桌面端显示偏好。模式控制明暗，主题控制强调色和对话界面样式。'
              : 'These are desktop-only display preferences. Mode controls brightness; theme controls the accent palette and chat surface styling.'}
          </p>
        </div>

        <section className="rounded-2xl border border-border/50 bg-card/55 p-4 shadow-sm">
          <div className="mb-3 flex items-center justify-between gap-3">
            <div>
              <SectionHeading icon={Globe} title={desktopLanguage === 'zh' ? '语言' : 'Language'} />
              <div className="mt-1 text-xs text-muted-foreground">
                {desktopLanguage === 'zh'
                  ? '选择桌面界面显示语言。部分旧界面会在接入翻译后逐步生效。'
                  : 'Choose the desktop display language. Some legacy labels update as they are migrated to translations.'}
              </div>
            </div>
            <Pill>{desktopLanguage === 'zh' ? '中文' : 'English'}</Pill>
          </div>
          <div className="grid gap-2 sm:grid-cols-2">
            {(
              [
                {
                  id: 'zh',
                  label: '中文',
                  description: '使用中文显示桌面界面。'
                },
                {
                  id: 'en',
                  label: 'English',
                  description: 'Use English for the desktop interface.'
                }
              ] as const satisfies readonly {
                id: DesktopLanguage
                label: string
                description: string
              }[]
            ).map(option => {
              const active = desktopLanguage === option.id

              return (
                <button
                  className={cn(
                    'group rounded-xl border border-border/45 bg-background/55 p-3 text-left transition hover:border-primary/35 hover:bg-accent/45',
                    active && 'border-primary/65 bg-primary/8 ring-2 ring-primary/25'
                  )}
                  key={option.id}
                  onClick={() => {
                    triggerHaptic('selection')
                    setDesktopLanguage(option.id)
                  }}
                  type="button"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="text-sm font-medium">{option.label}</div>
                    {active && (
                      <span className="grid size-5 place-items-center rounded-full bg-primary text-primary-foreground">
                        <Check className="size-3.5" />
                      </span>
                    )}
                  </div>
                  <div className="mt-1 text-xs leading-5 text-muted-foreground">{option.description}</div>
                </button>
              )
            })}
          </div>
        </section>

        <section className="rounded-2xl border border-border/50 bg-card/55 p-4 shadow-sm">
          <div className="mb-3 flex items-center justify-between gap-3">
            <div>
              <div className="text-sm font-medium">{desktopLanguage === 'zh' ? '颜色模式' : 'Color Mode'}</div>
              <div className="mt-1 text-xs text-muted-foreground">
                {desktopLanguage === 'zh'
                  ? '选择固定模式，或跟随 macOS 系统外观。'
                  : 'Pick a fixed mode or let Hermes follow your system setting.'}
              </div>
            </div>
            <Pill>{desktopLanguage === 'zh' ? modeNameZh(mode) : prettyName(mode)}</Pill>
          </div>
          <div className="grid gap-2 sm:grid-cols-3">
            {MODE_OPTIONS.map(({ id, label, description, icon: Icon }) => {
              const active = mode === id

              return (
                <button
                  className={cn(
                    'group rounded-xl border border-border/45 bg-background/55 p-3 text-left transition hover:border-primary/35 hover:bg-accent/45',
                    active && 'border-primary/65 bg-primary/8 ring-2 ring-primary/25'
                  )}
                  key={id}
                  onClick={() => {
                    triggerHaptic('crisp')
                    setMode(id)
                  }}
                  type="button"
                >
                  <div className="flex items-start justify-between gap-3">
                    <span className="flex size-9 items-center justify-center rounded-lg bg-muted text-foreground transition group-hover:bg-background">
                      <Icon className="size-4" />
                    </span>
                    {active && (
                      <span className="grid size-5 place-items-center rounded-full bg-primary text-primary-foreground">
                        <Check className="size-3.5" />
                      </span>
                    )}
                  </div>
                  <div className="mt-3 text-sm font-medium">
                    {desktopLanguage === 'zh' ? modeNameZh(id) : label}
                  </div>
                  <div className="mt-1 text-xs leading-5 text-muted-foreground">
                    {desktopLanguage === 'zh' ? modeDescriptionZh(id) : description}
                  </div>
                </button>
              )
            })}
          </div>
        </section>

        <section className="rounded-2xl border border-border/50 bg-card/55 p-4 shadow-sm">
          <div className="mb-3 flex items-center justify-between gap-3">
            <div>
              <div className="text-sm font-medium">{desktopLanguage === 'zh' ? '工具调用显示' : 'Tool Call Display'}</div>
              <div className="mt-1 text-xs text-muted-foreground">
                {desktopLanguage === 'zh'
                  ? '产品模式隐藏原始工具参数；技术模式显示完整输入和输出。'
                  : 'Product hides raw tool payloads; Technical shows full input/output.'}
              </div>
            </div>
            <Pill>
              {toolViewMode === 'technical'
                ? desktopLanguage === 'zh'
                  ? '技术'
                  : 'Technical'
                : desktopLanguage === 'zh'
                  ? '产品'
                  : 'Product'}
            </Pill>
          </div>
          <div className="grid gap-2 sm:grid-cols-2">
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
            ).map(option => {
              const active = toolViewMode === option.id

              return (
                <button
                  className={cn(
                    'group rounded-xl border border-border/45 bg-background/55 p-3 text-left transition hover:border-primary/35 hover:bg-accent/45',
                    active && 'border-primary/65 bg-primary/8 ring-2 ring-primary/25'
                  )}
                  key={option.id}
                  onClick={() => {
                    triggerHaptic('selection')
                    setToolViewMode(option.id)
                  }}
                  type="button"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="text-sm font-medium">
                      {desktopLanguage === 'zh' ? (option.id === 'technical' ? '技术' : '产品') : option.label}
                    </div>
                    {active && (
                      <span className="grid size-5 place-items-center rounded-full bg-primary text-primary-foreground">
                        <Check className="size-3.5" />
                      </span>
                    )}
                  </div>
                  <div className="mt-1 text-xs leading-5 text-muted-foreground">
                    {desktopLanguage === 'zh'
                      ? option.id === 'technical'
                        ? '包含原始工具参数、结果和底层细节。'
                        : '用面向人的简洁摘要显示工具活动。'
                      : option.description}
                  </div>
                </button>
              )
            })}
          </div>
        </section>

        <section className="rounded-2xl border border-border/50 bg-card/55 p-4 shadow-sm">
          <div className="mb-3 flex items-center justify-between gap-3">
            <div>
              <div className="text-sm font-medium">{desktopLanguage === 'zh' ? '主题' : 'Theme'}</div>
              <div className="mt-1 text-xs text-muted-foreground">
                {desktopLanguage === 'zh'
                  ? '仅影响桌面端配色。当前颜色模式会叠加应用。'
                  : 'Desktop palettes only. The selected mode is applied on top.'}
              </div>
            </div>
            {activeTheme && <Pill>{activeTheme.label}</Pill>}
          </div>
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
            {availableThemes.map(theme => {
              const active = themeName === theme.name

              return (
                <button
                  className={cn(
                    'rounded-2xl border border-border/45 bg-background/50 p-2.5 text-left transition hover:border-primary/35 hover:bg-accent/35',
                    active && 'border-primary/65 bg-primary/8 ring-2 ring-primary/25'
                  )}
                  key={theme.name}
                  onClick={() => {
                    triggerHaptic('crisp')
                    setTheme(theme.name)
                  }}
                  type="button"
                >
                  <ThemePreview name={theme.name} />
                  <div className="mt-3 flex items-start justify-between gap-3 px-1">
                    <div className="min-w-0">
                      <div className="truncate text-sm font-medium">{theme.label}</div>
                      <div className="mt-0.5 line-clamp-2 text-xs leading-5 text-muted-foreground">
                        {theme.description}
                      </div>
                    </div>
                    {active && (
                      <span className="mt-0.5 grid size-5 shrink-0 place-items-center rounded-full bg-primary text-primary-foreground">
                        <Check className="size-3.5" />
                      </span>
                    )}
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
