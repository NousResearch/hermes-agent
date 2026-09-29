import { useStore } from '@nanostores/react'
import type { KeyboardEvent, RefObject } from 'react'
import { createRef, useMemo } from 'react'

import logoUrl from '@/assets/czesiek-logo.png'
import { type Locale, useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import {
  Activity,
  Box,
  Brain,
  CheckCircle2,
  ChevronRight,
  Clock,
  LayoutDashboard,
  Link2,
  MessageCircle,
  Monitor,
  Moon,
  Network,
  Plus,
  Search,
  Settings2,
  Sparkles,
  Starmap,
  Sun,
  Users,
  Wrench
} from '@/lib/icons'
import { IS_MAC } from '@/lib/keybinds/combo'
import { cn } from '@/lib/utils'
import { openCommandPalette } from '@/store/command-palette'
import { $activeGatewayProfile, $profiles, newSessionInProfile, profileLabel } from '@/store/profile'
import { setSessionPickerOpen } from '@/store/session'
import { useTheme } from '@/themes/context'

import {
  JARVIS_MAIN_VIEWS,
  JARVIS_NAV_GROUPS,
  type JarvisMainView,
  type JarvisShellCopy,
  type JarvisShellView
} from './i18n'

type IconComponent = React.ComponentType<{ className?: string }>

const VIEW_ICONS: Record<Exclude<JarvisShellView, 'profile'>, IconComponent> = {
  prompts: MessageCircle,
  jarvis: LayoutDashboard,
  tasks: CheckCircle2,
  agents: Users,
  messaging: MessageCircle,
  webhooks: Link2,
  artifacts: Box,
  memory: Brain,
  starmap: Starmap,
  tools: Wrench,
  connections: Network,
  insights: Activity,
  settings: Settings2
}

const FOCUS_RING =
  'outline-none focus-visible:outline focus-visible:outline-solid focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-(--ui-accent)'

interface JarvisNavigationProps {
  activeView: JarvisShellView
  copy: JarvisShellCopy
  onSelect: (view: JarvisShellView) => void
}

interface NavButtonProps {
  active: boolean
  buttonRef?: RefObject<HTMLButtonElement | null>
  icon: IconComponent
  label: string
  onClick: () => void
  onKeyDown?: (event: KeyboardEvent<HTMLButtonElement>) => void
  view?: JarvisMainView
}

function NavButton({ active, buttonRef, icon: Icon, label, onClick, onKeyDown, view }: NavButtonProps) {
  return (
    <button
      aria-current={active ? 'page' : undefined}
      className={cn(
        'group flex h-11 min-h-11 shrink-0 items-center gap-3 rounded-xl px-3 text-left text-sm font-medium transition-[background-color,color] duration-150 md:w-full',
        FOCUS_RING,
        active
          ? 'jarvis-nav-active text-(--ui-text-primary)'
          : 'text-(--ui-text-secondary) hover:bg-(--chrome-action-hover) hover:text-(--ui-text-primary)'
      )}
      data-jarvis-nav-view={view}
      onClick={() => {
        triggerHaptic(active ? 'selection' : 'tap')
        onClick()
      }}
      onKeyDown={onKeyDown}
      ref={buttonRef}
      type="button"
    >
      <Icon
        className={cn(
          'size-[1.125rem] shrink-0 transition-colors',
          active ? 'text-(--ui-accent)' : 'text-(--ui-text-tertiary) group-hover:text-(--ui-text-secondary)'
        )}
      />
      <span className="truncate">{label}</span>
    </button>
  )
}

/** Opens the command palette — the same place `mod+k` goes. */
function SearchButton({ label }: { label: string }) {
  return (
    <button
      className={cn(
        'jarvis-glass jarvis-glass-hover hidden min-h-11 w-full items-center gap-2 rounded-xl px-3 text-sm text-(--ui-text-tertiary) hover:text-(--ui-text-secondary) md:flex',
        FOCUS_RING
      )}
      onClick={openCommandPalette}
      type="button"
    >
      <Search className="size-4 shrink-0" />
      <span className="flex-1 truncate text-left">{label}</span>
      <kbd className="jarvis-well rounded-md px-1.5 py-0.5 font-sans text-[0.65rem] text-(--ui-text-tertiary)">
        {IS_MAC ? '⌘K' : 'Ctrl K'}
      </kbd>
    </button>
  )
}

/** The person this workspace belongs to, one click from their profiles and agents. */
function ProfileCard({ active, copy, onClick }: { active: boolean; copy: JarvisShellCopy; onClick: () => void }) {
  const activeProfile = useStore($activeGatewayProfile)
  const profiles = useStore($profiles)
  const row = profiles.find(profile => profile.name === activeProfile)
  const name = row ? profileLabel(row) : activeProfile || 'default'

  return (
    <button
      aria-current={active ? 'page' : undefined}
      aria-label={copy.views.profile}
      className={cn(
        'jarvis-glass jarvis-glass-hover flex min-h-11 shrink-0 items-center gap-3 rounded-2xl px-2 py-1.5 text-left md:w-full',
        active && 'jarvis-nav-active',
        FOCUS_RING
      )}
      onClick={onClick}
      title={name}
      type="button"
    >
      <span
        aria-hidden="true"
        className="grid size-9 shrink-0 place-items-center rounded-full bg-linear-to-br from-[#5b8cff] to-[#b04cff] text-sm font-semibold uppercase text-white shadow-[0_0_14px_rgb(124_92_255/0.4)]"
      >
        {name.slice(0, 1)}
      </span>
      <span className="hidden min-w-0 flex-1 md:block">
        <span className="block truncate text-sm font-medium text-(--ui-text-primary)">{name}</span>
        <span className="block truncate text-xs text-(--ui-text-tertiary)">{copy.home.nav.profileHint}</span>
      </span>
      <ChevronRight className="hidden size-4 shrink-0 text-(--ui-text-tertiary) md:block" />
    </button>
  )
}

export function JarvisNavigation({ activeView, copy, onSelect }: JarvisNavigationProps) {
  const { locale } = useI18n()
  const mainRefs = useMemo(() => JARVIS_MAIN_VIEWS.map(() => createRef<HTMLButtonElement>()), [])
  const navCopy = copy.home.nav

  const focusMain = (index: number) => {
    mainRefs[index]?.current?.focus()
  }

  const handleMainKeyDown = (index: number) => (event: KeyboardEvent<HTMLButtonElement>) => {
    const last = JARVIS_MAIN_VIEWS.length - 1

    const targets: Partial<Record<string, number>> = {
      ArrowDown: (index + 1) % JARVIS_MAIN_VIEWS.length,
      ArrowRight: (index + 1) % JARVIS_MAIN_VIEWS.length,
      ArrowUp: (index - 1 + JARVIS_MAIN_VIEWS.length) % JARVIS_MAIN_VIEWS.length,
      ArrowLeft: (index - 1 + JARVIS_MAIN_VIEWS.length) % JARVIS_MAIN_VIEWS.length,
      Home: 0,
      End: last
    }

    const target = targets[event.key]

    if (target !== undefined) {
      event.preventDefault()
      focusMain(target)
    }
  }

  return (
    <aside
      className="jarvis-glass-strong flex w-full shrink-0 flex-col gap-3 border-x-0 border-t-0 p-3 md:m-2 md:mr-0 md:h-[calc(100%-1rem)] md:w-60 md:rounded-3xl"
      data-jarvis-nav-rail=""
    >
      <div className="flex min-w-0 items-center gap-3 px-1">
        {/* Static brand mark: the live orb belongs to the dashboard, not the chrome. */}
        <img
          alt=""
          className="size-10 shrink-0 object-contain drop-shadow-[0_0_10px_rgb(124_92_255/0.45)] md:size-12"
          draggable={false}
          src={logoUrl}
        />
        <div className="min-w-0">
          <div className="jarvis-wordmark truncate text-base font-semibold">{copy.productName}</div>
          <div className="hidden truncate text-[0.65rem] uppercase tracking-[0.08em] text-(--ui-text-tertiary) md:block">
            {navCopy.tagline}
          </div>
        </div>
      </div>

      <SearchButton label={navCopy.search} />

      <nav
        aria-label={copy.navigationLabel}
        className="min-h-0 min-w-0 md:flex-1 md:overflow-y-auto"
        data-jarvis-nav=""
      >
        <div className="flex gap-2 overflow-x-auto pb-1 md:flex-col md:gap-4 md:overflow-visible md:pb-0">
          {JARVIS_NAV_GROUPS.map(group => (
            <div className="contents md:flex md:flex-col md:gap-0.5" key={group.id}>
              <p className="hidden px-3 pb-1 text-[0.65rem] font-medium uppercase tracking-[0.12em] text-(--ui-text-tertiary) md:block">
                {navCopy.sections[group.id]}
              </p>
              {group.views.map((view: JarvisMainView) => {
                const index = JARVIS_MAIN_VIEWS.indexOf(view)

                return (
                  <NavButton
                    active={activeView === view || (view === 'connections' && activeView === 'messaging')}
                    buttonRef={mainRefs[index]}
                    icon={VIEW_ICONS[view]}
                    key={view}
                    label={copy.views[view]}
                    onClick={() => onSelect(view)}
                    onKeyDown={handleMainKeyDown(index)}
                    view={view}
                  />
                )
              })}
              {group.id === 'work' ? (
                <>
                  <NavButton
                    active={false}
                    icon={Plus}
                    label={locale === 'pl' ? 'Nowa sesja' : 'New session'}
                    onClick={() => {
                      onSelect('jarvis')
                      newSessionInProfile($activeGatewayProfile.get() || 'default')
                    }}
                  />
                  <NavButton
                    active={false}
                    icon={Clock}
                    label={locale === 'pl' ? 'Historia' : 'History'}
                    onClick={() => setSessionPickerOpen(true)}
                  />
                </>
              ) : null}
            </div>
          ))}
        </div>
      </nav>

      {/* Only where the rail has height to spare: it must never push the nav into a scroll. */}
      <button
        className={cn(
          'jarvis-glass jarvis-glass-hover relative hidden overflow-hidden rounded-2xl bg-linear-to-br from-(--ui-accent)/16 via-transparent to-[#b04cff]/14 p-4 text-left [@media(min-height:1100px)]:md:block',
          FOCUS_RING
        )}
        onClick={() => onSelect('tools')}
        type="button"
      >
        <Sparkles className="mb-2 size-4 text-(--ui-accent)" />
        <span className="block text-sm font-semibold text-(--ui-text-primary)">{navCopy.promoTitle}</span>
        <span className="mt-1 block text-xs leading-5 text-(--ui-text-secondary)">{navCopy.promoBody}</span>
      </button>

      <div className="flex shrink-0 items-center gap-2 overflow-x-auto pt-1 md:flex-col md:items-stretch md:overflow-visible">
        <div className="flex shrink-0 items-center justify-between gap-2">
          <ThemeToggle copy={navCopy} />
          <LanguageToggle label={navCopy.language} />
        </div>
        <ProfileCard active={activeView === 'profile'} copy={copy} onClick={() => onSelect('profile')} />
      </div>
    </aside>
  )
}

const LANGUAGE_CHOICES: readonly { id: Locale; label: string }[] = [
  { id: 'pl', label: 'PL' },
  { id: 'en', label: 'EN' }
]

/** Polish first: the product speaks Polish by default, English is one tap away. */
function LanguageToggle({ label }: { label: string }) {
  const { isSavingLocale, locale, setLocale } = useI18n()

  return (
    <div aria-label={label} className="jarvis-glass flex shrink-0 gap-0.5 rounded-xl p-0.5" role="radiogroup">
      {LANGUAGE_CHOICES.map(choice => (
        <button
          aria-checked={locale === choice.id}
          className={cn(
            'min-h-11 min-w-9 rounded-lg px-1.5 text-xs font-semibold outline-none focus-visible:outline focus-visible:outline-solid focus-visible:outline-2 focus-visible:outline-(--ui-accent)',
            locale === choice.id ? 'jarvis-segment-on' : 'text-(--ui-text-secondary) hover:text-(--ui-text-primary)'
          )}
          disabled={isSavingLocale}
          key={choice.id}
          onClick={() => void setLocale(choice.id)}
          role="radio"
          type="button"
        >
          {choice.label}
        </button>
      ))}
    </div>
  )
}

type ThemeChoice = 'dark' | 'light' | 'system'

const THEME_CHOICES: readonly { icon: IconComponent; id: ThemeChoice }[] = [
  { icon: Sun, id: 'light' },
  { icon: Moon, id: 'dark' },
  { icon: Monitor, id: 'system' }
]

/** Light, dark, or follow the system — the whole app, orb included, repaints. */
function ThemeToggle({ copy }: { copy: JarvisShellCopy['home']['nav'] }) {
  const { mode, setMode } = useTheme()
  const labels: Record<ThemeChoice, string> = { dark: copy.themeDark, light: copy.themeLight, system: copy.themeSystem }

  return (
    // Shares one row with the language switch: icons only, names in aria-label.
    <div aria-label={copy.theme} className="jarvis-glass flex shrink-0 gap-0.5 rounded-xl p-0.5" role="radiogroup">
      {THEME_CHOICES.map(({ icon: Icon, id }) => (
        <button
          aria-checked={mode === id}
          aria-label={labels[id]}
          className={cn(
            'grid min-h-11 min-w-9 place-items-center rounded-lg px-1.5 outline-none transition-colors focus-visible:outline focus-visible:outline-solid focus-visible:outline-2 focus-visible:outline-(--ui-accent)',
            mode === id ? 'jarvis-segment-on' : 'text-(--ui-text-secondary) hover:text-(--ui-text-primary)'
          )}
          key={id}
          onClick={() => setMode(id)}
          role="radio"
          title={labels[id]}
          type="button"
        >
          <Icon className="size-4" />
        </button>
      ))}
    </div>
  )
}
