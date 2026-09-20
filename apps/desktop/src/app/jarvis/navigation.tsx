import type { KeyboardEvent, RefObject } from 'react'
import { createRef, useMemo } from 'react'

import { Brain, CheckCircle2, KeyRound, Settings2, Wrench, Zap } from '@/lib/icons'
import { cn } from '@/lib/utils'

import {
  JARVIS_AUXILIARY_VIEWS,
  JARVIS_MAIN_VIEWS,
  type JarvisShellCopy,
  type JarvisShellView
} from './i18n'

type IconComponent = React.ComponentType<{ className?: string }>

const VIEW_ICONS: Record<JarvisShellView, IconComponent> = {
  jarvis: Zap,
  tasks: CheckCircle2,
  memory: Brain,
  tools: Wrench,
  settings: Settings2,
  profile: KeyRound
}

interface JarvisNavigationProps {
  activeView: JarvisShellView
  copy: JarvisShellCopy
  onSelect: (view: JarvisShellView) => void
}

interface NavButtonProps {
  active: boolean
  buttonRef: RefObject<HTMLButtonElement | null>
  icon: IconComponent
  label: string
  onClick: () => void
  onKeyDown?: (event: KeyboardEvent<HTMLButtonElement>) => void
}

function NavButton({ active, buttonRef, icon: Icon, label, onClick, onKeyDown }: NavButtonProps) {
  return (
    <button
      aria-current={active ? 'page' : undefined}
      className={cn(
        'group flex min-h-11 min-w-32 items-center gap-3 rounded-md px-3 text-left text-sm font-medium outline-none transition-colors md:min-w-0',
        'focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-(--ui-accent)',
        active
          ? 'bg-(--ui-bg-quaternary) text-(--ui-text-primary)'
          : 'text-(--ui-text-secondary) hover:bg-(--chrome-action-hover) hover:text-(--ui-text-primary)'
      )}
      onClick={onClick}
      onKeyDown={onKeyDown}
      ref={buttonRef}
      type="button"
    >
      <Icon className={cn('size-4 shrink-0', active ? 'text-(--ui-accent)' : 'text-(--ui-text-tertiary)')} />
      <span className="truncate">{label}</span>
    </button>
  )
}

export function JarvisNavigation({ activeView, copy, onSelect }: JarvisNavigationProps) {
  const mainRefs = useMemo(() => JARVIS_MAIN_VIEWS.map(() => createRef<HTMLButtonElement>()), [])
  const auxiliaryRefs = useMemo(() => JARVIS_AUXILIARY_VIEWS.map(() => createRef<HTMLButtonElement>()), [])

  const focusMain = (index: number) => {
    mainRefs[index]?.current?.focus()
  }

  const handleMainKeyDown = (index: number) => (event: KeyboardEvent<HTMLButtonElement>) => {
    if (event.key === 'ArrowDown' || event.key === 'ArrowRight') {
      event.preventDefault()
      focusMain((index + 1) % JARVIS_MAIN_VIEWS.length)
      return
    }

    if (event.key === 'ArrowUp' || event.key === 'ArrowLeft') {
      event.preventDefault()
      focusMain((index - 1 + JARVIS_MAIN_VIEWS.length) % JARVIS_MAIN_VIEWS.length)
      return
    }

    if (event.key === 'Home') {
      event.preventDefault()
      focusMain(0)
      return
    }

    if (event.key === 'End') {
      event.preventDefault()
      focusMain(JARVIS_MAIN_VIEWS.length - 1)
    }
  }

  return (
    <aside className="flex w-full shrink-0 flex-col gap-3 border-b border-(--ui-stroke-tertiary) bg-(--ui-bg-chrome) p-3 md:h-full md:w-64 md:border-b-0 md:border-r">
      <div className="min-w-0 px-1 py-1 md:py-2">
        <div className="truncate text-sm font-semibold text-(--ui-text-primary)">{copy.productName}</div>
      </div>

      <nav aria-label={copy.navigationLabel} className="min-w-0" data-jarvis-nav="">
        <div className="flex gap-2 overflow-x-auto pb-1 md:flex-col md:overflow-visible md:pb-0">
          {JARVIS_MAIN_VIEWS.map((view, index) => (
            <NavButton
              active={activeView === view}
              buttonRef={mainRefs[index]}
              icon={VIEW_ICONS[view]}
              key={view}
              label={copy.views[view]}
              onClick={() => onSelect(view)}
              onKeyDown={handleMainKeyDown(index)}
            />
          ))}
        </div>
      </nav>

      <div className="mt-auto flex gap-2 border-t border-(--ui-stroke-tertiary) pt-3 md:flex-col">
        {JARVIS_AUXILIARY_VIEWS.map((view, index) => (
          <NavButton
            active={activeView === view}
            buttonRef={auxiliaryRefs[index]}
            icon={VIEW_ICONS[view]}
            key={view}
            label={copy.views[view]}
            onClick={() => onSelect(view)}
          />
        ))}
      </div>
    </aside>
  )
}
