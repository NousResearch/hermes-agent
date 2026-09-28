/**
 * "Podpowiedzi" — the window that answers *co ja właściwie mogę mu zlecić?*
 *
 * Two rules keep it a help surface instead of an advert:
 *
 * - It only offers what this machine can already do. The deck is filtered by
 *   the capabilities chosen during setup, so nothing here fails the moment the
 *   user tries it.
 * - It opens itself exactly once, after setup, and never again on its own.
 *   Every later appearance is the button in the dashboard.
 *
 * Picking a tip fills the composer — it never sends. The user reads what Jarvis
 * is about to be asked before it is asked, which is the same contract the rest
 * of the product keeps around actions taken on their machine.
 */

import { useStore } from '@nanostores/react'
import { type ReactNode, useEffect, useMemo, useRef, useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { SearchField } from '@/components/ui/search-field'
import { useI18n } from '@/i18n'
import { Brain, Clock, FolderOpen, Globe, Lightbulb, Mic, Monitor, X } from '@/lib/icons'
import { cn } from '@/lib/utils'

import { requestComposerInsert } from '../chat/composer/focus'

import type { JarvisComputerMode } from './computer-capabilities'
import {
  $jarvisOnboardingCompletedAt,
  jarvisOnboardingComplete,
  type JarvisOnboardingScope,
  readJarvisOnboardingState
} from './onboarding-state'
import {
  jarvisPlaybookCategories,
  type JarvisPlaybookCategory,
  type JarvisPlaybookEntry,
  selectJarvisPlaybook
} from './playbook'
import {
  dismissJarvisTip,
  type JarvisTipsState,
  readJarvisTipsState,
  resetJarvisTips,
  shouldAutoOpenJarvisTips,
  writeJarvisTipsState
} from './tips-state'

type IconComponent = React.ComponentType<{ className?: string }>

const CATEGORY_ICONS: Record<JarvisPlaybookCategory, IconComponent> = {
  automation: Clock,
  computer: Monitor,
  files: FolderOpen,
  memory: Brain,
  voice: Mic,
  web: Globe
}

type TipsCopy = ReturnType<typeof useI18n>['t']['jarvisTips']

export interface JarvisTipsWindowProps {
  copy: TipsCopy
  entries: readonly JarvisPlaybookEntry[]
  onClose: () => void
  onDismiss: (id: string) => void
  onReset: () => void
  onUse: (prompt: string) => void
  open: boolean
}

function matches(text: string, query: string): boolean {
  return text.toLocaleLowerCase().includes(query)
}

export function JarvisTipsWindow({
  copy,
  entries,
  onClose,
  onDismiss,
  onReset,
  onUse,
  open
}: JarvisTipsWindowProps) {
  const [query, setQuery] = useState('')
  const [category, setCategory] = useState<'all' | JarvisPlaybookCategory>('all')
  const categories = useMemo(() => jarvisPlaybookCategories(entries), [entries])

  // A filter pinned to a category that the deck no longer has (the last tip in
  // it was hidden) would render an empty window with no way back.
  const activeCategory = category !== 'all' && !categories.includes(category) ? 'all' : category

  const visible = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase()

    return entries.filter(entry => {
      if (activeCategory !== 'all' && entry.category !== activeCategory) {
        return false
      }

      const text = copy.entries[entry.id]

      return !needle || matches(text.title, needle) || matches(text.detail, needle)
    })
  }, [activeCategory, copy.entries, entries, query])

  return (
    <Dialog onOpenChange={next => !next && onClose()} open={open}>
      <DialogContent
        bodyClassName="grid max-h-[min(40rem,calc(100vh-4rem))] grid-rows-[auto_auto_minmax(0,1fr)_auto] gap-3 p-4 sm:p-5"
        className="jarvis-tips-window w-[calc(100vw-2rem)] max-w-3xl"
        data-testid="jarvis-tips"
      >
        <DialogHeader>
          <DialogTitle icon={Lightbulb}>{copy.title}</DialogTitle>
          <DialogDescription>{copy.subtitle}</DialogDescription>
        </DialogHeader>

        <div className="flex flex-wrap items-center gap-2">
          <SearchField
            aria-label={copy.searchLabel}
            containerClassName="min-w-40 flex-1"
            onChange={setQuery}
            placeholder={copy.searchPlaceholder}
            value={query}
          />
          <div aria-label={copy.categoriesLabel} className="flex flex-wrap gap-1.5" role="group">
            <CategoryChip
              active={activeCategory === 'all'}
              label={copy.allCategories}
              onSelect={() => setCategory('all')}
            />
            {categories.map(item => (
              <CategoryChip
                active={activeCategory === item}
                icon={CATEGORY_ICONS[item]}
                key={item}
                label={copy.categories[item]}
                onSelect={() => setCategory(item)}
              />
            ))}
          </div>
        </div>

        <ul className="grid min-h-0 gap-2 overflow-y-auto pr-1" data-testid="jarvis-tips-list">
          {visible.map(entry => {
            const text = copy.entries[entry.id]
            const Icon = CATEGORY_ICONS[entry.category]

            return (
              <li
                className="jarvis-tips-card grid gap-2 p-3"
                key={entry.id}
              >
                <div className="flex items-start gap-2">
                  <Icon className="mt-0.5 size-4 shrink-0 text-(--ui-accent)" />
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium text-(--ui-text-primary)">{text.title}</p>
                    <p className="mt-1 text-sm text-(--ui-text-secondary)">{text.detail}</p>
                  </div>
                  <Badge size="xs" variant="muted">
                    {copy.categories[entry.category]}
                  </Badge>
                </div>
                <div className="flex flex-wrap items-center justify-end gap-2">
                  <Button
                    aria-label={copy.hideEntry(text.title)}
                    className="min-h-11"
                    onClick={() => onDismiss(entry.id)}
                    size="sm"
                    type="button"
                    variant="ghost"
                  >
                    <X className="size-4" />
                    {copy.hide}
                  </Button>
                  <Button className="min-h-11" onClick={() => onUse(text.prompt)} size="sm" type="button">
                    {copy.use}
                  </Button>
                </div>
              </li>
            )
          })}
          {visible.length === 0 && (
            <li className="rounded-md border border-dashed border-(--ui-stroke-tertiary) p-4 text-sm text-(--ui-text-secondary)">
              {entries.length === 0 ? copy.empty : copy.emptyFiltered}
            </li>
          )}
        </ul>

        <div className="flex flex-wrap items-center justify-between gap-2 border-t border-(--ui-stroke-tertiary) pt-3">
          <p className="text-xs text-(--ui-text-tertiary)">{copy.footerHint}</p>
          <div className="flex gap-2">
            <Button className="min-h-11" onClick={onReset} size="sm" type="button" variant="outline">
              {copy.reset}
            </Button>
            <Button className="min-h-11" onClick={onClose} size="sm" type="button" variant="secondary">
              {copy.close}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}

function CategoryChip({
  active,
  icon: Icon,
  label,
  onSelect
}: {
  active: boolean
  icon?: IconComponent
  label: string
  onSelect: () => void
}) {
  return (
    <button
      aria-pressed={active}
      className={cn(
        'inline-flex min-h-11 items-center gap-1.5 rounded-md border px-3 text-sm transition-colors',
        'focus-visible:outline-2 focus-visible:outline-solid focus-visible:outline-offset-2 focus-visible:outline-(--ui-accent)',
        active
          ? 'border-(--ui-accent) bg-(--ui-bg-quaternary) text-(--ui-text-primary)'
          : 'border-(--ui-stroke-tertiary) text-(--ui-text-secondary) hover:text-(--ui-text-primary)'
      )}
      onClick={onSelect}
      type="button"
    >
      {Icon ? <Icon className="size-3.5 shrink-0" /> : null}
      {label}
    </button>
  )
}

export interface JarvisTipsLauncherProps {
  /** A turn in flight — the window may be opened, but never opens itself. */
  busy?: boolean
  className?: string
  /** Test seam: the deck's capability context, normally read from setup. */
  computerMode?: JarvisComputerMode | null
  hasHistory?: boolean
  /** Test seam: where a picked tip goes. Defaults to the main composer. */
  onUse?: (prompt: string) => void
  scope?: JarvisOnboardingScope
  storage?: Storage
  /** Rendered instead of the default button — lets a host place its own trigger. */
  trigger?: (props: { onClick: () => void }) => ReactNode
}

/**
 * The button plus the window, with the remembered state behind them.
 *
 * Persistence is deliberately synchronous-on-change rather than an effect: a
 * hidden tip that came back after a reload would make the ✕ look broken.
 */
export function JarvisTipsLauncher({
  busy = false,
  className,
  computerMode: computerModeOverride,
  hasHistory = false,
  onUse,
  scope,
  storage,
  trigger
}: JarvisTipsLauncherProps) {
  const { t } = useI18n()
  const copy = t.jarvisTips
  const [open, setOpen] = useState(false)
  const [state, setState] = useState<JarvisTipsState>(() => readJarvisTipsState(storage, scope))
  const autoOpenChecked = useRef(false)

  // Setup usually finishes while this is already mounted (the wizard is an
  // overlay on top of it), and storage does not notify — so the completion
  // signal is what makes this re-read instead of answering from mid-wizard.
  const completedAt = useStore($jarvisOnboardingCompletedAt)

  const onboarding = useMemo(
    () => readJarvisOnboardingState(storage, scope),
    // `completedAt` is a cache key, not an input: a bump means the stored state
    // changed under us and has to be read again.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [completedAt, scope, storage]
  )

  const computerMode =
    computerModeOverride !== undefined ? computerModeOverride : (onboarding?.selections?.computerMode ?? null)

  const persist = (next: JarvisTipsState) => {
    setState(next)
    writeJarvisTipsState(next, storage, scope)
  }

  // eslint-disable-next-line no-restricted-syntax -- one-shot latch, not an atom mirror
  useEffect(() => {
    // Not latching on an unfinished setup or a busy moment: the window has not
    // had its turn yet, so it gets to ask again once the answer can change.
    if (autoOpenChecked.current || busy || !jarvisOnboardingComplete(onboarding)) {
      return
    }

    autoOpenChecked.current = true

    if (!shouldAutoOpenJarvisTips({ busy, onboardingComplete: true, state })) {
      return
    }

    setOpen(true)
    // Spending the one auto-open here (not on close) means a crash or a quick
    // dismissal still counts: the window has had its turn either way.
    const spent = { ...state, autoOpen: false }
    setState(spent)
    writeJarvisTipsState(spent, storage, scope)
  }, [busy, onboarding, scope, state, storage])

  const entries = useMemo(
    () => selectJarvisPlaybook({ computerMode, dismissedIds: state.dismissedIds, hasHistory }),
    [computerMode, hasHistory, state.dismissedIds]
  )

  const use = (prompt: string) => {
    if (onUse) {
      onUse(prompt)
    } else {
      requestComposerInsert(prompt, { mode: 'block', target: 'main' })
    }

    setOpen(false)
  }

  return (
    // The corner wrapper is what puts the trigger in the Pulpit's top corner
    // (`order: 2`, see glass.css): the host renders it wherever it likes and
    // the CSS keeps it the right-most control of the top bar.
    <div className="jarvis-tips-corner" data-testid="jarvis-tips-corner">
      {trigger ? (
        trigger({ onClick: () => setOpen(true) })
      ) : (
        <Button
          aria-label={copy.openLabel}
          className={cn(
            'jarvis-glass jarvis-glass-hover min-h-11 rounded-full px-4 text-(--ui-text-primary)',
            className
          )}
          onClick={() => setOpen(true)}
          size="sm"
          type="button"
          variant="secondary"
        >
          <Lightbulb className="size-4 text-(--ui-accent)" />
          {copy.openLabel}
        </Button>
      )}
      <JarvisTipsWindow
        copy={copy}
        entries={entries}
        onClose={() => setOpen(false)}
        onDismiss={id => persist(dismissJarvisTip(state, id))}
        onReset={() => persist(resetJarvisTips(state))}
        onUse={use}
        open={open}
      />
    </div>
  )
}
