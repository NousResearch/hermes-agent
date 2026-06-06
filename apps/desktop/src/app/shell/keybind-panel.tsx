import { useStore } from '@nanostores/react'
import { Dialog as DialogPrimitive } from 'radix-ui'
import { useState } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { DisclosureCaret } from '@/components/ui/disclosure-caret'
import { useI18n } from '@/i18n'
import {
  KEYBIND_ACTIONS,
  KEYBIND_CATEGORIES,
  KEYBIND_PANEL_ACTION,
  KEYBIND_READONLY,
  type KeybindActionMeta,
  type KeybindReadonly
} from '@/lib/keybinds/actions'
import { formatCombo } from '@/lib/keybinds/combo'
import {
  $bindings,
  $capture,
  $keybindPanelOpen,
  beginCapture,
  closeKeybindPanel,
  conflictsFor,
  endCapture,
  resetAllBindings,
  resetBinding
} from '@/store/keybinds'

// The full hotkey map. Quiet popover, click a row's chip to rebind.
export function KeybindPanel() {
  const { t } = useI18n()
  const open = useStore($keybindPanelOpen)
  const bindings = useStore($bindings)
  const k = t.keybinds
  const [collapsed, setCollapsed] = useState<ReadonlySet<string>>(new Set())

  const openCombo = bindings[KEYBIND_PANEL_ACTION]?.[0]

  const toggleCategory = (category: string) =>
    setCollapsed(prev => {
      const next = new Set(prev)

      if (next.has(category)) {
        next.delete(category)
      } else {
        next.add(category)
      }

      return next
    })

  return (
    <DialogPrimitive.Root onOpenChange={next => !next && closeKeybindPanel()} open={open}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-[200] bg-black/25 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:animate-in data-[state=open]:fade-in-0" />
        <DialogPrimitive.Content
          aria-describedby={undefined}
          className="fixed left-1/2 top-[9vh] z-[210] flex max-h-[82vh] w-[min(38rem,calc(100vw-2rem))] -translate-x-1/2 flex-col overflow-hidden rounded-xl border border-(--ui-stroke-secondary) bg-(--ui-chat-bubble-background) shadow-[0_20px_48px_-24px_rgba(0,0,0,0.55)] duration-150 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95 data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95"
        >
          {/* Header */}
          <div className="flex items-center justify-between gap-3 border-b border-(--ui-stroke-tertiary) px-4 py-3">
            <div className="min-w-0">
              <DialogPrimitive.Title className="text-sm font-semibold text-foreground">{k.title}</DialogPrimitive.Title>
              <DialogPrimitive.Description className="mt-0.5 text-[0.72rem] text-muted-foreground">
                {k.subtitle(openCombo ? formatCombo(openCombo) : '')}
              </DialogPrimitive.Description>
            </div>
            <HeaderButton icon="discard" label={k.resetAll} onClick={resetAllBindings} />
          </div>

          {/* Body */}
          <div className="min-h-0 flex-1 overflow-y-auto px-2 py-1.5">
            {KEYBIND_CATEGORIES.map(category => {
              const actions = KEYBIND_ACTIONS.filter(
                action => action.category === category && action.id !== KEYBIND_PANEL_ACTION
              )

              const readonly = KEYBIND_READONLY.filter(shortcut => shortcut.category === category)

              if (actions.length === 0 && readonly.length === 0) {
                return null
              }

              const sectionOpen = !collapsed.has(category)

              return (
                <section key={category}>
                  <CategoryHeader
                    label={k.categories[category] ?? category}
                    onToggle={() => toggleCategory(category)}
                    open={sectionOpen}
                  />
                  {sectionOpen && actions.map(action => <KeybindRow action={action} key={action.id} />)}
                  {sectionOpen && readonly.map(shortcut => <ReadonlyRow key={shortcut.id} shortcut={shortcut} />)}
                </section>
              )
            })}
          </div>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  )
}

// Collapsible category header — chevron fades in on hover, rotates when open
// (matches the sessions sidebar section pattern).
function CategoryHeader({ label, onToggle, open }: { label: string; onToggle: () => void; open: boolean }) {
  return (
    <button
      className="group/kbd-cat flex w-fit items-center gap-1 px-2.5 pb-1 pt-3 text-left leading-none"
      onClick={onToggle}
      type="button"
    >
      <span className="text-[0.64rem] font-semibold uppercase tracking-[0.12em] text-muted-foreground/70">{label}</span>
      <DisclosureCaret
        className="text-(--ui-text-tertiary) opacity-0 transition group-hover/kbd-cat:opacity-100"
        open={open}
        size="0.6875rem"
      />
    </button>
  )
}

function HeaderButton({ icon, label, onClick }: { icon: string; label: string; onClick: () => void }) {
  return (
    <button
      className="flex items-center gap-1.5 rounded-md px-2 py-1 text-[0.72rem] font-medium text-muted-foreground transition-colors hover:bg-(--chrome-action-hover) hover:text-foreground"
      onClick={onClick}
      type="button"
    >
      <Codicon name={icon} size="0.8125rem" />
      {label}
    </button>
  )
}

function KeybindRow({ action }: { action: KeybindActionMeta }) {
  const { t } = useI18n()
  const k = t.keybinds
  const bindings = useStore($bindings)
  const capture = useStore($capture)

  const combos = bindings[action.id] ?? []
  const capturing = capture === action.id
  const label = k.actions[action.id] ?? action.id

  const conflict = combos
    .flatMap(combo => conflictsFor(action.id, combo).map(other => k.actions[other] ?? other))
    .find(Boolean)

  return (
    <div className="group flex items-center gap-2.5 rounded-lg px-2.5 py-1 transition-colors hover:bg-(--chrome-action-hover)">
      {/* Mirrors the reset button's footprint on the right so rows stay uniform. */}
      <span aria-hidden className="size-6 shrink-0" />
      <span className="min-w-0 flex-1 truncate text-[0.82rem] text-foreground/90">{label}</span>

      {conflict && (
        <span className="flex size-4 items-center justify-center text-amber-500/90" title={k.conflictWith(conflict)}>
          <Codicon name="warning" size="0.8125rem" />
        </span>
      )}

      {/* Click the caps to rebind — the on-screen editor does the same thing. */}
      <button
        aria-label={k.rebind}
        className="flex shrink-0 items-center gap-1 rounded-lg outline-none"
        onClick={() => (capturing ? endCapture() : beginCapture(action.id))}
        title={k.rebind}
        type="button"
      >
        {capturing ? (
          <span className="kbd-cap kbd-capturing">{k.pressKey}</span>
        ) : combos.length > 0 ? (
          combos.map(combo => (
            <span className="kbd-cap" key={combo}>
              {formatCombo(combo)}
            </span>
          ))
        ) : (
          <span className="kbd-cap kbd-cap--ghost">{k.set}</span>
        )}
      </button>

      <button
        aria-label={k.reset}
        className="grid size-6 shrink-0 place-items-center rounded-md text-muted-foreground/70 opacity-0 transition-all hover:bg-(--ui-control-active-background) hover:text-foreground group-hover:opacity-100"
        onClick={() => resetBinding(action.id)}
        title={k.reset}
        type="button"
      >
        <Codicon name="discard" size="0.8125rem" />
      </button>
    </div>
  )
}

// Fixed shortcut: same layout as KeybindRow but the caps aren't interactive and
// the trailing reset slot stays empty (spacer keeps the columns aligned).
function ReadonlyRow({ shortcut }: { shortcut: KeybindReadonly }) {
  const { t } = useI18n()
  const k = t.keybinds
  const label = k.actions[shortcut.id] ?? shortcut.id

  return (
    <div className="flex items-center gap-2.5 rounded-lg px-2.5 py-1">
      <span aria-hidden className="size-6 shrink-0" />
      <span className="min-w-0 flex-1 truncate text-[0.82rem] text-foreground/75">{label}</span>
      <div className="flex shrink-0 items-center gap-1">
        {shortcut.keys.map(key => (
          <span className="kbd-cap" key={key}>
            {formatCombo(key)}
          </span>
        ))}
      </div>
      <span aria-hidden className="size-6 shrink-0" />
    </div>
  )
}
