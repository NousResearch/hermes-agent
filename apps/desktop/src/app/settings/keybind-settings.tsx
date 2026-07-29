import { useStore } from '@nanostores/react'
import { useMemo, useState } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { DisclosureCaret } from '@/components/ui/disclosure-caret'
import { Kbd, KbdCombo } from '@/components/ui/kbd'
import { SearchField } from '@/components/ui/search-field'
import { Tip } from '@/components/ui/tooltip'
import { useContributions } from '@/contrib/react/use-contributions'
import { useI18n } from '@/i18n'
import {
  allKeybindActions,
  KEYBIND_CATEGORIES,
  KEYBIND_PANEL_ACTION,
  KEYBIND_READONLY,
  type KeybindActionMeta,
  type KeybindReadonly,
  KEYBINDS_AREA
} from '@/lib/keybinds/actions'
import { formatCombo } from '@/lib/keybinds/combo'
import { arraysEqual } from '@/lib/storage'
import {
  $bindings,
  $capture,
  beginCapture,
  bindingsFor,
  conflictsFor,
  endCapture,
  resetAllBindings,
  resetBinding
} from '@/store/keybinds'

import { SettingsContent } from './primitives'

export function KeybindSettings() {
  const { t } = useI18n()
  const bindings = useStore($bindings)
  const k = t.keybinds
  const [collapsed, setCollapsed] = useState<ReadonlySet<string>>(new Set())
  // Subscribe so contributed actions appear/disappear live in the map.
  useContributions(KEYBINDS_AREA)
  const actionList = allKeybindActions()
  const [query, setQuery] = useState('')

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

  // Filter actions and readonly shortcuts by label match against the query.
  // When searching, categories auto-expand (collapsed state is ignored).
  const isSearching = query.trim().length > 0

  const filteredActions = useMemo(() => {
    if (!isSearching) {
      return null
    }

    const lower = query.toLowerCase()

    return actionList.filter(action => {
      if (action.id === KEYBIND_PANEL_ACTION) {
        return false
      }

      const label = k.actions[action.id] ?? action.id

      return label.toLowerCase().includes(lower) || action.id.includes(lower)
    })
  }, [actionList, isSearching, query, k.actions])

  const filteredReadonly = useMemo(() => {
    if (!isSearching) {
      return null
    }

    const lower = query.toLowerCase()

    return KEYBIND_READONLY.filter(shortcut => {
      const label = k.actions[shortcut.id] ?? shortcut.id

      return label.toLowerCase().includes(lower) || shortcut.id.includes(lower)
    })
  }, [isSearching, query, k.actions])

  return (
    <SettingsContent>
      <div className="flex items-center justify-between gap-3 pb-3">
        <div className="min-w-0">
          <h2 className="text-sm font-semibold text-foreground">{k.title}</h2>
          <p className="mt-0.5 text-[0.72rem] text-muted-foreground">
            {k.subtitle(openCombo ? formatCombo(openCombo) : '')}
          </p>
        </div>
        <button
          className="flex shrink-0 items-center gap-1 rounded-md text-[0.72rem] text-muted-foreground hover:text-foreground"
          onClick={resetAllBindings}
          type="button"
        >
          <Codicon name="discard" size="0.8125rem" />
          {k.resetAll}
        </button>
      </div>

      <div className="pb-3">
        <SearchField
          aria-label={k.search}
          containerClassName="w-full"
          onChange={setQuery}
          placeholder={k.search}
          value={query}
        />
      </div>

      {isSearching ? (
        <div className="px-2 py-1.5">
          {filteredActions?.length === 0 && filteredReadonly?.length === 0 ? (
            <p className="px-2.5 py-4 text-center text-[0.82rem] text-muted-foreground">—</p>
          ) : (
            <>
              {filteredActions?.map(action => (
                <KeybindRow action={action} key={action.id} />
              ))}
              {filteredReadonly?.map(shortcut => (
                <ReadonlyRow key={shortcut.id} shortcut={shortcut} />
              ))}
            </>
          )}
        </div>
      ) : (
        <div className="px-2 py-1.5">
          {KEYBIND_CATEGORIES.map(category => {
            const actions = actionList.filter(
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
      )}
    </SettingsContent>
  )
}

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

function KeybindRow({ action }: { action: KeybindActionMeta }) {
  const { t } = useI18n()
  const k = t.keybinds
  const bindings = useStore($bindings)
  const capture = useStore($capture)

  // bindingsFor resolves stored overrides for late-registered (contributed)
  // actions too — $bindings only carries built-ins, so a raw lookup would show
  // the default instead of the user's rebinding for a plugin/contrib action.
  const combos = bindingsFor(action.id, bindings)
  const capturing = capture === action.id
  const label = k.actions[action.id] ?? action.label ?? action.id
  const isDefault = arraysEqual(combos, [...action.defaults])

  const conflict = combos
    .flatMap(combo => conflictsFor(action.id, combo).map(other => k.actions[other] ?? other))
    .find(Boolean)

  return (
    <div className="group flex items-center gap-2.5 rounded-lg px-2.5 py-1 transition-colors hover:bg-(--chrome-action-hover)">
      <span className="min-w-0 flex-1 truncate text-[0.82rem] text-foreground/90">{label}</span>

      {conflict && (
        <span className="flex size-4 items-center justify-center text-amber-500/90" title={k.conflictWith(conflict)}>
          <Codicon name="warning" size="0.8125rem" />
        </span>
      )}

      {/* Click the caps to rebind — the on-screen editor does the same thing. */}
      <Tip label={k.rebind}>
        <button
          aria-label={k.rebind}
          className="flex shrink-0 items-center gap-1 rounded-lg outline-none"
          onClick={() => (capturing ? endCapture() : beginCapture(action.id))}
          type="button"
        >
          {capturing ? (
            <Kbd variant="capturing">{k.pressKey}</Kbd>
          ) : combos.length > 0 ? (
            combos.map(combo => <KbdCombo combo={combo} key={combo} />)
          ) : (
            <Kbd variant="ghost">{k.set}</Kbd>
          )}
        </button>
      </Tip>

      {/* Reset only shows once a binding diverges from its default; the spacer
          holds the column otherwise so rows stay aligned. */}
      {isDefault ? (
        <span aria-hidden className="size-6 shrink-0" />
      ) : (
        <Tip label={k.reset}>
          <button
            aria-label={k.reset}
            className="grid size-6 shrink-0 place-items-center rounded-md text-muted-foreground/70 opacity-0 transition-all hover:bg-(--ui-control-active-background) hover:text-foreground group-hover:opacity-100"
            onClick={() => resetBinding(action.id)}
            type="button"
          >
            <Codicon name="discard" size="0.8125rem" />
          </button>
        </Tip>
      )}
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
      <span className="min-w-0 flex-1 truncate text-[0.82rem] text-foreground/75">{label}</span>
      <div className="flex shrink-0 items-center gap-1">
        {shortcut.keys.map(key => (
          <KbdCombo combo={key} key={key} />
        ))}
      </div>
      <span aria-hidden className="size-6 shrink-0" />
    </div>
  )
}
