// Local state for one open tool list: baseline, local, dirty, saving, conflict.
//
// No RPC lives here. The caller hands in the saved rule and a `save` callback and
// gets back the editor's own machine, so the hook is provable without a backend
// and the wiring slice can swap what `save` actually does.

import { useCallback, useEffect, useMemo, useState } from 'react'

import {
  editorCounts,
  expandQuickAction,
  isUntouchedByQuickActions,
  matchingQuickAction,
  quickActionById,
  sameSet
} from './derive-tools'
import type { QuickAction, QuickActionId, ToolRowModel, ToolsEditorCounts, ToolsEditorPhase } from './types'

/** What a save came back as. `failed` leaves the editor dirty and the work on
 *  screen; only `saved` moves the baseline. */
export type SaveResult = 'conflict' | 'failed' | 'saved'

/** The phases the CALLER owns: they are facts about the fetch, not about the
 *  edit, and the hook never enters or leaves them on its own. */
export type ToolsEditorStatus = Extract<ToolsEditorPhase, 'gone' | 'loading' | 'signedOut' | 'unavailable'>

/** How this write is meant to land. `overwrite` is the reader's answer to a
 *  conflict: write over the version the other editor saved. It has to travel with
 *  the list, because a compare-and-set write cannot tell an overwrite from a
 *  second losing attempt. */
export interface SaveOptions {
  overwrite: boolean
}

export interface UseToolsEditorOptions {
  /** Identity of the thing being edited — the connector slug, usually. Two
   *  connectors can share a saved rule (`[]` and `[]` on first use), so the rule
   *  alone cannot say "this is a different editor now". */
  editorKey?: string
  onSave: (disabled: string[], options: SaveOptions) => Promise<SaveResult>
  /** The saved personal rule. A different list is a different editor: the hook
   *  reloads itself against it. */
  savedDisabled: readonly string[]
  status?: ToolsEditorStatus | null
  tools: readonly ToolRowModel[]
}

export interface ToolsEditor {
  applyQuickAction: (id: QuickActionId) => void
  counts: ToolsEditorCounts
  /** Which quick action the current list matches, if any. */
  currentAction: QuickAction | null
  dirty: boolean
  disabledSet: ReadonlySet<string>
  discard: () => void
  isOn: (slug: string) => boolean
  /** Keep this editor's work after a conflict, and write it — the button says
   *  "Save over their version", so pressing it has to save. */
  keepMine: () => Promise<SaveResult>
  local: string[]
  /** True once the reader chose to write over the other version. */
  overwrite: boolean
  phase: ToolsEditorPhase
  save: () => Promise<SaveResult>
  toggle: (slug: string) => void
}

export function useToolsEditor({
  editorKey,
  onSave,
  savedDisabled,
  status,
  tools
}: UseToolsEditorOptions): ToolsEditor {
  const [local, setLocal] = useState<string[]>([...savedDisabled])
  const [baseline, setBaseline] = useState<string[]>([...savedDisabled])
  const [editing, setEditing] = useState<'conflict' | 'ready' | 'saving'>('ready')
  /** The action the person pressed. Two actions can expand to the same list, so
   *  the choice is remembered rather than re-derived from the result. */
  const [pressed, setPressed] = useState<QuickActionId | null>(null)
  const [overwrite, setOverwrite] = useState(false)

  // Reset when the editor changes identity: another connector, another profile,
  // or a reload after a conflict. `editorKey` is part of that identity because
  // the saved rule is not one: swapping connector A for connector B when both
  // are saved as `[]` would otherwise carry A's unsaved edit onto B and then
  // write it as B's rule.
  const savedKey = `${editorKey ?? ''}\u0000${[...savedDisabled].sort().join('\u0000')}`

  useEffect(() => {
    setLocal([...savedDisabled])
    setBaseline([...savedDisabled])
    setEditing('ready')
    setPressed(null)
    setOverwrite(false)
    // Keyed on the CONTENT of the saved rule, not its array identity: a caller
    // that rebuilds the list every render would otherwise wipe the edit in
    // progress on every keystroke.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [savedKey])

  const byslug = useMemo(() => new Map(tools.map(tool => [tool.slug, tool])), [tools])
  const disabledSet = useMemo(() => new Set(local), [local])
  const dirty = !sameSet(local, baseline)
  const counts = useMemo(() => editorCounts(local, baseline), [local, baseline])
  const currentAction = useMemo(() => matchingQuickAction(local, tools, pressed), [local, tools, pressed])

  const toggle = useCallback(
    (slug: string) => {
      // A locked row has no switch to press; refusing here as well keeps a stray
      // keyboard event from writing a rule the org already owns.
      if (byslug.get(slug)?.lockedBy) {
        return
      }

      setPressed(null)
      setLocal(previous => (previous.includes(slug) ? previous.filter(s => s !== slug) : [...previous, slug]))
    },
    [byslug]
  )

  /** A quick action rewrites only the part of the list it owns. Anything the
   *  person turned off by hand that a quick action may never touch — unclassified
   *  or deprecated — is carried across untouched, including by `Everything on`. */
  const applyQuickAction = useCallback(
    (id: QuickActionId) => {
      const action = quickActionById(id)

      setPressed(id)
      setLocal(previous => {
        const kept = previous.filter(slug => {
          const tool = byslug.get(slug)

          // An org-locked slug is carried across for the same reason an
          // unclassified one is: `expandQuickAction` will not put it back, so
          // dropping it here would delete a rule the person wrote by hand and
          // report the tool as "back on" although no switch moved.
          return tool !== undefined && (isUntouchedByQuickActions(tool) || tool.lockedBy !== null)
        })

        return [...new Set([...kept, ...expandQuickAction(action, tools)])]
      })
    },
    [byslug, tools]
  )

  const discard = useCallback(() => {
    setLocal(baseline)
    setEditing('ready')
    setPressed(null)
    setOverwrite(false)
  }, [baseline])

  /** One writer, so a press and a write can never disagree about the flag. The
   *  caller reads `overwrite` off the options rather than off a later render. */
  const write = useCallback(
    async (asOverwrite: boolean) => {
      setEditing('saving')

      const result = await onSave(local, { overwrite: asOverwrite })

      setEditing(result === 'conflict' ? 'conflict' : 'ready')

      if (result === 'saved') {
        setBaseline(local)
        setOverwrite(false)
      }

      return result
    },
    [local, onSave]
  )

  /** Nothing merges: this writes over the version the other editor left. The
   *  button is labelled "Save over their version", so it saves — leaving the
   *  reader back at the dirty footer with nothing written would make the label
   *  a lie and lose the one control that says what happens next. */
  const keepMine = useCallback(() => {
    setOverwrite(true)

    return write(true)
  }, [write])

  const save = useCallback(() => write(overwrite), [overwrite, write])

  const isOn = useCallback(
    (slug: string) => byslug.get(slug)?.lockedBy === null && !disabledSet.has(slug),
    [byslug, disabledSet]
  )

  return {
    applyQuickAction,
    counts,
    currentAction,
    dirty,
    disabledSet,
    discard,
    isOn,
    keepMine,
    local,
    overwrite,
    // A failed fetch is not an edit state: it replaces the whole right column,
    // so it outranks whatever the editor thinks it is doing.
    phase: status ?? editing,
    save,
    toggle
  }
}
