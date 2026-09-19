import type { Unstable_TriggerItem } from '@assistant-ui/core'
import { type Extension, Prec } from '@codemirror/state'
import { EditorView, keymap, ViewPlugin } from '@codemirror/view'
import { useStore } from '@nanostores/react'
import { type ReactNode, useCallback, useMemo, useRef, useState } from 'react'

import { COMPOSER_AREAS, type ComposerAtCompletionSource } from '@/app/chat/composer/contrib'
import { ComposerTriggerPopover } from '@/app/chat/composer/trigger-popover'
import { useContributions } from '@/contrib/react/use-contributions'
import { useI18n } from '@/i18n'
import { $activeGatewayProfile } from '@/store/profile'

import { mentionInsertion, mentionTokenAt } from './mention'

interface MentionAnchor {
  left: number
  query: string
  top: number
}

/** The window's own profile answers as `@hermes`; any other name is its profile. */
export function selfMentionHandle(profile: string): string {
  return profile === 'default' ? 'hermes' : profile
}

/**
 * The agents this page can call: every bot the composer's `@` list offers —
 * one registered source, so a renamed bot keeps its tag — plus the profile this
 * window is on, which that source leaves out (in a chat a bot never @s itself).
 */
function mentionItems(
  sources: readonly { data?: unknown }[],
  query: string,
  profile: string,
  selfMeta: string
): Unstable_TriggerItem[] {
  const self = `@${selfMentionHandle(profile)}`
  const rows: Unstable_TriggerItem[] = []
  const seen = new Set<string>()

  const add = (insert: string, display: string, meta: string) => {
    if (!insert.startsWith('@') || seen.has(insert)) {
      return
    }

    seen.add(insert)
    rows.push({ id: insert, label: insert, metadata: { display, meta }, type: 'simple' })
  }

  if (self.slice(1).toLowerCase().startsWith(query.toLowerCase())) {
    add(self, self, selfMeta)
  }

  for (const contribution of sources) {
    const source = contribution.data as ComposerAtCompletionSource | undefined

    if (typeof source?.provide !== 'function') {
      continue
    }

    try {
      for (const item of source.provide(query) || []) {
        add(item.insert, item.display || item.insert, item.meta || '')
      }
    } catch {
      // A broken source drops its own rows, never the list.
    }
  }

  return rows.slice(0, 8)
}

/**
 * The backend that answers `handle`: the window's own profile for `@hermes`,
 * otherwise the route the completion row carries. A row without one (an older
 * source) is not reachable from here, and the page says so rather than sending
 * the question somewhere arbitrary.
 */
export function resolveMentionRoute(
  sources: readonly { data?: unknown }[],
  handle: string,
  self: { connectionId: null | string; profile: string }
): { connectionId: null | string; profile: string } | null {
  if (handle === `@${selfMentionHandle(self.profile)}`) {
    return self
  }

  for (const contribution of sources) {
    const source = contribution.data as ComposerAtCompletionSource | undefined

    if (typeof source?.provide !== 'function') {
      continue
    }

    try {
      for (const item of source.provide(handle.slice(1)) || []) {
        if (item.insert === handle && item.target) {
          return { connectionId: item.target.connectionId ?? null, profile: item.target.profile }
        }
      }
    } catch {
      // A broken source cannot answer; the next one might.
    }
  }

  return null
}

/** Where the caret sits, for placing the list. A position that has no
 *  rectangle — not laid out yet, or a test environment without layout — falls
 *  back to the editor's own box, so the list still opens on the page. */
function caretRect(view: EditorView, pos: number): { bottom: number; left: number } {
  try {
    return view.coordsAtPos(pos) ?? view.dom.getBoundingClientRect()
  } catch {
    return view.dom.getBoundingClientRect()
  }
}

/**
 * The `@` list for the page's editor: a CodeMirror extension that tracks the
 * mention being typed and drives this list, plus the list itself. The rows are
 * the composer's own popover component — one completion look in the app.
 */
export function useMentionPopup(): { extension: Extension; popover: ReactNode } {
  const { t } = useI18n()
  const profile = useStore($activeGatewayProfile)
  const sources = useContributions(COMPOSER_AREAS.atCompletions)
  const [anchor, setAnchor] = useState<MentionAnchor | null>(null)
  const [activeIndex, setActiveIndex] = useState(0)
  const viewRef = useRef<EditorView | null>(null)
  const itemsRef = useRef<Unstable_TriggerItem[]>([])
  const activeIndexRef = useRef(0)

  const items = useMemo(
    () => (anchor ? mentionItems(sources, anchor.query, profile, t.backworkspace.mentionSelf) : []),
    [anchor, profile, sources, t]
  )

  // Read inside CodeMirror key handlers, which run outside React's render.
  itemsRef.current = items
  activeIndexRef.current = Math.min(activeIndex, Math.max(0, items.length - 1))

  const pick = useCallback((item: Unstable_TriggerItem) => {
    const view = viewRef.current
    const token = view && mentionTokenAt(view.state.doc.toString(), view.state.selection.main.head)

    if (!view || !token) {
      return
    }

    view.dispatch(mentionInsertion(token, item.id))
    view.focus()
    setAnchor(null)
  }, [])

  const extension = useMemo(() => {
    const isOpen = () => itemsRef.current.length > 0

    const step = (delta: number) => {
      setActiveIndex(current => {
        const count = itemsRef.current.length

        return count === 0 ? 0 : (current + delta + count) % count
      })
    }

    const track = (view: EditorView) => {
      const selection = view.state.selection.main
      const token = selection.empty ? mentionTokenAt(view.state.doc.toString(), selection.head) : null

      if (!token) {
        setAnchor(null)

        return
      }

      setActiveIndex(0)
      // Layout must not be read while an update runs, so the caret is measured
      // in CodeMirror's own measure phase. A caret with no rectangle yet (fresh
      // mount) falls back to the editor's box, which still puts the list on the
      // page rather than dropping it.
      view.requestMeasure({
        read: measured => caretRect(measured, token.from),
        write: rect => setAnchor({ left: rect.left, query: token.query, top: rect.bottom })
      })
    }

    return [
      ViewPlugin.define(view => {
        viewRef.current = view

        return {
          destroy() {
            viewRef.current = null
            setAnchor(null)
          },
          update(update) {
            if (update.docChanged || update.selectionSet) {
              track(update.view)
            }
          }
        }
      }),
      EditorView.domEventHandlers({
        blur() {
          setAnchor(null)

          return false
        }
      }),
      // Above CodeMirror's own bindings AND the page's: while the list is open
      // these keys belong to it, and Escape closes the list rather than turning
      // the window back.
      Prec.highest(
        keymap.of([
          // A handler that returns true takes the key: CodeMirror stops its own
          // bindings and calls preventDefault, and the page's Escape stands down
          // on that flag rather than turning the window back.
          { key: 'ArrowDown', run: () => isOpen() && (step(1), true) },
          { key: 'ArrowUp', run: () => isOpen() && (step(-1), true) },
          {
            key: 'Escape',
            run: () => {
              if (!isOpen()) {
                return false
              }

              setAnchor(null)

              return true
            }
          },
          ...['Enter', 'Tab'].map(key => ({
            key,
            run: () => {
              const item = itemsRef.current[activeIndexRef.current]

              if (!item) {
                return false
              }

              pick(item)

              return true
            }
          }))
        ])
      )
    ]
  }, [pick])

  const popover =
    anchor && items.length > 0 ? (
      // The caret's own coordinates: the list opens where the word is, not at
      // the page edge. The box reaches the right edge so the popover's own
      // width rules resolve against real space.
      <div className="absolute right-0" style={{ left: anchor.left, top: anchor.top }}>
        <ComposerTriggerPopover
          activeIndex={activeIndexRef.current}
          items={items}
          kind="@"
          loading={false}
          onHover={setActiveIndex}
          onPick={pick}
          placement={anchor.top > window.innerHeight / 2 ? 'top' : 'bottom'}
        />
      </div>
    ) : null

  return { extension, popover }
}
