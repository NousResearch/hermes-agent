import { useEffect, useState } from 'react'

import { $terminalTakeover, setTerminalTakeover } from '@/app/right-sidebar/store'
import { type PreviewUndo, snapshotChatLayout } from '@/components/onboarding/first-run/assembly'
import { $fileBrowserOpen, setFileBrowserOpen } from '@/store/layout'
import { applyDesktopLayoutPreset, openLayoutPickPanes } from '@/store/pane-focus'
import { $accentOverride, setAccentOverride } from '@/themes/accent-override'
import { resolveAccentPick } from '@/themes/accents'
import { ownAccent } from '@/themes/persist-accent'

/** A staged pick; `value` null is the theme's own accent. */
export interface StagedLook {
  value: null | string
}

interface LiveLook {
  apply: (value: null | string, dark: boolean) => void
  snapshot: () => PreviewUndo
}

// The accent previews through the unsaved override so nothing is persisted before Continue; the theme's own
// accent previews as the base theme's colour because an empty override would keep the current (maybe saved) one.
const ACCENT: LiveLook = {
  apply: (value, dark) => setAccentOverride(resolveAccentPick(value ?? 'theme', dark)?.hex ?? ownAccent()),
  snapshot: () => {
    const override = $accentOverride.get()

    return { keep: send => send(), revert: () => setAccentOverride(override) }
  }
}

// A layout previews as an arrangement in the current mode; Continue applies it for real (switching
// Simple/Advanced when the preset needs it), so a cancelled preview never touches either mode's saved layout.
const LAYOUT: LiveLook = {
  apply: value => {
    if (value && applyDesktopLayoutPreset(value)) {
      openLayoutPickPanes(value)
    }
  },
  // The terminal and file tree keep their open state outside the layout tree.
  snapshot: () => {
    const chat = snapshotChatLayout()
    const terminal = $terminalTakeover.get()
    const files = $fileBrowserOpen.get()

    return {
      keep: chat.keep,
      revert: () => {
        chat.revert()
        setTerminalTakeover(terminal)
        setFileBrowserOpen(files)
      }
    }
  }
}

export interface LookPreview {
  /** Answer with the staged pick; leaving keeps it when `send` returns true. */
  commit: (send: (value: null | string) => boolean) => void
  stage: (value: null | string) => void
  staged: null | StagedLook
}

/** A look's open preview: the staged pick and the value from before the first preview. */
interface Held {
  mounts: number
  settled: boolean
  staged: null | StagedLook
  undo: null | PreviewUndo
}

// A layout preview re-parents the chat, which remounts the card that staged it. The preview outlives the
// card instance, so the remounted card shows the pick, and only a card that stays gone puts the old look back.
const held = new Map<'accent' | 'layout', Held>()

/** Click previews a look live; a card that unmounts without Continue (cancel, timeout, Skip setup,
 *  words in the composer) puts back the value from before the preview. */
export function useLookPreview(kind: 'accent' | 'layout', dark: boolean): LookPreview {
  const live = kind === 'accent' ? ACCENT : LAYOUT
  const [staged, setStaged] = useState<null | StagedLook>(() => held.get(kind)?.staged ?? null)

  useEffect(() => {
    const entry = held.get(kind)

    if (entry) {
      entry.mounts += 1
    }

    return () => {
      const left = held.get(kind)

      if (!left) {
        return
      }

      left.mounts -= 1
      // A remount mounts the new card in the same commit; wait past it before calling the card gone.
      setTimeout(() => {
        if (held.get(kind) === left && left.mounts === 0) {
          held.delete(kind)

          if (!left.settled) {
            left.undo?.revert()
          }
        }
      })
    }
  }, [kind])

  return {
    // Settled before sending: the answer can unmount the card before `send` returns.
    commit: send => {
      const entry = held.get(kind)

      if (!staged || !entry) {
        return
      }

      entry.settled = true
      entry.settled = entry.undo ? entry.undo.keep(() => send(staged.value)) : send(staged.value)

      // The sent value is the new truth. A failed send keeps the card, so its preview goes back on screen.
      if (entry.settled) {
        held.delete(kind)
      } else if (entry.undo) {
        live.apply(staged.value, dark)
      }
    },
    stage: value => {
      const entry = held.get(kind) ?? { mounts: 1, settled: false, staged: null, undo: null }

      held.set(kind, entry)
      entry.undo ??= live.snapshot()
      entry.staged = { value }
      live.apply(value, dark)
      setStaged(entry.staged)
    },
    staged
  }
}
