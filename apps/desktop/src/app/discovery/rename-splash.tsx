import { useStore } from '@nanostores/react'
import { useCallback } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import { FileText } from '@/lib/icons'
import { OLD_NAME_ALIASES, type RenamedDestination } from '@/lib/old-name-aliases'
import { $cronJobs } from '@/store/cron'
import { dismissRenameSplash, noteRenameSplashRow, type RenameSplashRow } from '@/store/rename-splash'
import { $sessionProfilesTruncated, $sessions } from '@/store/session'

import { requestComposerFocus } from '../chat/composer/focus'
import { useSimplifiedAgents } from '../chat/sidebar/use-simplified-agents'
import { useItemIndex } from '../library/use-item-index'

import { useDiscoveryCopy } from './discovery-copy'
import { useRenamedNames } from './renamed-names'
import { type SplashRow, SplashSurface } from './splash-surface'

type Count = null | { value: number; more: boolean }

/** What each renamed place holds, as its own page counts it. Chats are the loaded conversations,
 *  hidden and archived ones aside (more may sit past a profile's page: then "25+"); the Library is
 *  the item index its page shows, known once a check has finished: while one runs, the index marks
 *  the list checked before it has read the chats' items, so the number so far (0 on a first check)
 *  is not the count. Null hides a count rather than guessing one. */
function useRenamedCounts(): Record<RenamedDestination, Count> {
  const sessions = useStore($sessions)
  const truncated = useStore($sessionProfilesTruncated)
  const routines = useStore($cronJobs)
  const { agents } = useSimplifiedAgents(sessions)
  const library = useItemIndex()
  const chats = sessions.filter(row => !row.hidden && !row.archived).length

  return {
    chats: { value: chats, more: Object.values(truncated).some(Boolean) },
    agents: { value: agents.length, more: false },
    library: library.checkedAt === null || library.refreshing ? null : { value: library.items.length, more: false },
    routines: { value: routines.length, more: false },
    machines: { value: new Set(agents.map(agent => agent.connectionId)).size, more: false }
  }
}

/** After the splash's own way out, focus goes where the person was headed: back to the bell that
 *  reopened it, else into the composer to get to work. A nav click already put focus where it went. */
function leave(how: 'button' | 'escape', reopened: boolean) {
  dismissRenameSplash(how)
  // Next frame: the chat under the splash is inert until the splash has unmounted.
  requestAnimationFrame(() => {
    if (reopened) {
      document.querySelector<HTMLElement>('[data-nav-spot="notifications"]')?.focus({ preventScroll: true })
    } else {
      requestComposerFocus('active')
    }
  })
}

interface RenameSplashProps {
  /** Opened from the tray: it takes focus at once. */
  reopened: boolean
}

/** Each renamed place's glyph: the rail's own (Library the file the welcome draws); Machines the
 *  server Figma draws. */
function RenamedGlyph({ row }: { row: RenamedDestination }) {
  if (row === 'library') {
    return <FileText aria-hidden className="discovery-welcome-glyph" />
  }

  const name = { chats: 'comment-discussion', agents: 'robot', routines: 'clock', machines: 'server' }[row]

  return <Codicon aria-hidden className="discovery-welcome-glyph" name={name} />
}

/** "Hermes got a new look" (Figma E5 18677:29814): each classic noun beside the name the redesign
 *  gives it, what it is for, and the user's own count of it, in the welcome's card. */
export function RenameSplash({ reopened }: RenameSplashProps) {
  const { locale } = useI18n()
  const copy = useDiscoveryCopy()
  const counts = useRenamedCounts()
  const current = useRenamedNames()
  const number = new Intl.NumberFormat(locale)
  const onEscape = useCallback(() => leave('escape', reopened), [reopened])

  // The rows are the alias table's renames, in its order: one list of what was renamed.
  const rows: SplashRow<RenameSplashRow>[] = OLD_NAME_ALIASES.map(({ code: classic, destination: row }) => {
    const count = counts[row]

    const shown =
      count && (count.more ? copy.renameSplash.atLeast(number.format(count.value)) : number.format(count.value))

    const renamed = copy.renameSplash.renamed(copy.classicNames[classic], current[row])

    return {
      id: row,
      glyph: <RenamedGlyph row={row} />,
      name: shown ? `${renamed} (${shown})` : renamed,
      label: (
        <>
          {copy.classicNames[classic]}
          <Codicon className="discovery-rename-arrow" name="arrow-right" />
          {current[row]}
        </>
      ),
      line: copy.renameSplash.lines[row],
      meta: shown || undefined,
      spot: row
    }
  })

  return (
    <SplashSurface
      actions={
        <Button className="discovery-welcome-action" onClick={() => leave('button', reopened)}>
          {copy.renameSplash.done}
        </Button>
      }
      eyebrow={copy.renameSplash.eyebrow}
      footnote={copy.renameSplash.footnote}
      kind="rename"
      listLabel={copy.renameSplash.listLabel}
      onEscape={onEscape}
      onRowActivated={noteRenameSplashRow}
      rows={rows}
      subtitle={copy.renameSplash.subtitle}
      takeFocus={reopened}
      title={copy.renameSplash.title}
    />
  )
}
