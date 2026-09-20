import { type Extension, Prec } from '@codemirror/state'
import type { EditorView } from '@codemirror/view'
import { keymap } from '@codemirror/view'
import { useStore } from '@nanostores/react'
import { atom } from 'nanostores'
import { useCallback, useMemo, useRef } from 'react'

import { COMPOSER_AREAS } from '@/app/chat/composer/contrib'
import { useContributions } from '@/contrib/react/use-contributions'
import { type Translations, useI18n } from '@/i18n'
import { $activeConnectionId } from '@/store/connections'
import { $activeGatewayProfile } from '@/store/profile'

import { askBackworkspace } from './ask'
import { writeToPage } from './live-editor'
import { firstMentionIn, paragraphAt } from './mention'
import { backworkspaceOwnerKey, type BackworkspaceRoute } from './page'
import { replyBlock, replyInsertion } from './reply'
import { resolveMentionRoute } from './use-mention-popup'

interface AskState {
  handle: string
  status: 'asking' | 'failed' | 'nobody' | 'unreachable'
}

// Who each page last asked, so a conversation continues without retyping the
// mention. Keyed by owner, because a profile's page has its own thread.
const lastAsked = new Map<string, string>()

// What each page's question is doing, by owner. Out here rather than in the
// hook: the hook goes every time the window is turned, and a question does not.
// Turned away and back, the page still has to say who is answering, and the
// question out still has to be the only one.
const $asks = atom<Readonly<Record<string, AskState>>>({})

function setAsk(owner: string, state: AskState | null) {
  const { [owner]: _previous, ...rest } = $asks.get()

  $asks.set(state ? { ...rest, [owner]: state } : rest)
}

function noticeForAsk(state: AskState, copy: Translations['backworkspace']): string {
  if (state.status === 'nobody') {
    return copy.askNobody
  }

  if (state.status === 'unreachable') {
    return copy.askUnreachable(state.handle)
  }

  return state.status === 'failed' ? copy.askFailed(state.handle) : copy.askPending(state.handle)
}

/**
 * Mod-Enter sends the paragraph the caret is in to the agent it mentions, and
 * writes the reply back under it. Plain Enter stays a newline: this is a page
 * to write on, and a question is the exception.
 */
export function useAskAgent(pagePath: null | string): { extension: Extension; notice: null | string } {
  const { t } = useI18n()
  const sources = useContributions(COMPOSER_AREAS.atCompletions)
  const profile = useStore($activeGatewayProfile)
  const connectionId = useStore($activeConnectionId)
  const state = useStore($asks)[backworkspaceOwnerKey({ connectionId, profile })] ?? null
  // Read inside the key handler, which runs outside React's render.
  const context = useRef({ connectionId, pagePath, profile, sources })

  context.current = { connectionId, pagePath, profile, sources }

  const ask = useCallback((view: EditorView): boolean => {
    // One question at a time: two turns on the same session would both be
    // waiting on its next completion, and the first reply would answer both.
    if (Object.values($asks.get()).some(entry => entry.status === 'asking')) {
      return true
    }

    const current = context.current
    const paragraph = paragraphAt(view.state.doc.toString(), view.state.selection.main.head)

    if (!paragraph.text) {
      return false
    }

    const self: BackworkspaceRoute = { connectionId: current.connectionId, profile: current.profile }
    const owner = backworkspaceOwnerKey(self)
    // A paragraph with no mention continues the last conversation on this
    // page: the session is already open, and the status line names who it
    // went to, so the caret never has to go back for an `@`.
    const handle = firstMentionIn(paragraph.text) ?? lastAsked.get(owner) ?? null

    if (!handle) {
      setAsk(owner, { handle: '', status: 'nobody' })

      return true
    }

    const route = resolveMentionRoute(current.sources, handle, self)

    if (!route) {
      setAsk(owner, { handle, status: 'unreachable' })

      return true
    }

    const question = paragraph.text

    lastAsked.set(owner, handle)
    setAsk(owner, { handle, status: 'asking' })
    void askBackworkspace({ asker: self, handle, route }, question, current.pagePath)
      .then(text => {
        const block = replyBlock(handle, text, new Date())

        // The reply belongs to the page, not to the editor the question left
        // from. No caret either — the user may be typing somewhere else by now.
        writeToPage(owner, doc => replyInsertion(doc, question, block))
        setAsk(owner, null)
      })
      .catch(() => setAsk(owner, { handle, status: 'failed' }))

    return true
  }, [])

  // Above CodeMirror's own bindings: its default keymap owns Mod-Enter
  // (insert blank line), so without this the question is never sent.
  const extension = useMemo(() => Prec.highest(keymap.of([{ key: 'Mod-Enter', run: ask }])), [ask])

  const notice = state ? noticeForAsk(state, t.backworkspace) : null

  return { extension, notice }
}
