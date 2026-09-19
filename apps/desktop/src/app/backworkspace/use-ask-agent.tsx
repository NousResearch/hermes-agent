import { type Extension, Prec } from '@codemirror/state'
import type { EditorView } from '@codemirror/view'
import { keymap } from '@codemirror/view'
import { useStore } from '@nanostores/react'
import { useCallback, useMemo, useRef, useState } from 'react'

import { COMPOSER_AREAS } from '@/app/chat/composer/contrib'
import { useContributions } from '@/contrib/react/use-contributions'
import { type Translations, useI18n } from '@/i18n'
import { $activeConnectionId } from '@/store/connections'
import { $activeGatewayProfile } from '@/store/profile'

import { askBackworkspace } from './ask'
import { firstMentionIn, paragraphAt } from './mention'
import { $backworkspacePage, type BackworkspaceRoute, editBackworkspacePage } from './page'
import { replyBlock, replyInsertion } from './reply'
import { resolveMentionRoute } from './use-mention-popup'

interface AskState {
  handle: string
  status: 'asking' | 'failed' | 'nobody' | 'unreachable'
}

// Who each page last asked, so a conversation continues without retyping the
// mention. Keyed by owner, because a profile's page has its own thread.
const lastAsked = new Map<string, string>()

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
  const [state, setState] = useState<AskState | null>(null)
  // Read inside the key handler, which runs outside React's render.
  const asking = useRef(false)
  const context = useRef({ connectionId, pagePath, profile, sources })

  context.current = { connectionId, pagePath, profile, sources }

  // The reply belongs to the page, not to the editor: turning the window back
  // destroys the view while an agent is still thinking, and the text must land
  // in the document either way.
  const writeReply = useCallback((view: EditorView, question: string, block: string) => {
    if (view.dom.isConnected) {
      const insertion = replyInsertion(view.state.doc.toString(), question, block)

      // No selection change — the user may be typing somewhere else by now.
      view.dispatch({ changes: { from: insertion.from, insert: insertion.insert } })

      return
    }

    const page = $backworkspacePage.get()

    if (page?.status === 'ready') {
      const insertion = replyInsertion(page.content, question, block)

      editBackworkspacePage(
        page.content.slice(0, insertion.from) + insertion.insert + page.content.slice(insertion.from)
      )
    }
  }, [])

  const ask = useCallback(
    (view: EditorView): boolean => {
      // One question at a time: two turns on the same session would both be
      // waiting on its next completion, and the first reply would answer both.
      if (asking.current) {
        return true
      }

      const current = context.current
      const paragraph = paragraphAt(view.state.doc.toString(), view.state.selection.main.head)

      if (!paragraph.text) {
        return false
      }

      const owner = `${current.connectionId ?? ''}:${current.profile}`
      // A paragraph with no mention continues the last conversation on this
      // page: the session is already open, and the status line names who it
      // went to, so the caret never has to go back for an `@`.
      const handle = firstMentionIn(paragraph.text) ?? lastAsked.get(owner) ?? null

      if (!handle) {
        setState({ handle: '', status: 'nobody' })

        return true
      }

      const self: BackworkspaceRoute = { connectionId: current.connectionId, profile: current.profile }
      const route = resolveMentionRoute(current.sources, handle, self)

      if (!route) {
        setState({ handle, status: 'unreachable' })

        return true
      }

      const question = paragraph.text

      asking.current = true
      lastAsked.set(owner, handle)
      setState({ handle, status: 'asking' })
      void askBackworkspace({ handle, route }, question, current.pagePath)
        .then(text => {
          writeReply(view, question, replyBlock(handle, text, new Date()))
          setState(null)
        })
        .catch(() => setState({ handle, status: 'failed' }))
        .finally(() => {
          asking.current = false
        })

      return true
    },
    [writeReply]
  )

  // Above CodeMirror's own bindings: its default keymap owns Mod-Enter
  // (insert blank line), so without this the question is never sent.
  const extension = useMemo(() => Prec.highest(keymap.of([{ key: 'Mod-Enter', run: ask }])), [ask])

  const notice = state ? noticeForAsk(state, t.backworkspace) : null

  return { extension, notice }
}
