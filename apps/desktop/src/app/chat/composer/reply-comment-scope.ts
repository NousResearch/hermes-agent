import { atom } from 'nanostores'

import { MAX_REPLY_COMMENTS, type ReplyComment } from './reply-comments'

/**
 * Per-composer reply-comment state — one live chip set PER MOUNTED COMPOSER,
 * mirroring the attachment-scope rule: the main chat wraps the module-level
 * atom below, each session tile creates its own, so two composers on screen
 * never share chips.
 *
 * Comments are pre-send UI state only. At every consumption boundary (send,
 * steer, queue, session swap) they are frozen into ordinary Markdown text and
 * the scope is drained — the backend contract never changes.
 */
export interface ReplyCommentScope {
  $comments: ReturnType<typeof atom<ReplyComment[]>>
  /** Append a comment. False (no-op) when the batch cap is reached. */
  add(comment: ReplyComment): boolean
  clear(): void
  /** Snapshot of the live list (non-destructive — for branching, not send). */
  list(): ReplyComment[]
  remove(id: string): void
  /** Snapshot + drain. Exactly one consumption path may call this per send. */
  take(): ReplyComment[]
  update(id: string, patch: Pick<ReplyComment, 'note'>): boolean
}

export function createReplyCommentScope($comments = atom<ReplyComment[]>([])): ReplyCommentScope {
  return {
    $comments,
    add(comment) {
      const current = $comments.get()

      if (current.some(item => item.id === comment.id) || current.length >= MAX_REPLY_COMMENTS) {
        return false
      }

      $comments.set([...current, comment])

      return true
    },
    clear() {
      if ($comments.get().length > 0) {
        $comments.set([])
      }
    },
    list() {
      return $comments.get()
    },
    remove(id) {
      const current = $comments.get()

      if (current.some(item => item.id === id)) {
        $comments.set(current.filter(item => item.id !== id))
      }
    },
    take() {
      const current = $comments.get()
      $comments.set([])

      return current
    },
    update(id, patch) {
      const current = $comments.get()
      const index = current.findIndex(item => item.id === id)

      if (index < 0) {
        return false
      }

      const next = [...current]
      next[index] = { ...next[index]!, ...patch }
      $comments.set(next)

      return true
    }
  }
}

/** The main chat composer's live comment set. */
export const mainReplyCommentScope = createReplyCommentScope()
