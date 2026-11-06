import {
  Button,
  Checkbox,
  cn,
  Codicon,
  ComposerDrawer,
  composerSurfaceGlass,
  DisclosureCaret,
  ErrorState,
  Loader,
  type PluginTranslate,
  Textarea,
  usePluginI18n,
  useThisComposer,
  useValue
} from '@hermes/plugin-sdk'
import {
  Component,
  type KeyboardEvent,
  type MutableRefObject,
  type ReactNode,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState
} from 'react'

import { registerContextReader } from './chat-context'
import { PLUGIN_ID } from './i18n'
import { INTENT_KINDS } from './intent'
import { canApprove, checkPreservation, type PreservationIssue } from './preservation'
import type { ReviewState, ReviewStore } from './review-store'
import type { RefinementAnalysis } from './types'

/** Matches the composer status drawer's fold (200 ms + 70 ms delay). */
export const DRAWER_EXIT_MS = 300

const promptText =
  'whitespace-pre-wrap break-words text-[length:var(--conversation-text-font-size)] leading-relaxed text-(--ui-text-primary)'

const meta = 'text-xs text-(--ui-text-tertiary)'

/** A crash inside the drawer must cancel the review (draft restored), never
 *  leave a send hanging behind a surface that can't be used. */
class DrawerErrorBoundary extends Component<{ children: ReactNode; onError: () => void }, { failed: boolean }> {
  state = { failed: false }

  static getDerivedStateFromError() {
    return { failed: true }
  }

  componentDidCatch() {
    this.props.onError()
  }

  render() {
    return this.state.failed ? null : this.props.children
  }
}

function Disclosure({ children, label }: { children: ReactNode; label: string }) {
  const [open, setOpen] = useState(false)
  const id = useId()

  return (
    <div className="grid gap-1">
      <button
        aria-controls={id}
        aria-expanded={open}
        className={cn(meta, 'flex w-fit cursor-pointer items-center gap-1 hover:text-(--ui-text-secondary)')}
        onClick={() => setOpen(value => !value)}
        type="button"
      >
        <DisclosureCaret open={open} />
        {label}
      </button>
      {open && <div id={id}>{children}</div>}
    </div>
  )
}

const ANALYSIS_KEYS: readonly (keyof RefinementAnalysis)[] = ['professional', 'rawMaterial', 'methods', 'verification']

function AnalysisRows({ analysis }: { analysis: RefinementAnalysis }) {
  const t = usePluginI18n(PLUGIN_ID)

  return (
    <dl className="grid grid-cols-[minmax(0,9rem)_minmax(0,1fr)] gap-x-3 gap-y-1 pl-4 text-xs">
      {ANALYSIS_KEYS.filter(key => analysis[key]).map(key => (
        <div className="contents" key={key}>
          <dt className="text-(--ui-text-tertiary)">{t(`drawer.analysis.${key}`)}</dt>
          <dd className="text-(--ui-text-secondary)">{analysis[key]}</dd>
        </div>
      ))}
    </dl>
  )
}

function issueLabel(t: PluginTranslate, issue: PreservationIssue): string {
  if (issue.kind === 'negation' || issue.kind === 'requirement') {
    return t(`drawer.kinds.${issue.kind}`, Number(issue.value))
  }

  return INTENT_KINDS.has(issue.kind)
    ? `${t(`drawer.kinds.${issue.kind}`)}: “${issue.value}”`
    : `${t(`drawer.kinds.${issue.kind}`)}: ${issue.value}`
}

function IssueList({ hint, issues, title }: { hint: string; issues: PreservationIssue[]; title: string }) {
  const t = usePluginI18n(PLUGIN_ID)

  if (!issues.length) {
    return null
  }

  return (
    <div className="grid gap-1 text-xs" role="status">
      <p className="font-medium text-(--ui-text-primary)">{title}</p>
      <ul className="grid list-disc gap-0.5 pl-4 text-(--ui-text-secondary)">
        {issues.map(issue => (
          <li className="break-all" key={`${issue.kind}:${issue.value}`}>
            {issueLabel(t, issue)}
          </li>
        ))}
      </ul>
      <p className="text-(--ui-text-tertiary)">{hint}</p>
    </div>
  )
}

interface ChecksProps {
  acknowledged: boolean
  issues: PreservationIssue[]
  onAcknowledge: (value: boolean) => void
}

/** What the rewrite lost (blocking), what it added that the prompt did not ask
 *  for, and what may only be rephrased, with one explicit check for the last
 *  two; or one quiet line when nothing was flagged. */
function Checks({ acknowledged, issues, onAcknowledge }: ChecksProps) {
  const t = usePluginI18n(PLUGIN_ID)
  const blocking = issues.filter(issue => issue.severity === 'blocking')
  const soft = issues.filter(issue => issue.severity === 'review')
  const added = soft.filter(issue => INTENT_KINDS.has(issue.kind))
  const rephrased = soft.filter(issue => !INTENT_KINDS.has(issue.kind))

  if (!issues.length) {
    return (
      <p className={cn(meta, 'flex items-center gap-1.5')}>
        <Codicon className="text-emerald-500/80" name="pass" size="0.8rem" />
        {t('drawer.kept')}
      </p>
    )
  }

  return (
    <>
      <IssueList hint={t('drawer.blockingHint')} issues={blocking} title={t('drawer.blockingTitle')} />
      <IssueList hint={t('drawer.intentHint')} issues={added} title={t('drawer.intentTitle')} />
      <IssueList hint={t('drawer.reviewHint')} issues={rephrased} title={t('drawer.reviewTitle')} />
      {soft.length > 0 && blocking.length === 0 && (
        <label className="flex cursor-pointer items-center gap-2 text-xs text-(--ui-text-primary)">
          <Checkbox checked={acknowledged} onCheckedChange={value => onAcknowledge(value === true)} />
          {t('drawer.acknowledge')}
        </label>
      )}
    </>
  )
}

/** While the model works: one slim row. Cancel keeps the draft; Send
 *  original sends it as typed instead of waiting. */
function RefiningRow({ review, store }: { review: ReviewState; store: ReviewStore }) {
  const t = usePluginI18n(PLUGIN_ID)

  return (
    <div className="flex items-center gap-2 py-0.5">
      <Loader className="size-5 text-(--ui-text-tertiary)" label={t('drawer.refining')} type="lemniscate-bloom" />
      <span className={cn(meta, 'min-w-0 flex-1 truncate')}>{t('drawer.refiningSend')}</span>
      <Button onClick={() => store.decide(review.id, { kind: 'original' })} size="sm" type="button" variant="ghost">
        {t('drawer.sendOriginal')}
      </Button>
      <Button onClick={() => store.decide(review.id, { kind: 'cancel' })} size="sm" type="button" variant="ghost">
        {t('drawer.cancel')}
      </Button>
    </div>
  )
}

interface RefinedBodyProps {
  draft: string
  editing: boolean
  onDraftChange: (text: string) => void
  onRetry: () => void
  review: ReviewState
}

function RefinedBody({ draft, editing, onDraftChange, onRetry, review }: RefinedBodyProps) {
  const t = usePluginI18n(PLUGIN_ID)
  const retryable = review.failure !== 'too-long' && review.failure !== 'unsupported'

  if (review.phase === 'failed') {
    return (
      <ErrorState
        description={review.failure ? t(`failure.${review.failure}`) : ''}
        title={<p className="text-sm font-medium text-(--ui-text-primary)">{t('failure.title')}</p>}
      >
        {retryable && (
          <Button onClick={onRetry} size="sm" type="button" variant="secondary">
            {t('drawer.retry')}
          </Button>
        )}
      </ErrorState>
    )
  }

  return editing ? (
    <Textarea
      aria-label={t('drawer.editing')}
      autoFocus
      className={cn(promptText, 'max-h-[min(40vh,22rem)] min-h-28 resize-y')}
      onChange={event => onDraftChange(event.target.value)}
      value={draft}
    />
  ) : (
    <div className="max-h-[min(40vh,22rem)] overflow-y-auto">
      <p className={promptText}>{draft}</p>
    </div>
  )
}

const isApproveChord = (event: KeyboardEvent) =>
  event.key === 'Enter' && (event.metaKey || event.ctrlKey) && !event.shiftKey && !event.altKey

/** The full review: shown for Approval-First mode, and for an automatic send that
 *  stopped because refinement failed or flagged something. */
interface ReviewPanelProps {
  review: ReviewState
  store: ReviewStore
  /** Filled with this panel's Approve, so the drawer's own key handler can run
   *  it wherever focus sits inside the drawer (focus lands on the drawer itself
   *  when it opens, not on anything inside the panel). */
  approveRef: MutableRefObject<null | (() => void)>
}

function ReviewPanel({ approveRef, review, store }: ReviewPanelProps) {
  const t = usePluginI18n(PLUGIN_ID)
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(review.refined ?? '')
  const [acknowledged, setAcknowledged] = useState(false)

  // A new result (first answer or a retry) resets the candidate and the ack.
  useEffect(() => {
    setDraft(review.refined ?? '')
    setEditing(false)
    setAcknowledged(false)
  }, [review.refined])

  const ready = review.phase === 'ready'
  const candidate = ready ? draft : ''

  const report = useMemo(
    () => checkPreservation(review.original, candidate, review.context),
    [review.original, candidate, review.context]
  )

  const approvable = ready && Boolean(candidate.trim()) && canApprove(report, acknowledged)

  const approve = () => {
    if (approvable) {
      store.decide(review.id, { kind: 'approve', text: candidate, acknowledged })
    }
  }

  // Render-time mirror (the composer's dispatchSubmitRef pattern): the drawer's
  // key handler always runs this render's Approve, with this render's text.
  approveRef.current = approve

  return (
    <div className="grid gap-2">
      <div className="flex items-center gap-2">
        <Codicon className="text-(--ui-text-tertiary)" name="sparkle" size="0.875rem" />
        <span className="text-sm font-medium text-(--ui-text-primary)">{t('drawer.titleSend')}</span>
        {ready && review.chosenDepth && (
          <span className={cn(meta, 'ml-auto')}>
            {t('drawer.chosenDepth', t(`settings.depths.${review.chosenDepth}`))}
          </span>
        )}
      </div>

      {review.attention && (
        <p className="text-xs text-(--ui-text-secondary)">
          {t(review.attention === 'issues' ? 'drawer.attentionIssues' : 'drawer.attentionFailed')}
        </p>
      )}

      <Disclosure label={t('drawer.yourPrompt')}>
        <p className={cn(promptText, 'pl-4 text-(--ui-text-secondary)')}>{review.original}</p>
      </Disclosure>

      <RefinedBody
        draft={draft}
        editing={editing}
        onDraftChange={setDraft}
        onRetry={() => store.retry(review.id)}
        review={review}
      />

      {ready && review.unchanged && !editing && <p className={meta}>{t('drawer.unchanged')}</p>}

      {ready && review.analysis && (
        <Disclosure label={t('drawer.added')}>
          <AnalysisRows analysis={review.analysis} />
        </Disclosure>
      )}

      {ready && <Checks acknowledged={acknowledged} issues={report.issues} onAcknowledge={setAcknowledged} />}

      {review.attachmentLabels.length > 0 && (
        <p className={meta}>{t('drawer.attachments', review.attachmentLabels.length)}</p>
      )}

      <div className="flex flex-wrap items-center gap-2 pt-1">
        <Button onClick={() => store.decide(review.id, { kind: 'cancel' })} size="sm" type="button" variant="ghost">
          {t('drawer.cancel')}
        </Button>
        <Button
          onClick={() => store.decide(review.id, { kind: 'original' })}
          size="sm"
          type="button"
          variant="secondary"
        >
          {t('drawer.sendOriginal')}
        </Button>
        <span className={cn(meta, 'ml-auto hidden sm:inline')}>{t('drawer.shortcuts')}</span>
        <Button
          disabled={!ready}
          onClick={() => setEditing(value => !value)}
          size="sm"
          type="button"
          variant="secondary"
        >
          {t(editing ? 'drawer.doneEditing' : 'drawer.edit')}
        </Button>
        <Button
          aria-keyshortcuts="Meta+Enter Control+Enter"
          disabled={!approvable}
          onClick={approve}
          size="sm"
          type="button"
        >
          {t('drawer.approveSend')}
        </Button>
      </div>
    </div>
  )
}

/**
 * The refinement drawer, contributed to every composer's top slot. It renders
 * only in the composer the review belongs to (`review.target`), slides up out
 * of the typing box with the composer's own drawer motion, and folds away the
 * same way. Esc inside it cancels (and nothing else: the composer's
 * Esc-to-stop ignores a handled key).
 */
export function createReviewDrawer(store: ReviewStore) {
  return function ReviewDrawer() {
    const t = usePluginI18n(PLUGIN_ID)
    const { activate, recentMessages, target } = useThisComposer()
    const review = useValue(store.$review)
    const mine = review && review.target === target ? review : null
    const [shown, setShown] = useState<null | ReviewState>(null)
    const [collapsed, setCollapsed] = useState(true)
    const regionRef = useRef<HTMLDivElement>(null)
    const approveRef = useRef<null | (() => void)>(null)
    const id = useId()

    useEffect(() => store.registerHost(), [])

    // This composer's chat, for refinements that opted into chat context.
    useEffect(() => registerContextReader(target, recentMessages), [target, recentMessages])

    // This composer going away mid-review must not strand the send.
    useEffect(
      () => () => {
        if (store.$review.get()?.target === target) {
          store.cancelActive('no-host')
        }
      },
      [target]
    )

    useEffect(() => {
      if (mine) {
        setShown(mine)

        return undefined
      }

      // Closing with the caret inside: give it back to this composer.
      if (regionRef.current?.contains(document.activeElement)) {
        activate()
      }

      setCollapsed(true)
      const timer = setTimeout(() => setShown(null), DRAWER_EXIT_MS)

      return () => clearTimeout(timer)
    }, [mine, activate])

    // A new review: paint folded, then open on the next frame so it slides up;
    // tell the store it is on screen; take focus (the user just asked for it).
    const reviewId = mine?.id

    useEffect(() => {
      if (reviewId === undefined) {
        return undefined
      }

      store.acknowledge(reviewId)

      const frame = requestAnimationFrame(() => {
        setCollapsed(false)
        regionRef.current?.focus({ preventScroll: true })
      })

      return () => cancelAnimationFrame(frame)
    }, [reviewId])

    if (!shown) {
      return null
    }

    const current = mine ?? shown

    // Handled at the drawer itself, so both keys work wherever focus is inside
    // it — including the drawer, which is where focus lands when it opens.
    const onKeyDown = (event: KeyboardEvent) => {
      if (!mine) {
        return
      }

      if (event.key === 'Escape') {
        event.preventDefault()
        event.stopPropagation()
        store.decide(mine.id, { kind: 'cancel' })
      } else if (isApproveChord(event) && mine.phase === 'ready') {
        event.preventDefault()
        event.stopPropagation()
        approveRef.current?.()
      }
    }

    return (
      <ComposerDrawer collapsed={collapsed || !mine} id={id}>
        <div
          aria-label={t('drawer.label')}
          // The composer's glass: what scrolls behind the drawer never shows through.
          className={cn(composerSurfaceGlass, 'mb-1 border-b border-(--ui-stroke-tertiary) pt-0.5 pb-2.5 outline-none')}
          data-slot="prompt-refinement-drawer"
          onKeyDown={onKeyDown}
          ref={regionRef}
          role="region"
          tabIndex={-1}
        >
          <DrawerErrorBoundary key={current.id} onError={() => store.cancelActive('no-host')}>
            {current.phase === 'refining' ? (
              <RefiningRow review={current} store={store} />
            ) : (
              <ReviewPanel approveRef={approveRef} review={current} store={store} />
            )}
          </DrawerErrorBoundary>
        </div>
      </ComposerDrawer>
    )
  }
}
