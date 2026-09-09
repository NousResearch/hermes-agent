import { useEffect, useRef, useState } from 'react'

import { WisdomFileEditor } from '@/app/skills/wisdom-file-editor'
import { wisdomManifestValidationError } from '@/app/skills/wisdom-manifest'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { WisdomReviewTables } from '@/components/wisdom-checks'
import {
  decideWisdomDraft,
  type ProfileScope,
  type WisdomPublicationReview as Review,
  reviewWisdomPublication,
  reviseWisdomDraft,
  saveWisdomPreparedDraft,
  submitWisdomPublication,
  type WisdomPublicationResult
} from '@/hermes'
import { useI18n } from '@/i18n'

export function WisdomPublicationReview({
  draftId,
  profile,
  onClose,
  onSubmitted,
  consent,
  allowDecline = false
}: {
  draftId: string
  profile?: ProfileScope
  onClose: () => void
  onSubmitted?: (result: WisdomPublicationResult) => void
  consent?: { interaction_id: string; session_id: string }
  allowDecline?: boolean
}) {
  const { t } = useI18n()
  const copy = t.skills.collective
  const [review, setReview] = useState<Review | null>(null)
  const [description, setDescription] = useState('')
  const [files, setFiles] = useState<Record<string, string>>({})
  const [busy, setBusy] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<WisdomPublicationResult | null>(null)
  const acting = useRef(false)
  const generation = useRef(0)

  const apply = (next: Review) => {
    setReview(next)
    const state = next.draft.state
    setResult(
      next.portal_url && (state === 'published' || state === 'pending_moderation')
        ? { draft_id: next.draft.id, publication_state: state, portal_url: next.portal_url }
        : null
    )
    setDescription(next.draft.authorDescription || '')
    setFiles(Object.fromEntries(next.files.map(file => [file.path, file.content_utf8])))
  }

  useEffect(() => {
    const current = ++generation.current
    setReview(null)
    setResult(null)
    setBusy(true)
    setError(null)
    void reviewWisdomPublication(draftId, profile)
      .then(next => {
        if (generation.current === current) {
          apply(next)
        }
      })
      .catch(reason => {
        if (generation.current === current) {
          setError(String(reason instanceof Error ? reason.message : reason))
        }
      })
      .finally(() => {
        if (generation.current === current) {
          setBusy(false)
        }
      })

    return () => {
      generation.current++
    }
  }, [draftId, profile])

  const reload = async (id: string) => {
    const current = generation.current
    const next = await reviewWisdomPublication(id, profile)

    if (generation.current === current) {
      apply(next)
    }
  }

  const local = review?.draft.state === 'prepared'
  const editable = !!review && ['prepared', 'ready', 'changes_requested'].includes(review.draft.state)

  const dirty =
    !!review &&
    (description !== (review.draft.authorDescription || '') ||
      review.files.some(file => files[file.path] !== file.content_utf8))

  const manifestError = review ? wisdomManifestValidationError(files['skill.manifest.json'] || '') : null
  const canSubmit = !!review && ['prepared', 'ready', 'owner_approved', 'publishing'].includes(review.draft.state)
  const blocked = review?.draft.security_check?.status === 'blocked'

  const pendingChecks = [review?.draft.security_check, review?.draft.professionalism_check].some(
    check => check && ['pending', 'running', 'retry'].includes(check.status)
  )

  const run = async (operation: () => Promise<void>) => {
    if (acting.current) {
      return
    }

    acting.current = true
    const current = generation.current
    setBusy(true)
    setError(null)

    try {
      await operation()
    } catch (reason) {
      if (generation.current === current) {
        setError(String(reason instanceof Error ? reason.message : reason))
      }
    } finally {
      acting.current = false

      if (generation.current === current) {
        setBusy(false)
      }
    }
  }

  const save = () =>
    run(async () => {
      if (!review || !dirty || manifestError || !description.trim()) {
        return
      }

      const edited = review.files.map(file => ({ path: file.path, content_utf8: files[file.path] }))
      const current = generation.current

      const saved = local
        ? await saveWisdomPreparedDraft(review.draft.id, description, edited, profile)
        : await reviseWisdomDraft(review.draft.id, description, edited, review.hashes, profile)

      const id = 'local_draft_id' in saved ? saved.local_draft_id : saved.draft.id
      const next = await reviewWisdomPublication(id, profile)

      if (generation.current === current) {
        apply(next)
      }
    })

  const submit = () =>
    run(async () => {
      if (!review || dirty || !canSubmit || blocked || pendingChecks || manifestError) {
        return
      }

      const current = generation.current
      const submitted = await submitWisdomPublication(review, profile, consent)

      if (generation.current === current) {
        setResult(submitted)
        onSubmitted?.(submitted)
      }
    })

  return (
    <section aria-label={copy.ownerReviewExact} className="min-w-0 space-y-4 p-4">
      <header className="flex items-start justify-between gap-3">
        <h3 className="break-words text-sm font-semibold">{review?.draft.slug || copy.prepareTitle}</h3>
        <Button disabled={busy} onClick={onClose} size="sm" variant="outline">
          {copy.close}
        </Button>
      </header>
      {error && (
        <p className="text-sm text-destructive" role="alert">
          {error}
        </p>
      )}
      {result ? (
        <div role="status">
          <p className="text-sm font-medium">{copy.draftState(result.publication_state)}</p>
          <a
            className="mt-2 inline-block text-sm hover:underline"
            href={result.portal_url}
            rel="noreferrer"
            target="_blank"
          >
            {copy.viewInPortal}
          </a>
          {allowDecline && result.publication_state === 'pending_moderation' && (
            <Button disabled={busy} onClick={() => void run(async () => {
              await decideWisdomDraft(result.draft_id, 'decline', profile)
              onClose()
            })} size="sm" variant="outline">{copy.decline}</Button>
          )}
        </div>
      ) : review ? (
        <>
          <p className="text-xs text-muted-foreground">
            {review.publication_mode === 'open' ? copy.publishLocalNotice : copy.submitLocalNotice}
          </p>
          <WisdomReviewTables
            professionalism={review.draft.professionalism_check}
            security={review.draft.security_check}
          />
          <label className="block text-xs">
            {copy.ownerDescription}
            <Textarea
              className="mt-2 min-h-24"
              disabled={busy || !editable}
              maxLength={4096}
              onChange={event => setDescription(event.target.value)}
              value={description}
            />
          </label>
          {review.files.map(file => (
            <WisdomFileEditor
              disabled={busy || !editable}
              file={file}
              key={file.path}
              onChange={value => setFiles(current => ({ ...current, [file.path]: value }))}
              reviewSource={local ? 'local' : 'server'}
              value={files[file.path] ?? file.content_utf8}
            />
          ))}
          {dirty && (
            <p className="text-xs text-amber-600" role="status">
              {copy.unsavedChanges}
            </p>
          )}
          {manifestError && (
            <p className="text-xs text-destructive" role="alert">
              {manifestError}
            </p>
          )}
          <div className="flex flex-wrap justify-end gap-2">
            {allowDecline &&
              !local &&
              ['ready', 'changes_requested', 'pending_moderation'].includes(review.draft.state) && (
                <Button
                  disabled={busy}
                  onClick={() =>
                    void run(async () => {
                      await decideWisdomDraft(review.draft.id, 'decline', profile)
                      onClose()
                    })
                  }
                  size="sm"
                  variant="outline"
                >
                  {copy.decline}
                </Button>
              )}
            {dirty && (
              <Button disabled={busy} onClick={() => apply(review)} size="sm" variant="outline">
                {copy.resetChanges}
              </Button>
            )}
            <Button disabled={busy} onClick={() => void run(() => reload(review.draft.id))} size="sm" variant="outline">
              {copy.reloadReview}
            </Button>
            {dirty && (
              <Button disabled={busy || !!manifestError || !description.trim()} onClick={() => void save()} size="sm">
                {copy.saveAndRescan}
              </Button>
            )}
            <Button
              disabled={
                busy || dirty || !canSubmit || blocked || pendingChecks || !!manifestError || !description.trim()
              }
              onClick={() => void submit()}
              size="sm"
            >
              {busy
                ? copy.submitting
                : review.publication_mode === 'open'
                  ? copy.publishToTeam
                  : copy.submitForApproval}
            </Button>
          </div>
        </>
      ) : (
        !busy && (
          <Button onClick={() => void run(() => reload(draftId))} size="sm">
            {copy.reloadReview}
          </Button>
        )
      )}
      {busy && (
        <p className="text-xs text-muted-foreground" role="status">
          {copy.loading}
        </p>
      )}
    </section>
  )
}
