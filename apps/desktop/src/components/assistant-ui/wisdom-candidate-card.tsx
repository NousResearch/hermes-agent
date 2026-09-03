import { useEffect, useLayoutEffect, useRef, useState } from 'react'

import { WisdomFileEditor } from '@/app/skills/wisdom-file-editor'
import { parseWisdomManifest, wisdomManifestValidationError } from '@/app/skills/wisdom-manifest'
import { TooltipIconButton } from '@/components/assistant-ui/tooltip-icon-button'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import {
  approveWisdomCandidate,
  decideWisdomDraft,
  deferWisdomCandidate,
  getWisdomCandidates,
  getWisdomEvents,
  type ProfileScope,
  reviewWisdomDraft,
  reviseWisdomDraft,
  saveWisdomPreparedDraft,
  suggestWisdomSkill,
  type WisdomCandidateEvent,
  type WisdomDraftReview,
  type WisdomEditedFile,
  type WisdomLocalScan,
  type WisdomPreparedDraft
} from '@/hermes'
import { useI18n } from '@/i18n'
import { Volume2, VolumeX } from '@/lib/icons'
import { notifyError } from '@/store/notifications'

const MAX_INLINE_REVIEW_BYTES = 256 * 1024
const AUTO_REVEAL_PREVIOUS_BOTTOM_GAP_PX = 48

type PreparedState = WisdomPreparedDraft & { localSkillId: string; skill: string }

function fileValues(files: WisdomDraftReview['files']): Record<string, string> {
  return Object.fromEntries(files.map(file => [file.path, file.content_utf8]))
}

function editedFiles(files: WisdomDraftReview['files'], values: Record<string, string>): WisdomEditedFile[] {
  return files.map(file => ({ path: file.path, content_utf8: values[file.path] ?? '' }))
}

function sameFiles(files: WisdomDraftReview['files'], values: Record<string, string>): boolean {
  return files.every(file => values[file.path] === file.content_utf8)
}

function inlineSize(files: WisdomDraftReview['files']): number {
  return files.reduce((total, file) => total + new TextEncoder().encode(file.content_utf8).length, 0)
}

function readableReason(key: string): string {
  return key.replaceAll('_', ' ').replace(/^./, letter => letter.toUpperCase())
}

function manifestName(value: string): string {
  try {
    return parseWisdomManifest(value).name
  } catch {
    return ''
  }
}

function withManifestName(value: string, name: string): string {
  try {
    return `${JSON.stringify({ ...parseWisdomManifest(value), name })}\n`
  } catch {
    return value
  }
}

export function WisdomCandidateCard({ profile, sessionId }: { profile?: ProfileScope; sessionId: string }) {
  const { t } = useI18n()
  const copy = t.skills.collective
  const [event, setEvent] = useState<null | WisdomCandidateEvent>(null)
  const [prepared, setPrepared] = useState<null | PreparedState>(null)
  const [preparedDescription, setPreparedDescription] = useState('')
  const [preparedFiles, setPreparedFiles] = useState<Record<string, string>>({})
  const [review, setReview] = useState<null | WisdomDraftReview>(null)
  const [reviewDescription, setReviewDescription] = useState('')
  const [reviewFiles, setReviewFiles] = useState<Record<string, string>>({})
  const [localScan, setLocalScan] = useState<null | WisdomLocalScan>(null)
  const [editorOpen, setEditorOpen] = useState(false)
  const [detailedEditorOpen, setDetailedEditorOpen] = useState(false)
  const [notificationsMuted, setNotificationsMuted] = useState(false)

  const [busy, setBusy] = useState<null | 'approve' | 'defer' | 'prepare' | 'save-local' | 'save-server' | 'submit'>(
    null
  )

  const [preparationError, setPreparationError] = useState<null | string>(null)
  const [prepareAttempt, setPrepareAttempt] = useState(0)
  const cardRef = useRef<HTMLElement>(null)
  const resolvedEventIdsRef = useRef(new Set<string>())
  const eventId = event?.id
  const eventSkill = event?.payload.skill_name
  const eventContentHash = event?.content_hash

  const openFullReview = () => {
    window.location.hash = '#/skills?tab=collective'
  }

  const fullReviewLink = (
    <button
      className="text-[0.65rem] text-muted-foreground underline decoration-dotted underline-offset-4 hover:text-foreground"
      onClick={openFullReview}
      type="button"
    >
      {copy.openFullReview}
    </button>
  )

  const applyPrepared = (result: WisdomPreparedDraft, localSkillId: string, skill: string) => {
    setPrepared({ ...result, localSkillId, skill })
    setPreparedDescription(result.drafted_description)
    setPreparedFiles(fileValues(result.files))
    setLocalScan(result.local_scan)
  }

  const applyReview = (result: WisdomDraftReview) => {
    setReview(result)
    setReviewDescription(result.draft.authorDescription || '')
    setReviewFiles(fileValues(result.files))
  }

  useEffect(() => {
    let active = true
    let refreshSequence = 0
    resolvedEventIdsRef.current.clear()

    const refresh = async () => {
      const sequence = ++refreshSequence

      try {
        const result = await getWisdomEvents(sessionId, profile)

        if (active && sequence === refreshSequence) {
          setEvent(
            result.events.find(item => !resolvedEventIdsRef.current.has(item.id)) ?? null
          )
        }
      } catch {
        // Candidate promotion is optional transcript UI. Wisdom availability
        // must never make ordinary chat unusable.
        if (active && sequence === refreshSequence) {
          setEvent(null)
        }
      }
    }

    void refresh()
    const timer = window.setInterval(() => void refresh(), 10_000)

    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [profile, sessionId])

  useEffect(() => {
    setEditorOpen(false)
    setDetailedEditorOpen(false)
    setNotificationsMuted(false)
  }, [eventId])

  useEffect(() => {
    if (!eventId || !eventSkill || !eventContentHash) {
      return
    }

    let active = true
    setBusy('prepare')
    setPreparationError(null)

    void (async () => {
      try {
        const candidates = await getWisdomCandidates(profile)

        const candidate = candidates.candidates.find(
          item => item.name === eventSkill && item.content_hash === eventContentHash
        )

        if (!candidate) {
          if (active) {
            setEvent(null)
          }

          return
        }

        const result = await suggestWisdomSkill(eventSkill, profile, undefined, candidate.local_skill_id)

        if (!('network_submission' in result)) {
          throw new Error('Local package preparation returned an unexpected result')
        }

        if (active) {
          applyPrepared(result, candidate.local_skill_id, eventSkill)
        }
      } catch (error) {
        if (active) {
          setPreparationError(error instanceof Error ? error.message : String(error))
        }
      } finally {
        if (active) {
          setBusy(null)
        }
      }
    })()

    return () => {
      active = false
    }
  }, [eventContentHash, eventId, eventSkill, prepareAttempt, profile])

  useLayoutEffect(() => {
    const card = cardRef.current

    if (!eventId || !card) {
      return
    }

    const viewport = card.closest<HTMLElement>('[data-slot="aui_thread-viewport"]')

    if (!viewport || typeof card.scrollIntoView !== 'function') {
      return
    }

    const cardHeight = card.getBoundingClientRect().height || card.offsetHeight
    const previousBottomGap = viewport.scrollHeight - cardHeight - viewport.scrollTop - viewport.clientHeight

    if (previousBottomGap <= AUTO_REVEAL_PREVIOUS_BOTTOM_GAP_PX) {
      card.scrollIntoView({ block: 'nearest' })
    }
  }, [eventId])

  if (!event) {
    return null
  }

  const skill = event.payload.skill_name
  const displayName = event.payload.editorial_name?.trim() || skill
  const displayDescription = event.payload.editorial_description?.trim()

  const qualificationNotice =
    event.notice_variant === 'first' ? copy.qualificationFirst(event.organization_name) : copy.qualificationReturning

  const preparedManifest = preparedFiles['skill.manifest.json'] || ''
  const preparedManifestError = prepared ? wisdomManifestValidationError(preparedManifest) : null
  const preparedSkillName = manifestName(preparedManifest)

  const preparedDirty = Boolean(
    prepared && (preparedDescription !== prepared.drafted_description || !sameFiles(prepared.files, preparedFiles))
  )

  const reviewManifest = reviewFiles['skill.manifest.json'] || ''
  const reviewManifestError = review ? wisdomManifestValidationError(reviewManifest) : null
  const reviewSkillName = manifestName(reviewManifest)

  const reviewDirty = Boolean(
    review && (reviewDescription !== (review.draft.authorDescription || '') || !sameFiles(review.files, reviewFiles))
  )

  const reviewCanEdit = Boolean(review && ['ready', 'changes_requested', 'invalidated'].includes(review.draft.state))

  const saveLocal = async () => {
    if (!prepared || preparedManifestError) {
      return
    }

    setBusy('save-local')

    try {
      const saved = await saveWisdomPreparedDraft(
        prepared.local_draft_id,
        preparedDescription,
        editedFiles(prepared.files, preparedFiles),
        profile
      )

      applyPrepared(saved, prepared.localSkillId, prepared.skill)
    } catch (error) {
      notifyError(error, 'Collective Wisdom local save failed')
    } finally {
      setBusy(null)
    }
  }

  const submit = async () => {
    if (!prepared || preparedDirty || preparedManifestError) {
      return
    }

    setBusy('submit')

    try {
      if (!preparedDescription.trim()) {
        throw new Error('Add a description before submitting this private draft.')
      }

      const specification = parseWisdomManifest(preparedManifest).requirements

      const result = await suggestWisdomSkill(
        prepared.skill,
        profile,
        { description: preparedDescription, systemSpecification: specification },
        prepared.localSkillId
      )

      if (!('draft' in result)) {
        throw new Error('Gateway did not return an owner-private draft')
      }

      setLocalScan(result.local_scan)
      const exact = await reviewWisdomDraft(result.draft.id, false, profile)

      if (inlineSize(exact.files) > MAX_INLINE_REVIEW_BYTES) {
        openFullReview()

        return
      }

      applyReview(exact)
      setPrepared(null)
    } catch (error) {
      notifyError(error, 'Owner-private submission failed')
    } finally {
      setBusy(null)
    }
  }

  const saveServerRevision = async () => {
    if (!review || !reviewCanEdit || reviewManifestError) {
      return
    }

    setBusy('save-server')

    try {
      const revised = await reviseWisdomDraft(
        review.draft.id,
        reviewDescription,
        editedFiles(review.files, reviewFiles),
        review.hashes,
        profile
      )

      setLocalScan(revised.local_scan)
      const exact = await reviewWisdomDraft(revised.draft.id, false, profile)

      if (inlineSize(exact.files) > MAX_INLINE_REVIEW_BYTES) {
        openFullReview()

        return
      }

      applyReview(exact)
    } catch (error) {
      notifyError(error, 'Collective Wisdom save and rescan failed')
    } finally {
      setBusy(null)
    }
  }

  const notNow = async () => {
    setBusy('defer')

    try {
      await deferWisdomCandidate(event.id, profile)
      resolvedEventIdsRef.current.add(event.id)
      setPrepared(null)
      setReview(null)
      setEvent(null)
    } catch (error) {
      notifyError(error, 'Collective Wisdom notification could not be deferred')
    } finally {
      setBusy(null)
    }
  }

  const approve = async () => {
    if ((!review && (!prepared || preparedDirty || preparedManifestError)) || (review && reviewDirty)) {
      return
    }

    setBusy('approve')

    try {
      if (review) {
        const acknowledged = await reviewWisdomDraft(review.draft.id, true, profile)

        if (!acknowledged.receipt) {
          throw new Error('Complete-package review receipt was not created')
        }

        await decideWisdomDraft(review.draft.id, 'approve', profile)
      } else {
        await approveWisdomCandidate(event.id, profile)
      }

      resolvedEventIdsRef.current.add(event.id)
      setPrepared(null)
      setReview(null)
      setEvent(null)
    } catch (error) {
      notifyError(error, 'Collective Wisdom approval failed')
    } finally {
      setBusy(null)
    }
  }

  return (
    <section
      aria-label={copy.proposalTitle}
      className="mb-(--conversation-turn-gap) border border-emerald-600/40 bg-(--ui-chat-surface-background)"
      ref={cardRef}
    >
      <header className="flex items-center justify-between border-b border-(--ui-stroke-tertiary) px-4 py-3">
        <div>
          <div className="text-xs font-medium">{copy.proposalTitle}</div>
          <div className="text-[0.68rem] font-medium text-muted-foreground">{displayName}</div>
        </div>
        <span className="text-[0.62rem] text-muted-foreground">
          {review ? copy.serverEnforced : copy.localSuggestion}
        </span>
      </header>

      <div className="border-b border-(--ui-stroke-tertiary) px-4 py-3">
        <p className="text-xs leading-5">{qualificationNotice}</p>
        {displayDescription && <p className="mt-1 text-xs text-muted-foreground">{displayDescription}</p>}
        <p className="mt-1 text-xs text-muted-foreground">{copy.proposalNotice}</p>
        <p className="mt-2 text-xs font-medium">{copy.sharePrompt}</p>
      </div>

      {!prepared && !review && (
        <div className="p-4">
          {busy === 'prepare' && <p className="mt-3 text-xs">{copy.preparingLocal}</p>}
          {preparationError && (
            <div className="mt-3 text-xs text-destructive" role="alert">
              {preparationError}
            </div>
          )}
          {preparationError && (
            <div className="mt-3 flex items-center justify-between gap-2">
              {fullReviewLink}
              <Button
                onClick={() => {
                  setPrepareAttempt(attempt => attempt + 1)
                }}
                size="sm"
              >
                {copy.prepareExact}
              </Button>
            </div>
          )}
        </div>
      )}

      {prepared && (
        <div className="p-4">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <p className="line-clamp-2 text-xs text-muted-foreground">{preparedDescription}</p>
            </div>
            <span className="shrink-0 bg-(--ui-bg-quinary) px-2 py-1 text-[0.62rem]">
              {copy.localAdvisory}: {localScan?.guard.allowed === true ? copy.scanPassed : copy.reviewFindings}
            </span>
          </div>
          <details className="mt-3 border-y border-(--ui-stroke-tertiary) py-2 text-[0.68rem]">
            <summary className="cursor-pointer font-medium">{copy.whySuggested}</summary>
            <dl className="mt-2 grid gap-1 text-muted-foreground">
              <div>
                <dt className="inline font-medium">{copy.qualificationLabel}: </dt>
                <dd className="inline">{readableReason(event.payload.qualification)}</dd>
              </div>
              {Object.entries(event.payload.local_reasons).map(([key, value]) => (
                <div key={key}>
                  <dt className="inline font-medium">{readableReason(key)}: </dt>
                  <dd className="inline">{String(value)}</dd>
                </div>
              ))}
            </dl>
          </details>

          {editorOpen && (
            <div className="mt-3 border-t border-(--ui-stroke-tertiary) pt-3">
              <p className="text-xs text-muted-foreground">{copy.editDefaultsNotice}</p>
              <div className="mt-3 grid gap-3 sm:grid-cols-2">
                <label className="block text-[0.68rem]" htmlFor={`wisdom-name-${event.id}`}>
                  {copy.skillName}
                  <Input
                    className="mt-1 w-full"
                    disabled={busy !== null}
                    id={`wisdom-name-${event.id}`}
                    maxLength={512}
                    onChange={input =>
                      setPreparedFiles(current => ({
                        ...current,
                        'skill.manifest.json': withManifestName(
                          current['skill.manifest.json'] || '',
                          input.target.value
                        )
                      }))
                    }
                    value={preparedSkillName}
                  />
                </label>
                <label className="block text-[0.68rem]" htmlFor={`wisdom-description-${event.id}`}>
                  {copy.ownerDescription}
                  <Textarea
                    className="mt-1 min-h-16 w-full resize-y text-xs"
                    disabled={busy !== null}
                    id={`wisdom-description-${event.id}`}
                    maxLength={4096}
                    onChange={input => setPreparedDescription(input.target.value)}
                    value={preparedDescription}
                  />
                </label>
              </div>
              <div className="mt-3">
                {prepared.files
                  .filter(file => file.path === 'SKILL.md')
                  .map(file => (
                    <WisdomFileEditor
                      disabled={busy !== null}
                      file={file}
                      key={file.path}
                      onChange={value => setPreparedFiles(current => ({ ...current, [file.path]: value }))}
                      reviewSource="local"
                      value={preparedFiles[file.path] ?? ''}
                    />
                  ))}
              </div>
              <Button
                aria-expanded={detailedEditorOpen}
                className="mt-2 px-0"
                onClick={() => setDetailedEditorOpen(open => !open)}
                size="xs"
                variant="text"
              >
                {detailedEditorOpen ? copy.hideDetailedRequirements : copy.detailedRequirements}
              </Button>
              {detailedEditorOpen && (
                <div className="mt-1">
                  <p className="text-[0.68rem] text-muted-foreground">{copy.specificationNotice}</p>
                  {prepared.files
                    .filter(file => file.path !== 'SKILL.md')
                    .map(file => (
                      <WisdomFileEditor
                        disabled={busy !== null}
                        file={file}
                        key={file.path}
                        onChange={value => setPreparedFiles(current => ({ ...current, [file.path]: value }))}
                        reviewSource="local"
                        value={preparedFiles[file.path] ?? ''}
                      />
                    ))}
                </div>
              )}
              {preparedManifestError && (
                <div className="mt-2 text-[0.68rem] text-destructive" role="alert">
                  {preparedManifestError}
                </div>
              )}
              {preparedDirty && (
                <div className="mt-2 text-[0.68rem] text-amber-600" role="status">
                  {copy.unsavedChanges}
                </div>
              )}
            </div>
          )}

          <div className="mt-3 flex flex-wrap items-center justify-between gap-2 border-t border-(--ui-stroke-tertiary) pt-3">
            {fullReviewLink}
            {!editorOpen ? (
              <Button aria-expanded={false} disabled={busy !== null} onClick={() => setEditorOpen(true)} size="sm">
                {copy.prepareExact}
              </Button>
            ) : (
              <>
                <Button
                  disabled={busy !== null}
                  onClick={() => {
                    setEditorOpen(false)
                    setDetailedEditorOpen(false)
                  }}
                  size="sm"
                  variant="outline"
                >
                  {copy.close}
                </Button>
                <Button
                  disabled={busy !== null || !preparedDirty || Boolean(preparedManifestError)}
                  onClick={() => void saveLocal()}
                  size="sm"
                  variant="outline"
                >
                  {busy === 'save-local' ? copy.savingLocal : copy.saveLocal}
                </Button>
              </>
            )}
          </div>
          <footer className="mt-3 grid grid-cols-3 items-center gap-2">
            <div className="flex items-center gap-1 justify-self-start">
              <TooltipIconButton
                aria-pressed={notificationsMuted}
                disabled={busy !== null}
                onClick={() => setNotificationsMuted(value => !value)}
                tooltip={notificationsMuted ? copy.unmuteNotificationsSoon : copy.muteNotificationsSoon}
              >
                {notificationsMuted ? <VolumeX className="size-4" /> : <Volume2 className="size-4" />}
              </TooltipIconButton>
              <Button disabled={busy !== null} onClick={() => void notNow()} size="sm" variant="outline">
                {copy.notNow}
              </Button>
            </div>
            <div className="justify-self-center">
              <Button
                disabled={busy !== null || preparedDirty || Boolean(preparedManifestError)}
                onClick={() => void submit()}
                size="sm"
              >
                {busy === 'submit' ? copy.submitting : copy.reviewFirst}
              </Button>
            </div>
            <div className="justify-self-end">
              <Button
                disabled={busy !== null || preparedDirty || Boolean(preparedManifestError)}
                onClick={() => void approve()}
                size="sm"
              >
                {busy === 'approve' ? copy.publishing : copy.yes}
              </Button>
            </div>
          </footer>
        </div>
      )}

      {review && (
        <div className="p-4">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <p className="line-clamp-2 text-xs text-muted-foreground">{reviewDescription}</p>
            </div>
            <span className="shrink-0 bg-emerald-600/10 px-2 py-1 text-[0.62rem] text-emerald-700">
              {copy.serverEnforced}: {review.draft.scanVerdict || copy.reviewed}
            </span>
          </div>
          <details className="mt-3 border-y border-(--ui-stroke-tertiary) py-2 text-[0.68rem]">
            <summary className="cursor-pointer font-medium">{copy.serverFactsLabel}</summary>
            <div className="mt-2">
              <strong>{copy.serverFactsLabel}</strong>
              <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap text-muted-foreground">
                {JSON.stringify(
                  { verdict: review.draft.scanVerdict, scan: review.draft.scan, explanation: review.draft.explanation },
                  null,
                  2
                )}
              </pre>
            </div>
            <div className="mt-2 grid gap-1 break-all font-mono text-[0.6rem]">
              <strong>{copy.reviewedHashes}</strong>
              <span>
                {copy.contentHash} {review.hashes.content}
              </span>
              <span>
                {copy.authorDescriptionHash} {review.hashes.author_description}
              </span>
              <span>
                {copy.packageManifestHash} {review.hashes.package_manifest}
              </span>
            </div>
          </details>

          {editorOpen && (
            <div className="mt-3 border-t border-(--ui-stroke-tertiary) pt-3">
              <p className="text-xs text-muted-foreground">{copy.serverReviewNotice}</p>
              {reviewCanEdit && <p className="mt-1 text-[0.68rem] text-muted-foreground">{copy.editDefaultsNotice}</p>}
              <div className="mt-3 grid gap-3 sm:grid-cols-2">
                <label className="block text-[0.68rem]" htmlFor={`wisdom-server-name-${event.id}`}>
                  {copy.skillName}
                  <Input
                    className="mt-1 w-full"
                    disabled={busy !== null || !reviewCanEdit}
                    id={`wisdom-server-name-${event.id}`}
                    maxLength={512}
                    onChange={input =>
                      setReviewFiles(current => ({
                        ...current,
                        'skill.manifest.json': withManifestName(
                          current['skill.manifest.json'] || '',
                          input.target.value
                        )
                      }))
                    }
                    value={reviewSkillName}
                  />
                </label>
                <label className="block text-[0.68rem]" htmlFor={`wisdom-server-description-${event.id}`}>
                  {copy.editOwnerDescription}
                  <Textarea
                    className="mt-1 min-h-16 w-full resize-y text-xs"
                    disabled={busy !== null || !reviewCanEdit}
                    id={`wisdom-server-description-${event.id}`}
                    maxLength={4096}
                    onChange={input => setReviewDescription(input.target.value)}
                    value={reviewDescription}
                  />
                </label>
              </div>
              <div className="mt-3">
                {review.files
                  .filter(file => file.path === 'SKILL.md')
                  .map(file => (
                    <WisdomFileEditor
                      disabled={busy !== null || !reviewCanEdit}
                      file={file}
                      key={file.path}
                      onChange={value => setReviewFiles(current => ({ ...current, [file.path]: value }))}
                      reviewSource="server"
                      value={reviewFiles[file.path] ?? ''}
                    />
                  ))}
              </div>
              <Button
                aria-expanded={detailedEditorOpen}
                className="mt-2 px-0"
                onClick={() => setDetailedEditorOpen(open => !open)}
                size="xs"
                variant="text"
              >
                {detailedEditorOpen ? copy.hideDetailedRequirements : copy.detailedRequirements}
              </Button>
              {detailedEditorOpen && (
                <div className="mt-1">
                  <p className="text-[0.68rem] text-muted-foreground">{copy.specificationNotice}</p>
                  {review.files
                    .filter(file => file.path !== 'SKILL.md')
                    .map(file => (
                      <WisdomFileEditor
                        disabled={busy !== null || !reviewCanEdit}
                        file={file}
                        key={file.path}
                        onChange={value => setReviewFiles(current => ({ ...current, [file.path]: value }))}
                        reviewSource="server"
                        value={reviewFiles[file.path] ?? ''}
                      />
                    ))}
                </div>
              )}
              {reviewManifestError && (
                <div className="mt-2 text-[0.68rem] text-destructive" role="alert">
                  {reviewManifestError}
                </div>
              )}
              {reviewDirty && (
                <div className="mt-2 text-[0.68rem] text-amber-600" role="status">
                  {copy.unsavedChanges}
                </div>
              )}
            </div>
          )}

          <div className="mt-3 border-t border-(--ui-stroke-tertiary) pt-3">{fullReviewLink}</div>
          <footer className="mt-3 grid grid-cols-3 items-center gap-2">
            <div className="flex items-center gap-1 justify-self-start">
              <TooltipIconButton
                aria-pressed={notificationsMuted}
                disabled={busy !== null}
                onClick={() => setNotificationsMuted(value => !value)}
                tooltip={notificationsMuted ? copy.unmuteNotificationsSoon : copy.muteNotificationsSoon}
              >
                {notificationsMuted ? <VolumeX className="size-4" /> : <Volume2 className="size-4" />}
              </TooltipIconButton>
              <Button disabled={busy !== null} onClick={() => void notNow()} size="sm" variant="outline">
                {copy.notNow}
              </Button>
            </div>
            <div className="flex flex-wrap justify-center gap-2">
              {!editorOpen ? (
                <Button aria-expanded={false} disabled={busy !== null} onClick={() => setEditorOpen(true)} size="sm">
                  {copy.prepareExact}
                </Button>
              ) : (
                <>
                  <Button
                    disabled={busy !== null}
                    onClick={() => {
                      setEditorOpen(false)
                      setDetailedEditorOpen(false)
                    }}
                    size="sm"
                    variant="outline"
                  >
                    {copy.close}
                  </Button>
                  {reviewCanEdit && (
                    <Button
                      disabled={busy !== null || !reviewDirty || Boolean(reviewManifestError)}
                      onClick={() => void saveServerRevision()}
                      size="sm"
                      variant="outline"
                    >
                      {busy === 'save-server' ? copy.savingRevision : copy.saveAndRescan}
                    </Button>
                  )}
                </>
              )}
            </div>
            <div className="justify-self-end">
              <Button disabled={busy !== null || reviewDirty} onClick={() => void approve()} size="sm">
                {busy === 'approve' ? copy.publishing : copy.yes}
              </Button>
            </div>
          </footer>
        </div>
      )}
    </section>
  )
}
