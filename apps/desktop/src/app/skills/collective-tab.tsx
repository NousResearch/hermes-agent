import { useQuery } from '@tanstack/react-query'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { useLocation, useNavigate } from 'react-router'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { WisdomCheckBadge, WisdomReviewTables } from '@/components/wisdom-checks'
import { WisdomNotificationsCard } from '@/components/wisdom-notifications-card'
import {
  acknowledgeWisdomNotifications,
  applyWisdomInstall,
  applyWisdomUpdate,
  checkWisdom,
  decideWisdomDraft,
  getActionStatus,
  getWisdomCandidates,
  getWisdomDiscovery,
  getWisdomDrafts,
  getWisdomInstallations,
  getWisdomSkill,
  getWisdomStatus,
  getWisdomVersionContent,
  planWisdomInstall,
  planWisdomUpdate,
  type ProfileScope,
  profileScopeKey,
  reviewWisdomDraft,
  reviseWisdomDraft,
  scanWisdom,
  setupWisdom,
  suggestWisdomSkill,
  uninstallWisdomSkill,
  type WisdomActionPlan,
  type WisdomCandidate,
  type WisdomCheckResult,
  type WisdomDraftReview,
  type WisdomPreparedDraft,
  type WisdomReviewCheck,
  type WisdomUpdateMode
} from '@/hermes'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { notifyError } from '@/store/notifications'

import { DetailColumn, ListColumn, ListStrip, MasterDetail } from '../master-detail'

import { WisdomFileEditor } from './wisdom-file-editor'
import {
  parseWisdomSystemSpecification,
  wisdomManifestValidationError,
  type WisdomSystemSpecification,
  wisdomSystemSpecificationValidationError
} from './wisdom-manifest'
import { WisdomSystemSpecificationEditor } from './wisdom-manifest-editor'

const TERMINAL_DRAFT_STATES = new Set(['published', 'declined', 'invalidated', 'rejected'])
const UPDATE_CHECK_INTERVAL_MS = 5 * 60 * 1000

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : {}
}

const candidateDisplayName = (candidate: WisdomCandidate): string =>
  candidate.editorial_name?.trim() || candidate.name

const candidateDisplayDescription = (candidate: WisdomCandidate): string =>
  candidate.editorial_description?.trim() || ''

async function waitForWisdomAction(name: string, profile: ProfileScope): Promise<void> {
  for (let attempt = 0; attempt < 1200; attempt += 1) {
    const status = await getActionStatus(name, 80, profile)

    if (!status.running) {
      if (status.exit_code !== 0) {
        throw new Error(status.lines.at(-1) || `Collective Wisdom action failed (${status.exit_code ?? 'unknown'})`)
      }

      return
    }

    await new Promise(resolve => setTimeout(resolve, 500))
  }

  throw new Error('Collective Wisdom action timed out')
}

export function CollectiveTab({ profile, query }: { profile: ProfileScope; query: string }) {
  const { t } = useI18n()
  const copy = t.skills.collective
  const location = useLocation()
  const navigate = useNavigate()
  const scope = profileScopeKey(profile)
  const [selectedId, setSelectedId] = useState<null | string>(null)

  const [prepared, setPrepared] = useState<null | (WisdomPreparedDraft & { localSkillId: string; skill: string })>(null)

  const [description, setDescription] = useState('')
  const [specification, setSpecification] = useState<null | WisdomSystemSpecification>(null)
  const [review, setReview] = useState<null | WisdomDraftReview>(null)
  const [reviewDescription, setReviewDescription] = useState('')
  const [reviewFiles, setReviewFiles] = useState<Record<string, string>>({})
  const [busy, setBusy] = useState<null | string>(null)
  const [showManualCandidates, setShowManualCandidates] = useState(false)
  const [installReference, setInstallReference] = useState('')
  const [installUpdateMode, setInstallUpdateMode] = useState<'' | WisdomUpdateMode>('')

  const [actionPlan, setActionPlan] = useState<
    null | (WisdomActionPlan & { action: 'install' | 'uninstall' | 'update' })
  >(null)

  const [actionPlanReference, setActionPlanReference] = useState<null | string>(null)

  const [acceptSensitive, setAcceptSensitive] = useState(false)
  const [acceptPartial, setAcceptPartial] = useState(false)
  const [preserveModified, setPreserveModified] = useState(false)

  useEffect(() => {
    setSelectedId(null)
    setPrepared(null)
    setDescription('')
    setSpecification(null)
    setReview(null)
    setReviewDescription('')
    setReviewFiles({})
    setShowManualCandidates(false)
    setInstallReference('')
    setInstallUpdateMode('')
    setActionPlan(null)
    setActionPlanReference(null)
    setAcceptSensitive(false)
    setAcceptPartial(false)
    setPreserveModified(false)
    setBusy(null)
  }, [scope])

  const status = useQuery({
    queryKey: ['wisdom-status', scope],
    queryFn: () => getWisdomStatus(profile),
    staleTime: 30_000
  })

  const discovery = useQuery({
    queryKey: ['wisdom-discovery', scope],
    queryFn: () => getWisdomDiscovery(profile),
    staleTime: 30_000,
    enabled: status.data?.configured === true
  })

  const candidates = useQuery({
    queryKey: ['wisdom-candidates', scope],
    queryFn: () => getWisdomCandidates(profile),
    staleTime: 15_000,
    enabled: status.data?.configured === true
  })

  const drafts = useQuery({
    queryKey: ['wisdom-drafts', scope],
    queryFn: () => getWisdomDrafts(profile),
    staleTime: 15_000,
    enabled: status.data?.configured === true
  })

  const detail = useQuery({
    queryKey: ['wisdom-detail', scope, selectedId],
    queryFn: () => getWisdomSkill(selectedId || '', profile),
    enabled: status.data?.configured === true && Boolean(selectedId),
    staleTime: 30_000
  })

  const installations = useQuery({
    queryKey: ['wisdom-installations', scope],
    queryFn: () => getWisdomInstallations(profile),
    staleTime: 10_000,
    enabled: status.data?.configured === true
  })

  const refetchInstallations = installations.refetch

  const updateCheck = useQuery<WisdomCheckResult>({
    queryKey: ['wisdom-update-check', scope],
    queryFn: () => checkWisdom(profile),
    enabled: status.data?.configured === true,
    staleTime: UPDATE_CHECK_INTERVAL_MS,
    refetchInterval: UPDATE_CHECK_INTERVAL_MS,
    refetchIntervalInBackground: false,
    refetchOnWindowFocus: true
  })

  useEffect(() => {
    if (!updateCheck.dataUpdatedAt) {
      return
    }

    void refetchInstallations().catch(error => notifyError(error, 'Wisdom installation refresh failed'))
  }, [refetchInstallations, updateCheck.dataUpdatedAt])

  const latestSelectedVersion = useMemo(
    () => Math.max(0, ...(detail.data?.versions ?? []).map(version => Number(version.version) || 0)),
    [detail.data?.versions]
  )

  const content = useQuery({
    queryKey: ['wisdom-content', scope, selectedId, latestSelectedVersion],
    queryFn: () => getWisdomVersionContent(selectedId || '', latestSelectedVersion, profile),
    enabled: Boolean(selectedId && latestSelectedVersion),
    staleTime: 60_000
  })

  const rows = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase()

    return (discovery.data?.skills ?? []).filter(
      skill =>
        !needle ||
        skill.slug.toLocaleLowerCase().includes(needle) ||
        (skill.author_description ?? '').toLocaleLowerCase().includes(needle)
    )
  }, [discovery.data?.skills, query])

  const pendingUpdates = useMemo(
    () =>
      new Map(
        (updateCheck.data?.installations ?? [])
          .filter(item => item.state === 'update_available')
          .map(item => [item.skill_id, item] as const)
      ),
    [updateCheck.data?.installations]
  )

  const activeDrafts = useMemo(
    () => (drafts.data?.drafts ?? []).filter(draft => !TERMINAL_DRAFT_STATES.has(draft.state)),
    [drafts.data?.drafts]
  )

  const { manualCandidates, qualifiedCandidates } = useMemo(() => {
    const all = candidates.data?.candidates ?? []

    return {
      manualCandidates: all.filter(candidate => candidate.qualification === 'manual_selection'),
      qualifiedCandidates: all.filter(candidate => candidate.qualification !== 'manual_selection')
    }
  }, [candidates.data?.candidates])

  const filterCandidates = useCallback(
    (items: WisdomCandidate[]) => {
      const needle = query.trim().toLocaleLowerCase()

      return items
        .filter(candidate =>
          !needle
            ? true
            : [candidate.name, candidateDisplayName(candidate), candidateDisplayDescription(candidate)].some(value =>
                value.toLocaleLowerCase().includes(needle)
              )
        )
        .toSorted((left, right) => candidateDisplayName(left).localeCompare(candidateDisplayName(right)))
    },
    [query]
  )

  const visibleQualifiedCandidates = useMemo(
    () => filterCandidates(qualifiedCandidates),
    [filterCandidates, qualifiedCandidates]
  )

  const visibleManualCandidates = useMemo(
    () => filterCandidates(manualCandidates),
    [filterCandidates, manualCandidates]
  )

  const reviewDirty = useMemo(() => {
    if (!review) {
      return false
    }

    if (reviewDescription !== (review.draft.authorDescription || '')) {
      return true
    }

    return review.files.some(file => reviewFiles[file.path] !== file.content_utf8)
  }, [review, reviewDescription, reviewFiles])

  const reviewManifestError = useMemo(() => {
    if (!review) {
      return null
    }

    const manifest = reviewFiles['skill.manifest.json']

    return manifest === undefined
      ? 'The complete package must include skill.manifest.json.'
      : wisdomManifestValidationError(manifest)
  }, [review, reviewFiles])

  const reviewCanEdit = Boolean(review && ['ready', 'changes_requested'].includes(review.draft.state))

  const specificationError = specification
    ? wisdomSystemSpecificationValidationError(specification)
    : 'System Specification is unavailable.'

  const refreshContributionData = useCallback(async () => {
    await Promise.all([candidates.refetch(), drafts.refetch(), discovery.refetch()])
  }, [candidates, discovery, drafts])

  const candidateSummary = (candidate: WisdomCandidate): string => {
    if (candidate.eligibility !== 'eligible') {
      return candidate.reason || copy.localOnly
    }

    const qualification =
      candidate.qualification === 'manual_selection'
        ? copy.localOnly
        : candidate.notice_variant === 'first'
          ? copy.qualificationFirst(candidate.organization_name)
          : copy.qualificationReturning

    return candidate.contribution_state === 'prepared' ? `${qualification} ${copy.savedLocally}` : qualification
  }

  const prepare = async (candidate: WisdomCandidate) => {
    setBusy(candidate.local_skill_id)

    try {
      const result = await suggestWisdomSkill(candidate.name, profile, undefined, candidate.local_skill_id)

      if ('network_submission' in result) {
        setPrepared({ ...result, localSkillId: candidate.local_skill_id, skill: candidate.name })
        setDescription(result.drafted_description)
        setSpecification(parseWisdomSystemSpecification(result.system_specification))
      }
    } catch (error) {
      notifyError(error, 'Collective Wisdom preparation failed')
    } finally {
      setBusy(null)
    }
  }

  const submit = async () => {
    if (!prepared) {
      return
    }

    setBusy(prepared.local_draft_id)

    try {
      if (!description.trim()) {
        throw new Error('Add a description before submitting this private draft.')
      }

      if (!specification || specificationError) {
        throw new Error(specificationError || 'System Specification is unavailable.')
      }

      await suggestWisdomSkill(
        prepared.skill,
        profile,
        { description, systemSpecification: specification },
        prepared.localSkillId
      )
      setPrepared(null)
      setSpecification(null)
      await refreshContributionData()
    } catch (error) {
      notifyError(error, 'Owner-private submission failed')
    } finally {
      setBusy(null)
    }
  }

  const openReview = async (draftId: string) => {
    setBusy(draftId)

    try {
      const nextReview = await reviewWisdomDraft(draftId, false, profile)
      setReview(nextReview)
      setReviewDescription(nextReview.draft.authorDescription || '')
      setReviewFiles(Object.fromEntries(nextReview.files.map(file => [file.path, file.content_utf8])))
    } catch (error) {
      notifyError(error, 'Wisdom review failed')
    } finally {
      setBusy(null)
    }
  }

  const closeReview = () => {
    setReview(null)
    setReviewDescription('')
    setReviewFiles({})
  }

  const resetReviewEdits = () => {
    if (!review) {
      return
    }

    setReviewDescription(review.draft.authorDescription || '')
    setReviewFiles(Object.fromEntries(review.files.map(file => [file.path, file.content_utf8])))
  }

  const saveReviewRevision = async () => {
    if (!review || !reviewCanEdit || !reviewDirty) {
      return
    }

    setBusy(review.draft.id)

    try {
      if (!reviewDescription.trim()) {
        throw new Error('Add a description before saving this revision.')
      }

      if (reviewManifestError) {
        throw new Error(`Fix the System Specification before saving: ${reviewManifestError}`)
      }

      const revised = await reviseWisdomDraft(
        review.draft.id,
        reviewDescription,
        review.files.map(file => ({ path: file.path, content_utf8: reviewFiles[file.path] ?? file.content_utf8 })),
        review.hashes,
        profile
      )

      await refreshContributionData()
      const nextReview = await reviewWisdomDraft(revised.draft.id, false, profile)
      setReview(nextReview)
      setReviewDescription(nextReview.draft.authorDescription || '')
      setReviewFiles(Object.fromEntries(nextReview.files.map(file => [file.path, file.content_utf8])))
    } catch (error) {
      notifyError(error, 'Wisdom revision failed')
    } finally {
      setBusy(null)
    }
  }

  const approve = async () => {
    if (!review) {
      return
    }

    setBusy(review.draft.id)

    try {
      const acknowledged = await reviewWisdomDraft(review.draft.id, true, profile)

      if (!acknowledged.receipt) {
        throw new Error('Gateway review receipt was not created')
      }

      await decideWisdomDraft(review.draft.id, 'approve', profile)
      closeReview()
      await refreshContributionData()
    } catch (error) {
      notifyError(error, 'Wisdom publication failed')
    } finally {
      setBusy(null)
    }
  }

  const installed = installations.data?.installations.find(
    item => item.skill_id === selectedId && item.state === 'active'
  )

  const selectedUpdate = selectedId ? pendingUpdates.get(selectedId) : undefined

  const planManagedActionForSkill = useCallback(
    async (skillId: string, action: 'install' | 'uninstall' | 'update') => {
      setSelectedId(skillId)
      setBusy(skillId)

      try {
        const plan =
          action === 'install'
            ? await planWisdomInstall(skillId, profile, installUpdateMode || undefined)
            : action === 'update'
              ? await planWisdomUpdate(skillId, profile)
              : { skill_id: skillId, state: 'confirm_uninstall' }

        setActionPlan({ ...plan, action })
        setActionPlanReference(action === 'install' ? skillId : null)
        setAcceptSensitive(false)
        setAcceptPartial(false)
        setPreserveModified(false)
      } catch (error) {
        notifyError(error, `Wisdom ${action} planning failed`)
      } finally {
        setBusy(null)
      }
    },
    [installUpdateMode, profile]
  )

  const planManagedAction = async (action: 'install' | 'uninstall' | 'update') => {
    if (selectedId) {
      await planManagedActionForSkill(selectedId, action)
    }
  }

  useEffect(() => {
    if (!status.data?.configured) {
      return
    }

    const params = new URLSearchParams(location.search)
    const action = params.get('wisdomAction')
    const skillId = params.get('wisdomSkillId')

    if ((action !== 'install' && action !== 'update') || !skillId) {
      return
    }

    params.delete('wisdomAction')
    params.delete('wisdomSkillId')
    const search = params.toString()

    navigate(
      { hash: location.hash, pathname: location.pathname, search: search ? `?${search}` : '' },
      { replace: true }
    )
    void planManagedActionForSkill(skillId, action)
  }, [location.hash, location.pathname, location.search, navigate, planManagedActionForSkill, status.data?.configured])

  const planReferencedInstall = async () => {
    const reference = installReference.trim()

    if (!reference) {
      return
    }

    setBusy('install-reference')

    try {
      const plan = await planWisdomInstall(reference, profile, installUpdateMode || undefined)
      setSelectedId(plan.skill_id)
      setActionPlan({ ...plan, action: 'install' })
      setActionPlanReference(reference)
      setAcceptSensitive(false)
      setAcceptPartial(false)
      setPreserveModified(false)
    } catch (error) {
      notifyError(error, 'Wisdom install planning failed')
    } finally {
      setBusy(null)
    }
  }

  const replanInstallUpdateMode = async (value: string) => {
    if (!actionPlan || actionPlan.action !== 'install') {
      return
    }

    const previousMode = installUpdateMode
    const nextMode = value === 'DEFAULT' ? '' : (value as WisdomUpdateMode)
    const reference = actionPlanReference || actionPlan.skill_id

    setInstallUpdateMode(nextMode)
    setBusy('install-mode')

    try {
      const plan = await planWisdomInstall(reference, profile, nextMode || undefined)
      setActionPlan({ ...plan, action: 'install' })
      setAcceptSensitive(false)
      setAcceptPartial(false)
      setPreserveModified(false)
    } catch (error) {
      setInstallUpdateMode(previousMode)
      notifyError(error, 'Wisdom install planning failed')
    } finally {
      setBusy(null)
    }
  }

  const applyManagedAction = async () => {
    if (!actionPlan) {
      return
    }

    setBusy(actionPlan.skill_id)

    try {
      if (actionPlan.action === 'uninstall') {
        await uninstallWisdomSkill(actionPlan.skill_id, profile)
      } else if (!actionPlan.receipt) {
        throw new Error('Verified action receipt is missing')
      } else if (actionPlan.action === 'install') {
        await applyWisdomInstall(actionPlan.receipt, acceptPartial, profile)
      } else {
        await applyWisdomUpdate(actionPlan.receipt, { acceptPartial, acceptSensitive, preserveModified }, profile)
      }

      setActionPlan(null)
      setActionPlanReference(null)

      if (actionPlan.action === 'install') {
        setInstallReference('')
        setInstallUpdateMode('')
      }

      await Promise.all([installations.refetch(), discovery.refetch(), detail.refetch(), updateCheck.refetch()])
    } catch (error) {
      notifyError(error, `Wisdom ${actionPlan.action} failed`)
    } finally {
      setBusy(null)
    }
  }

  const setupProfile = async () => {
    setBusy('setup')

    try {
      const action = await setupWisdom(profile)
      await waitForWisdomAction(action.name, profile)
      await status.refetch()
      await Promise.all([discovery.refetch(), candidates.refetch(), drafts.refetch(), installations.refetch()])
    } catch (error) {
      notifyError(error, 'Collective Wisdom setup failed')
    } finally {
      setBusy(null)
    }
  }

  if (status.isPending) {
    return <div className="grid h-full place-items-center text-xs text-muted-foreground">{copy.loading}</div>
  }

  if (status.isError) {
    const error = status.error

    return (
      <div className="grid h-full place-items-center px-8 text-center text-xs text-muted-foreground">
        {copy.unavailable} {error instanceof Error ? error.message : ''}
      </div>
    )
  }

  if (!status.data.configured) {
    return (
      <div className="grid h-full place-items-center p-8">
        <section
          aria-label={copy.title}
          className="max-w-xl space-y-4 border border-(--ui-stroke-tertiary) p-5 text-sm"
        >
          <div>
            <h2 className="font-medium">{copy.title}</h2>
            <p className="mt-1 text-xs text-muted-foreground">{copy.setup}</p>
          </div>
          <p className="text-xs leading-5 text-muted-foreground">{copy.setupDisclosure}</p>
          {status.data.error && (
            <div className="text-xs text-destructive" role="alert">
              {status.data.error}
            </div>
          )}
          <Button disabled={busy === 'setup'} onClick={setupProfile} size="sm">
            {busy === 'setup' ? copy.settingUp : copy.setupAction}
          </Button>
        </section>
      </div>
    )
  }

  if (discovery.isPending || candidates.isPending || drafts.isPending || installations.isPending) {
    return <div className="grid h-full place-items-center text-xs text-muted-foreground">{copy.loading}</div>
  }

  if (discovery.isError || candidates.isError || drafts.isError || installations.isError) {
    const error = discovery.error || candidates.error || drafts.error || installations.error

    return (
      <div className="grid h-full place-items-center px-8 text-center text-xs text-muted-foreground">
        {copy.unavailable} {error instanceof Error ? error.message : ''}
      </div>
    )
  }

  const statusCopy = status.data?.verified_org_id ? `${status.data.verified_org_id} · ${copy.orgWide}` : copy.setup

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="shrink-0 border-b border-(--ui-stroke-tertiary) px-3 py-2">
        <div className="flex items-center justify-between gap-3">
          <div>
            <div className="text-xs font-medium">{copy.title}</div>
            <div className="text-[0.65rem] text-muted-foreground">{statusCopy}</div>
          </div>
          <div className="flex gap-2">
            <Button
              onClick={async () => {
                setBusy('refresh-shared')

                try {
                  await Promise.all([
                    discovery.refetch(),
                    installations.refetch(),
                    ...(selectedId ? [detail.refetch()] : [])
                  ])
                } catch (error) {
                  notifyError(error, 'Wisdom registry refresh failed')
                } finally {
                  setBusy(null)
                }
              }}
              size="sm"
              variant="outline"
            >
              {busy === 'refresh-shared' ? copy.refreshingShared : copy.refreshShared}
            </Button>
            <Button
              onClick={async () => {
                setBusy('scan')

                try {
                  const action = await scanWisdom(undefined, profile)
                  await waitForWisdomAction(action.name, profile)
                  await candidates.refetch()
                } catch (error) {
                  notifyError(error, 'Wisdom local scan failed')
                } finally {
                  setBusy(null)
                }
              }}
              size="sm"
              variant="outline"
            >
              {busy === 'scan' ? copy.checking : copy.scanLocal}
            </Button>
            <Button
              onClick={async () => {
                setBusy('check')

                try {
                  await updateCheck.refetch()
                  await Promise.all([
                    installations.refetch(),
                    discovery.refetch(),
                    ...(selectedId ? [detail.refetch()] : [])
                  ])
                } catch (error) {
                  notifyError(error, 'Wisdom update check failed')
                } finally {
                  setBusy(null)
                }
              }}
              size="sm"
              variant="outline"
            >
              {busy === 'check' ? copy.checking : copy.checkUpdates(pendingUpdates.size)}
            </Button>
          </div>
        </div>
        <form
          className="mt-2 grid grid-cols-[minmax(0,1fr)_minmax(11rem,0.55fr)_auto] items-end gap-2 border-t border-(--ui-stroke-tertiary) pt-2"
          onSubmit={event => {
            event.preventDefault()
            void planReferencedInstall()
          }}
        >
          <div className="min-w-0 flex-1">
            <label className="mb-1 block text-[0.65rem] font-medium" htmlFor="desktop-wisdom-install-reference">
              {copy.installReferenceLabel}
            </label>
            <Input
              aria-describedby="desktop-wisdom-install-reference-help"
              id="desktop-wisdom-install-reference"
              onChange={event => setInstallReference(event.target.value)}
              placeholder={copy.installReferencePlaceholder}
              size="sm"
              value={installReference}
            />
            <p className="mt-1 truncate text-[0.6rem] text-muted-foreground" id="desktop-wisdom-install-reference-help">
              {copy.installReferenceHelp}
            </p>
          </div>
          <div className="min-w-0">
            <label className="mb-1 block text-[0.65rem] font-medium" id="desktop-wisdom-update-mode-label">
              {copy.updateModeLabel}
            </label>
            <Select
              onValueChange={value => setInstallUpdateMode(value === 'DEFAULT' ? '' : (value as WisdomUpdateMode))}
              value={installUpdateMode || 'DEFAULT'}
            >
              <SelectTrigger
                aria-describedby="desktop-wisdom-update-mode-help"
                aria-labelledby="desktop-wisdom-update-mode-label"
                className="w-full"
                size="sm"
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="DEFAULT">{copy.updateModeDefault}</SelectItem>
                <SelectItem value="MANUAL">{copy.updateModeManual}</SelectItem>
                <SelectItem value="AUTO_WITH_NOTICE">{copy.updateModeAutomatic}</SelectItem>
                <SelectItem value="REQUIRED">{copy.updateModeRequired}</SelectItem>
              </SelectContent>
            </Select>
            <p className="mt-1 truncate text-[0.6rem] text-muted-foreground" id="desktop-wisdom-update-mode-help">
              {copy.updateModeHelp}
            </p>
          </div>
          <Button
            disabled={!installReference.trim() || busy === 'install-reference'}
            size="sm"
            type="submit"
            variant="outline"
          >
            {busy === 'install-reference' ? copy.planningInstall : copy.reviewInstall}
          </Button>
        </form>
        <WisdomNotificationsCard
          className="mt-2"
          events={installations.data.notifications}
          onMarkAllRead={async () => {
            try {
              await acknowledgeWisdomNotifications(profile)
              await installations.refetch()
            } catch (error) {
              notifyError(error, 'Could not acknowledge Wisdom notifications')
            }
          }}
          onPlanAction={(action, event) => planManagedActionForSkill(event.skill_id, action)}
        />
      </div>
      <MasterDetail resizeId="collective-capabilities-split" split="wide">
        <ListColumn
          header={
            <ListStrip
              left={<span className="text-[0.68rem] text-muted-foreground">{copy.sharedSkills(rows.length)}</span>}
              right={
                qualifiedCandidates.length > 0 ? (
                  <span className="text-[0.62rem] text-muted-foreground">
                    {copy.localCandidates(qualifiedCandidates.length)}
                  </span>
                ) : undefined
              }
            />
          }
        >
          {rows.map(skill => (
            <button
              className={cn(
                'row-hover flex h-12 w-full shrink-0 items-center rounded-md px-2 text-left',
                selectedId === skill.id && 'bg-(--ui-row-active-background)'
              )}
              key={skill.id}
              onClick={() => setSelectedId(skill.id)}
              type="button"
            >
              <span className="min-w-0 flex-1">
                <span className="block truncate text-[0.78rem] font-medium">{skill.slug}</span>
                <span className="block truncate text-[0.62rem] text-muted-foreground">
                  {skill.author_description || copy.noDescription}
                </span>
              </span>
              <span className="ml-2 flex shrink-0 flex-col items-end gap-0.5 text-[0.6rem]">
                <WisdomCheckBadge label="Security" value={skill.security_check} />
                <WisdomCheckBadge label="Professionalism" value={skill.professionalism_check} />
                {pendingUpdates.has(skill.id) && (
                  <span className="text-amber-500">
                    {copy.updateAvailable(pendingUpdates.get(skill.id)?.plan?.version)}
                  </span>
                )}
              </span>
            </button>
          ))}
          {rows.length === 0 && (
            <div className="px-3 py-8 text-center text-xs text-muted-foreground">{copy.noShared}</div>
          )}
        </ListColumn>
        <DetailColumn footer={copy.authoritative}>
          <div className="space-y-5 p-4">
            {((candidates.data?.candidates.length ?? 0) > 0 || activeDrafts.length > 0) && (
              <section aria-label={copy.contributionWorkflow} className="grid gap-5 lg:grid-cols-2">
                <div>
                  <h2 className="text-xs font-medium">{copy.potential}</h2>
                  <p className="mt-1 text-[0.65rem] leading-4 text-muted-foreground">{copy.potentialHelp}</p>
                  <div className="mt-2">
                    {visibleQualifiedCandidates.map(candidate => (
                      <div
                        className="flex items-start gap-3 border-t border-(--ui-stroke-tertiary) py-3 first:border-0"
                        key={candidate.local_skill_id}
                      >
                        <div className="min-w-0 flex-1">
                          <div className="truncate text-xs font-medium">{candidateDisplayName(candidate)}</div>
                          {candidateDisplayDescription(candidate) && (
                            <div className="line-clamp-2 text-[0.65rem] leading-4 text-muted-foreground">
                              {candidateDisplayDescription(candidate)}
                            </div>
                          )}
                          <div className="text-[0.65rem] leading-4 text-muted-foreground">
                            {candidateSummary(candidate)}
                          </div>
                          {candidate.professionalism_check && (
                            <div className="mt-1">
                              <WisdomCheckBadge label="Professionalism" value={candidate.professionalism_check} />
                            </div>
                          )}
                        </div>
                        <Button
                          disabled={busy === candidate.local_skill_id || candidate.eligibility !== 'eligible'}
                          onClick={() => void prepare(candidate)}
                          size="xs"
                          variant="outline"
                        >
                          {candidate.contribution_state === 'prepared' ? copy.continueDraft : copy.prepare}
                        </Button>
                      </div>
                    ))}
                    {visibleQualifiedCandidates.length === 0 && (
                      <p className="py-3 text-[0.65rem] text-muted-foreground">{copy.noSuggestions}</p>
                    )}
                    {manualCandidates.length > 0 && (
                      <div className="border-t border-(--ui-stroke-tertiary) py-3">
                        <button
                          aria-expanded={showManualCandidates}
                          className="text-[0.68rem] font-medium"
                          onClick={() => setShowManualCandidates(value => !value)}
                          type="button"
                        >
                          {copy.browseLocal(manualCandidates.length)}
                        </button>
                        {showManualCandidates && (
                          <>
                            <p className="mt-1 text-[0.65rem] leading-4 text-muted-foreground">
                              {copy.browseLocalHelp}
                            </p>
                            <div className="mt-2 max-h-64 overflow-y-auto pr-1">
                              {visibleManualCandidates.map(candidate => (
                                <div
                                  className="flex items-start gap-3 border-t border-(--ui-stroke-tertiary) py-3 first:border-0"
                                  key={candidate.local_skill_id}
                                >
                                  <div className="min-w-0 flex-1">
                                    <div className="truncate text-xs font-medium">{candidateDisplayName(candidate)}</div>
                                    {candidateDisplayDescription(candidate) && (
                                      <div className="line-clamp-2 text-[0.65rem] leading-4 text-muted-foreground">
                                        {candidateDisplayDescription(candidate)}
                                      </div>
                                    )}
                                    <div className="text-[0.65rem] leading-4 text-muted-foreground">
                                      {candidateSummary(candidate)}
                                    </div>
                                    {candidate.professionalism_check && (
                                      <div className="mt-1">
                                        <WisdomCheckBadge
                                          label="Professionalism"
                                          value={candidate.professionalism_check}
                                        />
                                      </div>
                                    )}
                                  </div>
                                  <Button
                                    disabled={busy === candidate.local_skill_id || candidate.eligibility !== 'eligible'}
                                    onClick={() => void prepare(candidate)}
                                    size="xs"
                                    variant="outline"
                                  >
                                    {candidate.contribution_state === 'prepared' ? copy.continueDraft : copy.prepare}
                                  </Button>
                                </div>
                              ))}
                            </div>
                          </>
                        )}
                      </div>
                    )}
                  </div>
                </div>

                <div>
                  <h2 className="text-xs font-medium">{copy.ownerReview}</h2>
                  <p className="mt-1 text-[0.65rem] leading-4 text-muted-foreground">{copy.ownerReviewHelp}</p>
                  <div className="mt-2">
                    {activeDrafts.length === 0 ? (
                      <p className="py-3 text-[0.65rem] text-muted-foreground">{copy.noDrafts}</p>
                    ) : (
                      activeDrafts.map(draft => (
                        <button
                          className="row-hover flex w-full items-center justify-between border-t border-(--ui-stroke-tertiary) py-3 text-left first:border-0 focus-visible:outline focus-visible:outline-2"
                          key={draft.id}
                          onClick={() => void openReview(draft.id)}
                          type="button"
                        >
                          <span>
                            <span className="block font-mono text-xs">{draft.slug}</span>
                            <span className="text-[0.65rem] text-muted-foreground">{copy.draftState(draft.state)}</span>
                          </span>
                          <span className="text-[0.65rem]">{copy.openDraft}</span>
                        </button>
                      ))
                    )}
                  </div>
                </div>
              </section>
            )}

            {detail.data && (
              <section aria-label="Collective Wisdom skill detail">
                <h2 className="font-mono text-base">{String(detail.data.skill.slug || detail.data.skill.id)}</h2>
                <p className="mt-2 text-sm text-muted-foreground">
                  {String(detail.data.skill.authorDescription || detail.data.skill.author_description || '')}
                </p>
                <WisdomReviewTables
                  professionalism={
                    asRecord(asRecord(detail.data.latest_version_detail).version).professionalism_check as
                      WisdomReviewCheck | undefined
                  }
                  security={
                    asRecord(asRecord(detail.data.latest_version_detail).version).security_check as
                      WisdomReviewCheck | undefined
                  }
                />
                <h3 className="mt-5 text-xs font-medium">{copy.versionHistory}</h3>
                <pre className="mt-2 max-h-72 overflow-auto whitespace-pre-wrap text-[0.67rem] text-muted-foreground">
                  {JSON.stringify(
                    {
                      latest_version: detail.data.latest_version_detail,
                      version_history: detail.data.versions,
                      local_compatibility: detail.data.local_compatibility
                    },
                    null,
                    2
                  )}
                </pre>
                {content.data && (
                  <div className="mt-4">
                    <div className="break-all font-mono text-[0.62rem]">content {content.data.content_hash}</div>
                    {content.data.files.map(file => (
                      <details className="border-t border-(--ui-stroke-tertiary) py-2" key={file.path} open>
                        <summary className="cursor-pointer font-mono text-[0.68rem]">
                          {file.path} · {file.hash}
                        </summary>
                        <pre className="mt-2 max-h-72 overflow-auto whitespace-pre-wrap text-[0.65rem]">
                          {file.content_utf8}
                        </pre>
                      </details>
                    ))}
                  </div>
                )}
                <div className="mt-4 flex flex-wrap items-center justify-end gap-2">
                  {installed ? (
                    <>
                      <span className="text-[0.62rem] text-muted-foreground">
                        {copy.installed(installed.version, installed.update_mode)}
                      </span>
                      {selectedUpdate && (
                        <span className="text-[0.62rem] font-medium text-amber-500">
                          {copy.updateAvailable(selectedUpdate.plan?.version)}
                        </span>
                      )}
                      <Button onClick={() => void planManagedAction('uninstall')} size="sm" variant="outline">
                        {copy.uninstall}
                      </Button>
                      <Button onClick={() => void planManagedAction('update')} size="sm">
                        {selectedUpdate ? copy.reviewUpdate : copy.checkSkill}
                      </Button>
                    </>
                  ) : (
                    <Button onClick={() => void planManagedAction('install')} size="sm">
                      {copy.install}
                    </Button>
                  )}
                </div>
              </section>
            )}
          </div>
        </DetailColumn>
      </MasterDetail>

      {prepared && (
        <div
          aria-label={copy.prepareTitle}
          aria-modal="true"
          className="shadow-nous absolute inset-6 z-20 overflow-auto border border-(--stroke-nous) bg-background p-5"
          role="dialog"
        >
          <h2 className="font-mono text-sm">{copy.prepareTitle}</h2>
          <p className="mt-1 text-xs text-muted-foreground">{copy.prepareNotice}</p>
          <WisdomReviewTables professionalism={prepared.professionalism_check} />
          <label className="mt-4 block text-xs" htmlFor="desktop-wisdom-description">
            {copy.ownerDescription}
          </label>
          <Textarea
            className="mt-1 min-h-20 w-full resize-y text-xs"
            id="desktop-wisdom-description"
            maxLength={4096}
            onChange={event => setDescription(event.target.value)}
            value={description}
          />
          <h3 className="mt-5 text-xs font-medium">{copy.systemSpecification}</h3>
          {specification && (
            <div className="mt-3">
              <WisdomSystemSpecificationEditor
                disabled={busy === prepared.local_draft_id}
                onChange={setSpecification}
                value={specification}
              />
            </div>
          )}
          {specificationError && (
            <div className="mt-3 text-xs text-destructive" role="alert">
              {specificationError}
            </div>
          )}
          <p className="mt-3 break-all font-mono text-[0.62rem] text-muted-foreground">
            {copy.localOverlay}: {prepared.overlay_path}
          </p>
          <div className="mt-4 flex justify-end gap-2">
            <Button
              onClick={() => {
                setPrepared(null)
                setSpecification(null)
              }}
              size="sm"
              variant="outline"
            >
              {copy.cancel}
            </Button>
            <Button
              disabled={busy === prepared.local_draft_id || Boolean(specificationError)}
              onClick={() => void submit()}
              size="sm"
            >
              {busy === prepared.local_draft_id ? copy.submitting : copy.submit}
            </Button>
          </div>
        </div>
      )}

      {review && (
        <div
          aria-label={copy.ownerReviewExact}
          aria-modal="true"
          className="shadow-nous absolute inset-6 z-20 overflow-auto border border-(--stroke-nous) bg-background p-5"
          role="dialog"
        >
          <h2 className="font-mono text-sm">{review.draft.slug}</h2>
          <p className="mt-1 text-xs text-muted-foreground">{copy.readEvery}</p>
          {reviewCanEdit && <p className="mt-2 text-xs leading-5 text-muted-foreground">{copy.editReview}</p>}
          {reviewDirty && (
            <div className="mt-3 border-l-2 border-amber-500 pl-3 text-xs text-amber-500" role="status">
              {copy.unsavedChanges}
            </div>
          )}
          <div className="mt-3 grid gap-3 border-y border-(--ui-stroke-tertiary) py-3 text-xs">
            <div>
              <strong>{copy.ownerCopyLabel}</strong>
              {reviewCanEdit ? (
                <Textarea
                  aria-label={copy.editOwnerDescription}
                  className="mt-2 min-h-20 w-full resize-y text-xs leading-relaxed"
                  disabled={busy === review.draft.id}
                  maxLength={4096}
                  onChange={event => setReviewDescription(event.target.value)}
                  value={reviewDescription}
                />
              ) : (
                <p className="mt-1 whitespace-pre-wrap text-muted-foreground">
                  {review.draft.authorDescription || copy.noDescription}
                </p>
              )}
            </div>
            <div>
              <strong>{copy.serverFactsLabel}</strong>
              <WisdomReviewTables
                professionalism={review.draft.professionalism_check}
                security={review.draft.security_check}
              />
              <pre className="mt-1 max-h-48 overflow-auto whitespace-pre-wrap text-muted-foreground">
                {JSON.stringify(
                  {
                    verdict: review.draft.scanVerdict,
                    scan: review.draft.scan,
                    explanation: review.draft.explanation
                  },
                  null,
                  2
                )}
              </pre>
            </div>
          </div>
          <div className="my-3 grid gap-1 break-all font-mono text-[0.62rem]">
            <strong className="font-sans text-xs">{copy.reviewedHashes}</strong>
            <span>content {review.hashes.content}</span>
            <span>author description {review.hashes.author_description}</span>
            <span>package manifest {review.hashes.package_manifest}</span>
          </div>
          {review.files.map(file => (
            <WisdomFileEditor
              disabled={!reviewCanEdit || busy === review.draft.id}
              file={file}
              key={`${review.draft.id}:${file.path}`}
              onChange={value => setReviewFiles(current => ({ ...current, [file.path]: value }))}
              value={reviewFiles[file.path] ?? file.content_utf8}
            />
          ))}
          <div className="mt-4 flex justify-end gap-2">
            <Button onClick={closeReview} size="sm" variant="outline">
              {copy.close}
            </Button>
            {reviewCanEdit && reviewDirty && (
              <>
                <Button disabled={busy === review.draft.id} onClick={resetReviewEdits} size="sm" variant="outline">
                  {copy.resetChanges}
                </Button>
                <Button
                  disabled={busy === review.draft.id || Boolean(reviewManifestError)}
                  onClick={() => void saveReviewRevision()}
                  size="sm"
                >
                  {busy === review.draft.id ? copy.savingRevision : copy.saveAndRescan}
                </Button>
              </>
            )}
            <Button
              disabled={busy === review.draft.id}
              onClick={async () => {
                setBusy(review.draft.id)

                try {
                  await decideWisdomDraft(review.draft.id, 'decline', profile)
                  await refreshContributionData()
                  closeReview()
                } catch (error) {
                  notifyError(error, 'Wisdom decline failed')
                } finally {
                  setBusy(null)
                }
              }}
              size="sm"
              variant="outline"
            >
              {copy.decline}
            </Button>
            <Button disabled={busy === review.draft.id || reviewDirty} onClick={() => void approve()} size="sm">
              {busy === review.draft.id ? copy.publishing : copy.approve}
            </Button>
          </div>
        </div>
      )}

      {actionPlan && (
        <div
          aria-label="Verified managed action plan"
          className="absolute inset-6 z-30 flex min-h-0 flex-col overflow-hidden border border-amber-600/50 bg-background p-5 shadow-xl"
          role="dialog"
        >
          <h2 className="shrink-0 font-mono text-sm">{copy.confirmAction(actionPlan.action)}</h2>
          <div
            aria-label={copy.confirmAction(actionPlan.action)}
            className="mt-3 min-h-0 flex-1 overflow-auto"
            role="region"
          >
            <pre className="whitespace-pre-wrap text-[0.65rem]">{JSON.stringify(actionPlan, null, 2)}</pre>
          </div>
          <div className="shrink-0 pt-3">
            {actionPlan.action === 'install' && (
              <div className="max-w-sm">
                <label className="mb-1 block text-xs font-medium" id="desktop-wisdom-plan-update-mode-label">
                  {copy.updateModeLabel}
                </label>
                <Select
                  disabled={busy === 'install-mode'}
                  onValueChange={value => void replanInstallUpdateMode(value)}
                  value={installUpdateMode || 'DEFAULT'}
                >
                  <SelectTrigger
                    aria-describedby="desktop-wisdom-plan-update-mode-help"
                    aria-labelledby="desktop-wisdom-plan-update-mode-label"
                    className="w-full"
                    size="sm"
                  >
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="DEFAULT">{copy.updateModeDefault}</SelectItem>
                    <SelectItem value="MANUAL">{copy.updateModeManual}</SelectItem>
                    <SelectItem value="AUTO_WITH_NOTICE">{copy.updateModeAutomatic}</SelectItem>
                    <SelectItem value="REQUIRED">{copy.updateModeRequired}</SelectItem>
                  </SelectContent>
                </Select>
                <p className="mt-1 text-[0.65rem] text-muted-foreground" id="desktop-wisdom-plan-update-mode-help">
                  {copy.updateModeHelp}
                </p>
              </div>
            )}
            {actionPlan.state === 'current' && <p className="text-xs">{copy.alreadyCurrent}</p>}
            {actionPlan.compatibility && actionPlan.compatibility.outcome !== 'compatible' && (
              <label className="mt-3 flex gap-2 text-xs">
                <input
                  checked={acceptPartial}
                  onChange={event => setAcceptPartial(event.target.checked)}
                  type="checkbox"
                />
                {copy.acceptCompatibility}
              </label>
            )}
            {(actionPlan.sensitive_expansion?.length ?? 0) > 0 && (
              <label className="mt-2 flex gap-2 text-xs">
                <input
                  checked={acceptSensitive}
                  onChange={event => setAcceptSensitive(event.target.checked)}
                  type="checkbox"
                />
                {copy.acceptSensitive}
              </label>
            )}
            {actionPlan.modified && actionPlan.update_mode !== 'REQUIRED' && (
              <label className="mt-2 flex gap-2 text-xs">
                <input
                  checked={preserveModified}
                  onChange={event => setPreserveModified(event.target.checked)}
                  type="checkbox"
                />
                {copy.preserveModified}
              </label>
            )}
            <div className="mt-4 flex justify-end gap-2">
              <Button
                onClick={() => {
                  setActionPlan(null)
                  setActionPlanReference(null)
                }}
                size="sm"
                variant="outline"
              >
                {t.common.cancel}
              </Button>
              {actionPlan.state !== 'current' && (
                <Button disabled={busy === 'install-mode'} onClick={() => void applyManagedAction()} size="sm">
                  {busy === 'install-mode' ? copy.planningInstall : copy.confirmAction(actionPlan.action)}
                </Button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
