import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useState } from 'react'

import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import {
  getStudyProfile,
  getStudyReviewDue,
  getStudyReviewQueue,
  getStudyReviewStats,
  updateStudyProfile
} from '@/hermes'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import {
  $allTags,
  $filteredDueItems,
  $reviewCompletedToday,
  $reviewDueItems,
  $reviewError,
  $reviewLevelFilter,
  $reviewLoadState,
  $reviewProfile,
  $reviewStats,
  $reviewSubjectFilter,
  $reviewTab,
  resetReviewState
} from '@/store/study-review'
import type { StudyLearningItem, StudyReviewItem } from '@/types/hermes'

const LEVEL_COLORS: Record<number, string> = {
  0: 'bg-red-500/10 text-red-600 border-red-500/30',
  1: 'bg-orange-500/10 text-orange-600 border-orange-500/30',
  2: 'bg-yellow-500/10 text-yellow-600 border-yellow-500/30',
  3: 'bg-green-500/10 text-green-600 border-green-500/30',
  4: 'bg-blue-500/10 text-blue-600 border-blue-500/30',
  5: 'bg-purple-500/10 text-purple-600 border-purple-500/30'
}

function LevelBadge({ level }: { level: number }) {
  return (
    <span
      className={cn(
        'inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium',
        LEVEL_COLORS[level] ?? 'bg-muted text-muted-foreground border-muted'
      )}
    >
      Lv.{level}
    </span>
  )
}

function formatDate(value: string | null, fallback: string): string {
  if (!value) return fallback
  return value.slice(0, 10)
}

function DueItemCard({ item }: { item: StudyReviewItem }) {
  const { t } = useI18n()
  return (
    <article className="rounded-2xl border bg-card/60 p-4 shadow-sm transition-colors hover:bg-card/80">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <h5 className="text-base font-semibold">{item.title}</h5>
        <div className="flex items-center gap-2">
          <LevelBadge level={item.review_level} />
          <span className="text-xs text-muted-foreground">{item.difficulty}</span>
        </div>
      </div>
      <div className="mt-2 text-xs text-muted-foreground">
        {t.study.lastReviewed}: {formatDate(item.last_reviewed_at, t.study.neverReviewed)}
      </div>
      <div className="mt-2 flex flex-wrap gap-1.5">
        {item.concepts?.map(c => (
          <span
            className="rounded-md bg-accent/50 px-2 py-0.5 text-xs text-muted-foreground"
            key={c}
          >
            {c}
          </span>
        ))}
      </div>
    </article>
  )
}

function LearningItemCard({ item }: { item: StudyLearningItem }) {
  const { t } = useI18n()
  return (
    <article className="rounded-2xl border bg-card/60 p-4 shadow-sm">
      <h5 className="text-base font-semibold">{item.title}</h5>
      {item.learning_state && (
        <span className="mt-1 inline-block text-xs text-muted-foreground">
          {t.study.learningState}: {item.learning_state}
        </span>
      )}
      {item.prerequisites && item.prerequisites.length > 0 && (
        <div className="mt-1 text-xs text-muted-foreground">
          {t.study.prerequisites}: {item.prerequisites.join(', ')}
        </div>
      )}
    </article>
  )
}

function StatsBar({ label, count, max, color }: { label: string; count: number; max: number; color: string }) {
  const pct = max > 0 ? (count / max) * 100 : 0
  return (
    <div className="flex items-center gap-2">
      <span className="w-8 text-right text-xs font-medium text-muted-foreground">{label}</span>
      <div className="h-3 flex-1 rounded-full bg-muted">
        <div
          className={cn('h-full rounded-full transition-all', color)}
          style={{ width: `${Math.max(2, pct)}%` }}
        />
      </div>
      <span className="w-6 text-right text-xs text-muted-foreground">{count}</span>
    </div>
  )
}

export function ReviewView() {
  const { t } = useI18n()
  const dueItems = useStore($reviewDueItems)
  const filteredItems = useStore($filteredDueItems)
  const stats = useStore($reviewStats)
  const loadState = useStore($reviewLoadState)
  const error = useStore($reviewError)
  const profile = useStore($reviewProfile)
  const completed = useStore($reviewCompletedToday)
  const subjectFilter = useStore($reviewSubjectFilter)
  const levelFilter = useStore($reviewLevelFilter)
  const activeTab = useStore($reviewTab)
  const allTags = useStore($allTags)

  const [queue, setQueue] = useState<{ newConcepts: StudyLearningItem[]; newExamples: StudyLearningItem[] }>({
    newConcepts: [],
    newExamples: []
  })
  const [editingQuota, setEditingQuota] = useState(false)
  const [quotaInput, setQuotaInput] = useState('')

  const loadData = useCallback(async () => {
    $reviewLoadState.set('loading')
    $reviewError.set(null)
    try {
      const [dueRes, statsRes, profileRes] = await Promise.all([
        getStudyReviewDue({
          subject: subjectFilter ?? undefined,
          level: levelFilter ?? undefined
        }),
        getStudyReviewStats(),
        getStudyProfile()
      ])
      $reviewDueItems.set(dueRes.due)
      $reviewStats.set(statsRes)
      $reviewProfile.set(profileRes)
      $reviewLoadState.set('ready')
    } catch (err) {
      $reviewError.set(err instanceof Error ? err.message : String(err))
      $reviewLoadState.set('error')
    }
  }, [subjectFilter, levelFilter])

  const loadQueue = useCallback(async () => {
    try {
      const q = await getStudyReviewQueue()
      setQueue({ newConcepts: q.new_concepts, newExamples: q.new_examples })
    } catch {
      // queue is non-critical, silently fail
    }
  }, [])

  useEffect(() => {
    void loadData()
    return () => resetReviewState()
  }, [loadData])

  useEffect(() => {
    if (activeTab === 'queue') void loadQueue()
  }, [activeTab, loadQueue])

  const handleEditQuota = useCallback(() => {
    setQuotaInput(String(profile.daily_review_limit))
    setEditingQuota(true)
  }, [profile.daily_review_limit])

  const handleSaveQuota = useCallback(async () => {
    const value = parseInt(quotaInput, 10)
    if (Number.isFinite(value) && value >= 1 && value <= 200) {
      try {
        await updateStudyProfile({ daily_review_limit: value })
        await loadData()
      } catch {
        // keep current value on failure
      }
    }
    setEditingQuota(false)
  }, [quotaInput, loadData])

  const handleQuotaKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter') void handleSaveQuota()
      if (e.key === 'Escape') setEditingQuota(false)
    },
    [handleSaveQuota]
  )

  const maxLevel = stats ? Math.max(...Object.values(stats.by_level), 1) : 1
  const progress = profile.daily_review_limit > 0
    ? Math.round((completed / profile.daily_review_limit) * 100)
    : 0

  return (
    <div className="flex h-full min-h-0 gap-0">
      <aside className="flex w-48 shrink-0 flex-col gap-4 overflow-y-auto border-r border-border/40 p-4">
        <div>
          <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            {t.study.filterBySubject}
          </div>
          <div className="flex flex-wrap gap-1">
            <button
              className={cn(
                'rounded-md px-2 py-1 text-xs transition-colors',
                subjectFilter === null
                  ? 'bg-primary/10 text-primary'
                  : 'text-muted-foreground hover:bg-accent/50 hover:text-foreground'
              )}
              onClick={() => $reviewSubjectFilter.set(null)}
              type="button"
            >
              {t.study.allLevels}
            </button>
            {allTags.map(tag => (
              <button
                className={cn(
                  'rounded-md px-2 py-1 text-xs transition-colors',
                  subjectFilter === tag
                    ? 'bg-primary/10 text-primary'
                    : 'text-muted-foreground hover:bg-accent/50 hover:text-foreground'
                )}
                key={tag}
                onClick={() => $reviewSubjectFilter.set(subjectFilter === tag ? null : tag)}
                type="button"
              >
                {tag}
              </button>
            ))}
          </div>
        </div>

        <div>
          <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            {t.study.filterByLevel}
          </div>
          <div className="flex flex-wrap gap-1">
            <button
              className={cn(
                'rounded-md px-2 py-1 text-xs transition-colors',
                levelFilter === null
                  ? 'bg-primary/10 text-primary'
                  : 'text-muted-foreground hover:bg-accent/50 hover:text-foreground'
              )}
              onClick={() => $reviewLevelFilter.set(null)}
              type="button"
            >
              {t.study.allLevels}
            </button>
            {[0, 1, 2, 3, 4, 5].map(lv => (
              <button
                className={cn(
                  'rounded-md px-2 py-1 text-xs transition-colors',
                  LEVEL_COLORS[lv],
                  levelFilter === lv ? 'ring-1 ring-current' : 'opacity-60 hover:opacity-100'
                )}
                key={lv}
                onClick={() => $reviewLevelFilter.set(levelFilter === lv ? null : lv)}
                type="button"
              >
                {lv}
              </button>
            ))}
          </div>
        </div>

        <div>
          <div className="mb-1 flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              {t.study.dailyQuota}
            </span>
            {!editingQuota && (
              <button
                className="rounded p-0.5 text-muted-foreground/60 transition-colors hover:bg-accent/50 hover:text-foreground"
                onClick={handleEditQuota}
                type="button"
              >
                <svg className="size-3.5" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} viewBox="0 0 24 24">
                  <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
                  <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
                </svg>
              </button>
            )}
          </div>
          {editingQuota ? (
            <div className="flex items-center gap-1">
              <input
                autoFocus
                className="h-7 w-16 rounded-md border bg-background px-2 text-sm"
                max={200}
                min={1}
                onChange={e => setQuotaInput(e.target.value)}
                onKeyDown={handleQuotaKeyDown}
                type="number"
                value={quotaInput}
              />
              <button
                className="rounded-md px-2 py-1 text-xs font-medium text-primary transition-colors hover:bg-primary/10"
                onClick={() => void handleSaveQuota()}
                type="button"
              >
                OK
              </button>
            </div>
          ) : (
            <div className="text-sm">
              <span className="font-medium">{completed}</span>
              <span className="text-muted-foreground"> {t.study.of} </span>
              <span className="font-medium">{profile.daily_review_limit}</span>
              <span className="ml-2 text-xs text-muted-foreground">
                {t.study.reviewed}
              </span>
            </div>
          )}
          <div className="mt-1 h-1.5 w-full rounded-full bg-muted">
            <div
              className="h-full rounded-full bg-primary transition-all"
              style={{ width: `${Math.min(100, progress)}%` }}
            />
          </div>
        </div>
      </aside>

      <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
        <div className="flex gap-1 border-b border-border/40 px-6 pt-2">
          {(['due', 'queue', 'stats'] as const).map(tab => (
            <button
              className={cn(
                'border-b-2 px-4 py-2 text-sm font-medium transition-colors',
                activeTab === tab
                  ? 'border-primary text-primary'
                  : 'border-transparent text-muted-foreground hover:text-foreground'
              )}
              key={tab}
              onClick={() => $reviewTab.set(tab)}
              type="button"
            >
              {tab === 'due' ? t.study.dueReviews : tab === 'queue' ? t.study.newMaterial : t.study.stats}
            </button>
          ))}
        </div>

        <div className="flex-1 overflow-y-auto p-6">
          {loadState === 'loading' && <PageLoader label={t.study.loading} />}

          {error && (
            <div className="rounded-2xl border border-destructive/40 bg-destructive/10 p-5 text-sm">
              <div className="font-semibold">{error}</div>
              <Button className="mt-4" onClick={() => void loadData()} size="sm" variant="secondary">
                {t.study.retry}
              </Button>
            </div>
          )}

          {loadState === 'ready' && activeTab === 'due' && (
            <>
              {filteredItems.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-16 text-center">
                  <div className="text-4xl">🎉</div>
                  <h3 className="mt-3 text-lg font-semibold">{t.study.noDueReviews}</h3>
                  <p className="mt-1 text-sm text-muted-foreground">{t.study.noDueReviewsDesc}</p>
                </div>
              ) : (
                <>
                  <div className="mb-3 text-sm text-muted-foreground">
                    {filteredItems.length} {t.study.dueReviews.toLowerCase()}
                    {filteredItems.length !== dueItems.length && (
                      <span className="ml-1">({dueItems.length} total)</span>
                    )}
                  </div>
                  <div className="space-y-3">
                    {filteredItems.map(item => (
                      <DueItemCard item={item} key={item.path} />
                    ))}
                  </div>
                </>
              )}
            </>
          )}

          {loadState === 'ready' && activeTab === 'queue' && (
            <div className="space-y-6">
              <section>
                <h3 className="text-lg font-semibold">{t.study.newConcepts}</h3>
                {queue.newConcepts.length === 0 ? (
                  <p className="mt-2 text-sm text-muted-foreground">{t.study.noNewConcepts}</p>
                ) : (
                  <div className="mt-3 space-y-3">
                    {queue.newConcepts.map(item => (
                      <LearningItemCard item={item} key={item.path} />
                    ))}
                  </div>
                )}
              </section>
              <section>
                <h3 className="text-lg font-semibold">{t.study.newExamples}</h3>
                {queue.newExamples.length === 0 ? (
                  <p className="mt-2 text-sm text-muted-foreground">{t.study.noNewExamples}</p>
                ) : (
                  <div className="mt-3 space-y-3">
                    {queue.newExamples.map(item => (
                      <LearningItemCard item={item} key={item.path} />
                    ))}
                  </div>
                )}
              </section>
            </div>
          )}

          {loadState === 'ready' && activeTab === 'stats' && stats && (
            <div className="space-y-6">
              <div className="grid gap-4 sm:grid-cols-3">
                <div className="rounded-2xl border bg-card/60 p-4 text-center">
                  <div className="text-3xl font-bold">{stats.progress}%</div>
                  <div className="mt-1 text-xs text-muted-foreground">{t.study.masteryProgress}</div>
                </div>
                <div className="rounded-2xl border bg-card/60 p-4 text-center">
                  <div className="text-3xl font-bold">{stats.review_streak}</div>
                  <div className="mt-1 text-xs text-muted-foreground">{t.study.streak} ({t.study.days})</div>
                </div>
                <div className="rounded-2xl border bg-card/60 p-4 text-center">
                  <div className="text-3xl font-bold">{stats.due_count}</div>
                  <div className="mt-1 text-xs text-muted-foreground">{t.study.dueReviews}</div>
                </div>
              </div>

              <div>
                <h3 className="mb-3 text-lg font-semibold">{t.study.reviewLevel}</h3>
                <div className="space-y-2">
                  {[0, 1, 2, 3, 4, 5].map(lv => (
                    <StatsBar
                      color={lv === 0 ? 'bg-red-500' : lv === 1 ? 'bg-orange-500' : lv === 2 ? 'bg-yellow-500' : lv === 3 ? 'bg-green-500' : lv === 4 ? 'bg-blue-500' : 'bg-purple-500'}
                      count={stats.by_level[String(lv)] ?? 0}
                      key={lv}
                      label={`Lv.${lv}`}
                      max={maxLevel}
                    />
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
