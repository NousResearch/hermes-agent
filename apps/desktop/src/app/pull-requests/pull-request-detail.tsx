import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { EmptyState } from '@/components/ui/empty-state'
import { Loader } from '@/components/ui/loader'
import type { HermesGithubPullRequestDetail } from '@/global'

interface Props {
  detail?: HermesGithubPullRequestDetail
  loading: boolean
  error: boolean
  onRetry: () => void
  onBack: () => void
  onOpen: (url: string) => void
  onCopy: (url: string) => void
  copy: Record<string, string>
}

export function PullRequestDetail({ detail, loading, error, onRetry, onBack, onOpen, onCopy, copy }: Props) {
  if (loading) {
    return (
      <div className="grid h-full place-items-center">
        <Loader label={copy.loadingDetails} />
      </div>
    )
  }

  if (error) {
    return (
      <div className="grid h-full place-items-center text-center">
        <div>
          <EmptyState title={copy.detailFailed} />
          <Button onClick={onRetry} size="sm" variant="secondary">
            {copy.retry}
          </Button>
        </div>
      </div>
    )
  }

  if (!detail) {
    return <EmptyState title={copy.title} />
  }
  const state = detail.isDraft
    ? copy.draft
    : detail.state === 'MERGED'
      ? copy.merged
      : detail.state === 'CLOSED'
        ? copy.closed
        : copy.open

  return (
    <article className="h-full overflow-y-auto px-5 py-4">
      <Button className="mb-3 md:hidden" onClick={onBack} size="sm" variant="ghost">
        <Codicon name="arrow-left" />
        {copy.back}
      </Button>
      <div className="text-xs text-(--ui-text-secondary)">
        {detail.repository} #{detail.number}
      </div>
      <h1 className="mt-1 text-xl font-semibold leading-tight text-(--ui-text-primary)">{detail.title}</h1>
      <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-(--ui-text-secondary)">
        <span>{state}</span>
        {detail.author && <span>@{detail.author.login}</span>}
        {detail.labels.map(label => (
          <span key={label.name}>{label.name}</span>
        ))}
      </div>
      <div className="mt-2 text-xs text-(--ui-text-tertiary)">
        {copy.updated}: {new Date(detail.updatedAt).toLocaleString()} · {copy.created}:{' '}
        {new Date(detail.createdAt).toLocaleString()}
      </div>
      <div className="mt-4 flex flex-wrap gap-2">
        <Button onClick={() => onOpen(detail.url)} size="sm" variant="secondary">
          <Codicon name="link-external" />
          {copy.openGithub}
        </Button>
        <Button onClick={() => onCopy(detail.url)} size="sm" variant="ghost">
          <Codicon name="copy" />
          {copy.copyUrl}
        </Button>
      </div>
      <dl className="mt-5 grid grid-cols-2 gap-x-5 gap-y-3 text-xs sm:grid-cols-3">
        <div>
          <dt className="text-(--ui-text-tertiary)">{copy.changedFiles}</dt>
          <dd className="mt-0.5 font-medium">{detail.changedFiles}</dd>
        </div>
        <div>
          <dt className="text-(--ui-text-tertiary)">{copy.additions}</dt>
          <dd className="mt-0.5 font-medium text-green-600">+{detail.additions}</dd>
        </div>
        <div>
          <dt className="text-(--ui-text-tertiary)">{copy.deletions}</dt>
          <dd className="mt-0.5 font-medium text-red-600">-{detail.deletions}</dd>
        </div>
        <div>
          <dt className="text-(--ui-text-tertiary)">{copy.reviewDecision}</dt>
          <dd className="mt-0.5 font-medium">{detail.reviewDecision ?? '—'}</dd>
        </div>
        <div>
          <dt className="text-(--ui-text-tertiary)">Merge state</dt>
          <dd className="mt-0.5 font-medium">{detail.mergeStateStatus ?? '—'}</dd>
        </div>
        <div>
          <dt className="text-(--ui-text-tertiary)">{copy.checks}</dt>
          <dd className="mt-0.5 font-medium">
            {detail.checks.passed}/{detail.checks.total} · {detail.checks.failed} failed
          </dd>
        </div>
      </dl>
      <div className="mt-5 overflow-wrap-anywhere text-xs text-(--ui-text-secondary)">
        <Codicon name="git-merge" /> {detail.headRefName} → {detail.baseRefName}
      </div>
      <div className="mt-5 whitespace-pre-wrap break-words text-sm leading-6 text-(--ui-text-secondary)">
        {detail.body || '—'}
      </div>
    </article>
  )
}
