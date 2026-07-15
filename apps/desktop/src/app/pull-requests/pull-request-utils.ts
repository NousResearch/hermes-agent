import type { HermesGithubPullRequestSummary } from '@/global'

export function matchesPullRequest(item: HermesGithubPullRequestSummary, query: string): boolean {
  const needle = query.trim().toLowerCase()

  if (!needle) {
    return true
  }

  return [
    item.repository,
    item.title,
    String(item.number),
    item.author?.login ?? '',
    ...item.labels.map(label => label.name)
  ].some(value => value.toLowerCase().includes(needle))
}

export function relativeTime(value: string, locale?: string): string {
  const timestamp = Date.parse(value)

  if (!Number.isFinite(timestamp)) {
    return '—'
  }
  const seconds = Math.round((timestamp - Date.now()) / 1000)
  const abs = Math.abs(seconds)
  const [amount, unit] =
    abs < 60
      ? [seconds, 'second']
      : abs < 3600
        ? [Math.round(seconds / 60), 'minute']
        : abs < 86400
          ? [Math.round(seconds / 3600), 'hour']
          : [Math.round(seconds / 86400), 'day']

  return new Intl.RelativeTimeFormat(locale, { numeric: 'auto' }).format(amount, unit as Intl.RelativeTimeFormatUnit)
}
