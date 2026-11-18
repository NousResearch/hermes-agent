import type { VariantProps } from 'class-variance-authority'
import type { ComponentProps } from 'react'

import { ActionStatus } from '@/components/ui/action-status'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge, type badgeVariants } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { EmptyState } from '@/components/ui/empty-state'
import { ErrorBanner, ErrorState } from '@/components/ui/error-state'
import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { Loader, LOADER_TYPES } from '@/components/ui/loader'
import { Progress } from '@/components/ui/progress'
import { CountSkeleton, Skeleton } from '@/components/ui/skeleton'
import { StatusPulse } from '@/components/ui/status-pulse'
import { AlertTriangle } from '@/lib/icons'

import { everyOf, type GalleryEntry } from './types'

type BadgeVariant = NonNullable<VariantProps<typeof badgeVariants>['variant']>
type BadgeSize = NonNullable<VariantProps<typeof badgeVariants>['size']>
type AlertVariant = NonNullable<ComponentProps<typeof Alert>['variant']>

const BADGE_VARIANTS = everyOf<BadgeVariant>()('default', 'muted', 'success', 'warn', 'destructive', 'outline', 'solid')
const BADGE_SIZES = everyOf<BadgeSize>()('default', 'xs', 'overlay')
const ALERT_VARIANTS = everyOf<AlertVariant>()('default', 'warning', 'success', 'destructive')
const PROGRESS_SIZES = everyOf<NonNullable<ComponentProps<typeof Progress>['size']>>()('sm', 'default', 'lg')
const ACTION_STATES = everyOf<ComponentProps<typeof ActionStatus>['state']>()('idle', 'saving', 'done')

export const FEEDBACK: readonly GalleryEntry[] = [
  {
    title: 'Badge',
    source: 'components/ui/badge.tsx',
    use: 'A short status or count beside a label.',
    group: 'feedback',
    examples: [
      ...BADGE_VARIANTS.map(variant => ({ label: variant, render: () => <Badge variant={variant}>Running</Badge> })),
      ...BADGE_SIZES.map(size => ({
        label: `size ${size}`,
        render: () => (
          <Badge size={size} variant="solid">
            {size === 'overlay' ? '3' : 'New'}
          </Badge>
        )
      }))
    ]
  },
  {
    title: 'Alert',
    source: 'components/ui/alert.tsx',
    use: 'A notice inside a page about the page itself.',
    group: 'feedback',
    examples: ALERT_VARIANTS.map(variant => ({
      label: variant,
      render: () => (
        <Alert variant={variant}>
          <AlertTriangle />
          <AlertTitle>Machine offline</AlertTitle>
          <AlertDescription>Routines on it wait until it is back.</AlertDescription>
        </Alert>
      )
    }))
  },
  {
    title: 'EmptyState',
    source: 'components/ui/empty-state.tsx',
    use: 'A list or page with nothing in it yet, and the one action that starts it.',
    group: 'feedback',
    examples: [
      {
        label: 'with action',
        render: () => (
          <EmptyState
            action={<Button variant="secondary">New routine</Button>}
            description="Routines run tasks on a schedule."
            title="No routines yet"
          />
        )
      }
    ]
  },
  {
    title: 'ErrorState',
    source: 'components/ui/error-state.tsx',
    use: 'Something failed: say what, and offer the way forward. ErrorBanner for a line inside content.',
    group: 'feedback',
    examples: [
      {
        label: 'ErrorState',
        render: () => (
          <ErrorState description="Check the connection, then try again." title="The library could not load">
            <Button variant="secondary">Try again</Button>
          </ErrorState>
        )
      },
      { label: 'ErrorBanner', render: () => <ErrorBanner>The upload stopped at 40%.</ErrorBanner> }
    ]
  },
  {
    title: 'Skeleton',
    source: 'components/ui/skeleton.tsx',
    use: 'A placeholder shaped like the content that replaces it.',
    group: 'feedback',
    examples: [
      { label: 'Skeleton', render: () => <Skeleton className="ds-skeleton-line" /> },
      { label: 'CountSkeleton', render: () => <CountSkeleton /> }
    ]
  },
  {
    title: 'Progress',
    source: 'components/ui/progress.tsx',
    use: 'How far a known amount of work has got, or that it is underway.',
    group: 'feedback',
    examples: [
      ...PROGRESS_SIZES.map(size => ({ label: `size ${size}`, render: () => <Progress size={size} value={0.6} /> })),
      { label: 'indeterminate', render: () => <Progress indeterminate /> },
      { label: 'animated', render: () => <Progress animated indeterminate /> },
      { label: 'destructive', render: () => <Progress destructive value={0.92} /> }
    ]
  },
  {
    title: 'Loader',
    source: 'components/ui/loader.tsx',
    use: 'A page or panel waiting on something long; each curve is a named type.',
    group: 'feedback',
    examples: LOADER_TYPES.map(type => ({
      label: type,
      render: () => <Loader className="ds-loader" label={type} type={type} />
    }))
  },
  {
    title: 'GlyphSpinner',
    source: 'components/ui/glyph-spinner.tsx',
    use: 'Work in progress inside a line of text, matching the terminal spinner.',
    group: 'feedback',
    examples: [{ label: 'default', render: () => <GlyphSpinner ariaLabel="Working" /> }]
  },
  {
    title: 'StatusPulse',
    source: 'components/ui/status-pulse.tsx',
    use: 'A live status dot that pulses with a rest between plays, so an idle window can sleep.',
    group: 'feedback',
    examples: [
      { label: 'opacity', render: () => <StatusPulse className="ds-status-dot" kind="opacity" /> },
      { label: 'ping', render: () => <StatusPulse className="ds-status-dot" kind="ping" /> }
    ]
  },
  {
    title: 'ActionStatus',
    source: 'components/ui/action-status.tsx',
    use: "A button's label through idle, working and done.",
    group: 'feedback',
    examples: ACTION_STATES.map(state => ({
      label: state,
      render: () => (
        <Button variant="secondary">
          <ActionStatus busy="Saving" done="Saved" idle="Save" state={state} />
        </Button>
      )
    }))
  }
]
