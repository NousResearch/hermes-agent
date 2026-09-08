import { act, cleanup, render, screen } from '@testing-library/react'
import { memo, useEffect, useState } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import { PageLoader } from '@/components/page-loader'
import { PaneVisibleContext } from '@/components/pane-shell/pane-visibility'
import { createPluginContext } from '@/contrib/plugin'

import {
  SYSTEM_ACTIVITY_AREA,
  type SystemActivityContribution,
  type SystemActivityProps,
  SystemActivitySlot
} from './system-activity'

const disposers: Array<() => void> = []
const ctx = createPluginContext('activity-test', dispose => disposers.push(dispose))

afterEach(() => {
  cleanup()
  disposers.splice(0).forEach(dispose => dispose())
  vi.restoreAllMocks()
})

it('keeps the region status host-owned while a scoped renderer mounts, changes kind and disposes', () => {
  const stopped = vi.fn()

  function CustomActivity({ activity, placement }: SystemActivityProps) {
    const [identity] = useState('same mount')
    useEffect(() => stopped, [])

    return (
      <span data-testid="custom">
        {activity}:{placement}:{identity}
      </span>
    )
  }

  const renderer: SystemActivityContribution = { render: CustomActivity }
  const view = render(<PageLoader label="Loading saved files" />)
  expect(screen.getByRole('status', { name: 'Loading saved files' }).querySelector('svg')).not.toBeNull()

  let remove = () => {}
  act(() => {
    remove = ctx.register({ area: SYSTEM_ACTIVITY_AREA, id: 'mark', data: renderer })
  })
  expect(screen.getByTestId('custom').textContent).toBe('loading:region:same mount')
  expect(screen.getAllByRole('status')).toHaveLength(1)
  expect(view.container.querySelector('svg')).toBeNull()

  view.rerender(<PageLoader activity="connecting" label="Connecting the server" />)
  expect(screen.getByTestId('custom').textContent).toBe('connecting:region:same mount')
  expect(stopped).not.toHaveBeenCalled()

  act(remove)
  expect(stopped).toHaveBeenCalledOnce()
  expect(screen.getByRole('status', { name: 'Connecting the server' }).querySelector('svg')).not.toBeNull()
})

it('renders a memoized plugin component through the same status host', () => {
  const MemoizedActivity = memo(function Activity({ activity }: SystemActivityProps) {
    return <span>Memoized {activity}</span>
  })

  ctx.register({
    area: SYSTEM_ACTIVITY_AREA,
    id: 'memoized',
    data: { render: MemoizedActivity } satisfies SystemActivityContribution
  })

  render(<PageLoader label="Loading saved files" />)
  expect(screen.getByRole('status', { name: 'Loading saved files' }).textContent).toBe('Memoized loading')
  expect(screen.getAllByRole('status')).toHaveLength(1)
})

it('honors intentional omission beside progress but restores native feedback after renderer failure', () => {
  const error = vi.spyOn(console, 'error').mockImplementation(() => undefined)

  const view = render(
    <SystemActivitySlot activity="processing" fallback={<span>Native mark</span>} hasProgress placement="region" />
  )

  let remove = () => {}
  act(() => {
    remove = ctx.register({
      area: SYSTEM_ACTIVITY_AREA,
      id: 'mark',
      data: {
        render: ({ hasProgress }) => (hasProgress ? null : <span>Custom mark</span>)
      } satisfies SystemActivityContribution
    })
  })
  expect(view.container.textContent).toBe('')
  act(remove)
  expect(screen.getByText('Native mark')).toBeTruthy()

  act(() => {
    ctx.register({
      area: SYSTEM_ACTIVITY_AREA,
      id: 'broken',
      data: {
        render: () => {
          throw new Error('Broken custom renderer')
        }
      } satisfies SystemActivityContribution
    })
  })
  expect(screen.getByText('Native mark')).toBeTruthy()
  expect(error).toHaveBeenCalled()

  act(() => {
    ctx.register({
      area: SYSTEM_ACTIVITY_AREA,
      id: 'broken',
      data: { render: () => <span>Recovered custom mark</span> } satisfies SystemActivityContribution
    })
  })
  expect(screen.getByText('Recovered custom mark')).toBeTruthy()
  expect(screen.queryByText('Native mark')).toBeNull()
})

it('passes Hermes pane and window visibility into the renderer without changing operation state', () => {
  vi.spyOn(window.document, 'hasFocus').mockReturnValue(true)
  const stopped = vi.fn()

  function ObservableActivity({ activity, paused }: SystemActivityProps) {
    useEffect(() => stopped, [])

    return (
      <span>
        {activity}:{paused ? 'still' : 'moving'}
      </span>
    )
  }

  ctx.register({
    area: SYSTEM_ACTIVITY_AREA,
    id: 'mark',
    data: {
      render: ObservableActivity
    } satisfies SystemActivityContribution
  })

  const host = (visible: boolean) => (
    <PaneVisibleContext value={visible}>
      <SystemActivitySlot activity="loading" fallback={null} placement="inline" />
    </PaneVisibleContext>
  )

  const view = render(host(true))
  expect(screen.getByText('loading:moving')).toBeTruthy()
  view.rerender(host(false))
  expect(screen.getByText('loading:still')).toBeTruthy()
  view.rerender(host(true))
  act(() => window.dispatchEvent(new Event('blur')))
  expect(screen.getByText('loading:still')).toBeTruthy()
  act(() => window.dispatchEvent(new Event('focus')))
  expect(screen.getByText('loading:moving')).toBeTruthy()
  expect(stopped).not.toHaveBeenCalled()
  view.unmount()
  expect(stopped).toHaveBeenCalledOnce()
})
