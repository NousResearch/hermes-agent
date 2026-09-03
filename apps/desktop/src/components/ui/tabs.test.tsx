import { render, screen } from '@testing-library/react'
import { cleanup } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { Tabs, TabsContent, TabsList, TabsTrigger } from './tabs'

afterEach(cleanup)

describe('Tabs line variant', () => {
  it('uses flat chrome and an active underline instead of a segmented surface', () => {
    render(
      <Tabs defaultValue="overview">
        <TabsList aria-label="Task sections" variant="line">
          <TabsTrigger value="overview" variant="line">
            Overview
          </TabsTrigger>
          <TabsTrigger value="activity" variant="line">
            Activity
          </TabsTrigger>
        </TabsList>
      </Tabs>
    )

    const list = screen.getByRole('tablist', { name: 'Task sections' })
    const active = screen.getByRole('tab', { name: 'Overview' })

    expect(list.className).toContain('border-b')
    expect(list.className).toContain('bg-transparent')
    expect(active.className).toContain('after:bg-(--ui-accent)')
    expect(active.className).not.toContain('data-[state=active]:shadow-xs')
  })

  it('keeps force-mounted panels linked to their tabs', () => {
    render(
      <Tabs defaultValue="overview">
        <TabsList aria-label="Task sections" variant="line">
          <TabsTrigger value="overview" variant="line">
            Overview
          </TabsTrigger>
          <TabsTrigger value="timeline" variant="line">
            Timeline
          </TabsTrigger>
        </TabsList>
        <TabsContent forceMount value="overview">
          Overview panel
        </TabsContent>
        <TabsContent forceMount value="timeline">
          Timeline panel
        </TabsContent>
      </Tabs>
    )

    const overview = screen.getByRole('tab', { name: 'Overview' })
    const timeline = screen.getByRole('tab', { name: 'Timeline' })
    const panels = screen.getAllByRole('tabpanel', { hidden: true })

    expect(panels).toHaveLength(2)
    expect(panels.map(panel => panel.id)).toContain(overview.getAttribute('aria-controls'))
    expect(panels.map(panel => panel.id)).toContain(timeline.getAttribute('aria-controls'))
    expect(screen.getByText('Timeline panel').getAttribute('data-state')).toBe('inactive')
  })
})
