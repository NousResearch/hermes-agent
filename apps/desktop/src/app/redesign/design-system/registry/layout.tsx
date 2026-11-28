import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { FadeScroll } from '@/components/ui/fade-scroll'
import { LogView } from '@/components/ui/log-view'
import { Masonry } from '@/components/ui/masonry'
import {
  Pagination,
  PaginationButton,
  PaginationContent,
  PaginationEllipsis,
  PaginationItem,
  PaginationNext,
  PaginationPrevious
} from '@/components/ui/pagination'
import { Reel } from '@/components/ui/reel'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { Skeleton } from '@/components/ui/skeleton'
import { ResponsiveTabs } from '@/components/ui/tab-dropdown'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { TextTab, TextTabMeta } from '@/components/ui/text-tab'

import { SettingsAddRow, SettingsCard, SettingsGroup, SettingsLinkRow } from '../../../settings/settings-card'
import { PageHeader, PageTabs } from '../../page-header'

import type { GalleryEntry } from './types'

const LINES = ['Started the morning digest', 'Read 14 new messages', 'Wrote a summary', 'Sent it to your chat', 'Done']

const PAGE_TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'history', label: 'History', meta: 12 },
  { id: 'settings', label: 'Settings' }
]

function PageHeaderDemo() {
  const [tab, setTab] = useState('overview')

  return (
    <PageHeader
      actions={<Button variant="secondary">Open chat</Button>}
      description="Reads the news each morning and sends a summary."
      tabs={<PageTabs label="Agent" onChange={setTab} tabs={PAGE_TABS} value={tab} />}
      title="Research"
    />
  )
}

function ResponsiveTabsDemo() {
  const [tab, setTab] = useState('overview')

  return <ResponsiveTabs onChange={setTab} tabs={PAGE_TABS} value={tab} />
}

function TextTabsDemo() {
  const [tab, setTab] = useState('all')

  return (
    <div className="ds-row">
      {['all', 'images', 'files'].map(id => (
        <TextTab active={tab === id} key={id} onClick={() => setTab(id)}>
          {id}
          {id === 'all' && <TextTabMeta>24</TextTabMeta>}
        </TextTab>
      ))}
    </div>
  )
}

const tiles = (count: number) =>
  Array.from({ length: count }, (_, index) => <Skeleton className="ds-tile" data-size={index % 3} key={index} />)

export const LAYOUT: readonly GalleryEntry[] = [
  {
    title: 'PageHeader',
    source: 'app/redesign/page-header.tsx',
    use: "Every routed page's H1, description, actions and tab bar, at one place on all of them.",
    group: 'layout',
    examples: [{ label: 'with tabs and an action', render: () => <PageHeaderDemo /> }]
  },
  {
    title: 'SettingsCard',
    source: 'app/settings/settings-card.tsx',
    use: 'Grouped rows on one card under a section heading; a link row opens, an add row starts.',
    group: 'layout',
    examples: [
      {
        label: 'group, link rows, add row',
        render: () => (
          <SettingsGroup description="Where this agent can be reached." title="Channels">
            <SettingsCard>
              <SettingsLinkRow description="Connected" meta="2 chats" onClick={() => undefined} title="Telegram" />
              <SettingsLinkRow description="Not connected" onClick={() => undefined} title="Slack" />
              <SettingsAddRow>Add a channel</SettingsAddRow>
            </SettingsCard>
          </SettingsGroup>
        )
      }
    ]
  },
  {
    title: 'Tabs',
    source: 'components/ui/tabs.tsx',
    use: 'Switch between views of the same thing inside a panel.',
    group: 'layout',
    examples: [
      {
        label: 'default',
        render: () => (
          <Tabs defaultValue="preview">
            <TabsList>
              <TabsTrigger value="preview">Preview</TabsTrigger>
              <TabsTrigger value="source">Source</TabsTrigger>
            </TabsList>
          </Tabs>
        )
      }
    ]
  },
  {
    title: 'TextTab',
    source: 'components/ui/text-tab.tsx',
    use: 'A quiet text tab in a header row, with an optional count.',
    group: 'layout',
    examples: [{ label: 'active, rest, meta', render: () => <TextTabsDemo /> }]
  },
  {
    title: 'ResponsiveTabs',
    source: 'components/ui/tab-dropdown.tsx',
    use: 'Text tabs on a wide header that fold into one dropdown when they no longer fit.',
    group: 'layout',
    examples: [{ label: 'default', render: () => <ResponsiveTabsDemo /> }]
  },
  {
    title: 'Pagination',
    source: 'components/ui/pagination.tsx',
    use: 'Move through pages of a long table.',
    group: 'layout',
    examples: [
      {
        label: 'default',
        render: () => (
          <Pagination>
            <PaginationContent>
              <PaginationItem>
                <PaginationPrevious />
              </PaginationItem>
              <PaginationItem>
                <PaginationButton isActive>1</PaginationButton>
              </PaginationItem>
              <PaginationItem>
                <PaginationButton>2</PaginationButton>
              </PaginationItem>
              <PaginationItem>
                <PaginationEllipsis />
              </PaginationItem>
              <PaginationItem>
                <PaginationNext />
              </PaginationItem>
            </PaginationContent>
          </Pagination>
        )
      }
    ]
  },
  {
    title: 'Separator',
    source: 'components/ui/separator.tsx',
    use: 'One hairline between groups that space alone cannot separate.',
    group: 'layout',
    examples: [{ label: 'horizontal', render: () => <Separator /> }]
  },
  {
    title: 'ScrollArea',
    source: 'components/ui/scroll-area.tsx',
    use: 'A bounded region with the app scrollbar.',
    group: 'layout',
    examples: [
      {
        label: 'default',
        render: () => (
          <ScrollArea className="ds-scroll-box">
            {LINES.concat(LINES).map((line, index) => (
              <p key={index}>{line}</p>
            ))}
          </ScrollArea>
        )
      }
    ]
  },
  {
    title: 'FadeScroll',
    source: 'components/ui/fade-scroll.tsx',
    use: 'A short scroller whose edges fade only while content is clipped behind them.',
    group: 'layout',
    examples: [
      {
        label: 'default',
        render: () => (
          <FadeScroll maxHeight="5rem">
            {LINES.concat(LINES).map((line, index) => (
              <p key={index}>{line}</p>
            ))}
          </FadeScroll>
        )
      }
    ]
  },
  {
    title: 'LogView',
    source: 'components/ui/log-view.tsx',
    use: 'Selectable, wrapping monospace output (logs, a command result).',
    group: 'layout',
    examples: [{ label: 'default', render: () => <LogView>{LINES.join('\n')}</LogView> }]
  },
  {
    title: 'Reel',
    source: 'components/ui/reel.tsx',
    use: 'A sideways row of cards that snaps as it scrolls.',
    group: 'layout',
    examples: [{ label: 'default', render: () => <Reel>{tiles(6)}</Reel> }]
  },
  {
    title: 'Masonry',
    source: 'components/ui/masonry.tsx',
    use: 'Cards of uneven height in balanced columns, in source order.',
    group: 'layout',
    examples: [{ label: 'default', render: () => <Masonry className="ds-masonry">{tiles(6)}</Masonry> }]
  }
]
