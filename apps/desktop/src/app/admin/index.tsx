/**
 * Native Admin hub — deep-links into existing Desktop surfaces (settings,
 * messaging, skills/MCP) plus the native Kanban board. No web SPA embed.
 */
import type * as React from 'react'
import { useNavigate } from 'react-router-dom'

import { Codicon } from '@/components/ui/codicon'
import { cn } from '@/lib/utils'

import {
  KANBAN_ROUTE,
  MESSAGING_ROUTE,
  SETTINGS_ROUTE,
  SKILLS_ROUTE
} from '../routes'

interface AdminViewProps extends React.ComponentProps<'section'> {}

interface AdminCard {
  title: string
  description: string
  codicon: string
  path: string
  keywords?: string
}

const CARDS: AdminCard[] = [
  {
    title: 'Kanban board',
    description: 'Multi-agent task board — create, inspect, and move tasks (native API client).',
    codicon: 'project',
    path: KANBAN_ROUTE,
    keywords: 'tasks workers dispatch'
  },
  {
    title: 'Model & config',
    description: 'Provider, model, agent, terminal, and display settings.',
    codicon: 'settings-gear',
    path: `${SETTINGS_ROUTE}?tab=config:model`
  },
  {
    title: 'API keys',
    description: 'Manage credentials and env keys for tools and providers.',
    codicon: 'key',
    path: `${SETTINGS_ROUTE}?tab=keys`
  },
  {
    title: 'Messaging channels',
    description: 'Telegram, Discord, Slack, and other gateway platforms.',
    codicon: 'comment-discussion',
    path: MESSAGING_ROUTE
  },
  {
    title: 'MCP servers',
    description: 'Browse catalog, enable servers, and test connections.',
    codicon: 'server-process',
    path: `${SKILLS_ROUTE}?tab=mcp`
  },
  {
    title: 'Plugins',
    description: 'Desktop / agent plugins and extensions.',
    codicon: 'extensions',
    path: `${SETTINGS_ROUTE}?tab=plugins`
  },
  {
    title: 'Gateway & connection',
    description: 'Local vs remote backend, gateway status, reconnect.',
    codicon: 'globe',
    path: `${SETTINGS_ROUTE}?tab=gateway`
  },
  {
    title: 'Appearance',
    description: 'Theme, density, and desktop chrome.',
    codicon: 'symbol-color',
    path: `${SETTINGS_ROUTE}?tab=config:appearance`
  }
]

export function AdminView({ className, ...rest }: AdminViewProps) {
  const navigate = useNavigate()

  return (
    <section className={cn('flex h-full min-h-0 flex-col overflow-y-auto bg-(--ui-editor-background)', className)} {...rest}>
      <header className="border-b border-(--ui-border) px-5 py-4">
        <h1 className="text-base font-semibold text-foreground">Admin</h1>
        <p className="mt-1 max-w-2xl text-xs text-(--ui-text-secondary)">
          Machine management without leaving Desktop. These open native panes and call the same
          backend HTTP APIs as the web dashboard — no browser UI is embedded.
        </p>
      </header>
      <div className="grid gap-3 p-5 sm:grid-cols-2 xl:grid-cols-3">
        {CARDS.map(card => (
          <button
            className={cn(
              'flex flex-col items-start gap-2 rounded-lg border border-(--ui-border) bg-(--ui-sidebar-surface-background)/35 p-4 text-left',
              'transition-colors hover:border-(--ui-focus-border) hover:bg-(--ui-control-hover-background)'
            )}
            key={card.path + card.title}
            onClick={() => navigate(card.path)}
            type="button"
          >
            <div className="flex items-center gap-2">
              <Codicon className="text-(--ui-text-secondary)" name={card.codicon as never} />
              <span className="text-sm font-semibold text-foreground">{card.title}</span>
            </div>
            <p className="text-xs leading-relaxed text-(--ui-text-secondary)">{card.description}</p>
          </button>
        ))}
      </div>
    </section>
  )
}
