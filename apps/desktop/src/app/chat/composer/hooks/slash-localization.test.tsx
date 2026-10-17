import type { Unstable_TriggerItem } from '@assistant-ui/core'
import { cleanup, render, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { renderCommandsCatalog } from '@/app/session/hooks/use-prompt-actions/utils'
import type { HermesGateway } from '@/hermes'
import { I18nProvider } from '@/i18n'
import { TRANSLATIONS } from '@/i18n/catalog'
import { setRuntimeI18nLocale } from '@/i18n/runtime'
import skillCatalog from '@/i18n/skill-catalog.zh.json'
import type { Locale } from '@/i18n/types'
import {
  type CommandsCatalogLike,
  desktopSlashDescription,
  rememberDesktopCommandsCatalog
} from '@/lib/desktop-slash-commands'
import { queryClient } from '@/lib/query-client'

import { useSlashCompletions } from './use-slash-completions'

const catalog: CommandsCatalogLike = {
  canon: { '/reset': '/new' },
  categories: [
    {
      name: 'Session',
      pairs: [
        ['/new', 'Backend description'],
        ['/retry', 'Retry the last message']
      ]
    }
  ],
  pairs: [
    ['/new', 'Backend description'],
    ['/retry', 'Retry the last message'],
    ['/my-skill', 'Author-owned description']
  ]
}

afterEach(() => {
  cleanup()
  queryClient.clear()
  setRuntimeI18nLocale('en')
  rememberDesktopCommandsCatalog(undefined)
})

describe('desktop slash description localization', () => {
  it('translates the remaining native desktop commands without changing usage tokens', () => {
    setRuntimeI18nLocale('zh')

    for (const command of ['/skills']) {
      const description = desktopSlashDescription(command, `Backend (usage: ${command} [name|off])`)

      expect(description).toMatch(/\p{Script=Han}/u)
      // usage 片段由后端拥有，必须逐字保留。
      expect(description).toContain(`(usage: ${command} [name|off])`)
    }
  })

  it('uses the active catalog across aliases and help while preserving unknown descriptions and usage syntax', () => {
    for (const [locale, copy] of Object.entries(TRANSLATIONS)) {
      setRuntimeI18nLocale(locale as Locale)

      for (const command of [
        '/new',
        '/save',
        '/retry',
        '/undo',
        '/title',
        '/branch',
        '/worktree',
        '/compress',
        '/stop'
      ]) {
        const expected = copy.composer.commandDescs[command]
        expect(expected, `${locale}: ${command}`).toBeTruthy()
        expect(desktopSlashDescription(command, 'backend')).toBe(expected)
      }

      for (const command of Object.keys(TRANSLATIONS.en.composer.commandDescs)) {
        expect(copy.composer.commandDescs[command], `${locale}: ${command}`).toBeTruthy()

        if (locale !== 'en') {
          expect(copy.composer.commandDescs[command], `${locale}: ${command}`).not.toBe(
            TRANSLATIONS.en.composer.commandDescs[command]
          )
        }
      }

      expect(desktopSlashDescription('/reset')).toBe(copy.composer.commandDescs['/new'])
      expect(desktopSlashDescription('/my-skill', 'Author-owned description')).toBe('Author-owned description')
      const syntax = '/retry [instructions]'
      expect(desktopSlashDescription('/retry', `Retry (usage: ${syntax})`)).toContain(syntax)
      const help = renderCommandsCatalog(catalog, copy.desktop)
      expect(help).toContain(copy.composer.commandDescs['/new'])
      expect(help).toContain(copy.composer.commandDescs['/retry'])
    }
  })

  it('refreshes both cached bare and typed suggestions when the UI locale changes', async () => {
    const request = vi.fn(async (method: string) =>
      method === 'commands.catalog'
        ? catalog
        : {
            items: [
              { text: '/new', meta: 'Backend description', kind: 'command' },
              { text: '/my-skill', meta: 'Author-owned description', kind: 'skill' }
            ]
          }
    )

    const api: {
      search?: (query: string) => readonly Unstable_TriggerItem[]
      setLocale?: ReturnType<typeof useI18n>['setLocale']
    } = {}

    function Probe() {
      api.setLocale = useI18n().setLocale
      api.search = useSlashCompletions({ gateway: { request } as unknown as HermesGateway }).adapter.search

      return null
    }

    render(
      <I18nProvider configClient={null} initialLocale="en">
        <Probe />
      </I18nProvider>
    )

    for (const query of ['', 'ne']) {
      for (const locale of ['en', 'zh', 'ja', 'zh-hant', 'ar', 'ru', 'fr', 'de', 'es', 'en'] as const) {
        await act(async () => {
          await api.setLocale!(locale)
        })
        await waitFor(
          () => {
            const items = api.search!(query)

            expect(items.find(item => item.metadata?.command === '/new')?.description).toBe(
              TRANSLATIONS[locale].composer.commandDescs['/new']
            )
            expect(items.find(item => item.metadata?.command === '/my-skill')?.description).toBe(
              'Author-owned description'
            )
          },
          { timeout: 2000 }
        )
      }
    }

    expect(request).toHaveBeenCalledTimes(2)
  })

  // 输入框敲 `/` 弹出的补全行：右侧灰色说明来自后端（英文），显示层要换成中文，
  // 但行首标识与插入内容必须还是真实的 `/pdf`。
  it('translates a skill row blurb while the command itself stays raw', async () => {
    const entry = (skillCatalog.skills as Record<string, { description: string; sourceDescription: string }>)[
      'pdf'
    ]

    const request = vi.fn(async (method: string) =>
      method === 'commands.catalog'
        ? catalog
        : {
            items: [{ text: '/pdf', display: '/pdf', kind: 'skill', meta: entry.sourceDescription }]
          }
    )

    const api: { search?: (query: string) => readonly Unstable_TriggerItem[] } = {}

    function Probe() {
      api.search = useSlashCompletions({ gateway: { request } as unknown as HermesGateway }).adapter.search

      return null
    }

    render(
      <I18nProvider configClient={null} initialLocale="zh">
        <Probe />
      </I18nProvider>
    )

    const row = () => api.search!('pd').find(item => item.metadata?.command === '/pdf')

    await waitFor(() => expect(row()?.description).toBe(entry.description), { timeout: 2000 })

    expect(row()?.description).toMatch(/\p{Script=Han}/u)
    expect(row()?.metadata?.command).toBe('/pdf')
    expect(row()?.metadata?.rawText).toBe('/pdf')
    expect(row()?.metadata?.group).toBe('Skills')
  })
})
