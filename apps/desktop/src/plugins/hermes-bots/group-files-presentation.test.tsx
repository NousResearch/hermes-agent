import type * as HermesSdk from '@hermes/plugin-sdk'
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { GroupFilesRows } from './group-files-rows'
import { fileItem, parsedFilePage } from './group-files-test-fixtures'
import { translateBots } from './i18n-test-helper'

const state = vi.hoisted(() => ({ locale: 'en' }))
vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  return { ...sdk, useI18n: () => ({ locale: state.locale }), usePluginI18n: () => translateBots }
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

describe('compact Files metadata through real rows', () => {
  it.each(['en', 'ja', 'zh', 'zh-Hant', 'ar', 'ru'])(
    'localizes %s while retaining full time and same-name precision',
    locale => {
      state.locale = locale
      vi.useFakeTimers()
      const now = new Date(2026, 8, 3, 14, 30, 45)
      vi.setSystemTime(now)
      const other = new Date(2026, 8, 3, 14, 30, 10)
      const past = new Date(2025, 8, 2, 14, 30, 45)

      const items = parsedFilePage([
        { ...fileItem(20, 'single.txt'), shared_at: now.getTime() / 1000 },
        { ...fileItem(19, 'same.txt'), shared_at: now.getTime() / 1000 },
        { ...fileItem(18, 'same.txt'), shared_at: other.getTime() / 1000 },
        { ...fileItem(17, 'past.txt'), shared_at: past.getTime() / 1000 }
      ]).items

      render(
        <GroupFilesRows
          group="Core"
          intentSignal={new AbortController().signal}
          items={items}
          loading={false}
          onRefresh={vi.fn()}
          onRoomAccessDenied={vi.fn()}
          roomId="room-1"
        />
      )
      const rows = screen.getAllByRole('listitem')
      const metadata = rows.map(row => row.querySelector<HTMLDivElement>('div[title]')!)
      const compact = new Intl.DateTimeFormat(locale, { hour: 'numeric', minute: '2-digit' }).format(now)
      expect(metadata[0].textContent).toContain(compact)
      expect(metadata[0].textContent).not.toContain(now.toLocaleString(locale))
      expect(metadata[0].title).toContain(now.toLocaleString(locale))
      expect(rows[0].getAttribute('aria-label')).toContain(now.toLocaleString(locale))
      expect(metadata[1].textContent).toContain(
        new Intl.DateTimeFormat(locale, { hour: 'numeric', minute: '2-digit', second: '2-digit' }).format(now)
      )
      expect(metadata[1].textContent).not.toBe(metadata[2].textContent)
      expect(metadata[3].textContent).toContain(
        new Intl.DateTimeFormat(locale, {
          year: 'numeric',
          month: 'short',
          day: 'numeric',
          hour: 'numeric',
          minute: '2-digit'
        }).format(past)
      )
      expect(metadata[3].title).toContain(past.toLocaleString(locale))
    }
  )
})
