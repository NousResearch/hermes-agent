import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { SidebarProvider } from '@/components/ui/sidebar'
import { I18nProvider } from '@/i18n'

import { ChatSidebar } from './index'

afterEach(() => {
  cleanup()
})

describe('ChatSidebar account management nav', () => {
  // 账号管理已从左侧栏移除,统一收进右上角账号弹窗(account-dialog 的左菜单+右内容)。
  it('no longer renders the account section in the sidebar', () => {
    render(
      <I18nProvider configClient={null} initialLocale="zh">
        <MemoryRouter>
          <SidebarProvider open>
            <ChatSidebar
              currentView="chat"
              onArchiveSession={vi.fn()}
              onDeleteSession={vi.fn()}
              onLoadMoreSessions={vi.fn()}
              onManageCronJob={vi.fn()}
              onNavigate={vi.fn()}
              onNewSessionInWorkspace={vi.fn()}
              onResumeSession={vi.fn()}
              onTriggerCronJob={vi.fn()}
            />
          </SidebarProvider>
        </MemoryRouter>
      </I18nProvider>
    )

    expect(screen.queryByRole('button', { name: 'EasyHermes账号管理' })).toBeNull()

    for (const label of ['消费明细', '充值', '团队账号']) {
      expect(screen.queryByRole('button', { name: label })).toBeNull()
    }
  })
})
