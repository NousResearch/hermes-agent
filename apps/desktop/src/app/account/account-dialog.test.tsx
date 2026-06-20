import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { setAccountDialogOpen } from '@/store/account'

import { AccountDialog } from './account-dialog'

afterEach(() => {
  cleanup()
  setAccountDialogOpen(false)
  Reflect.deleteProperty(window, 'hermesDesktop')
})

function setBridge(account: unknown) {
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { account } })
}

describe('AccountDialog', () => {
  it('shows the login / register form when signed out', async () => {
    setBridge({
      status: vi.fn(async () => ({ loggedIn: false, cloudBaseUrl: 'https://flow.karivibe.com' }))
    })
    setAccountDialogOpen(true)
    render(<AccountDialog />)

    expect(await screen.findByRole('button', { name: '登录' })).toBeTruthy()
    expect(screen.getByRole('button', { name: '注册' })).toBeTruthy()
  })

  it('shows balance, left menu and consumption detail in the two-pane popup when signed in', async () => {
    setBridge({
      status: vi.fn(async () => ({ loggedIn: true, username: 'Alice', email: 'a@b.com', balance: 140 })),
      usage: vi.fn(async () => ({
        ok: true,
        items: [{ ts: 1781600000, kind: 'llm', credits: 0.19, provider: 'sub2api', model: '极致' }],
        total: 1
      }))
    })
    setAccountDialogOpen(true)
    render(<AccountDialog />)

    // 侧栏:头像名 + 余额(来自 status)
    expect(await screen.findByText('Alice')).toBeTruthy()
    expect(screen.getByText('余额')).toBeTruthy()
    await waitFor(() => expect(screen.getByText(/140/)).toBeTruthy())
    // 左侧菜单项(默认 usage 区,这些文案此时只在菜单出现)
    expect(screen.getByText('充值')).toBeTruthy()
    expect(screen.getByText('个人')).toBeTruthy()
    expect(screen.getByText('团队账号')).toBeTruthy()
    // 右侧默认「消费明细」区:标题(菜单+区头)、条目、筛选 chips
    expect(screen.getAllByText('消费明细').length).toBeGreaterThan(0)
    expect(await screen.findByText('极致')).toBeTruthy()
    expect(screen.getByRole('button', { name: '全部' })).toBeTruthy()
    expect(screen.getByRole('button', { name: '图片' })).toBeTruthy()

    // 切到「个人」→ 出现退出登录
    fireEvent.click(screen.getByText('个人'))
    expect(await screen.findByRole('button', { name: /退出登录/ })).toBeTruthy()
  })

  it('filters consumption by kind and paginates with 上一页/下一页', async () => {
    const usage = vi.fn(async (opts?: { kind?: string; offset?: number }) => {
      if (opts?.kind === 'llm') {
        return { ok: true, items: [{ kind: 'llm', credits: 1, model: 'gpt' }], total: 1 }
      }
      if (opts?.offset) {
        return { ok: true, items: [{ kind: 'image', credits: 2, model: 'flux' }], total: 25 }
      }
      return { ok: true, items: [{ kind: 'llm', credits: 1, model: 'first' }], total: 25 }
    })
    setBridge({ status: vi.fn(async () => ({ loggedIn: true, username: 'Bob', balance: 5 })), usage })
    setAccountDialogOpen(true)
    render(<AccountDialog />)

    await screen.findByText('first')
    expect(usage).toHaveBeenCalledWith({ kind: undefined, limit: 20, offset: 0 })

    // 共 25 条 → 多页,翻到下一页(offset = 20)
    const next = await screen.findByRole('button', { name: '下一页' })
    fireEvent.click(next)
    await waitFor(() => expect(usage).toHaveBeenCalledWith({ kind: undefined, limit: 20, offset: 20 }))
    expect(await screen.findByText('flux')).toBeTruthy() // 第二页(替换,非追加)
    expect(screen.queryByText('first')).toBeNull()

    // 点 LLM 筛选 → 回第 1 页带 kind
    fireEvent.click(screen.getByRole('button', { name: 'LLM' }))
    await waitFor(() => expect(usage).toHaveBeenCalledWith({ kind: 'llm', limit: 20, offset: 0 }))
  })
})
