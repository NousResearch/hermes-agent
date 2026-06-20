import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { AccountManagementView } from './index'

afterEach(() => {
  cleanup()
  Reflect.deleteProperty(window, 'hermesDesktop')
})

function setBridge(account: unknown) {
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { account } })
}

function renderAccount(path = '/account') {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <AccountManagementView />
    </MemoryRouter>
  )
}

describe('AccountManagementView', () => {
  it('keeps account management compact and switches between requested sections', async () => {
    const usage = vi.fn(async () => ({
      ok: true,
      items: [{ ts: 1781600000, kind: 'llm', credits: 0.19, provider: 'sub2api', model: '极致' }],
      total: 1,
      balance: 139.81,
      creditRmb: 0.04
    }))

    const createOrder = vi.fn(async (yuan: number) => ({
      order_id: 'order-1',
      yuan,
      credits: yuan * 25,
      mock: true
    }))

    const createSubaccount = vi.fn(async () => ({ user_id: 'child-2', email: 'child@example.com', name: '子爱马仕' }))
    const createRelation = vi.fn(async () => ({ manager_id: 'root-1', member_id: 'child-1' }))
    setBridge({
      createOrder,
      createRelation,
      createSubaccount,
      payConfig: vi.fn(async () => ({ enabled: true, mock: true, quick_yuan: [5, 10], min_yuan: 5, credits_per_yuan: 25 })),
      status: vi.fn(async () => ({ loggedIn: true, username: 'Alice', email: 'a@b.com', balance: 140 })),
      subtree: vi.fn(async () => ({
        root: 'root-1',
        nodes: [
          { user_id: 'root-1', email: 'a@b.com', name: 'Alice' },
          { user_id: 'child-1', email: 'child@example.com', name: '子爱马仕', parent_id: 'root-1' }
        ],
        edges: [{ manager_id: 'root-1', member_id: 'child-1', primary: true }]
      })),
      usage,
      wallet: vi.fn(async () => ({ balance: 139.81, credit_rmb: 0.04 })),
      logout: vi.fn(async () => ({ ok: true }))
    })

    renderAccount('/account?section=usage')

    expect(await screen.findByRole('heading', { name: 'EasyHermes账号管理' })).toBeTruthy()
    expect(screen.getByRole('button', { name: '消费明细' })).toBeTruthy()
    expect(screen.getByRole('button', { name: '充值' })).toBeTruthy()
    expect(screen.getByRole('button', { name: '个人' })).toBeTruthy()
    expect(screen.getByRole('button', { name: '团队账号' })).toBeTruthy()
    await waitFor(() => expect(usage).toHaveBeenCalledWith({ kind: undefined, limit: 20, offset: 0 }))
    expect(await screen.findByText('极致')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: '个人' }))
    await waitFor(() => expect(screen.getAllByText('Alice').length).toBeGreaterThan(0))
    expect(screen.getByText('a@b.com')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: '充值' }))
    expect(await screen.findByRole('button', { name: '5 元' })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: '5 元' }))
    await waitFor(() => expect(createOrder).toHaveBeenCalledWith(5))
    expect(await screen.findByText('order-1')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: '团队账号' }))
    // 团队账号现在是思维导图画布:子账号作为节点出现
    expect((await screen.findAllByText('子爱马仕')).length).toBeGreaterThan(0)
    // 工具条"子账号"按钮打开创建表单(placeholder 输入)
    fireEvent.click(screen.getByRole('button', { name: '子账号' }))
    fireEvent.change(screen.getByPlaceholderText('邮箱'), { target: { value: 'new@example.com' } })
    fireEvent.change(screen.getByPlaceholderText('初始密码'), { target: { value: 'secret1' } })
    fireEvent.change(screen.getByPlaceholderText(/名称/), { target: { value: '新节点' } })
    fireEvent.click(screen.getByRole('button', { name: '创建' }))
    await waitFor(() =>
      expect(createSubaccount).toHaveBeenCalledWith({ name: '新节点', email: 'new@example.com', password: 'secret1' })
    )
  })

  it('grants a sub-account resource to a role from the permissions panel', async () => {
    const api = vi.fn(async (req: { path: string; method?: string; body?: unknown }) => {
      if (req.path === '/api/kari/resources') {
        return {
          langflow_capable: true,
          by_node: { 'child-1': [{ node_uid: 'child-1', kind: 'workflow', resource_id: 'flow_a', name: '财务流程' }] }
        }
      }

      if (req.path === '/api/kari/grants' && (!req.method || req.method === 'GET')) {
        return { grants: [] }
      }

      return { ok: true }
    })

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        account: {
          status: vi.fn(async () => ({ loggedIn: true, username: 'Alice', email: 'a@b.com', balance: 0 })),
          subtree: vi.fn(async () => ({
            root: 'root-1',
            nodes: [
              { user_id: 'root-1', email: 'a@b.com', name: 'Alice' },
              { user_id: 'child-1', email: 'child@example.com', name: '子A', parent_id: 'root-1' }
            ],
            edges: []
          })),
          roles: vi.fn(async () => ({ roles: ['财务'] })),
          wallet: vi.fn(async () => ({ balance: 0, credit_rmb: 0.04 }))
        },
        api
      }
    })

    renderAccount('/account?section=permissions')

    // 权限管理是独立菜单分区;等本地资源/授权拉取完成。
    await waitFor(() => expect(api).toHaveBeenCalledWith({ path: '/api/kari/grants' }))

    // 点角色 → 出现资源勾选(此前没有 checkbox)。
    fireEvent.click(await screen.findByRole('button', { name: '财务' }))

    const checkbox = (await screen.findByRole('checkbox')) as HTMLInputElement
    expect(checkbox.checked).toBe(false)

    // 勾选 → POST 授权 + 乐观勾上。
    fireEvent.click(checkbox)
    await waitFor(() =>
      expect(api).toHaveBeenCalledWith({
        body: { role: '财务', node_uid: 'child-1', kind: 'workflow', resource_id: 'flow_a' },
        method: 'POST',
        path: '/api/kari/grants'
      })
    )
    await waitFor(() => expect((screen.getByRole('checkbox') as HTMLInputElement).checked).toBe(true))
    // 角色 chip 出现「实际可用」数徽标(grant ∩ 现存资源)。
    expect(await screen.findByRole('button', { name: /^财务\s*1$/ })).toBeTruthy()
  })

  it('hides MCP config when the account has no langflow capability', async () => {
    const api = vi.fn(async (req: { path: string }) => {
      if (req.path === '/api/kari/resources') {
        return { langflow_capable: false, by_node: {} }
      }

      if (req.path === '/api/kari/grants') {
        return { grants: [] }
      }

      return { ok: true }
    })

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        account: {
          status: vi.fn(async () => ({ loggedIn: true, username: 'Bob', email: 'b@b.com', balance: 0 })),
          subtree: vi.fn(async () => ({
            root: 'root-1',
            nodes: [{ user_id: 'root-1', email: 'b@b.com', name: 'Bob' }],
            edges: []
          })),
          roles: vi.fn(async () => ({ roles: ['财务'] }))
        },
        api
      }
    })

    renderAccount('/account?section=permissions')

    // 没 langflow 能力 → 出现提示,且不显示角色 chip(无法配 MCP)。
    expect(await screen.findByText(/不能配 MCP 给下级/)).toBeTruthy()
    expect(screen.queryByRole('button', { name: '财务' })).toBeNull()
  })

  it('grants a workflow to a specific sub-account in 按账号微调 mode', async () => {
    const api = vi.fn(async (req: { path: string; method?: string; body?: unknown }) => {
      if (req.path === '/api/kari/resources') {
        return {
          langflow_capable: true,
          by_node: { 'child-1': [{ node_uid: 'child-1', kind: 'workflow', resource_id: 'flow_a', name: '财务流程' }] }
        }
      }

      if ((req.path === '/api/kari/grants' || req.path === '/api/kari/grants/user') && (!req.method || req.method === 'GET')) {
        return { grants: [] }
      }

      return { ok: true }
    })

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        account: {
          status: vi.fn(async () => ({ loggedIn: true, username: 'Alice', email: 'a@b.com', balance: 0 })),
          subtree: vi.fn(async () => ({
            root: 'root-1',
            nodes: [
              { user_id: 'root-1', email: 'a@b.com', name: 'Alice' },
              { user_id: 'child-1', email: 'child@example.com', name: '子A', parent_id: 'root-1' }
            ],
            edges: []
          })),
          roles: vi.fn(async () => ({ roles: ['财务'] }))
        },
        api
      }
    })

    renderAccount('/account?section=permissions')
    await waitFor(() => expect(api).toHaveBeenCalledWith({ path: '/api/kari/grants/user' }))

    // 切到「按账号微调」→ 点下级账号 子A → 勾工作流 → 写 grant_user(不是 grant_policy)。
    fireEvent.click(screen.getByRole('button', { name: '按账号微调' }))
    fireEvent.click(await screen.findByRole('button', { name: '子A' }))
    fireEvent.click(await screen.findByRole('checkbox'))

    await waitFor(() =>
      expect(api).toHaveBeenCalledWith({
        body: { user_id: 'child-1', node_uid: 'child-1', kind: 'workflow', resource_id: 'flow_a' },
        method: 'POST',
        path: '/api/kari/grants/user'
      })
    )
  })
})
