import { type FormEvent, type ReactNode, useCallback, useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Input } from '@/components/ui/input'
import type {
  DesktopAccountPayConfig,
  DesktopAccountPayOrder,
  DesktopAccountStatus,
  DesktopAccountSubtree,
  DesktopAccountTransactionItem,
  DesktopAccountTreeNode,
  DesktopAccountUsageItem,
  DesktopAccountWallet
} from '@/global'
import { Loader2, LogIn } from '@/lib/icons'
import { cn } from '@/lib/utils'

import { TeamCanvas } from './team-canvas'
import { PermissionsPanel, RolesPanel } from './team-roles'
import type { TeamGrant, TeamResource } from './team-types'

const DEFAULT_CLOUD_URL =
  (import.meta.env?.VITE_KARI_CLOUD_URL as string | undefined)?.trim() || 'https://flow.karivibe.com'

const PAGE_SIZE = 20

type AccountSection = 'permissions' | 'profile' | 'recharge' | 'roles' | 'team' | 'usage'

const SECTION_LABELS: Record<AccountSection, string> = {
  usage: '消费明细',
  recharge: '充值',
  profile: '个人',
  team: '团队账号',
  roles: '角色管理',
  permissions: '权限管理'
}

const SECTION_ICONS: Record<AccountSection, string> = {
  usage: 'history',
  recharge: 'credit-card',
  profile: 'account',
  team: 'type-hierarchy',
  roles: 'tag',
  permissions: 'key'
}

const KIND_LABEL: Record<string, string> = {
  grant: '发放',
  image: '图片',
  kie: 'KIE',
  llm: 'LLM',
  redeem: '兑换',
  video: '视频'
}

const KIND_FILTERS: { label: string; value: string }[] = [
  { label: '全部', value: '' },
  { label: 'LLM', value: 'llm' },
  { label: '图片', value: 'image' },
  { label: '视频', value: 'video' }
]

const SECTION_ORDER: AccountSection[] = ['usage', 'recharge', 'profile', 'team', 'roles', 'permissions']

function money(value?: number) {
  return typeof value === 'number' ? value.toFixed(2) : '0.00'
}

function formatTs(ts?: number) {
  if (!ts) {
    return '—'
  }

  const ms = ts > 1e12 ? ts : ts * 1000

  try {
    return new Date(ms).toLocaleString(undefined, {
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit'
    })
  } catch {
    return '—'
  }
}

function notifyAccountChanged() {
  window.dispatchEvent(new Event('hermes-account-changed'))
}

function getSection(value: string | null): AccountSection {
  return value === 'usage' ||
    value === 'recharge' ||
    value === 'team' ||
    value === 'profile' ||
    value === 'roles' ||
    value === 'permissions'
    ? value
    : 'profile'
}

function accountBridge() {
  return window.hermesDesktop?.account
}

function sameGrant(a: TeamGrant, b: TeamGrant) {
  return a.role === b.role && a.node_uid === b.node_uid && a.kind === b.kind && a.resource_id === b.resource_id
}

export function AccountManagementView() {
  const [searchParams, setSearchParams] = useSearchParams()
  const section = getSection(searchParams.get('section'))
  const [status, setStatus] = useState<DesktopAccountStatus | null>(null)
  const [loadingStatus, setLoadingStatus] = useState(true)
  const [statusError, setStatusError] = useState('')

  const setSection = (next: AccountSection) => setSearchParams({ section: next })

  const refreshStatus = useCallback(async () => {
    const account = accountBridge()

    if (!account?.status) {
      setStatusError('桌面账号服务不可用')
      setLoadingStatus(false)

      return
    }

    setLoadingStatus(true)
    setStatusError('')

    try {
      setStatus(await account.status())
    } catch (err) {
      setStatusError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoadingStatus(false)
    }
  }, [])

  useEffect(() => {
    void refreshStatus()
  }, [refreshStatus])

  const loggedIn = Boolean(status?.loggedIn)

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden bg-(--ui-chat-surface-background) text-foreground">
      <header className="shrink-0 border-b border-(--ui-stroke-tertiary) px-5 pb-3 pt-[calc(var(--titlebar-height)+0.625rem)]">
        <div className="flex min-h-7 flex-wrap items-center justify-between gap-2">
          <div className="flex min-w-0 items-center gap-2">
            <span className="grid size-6 shrink-0 place-items-center rounded-md bg-(--ui-bg-tertiary) text-(--ui-text-secondary)">
              <Codicon name="account" size="0.875rem" />
            </span>
            <div className="min-w-0">
              <h1 className="truncate text-[length:var(--conversation-text-font-size)] font-semibold text-foreground">
                EasyHermes账号管理
              </h1>
              {loggedIn ? (
                <p className="truncate text-[length:var(--conversation-caption-font-size)] leading-4 text-(--ui-text-tertiary)">
                  {status?.cloudBaseUrl || 'EasyHermes Cloud'}
                </p>
              ) : null}
            </div>
          </div>
          {loggedIn ? (
            <div className="flex min-w-0 items-center gap-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-secondary)">
              <span className="truncate">{status?.username || status?.email || '已登录'}</span>
              <span className="shrink-0 rounded bg-(--ui-bg-quinary) px-1.5 py-0.5 font-mono text-[0.6875rem] text-(--ui-text-tertiary)">
                {money(status?.balance)}
              </span>
            </div>
          ) : null}
        </div>
        <div className="mt-3 flex flex-wrap gap-1">
          {SECTION_ORDER.map(item => (
            <button
              className={cn(
                'flex h-7 items-center gap-1.5 rounded-md border border-transparent px-2 text-[length:var(--conversation-text-font-size)] text-(--ui-text-secondary) transition-colors hover:bg-(--chrome-action-hover) hover:text-foreground',
                section === item
                  ? 'border-(--ui-stroke-tertiary) bg-(--ui-bg-tertiary) font-medium text-foreground'
                  : undefined
              )}
              key={item}
              onClick={() => setSection(item)}
              type="button"
            >
              <Codicon name={SECTION_ICONS[item]} size="0.85rem" />
              {SECTION_LABELS[item]}
            </button>
          ))}
        </div>
      </header>

      <main className="min-h-0 flex-1 overflow-y-auto px-5 py-4">
        {loadingStatus ? (
          <div className="grid min-h-56 place-items-center">
            <Loader2 className="size-5 animate-spin text-muted-foreground" />
          </div>
        ) : statusError ? (
          <Panel>
            <p className="text-sm text-destructive">{statusError}</p>
          </Panel>
        ) : !loggedIn ? (
          <div className="grid min-h-full place-items-center px-4 py-8">
            <SignInPanel onChanged={() => void refreshStatus()} status={status} />
          </div>
        ) : section === 'usage' ? (
          <UsageSection />
        ) : section === 'recharge' ? (
          <RechargeSection onChanged={() => void refreshStatus()} />
        ) : section === 'team' ? (
          <TeamSection />
        ) : section === 'roles' ? (
          <RolesSection />
        ) : section === 'permissions' ? (
          <PermissionsSection />
        ) : (
          <ProfileSection onChanged={() => void refreshStatus()} status={status} />
        )}
      </main>
    </div>
  )
}

function Panel({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <section className={cn('rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) p-3', className)}>
      {children}
    </section>
  )
}

function SectionHeader({ title, description, action }: { title: string; description?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-2">
      <div className="min-w-0">
        <h2 className="text-[length:var(--conversation-text-font-size)] font-semibold text-foreground">{title}</h2>
        {description ? (
          <p className="mt-1 text-[length:var(--conversation-caption-font-size)] leading-5 text-(--ui-text-tertiary)">
            {description}
          </p>
        ) : null}
      </div>
      {action}
    </div>
  )
}

function SignInPanel({ onChanged, status }: { onChanged: () => void; status: DesktopAccountStatus | null }) {
  const [cloudBaseUrl, setCloudBaseUrl] = useState(status?.cloudBaseUrl || DEFAULT_CLOUD_URL)
  const [email, setEmail] = useState(status?.email || '')
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const submit = async (mode: 'login' | 'register', event?: FormEvent) => {
    event?.preventDefault()
    const account = accountBridge()
    const fn = mode === 'register' ? account?.register : account?.login

    if (!fn) {
      setError('桌面账号服务不可用')

      return
    }

    setBusy(true)
    setError('')

    try {
      const result = await fn({ cloudBaseUrl, email, password })

      if (!result.ok) {
        setError(result.error || (mode === 'register' ? '注册失败' : '登录失败'))

        return
      }

      setPassword('')
      notifyAccountChanged()
      onChanged()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Panel className="max-w-md">
      <form className="grid gap-3" onSubmit={event => void submit('login', event)}>
        <SectionHeader description="登录后在桌面端管理消费、充值和团队账号。" title="登录 EasyHermes" />
        <label className="grid gap-1 text-xs font-medium text-(--ui-text-secondary)">
          云端地址
          <Input onChange={event => setCloudBaseUrl(event.target.value)} type="url" value={cloudBaseUrl} />
        </label>
        <label className="grid gap-1 text-xs font-medium text-(--ui-text-secondary)">
          邮箱
          <Input autoComplete="username" onChange={event => setEmail(event.target.value)} type="email" value={email} />
        </label>
        <label className="grid gap-1 text-xs font-medium text-(--ui-text-secondary)">
          密码
          <Input
            autoComplete="current-password"
            onChange={event => setPassword(event.target.value)}
            type="password"
            value={password}
          />
        </label>
        {error ? <p className="text-xs text-destructive">{error}</p> : null}
        <div className="flex gap-2">
          <Button disabled={busy} type="submit">
            {busy ? <Loader2 className="size-3.5 animate-spin" /> : <LogIn className="size-3.5" />}
            登录
          </Button>
          <Button disabled={busy} onClick={() => void submit('register')} type="button" variant="ghost">
            注册
          </Button>
        </div>
      </form>
    </Panel>
  )
}

export function ProfileSection({ onChanged, status }: { onChanged: () => void; status: DesktopAccountStatus | null }) {
  const [oldPassword, setOldPassword] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')

  const logout = async () => {
    const account = accountBridge()

    if (!account?.logout) {
      setError('桌面账号服务不可用')

      return
    }

    setBusy(true)
    setError('')

    try {
      await account.logout()
      notifyAccountChanged()
      onChanged()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  const changePassword = async (event: FormEvent) => {
    event.preventDefault()
    const account = accountBridge()

    if (!account?.changePassword) {
      setError('桌面账号服务不可用')

      return
    }

    setBusy(true)
    setError('')
    setMessage('')

    try {
      await account.changePassword({ new_password: newPassword, old_password: oldPassword })
      setOldPassword('')
      setNewPassword('')
      setMessage('密码已更新')
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="grid max-w-4xl gap-4 lg:grid-cols-[minmax(0,1fr)_20rem]">
      <Panel>
        <SectionHeader title="个人" />
        <div className="mt-4 grid gap-3 sm:grid-cols-2">
          <InfoBlock label="账号" value={status?.username || status?.email || '已登录'} />
          <InfoBlock label="余额" value={`${money(status?.balance)} 积分`} />
          <InfoBlock label="邮箱" value={status?.email || '—'} />
          <InfoBlock label="云端" value={status?.cloudBaseUrl || '—'} />
        </div>
        <Button className="mt-4" disabled={busy} onClick={() => void logout()} type="button" variant="secondary">
          {busy ? <Loader2 className="size-3.5 animate-spin" /> : <Codicon name="sign-out" size="0.85rem" />}
          退出登录
        </Button>
      </Panel>

      <Panel>
        <form className="grid gap-3" onSubmit={changePassword}>
          <h3 className="text-[length:var(--conversation-text-font-size)] font-semibold">修改密码</h3>
          <label className="grid gap-1 text-xs font-medium text-(--ui-text-secondary)">
            原密码
            <Input onChange={event => setOldPassword(event.target.value)} type="password" value={oldPassword} />
          </label>
          <label className="grid gap-1 text-xs font-medium text-(--ui-text-secondary)">
            新密码
            <Input onChange={event => setNewPassword(event.target.value)} type="password" value={newPassword} />
          </label>
          {message ? <p className="text-xs text-emerald-400">{message}</p> : null}
          {error ? <p className="text-xs text-destructive">{error}</p> : null}
          <Button disabled={busy || !oldPassword || !newPassword} size="sm" type="submit">
            更新密码
          </Button>
        </form>
      </Panel>
    </div>
  )
}

function InfoBlock({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-chat-surface-background) px-3 py-2">
      <div className="text-[0.6875rem] text-(--ui-text-tertiary)">{label}</div>
      <div className="mt-1 truncate text-sm font-medium">{value}</div>
    </div>
  )
}

export function UsageSection() {
  const [items, setItems] = useState<DesktopAccountUsageItem[]>([])
  const [kindFilter, setKindFilter] = useState('')
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(0)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const loadUsage = useCallback(async (kind: string, nextPage: number) => {
    const account = accountBridge()

    if (!account?.usage) {
      setError('桌面账号服务不可用')

      return
    }

    setBusy(true)
    setError('')

    try {
      const result = await account.usage({ kind: kind || undefined, limit: PAGE_SIZE, offset: nextPage * PAGE_SIZE })
      setItems(result.items || [])
      setTotal(typeof result.total === 'number' ? result.total : result.items?.length || 0)
      setPage(nextPage)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }, [])

  useEffect(() => {
    void loadUsage('', 0)
  }, [loadUsage])

  const applyFilter = (kind: string) => {
    setKindFilter(kind)
    void loadUsage(kind, 0)
  }

  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE))

  return (
    <Panel>
      <SectionHeader
        action={
          <div className="flex flex-wrap gap-1">
            {KIND_FILTERS.map(filter => (
              <button
                className={cn(
                  'h-6 rounded-md px-2 text-[0.75rem] font-medium text-(--ui-text-secondary) transition-colors hover:bg-(--chrome-action-hover) hover:text-foreground',
                  kindFilter === filter.value && 'bg-(--ui-bg-tertiary) text-foreground'
                )}
                disabled={busy}
                key={filter.value}
                onClick={() => applyFilter(filter.value)}
                type="button"
              >
                {filter.label}
              </button>
            ))}
          </div>
        }
        description="LLM、KIE、图片/视频等消耗统一记在这里。"
        title="消费明细"
      />
      {error ? <p className="mt-3 text-xs text-destructive">{error}</p> : null}
      <div className="mt-3 overflow-hidden rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-chat-surface-background)">
        <table className="w-full table-fixed text-left text-xs">
          <thead className="border-b border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) text-[0.6875rem] text-(--ui-text-tertiary)">
            <tr>
              <th className="w-32 px-3 py-2 font-medium">时间</th>
              <th className="w-20 px-3 py-2 font-medium">类型</th>
              <th className="px-3 py-2 font-medium">内容</th>
              <th className="w-24 px-3 py-2 text-right font-medium">积分</th>
            </tr>
          </thead>
          <tbody>
            {items.length ? (
              items.map((item, index) => (
                <tr className="border-t border-(--ui-stroke-tertiary)" key={`${item.ts || 0}-${index}`}>
                  <td className="px-3 py-2 text-(--ui-text-secondary)">{formatTs(item.ts)}</td>
                  <td className="px-3 py-2">{KIND_LABEL[item.kind || ''] || item.kind || '—'}</td>
                  <td className="truncate px-3 py-2">
                    {item.model || item.provider || item.note || item.user_id || '消费'}
                  </td>
                  <td className="px-3 py-2 text-right font-semibold tabular-nums">-{item.credits ?? 0}</td>
                </tr>
              ))
            ) : (
              <tr>
                <td className="px-3 py-8 text-center text-(--ui-text-tertiary)" colSpan={4}>
                  {busy ? '加载中…' : '暂无消费记录'}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      {total > PAGE_SIZE ? (
        <div className="mt-3 flex items-center justify-between gap-2">
          <span className="text-xs text-(--ui-text-tertiary)">
            共 {total} 条 · 第 {page + 1}/{pageCount} 页
          </span>
          <div className="flex gap-1.5">
            <Button
              disabled={busy || page <= 0}
              onClick={() => void loadUsage(kindFilter, page - 1)}
              size="sm"
              variant="secondary"
            >
              上一页
            </Button>
            <Button
              disabled={busy || page >= pageCount - 1}
              onClick={() => void loadUsage(kindFilter, page + 1)}
              size="sm"
              variant="secondary"
            >
              下一页
            </Button>
          </div>
        </div>
      ) : null}
    </Panel>
  )
}

export function RechargeSection({ onChanged }: { onChanged: () => void }) {
  const [wallet, setWallet] = useState<DesktopAccountWallet | null>(null)
  const [config, setConfig] = useState<DesktopAccountPayConfig | null>(null)
  const [order, setOrder] = useState<DesktopAccountPayOrder | null>(null)
  const [transactions, setTransactions] = useState<DesktopAccountTransactionItem[]>([])
  const [redeemCode, setRedeemCode] = useState('')
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')

  const refresh = useCallback(async () => {
    const account = accountBridge()

    if (!account?.payConfig || !account.wallet) {
      setError('桌面账号服务不可用')

      return
    }

    setBusy(true)
    setError('')

    try {
      const [nextConfig, nextWallet, tx] = await Promise.all([
        account.payConfig(),
        account.wallet(),
        account.transactions?.({ limit: 8, offset: 0 }).catch(() => ({ items: [] }))
      ])

      setConfig(nextConfig)
      setWallet(nextWallet)
      setTransactions(tx?.items || [])
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const createOrder = async (yuan: number) => {
    const account = accountBridge()

    if (!account?.createOrder) {
      setError('桌面账号服务不可用')

      return
    }

    setBusy(true)
    setError('')
    setMessage('')

    try {
      setOrder(await account.createOrder(yuan))
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  const confirmMock = async () => {
    const account = accountBridge()

    if (!order?.order_id || !account?.mockConfirm) {
      return
    }

    setBusy(true)
    setError('')

    try {
      const result = await account.mockConfirm(order.order_id)
      const creditRmb = wallet?.credit_rmb ?? (config?.credits_per_yuan ? 1 / config.credits_per_yuan : 0.04)
      setWallet({ balance: result.balance, credit_rmb: creditRmb })
      setMessage(result.credited ? '模拟充值已到账' : '订单已确认')
      notifyAccountChanged()
      onChanged()
      void refresh()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  const redeem = async (event: FormEvent) => {
    event.preventDefault()
    const account = accountBridge()

    if (!account?.redeem) {
      setError('桌面账号服务不可用')

      return
    }

    setBusy(true)
    setError('')
    setMessage('')

    try {
      const result = await account.redeem(redeemCode)
      setWallet(current => ({ balance: result.balance, credit_rmb: current?.credit_rmb ?? 0.04 }))
      setRedeemCode('')
      setMessage(`兑换成功,+${result.added} 积分`)
      notifyAccountChanged()
      onChanged()
      void refresh()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="grid max-w-5xl gap-4 lg:grid-cols-[minmax(0,1fr)_22rem]">
      <Panel>
        <SectionHeader
          action={
            <div className="text-right text-[length:var(--conversation-caption-font-size)]">
              <div className="text-(--ui-text-tertiary)">当前余额</div>
              <div className="text-sm font-semibold tabular-nums">{money(wallet?.balance)} 积分</div>
              <div className="text-(--ui-text-tertiary)">
                ≈ ¥{((wallet?.balance || 0) * (wallet?.credit_rmb || 0)).toFixed(2)}
              </div>
            </div>
          }
          description="直接在桌面端生成订单或兑换码。"
          title="充值"
        />
        <div className="mt-4 flex flex-wrap gap-2">
          {(config?.quick_yuan || []).map(yuan => (
            <Button disabled={busy} key={yuan} onClick={() => void createOrder(yuan)} type="button" variant="secondary">
              {yuan} 元
            </Button>
          ))}
        </div>
        {order ? (
          <div className="mt-4 rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-chat-surface-background) p-3">
            <div className="text-xs text-(--ui-text-tertiary)">订单</div>
            <div className="mt-1 truncate font-mono text-sm">{order.order_id}</div>
            <div className="mt-1 text-xs text-(--ui-text-secondary)">
              {order.yuan} 元 / {order.credits} 积分
            </div>
            {order.qr_image_url ? (
              <img alt="充值二维码" className="mt-3 size-44 rounded-md bg-white p-2" src={order.qr_image_url} />
            ) : order.mock ? (
              <Button className="mt-3" disabled={busy} onClick={() => void confirmMock()} size="sm" type="button">
                模拟确认到账
              </Button>
            ) : null}
          </div>
        ) : null}
        {error ? <p className="mt-3 text-xs text-destructive">{error}</p> : null}
        {message ? <p className="mt-3 text-xs text-emerald-400">{message}</p> : null}
      </Panel>

      <Panel>
        <form className="grid gap-3" onSubmit={redeem}>
          <h3 className="text-[length:var(--conversation-text-font-size)] font-semibold">兑换码</h3>
          <Input onChange={event => setRedeemCode(event.target.value)} placeholder="输入兑换码" value={redeemCode} />
          <Button disabled={busy || !redeemCode.trim()} size="sm" type="submit">
            兑换
          </Button>
        </form>
        <div className="mt-5 border-t border-(--ui-stroke-tertiary) pt-4">
          <h3 className="text-[length:var(--conversation-text-font-size)] font-semibold">最近流水</h3>
          <div className="mt-2 grid gap-1.5">
            {transactions.length ? (
              transactions.map((item, index) => (
                <div
                  className="grid grid-cols-[1fr_auto] gap-2 rounded-md bg-(--ui-chat-surface-background) px-3 py-2 text-xs"
                  key={`${item.ts || 0}-${index}`}
                >
                  <span className="truncate text-(--ui-text-secondary)">{item.note || item.kind}</span>
                  <span className="font-semibold tabular-nums">
                    {item.delta > 0 ? '+' : ''}
                    {money(item.delta)}
                  </span>
                </div>
              ))
            ) : (
              <div className="rounded-md border border-dashed border-(--ui-stroke-tertiary) px-3 py-6 text-center text-xs text-(--ui-text-tertiary)">
                暂无钱包流水
              </div>
            )}
          </div>
        </div>
      </Panel>
    </div>
  )
}

export function TeamSection() {
  const [subtree, setSubtree] = useState<DesktopAccountSubtree | null>(null)
  const [wallet, setWallet] = useState<DesktopAccountWallet | null>(null)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const [roles, setRoles] = useState<string[]>([])
  const [resourcesByNode, setResourcesByNode] = useState<Record<string, TeamResource[]>>({})

  const nodes = useMemo(() => subtree?.nodes ?? [], [subtree])
  const edges = useMemo(() => subtree?.edges ?? [], [subtree])

  const load = useCallback(async () => {
    const account = accountBridge()

    if (!account?.subtree) {
      setError('桌面账号服务不可用')

      return
    }

    setBusy(true)
    setError('')

    try {
      const [tree, nextWallet, roleList] = await Promise.all([
        account.subtree(),
        account.wallet?.().catch(() => null),
        account.roles?.().catch(() => ({ roles: [] }))
      ])

      setSubtree(tree)
      setWallet(nextWallet)
      setRoles(roleList?.roles ?? [])

      // 卡片上展示「每个子有哪些知识库」,从**本地**后端读(authoritative 副本在主本地);失败不影响团队树。
      // 角色集 / 授权(角色→资源)已挪到「角色管理」「权限管理」两个菜单分区。
      try {
        const res = await window.hermesDesktop?.api?.<{ by_node?: Record<string, TeamResource[]> }>({
          path: '/api/kari/resources'
        })

        setResourcesByNode(res?.by_node ?? {})
      } catch {
        setResourcesByNode({})
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  const connect = useCallback(
    async (managerId: string, memberId: string) => {
      const account = accountBridge()

      if (!account?.createRelation) {
        setError('桌面账号服务不可用')

        return
      }

      setBusy(true)
      setError('')
      setMessage('')

      try {
        await account.createRelation({ manager_id: managerId, member_id: memberId })
        setMessage('协同连接已建立')
        await load()
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err))
      } finally {
        setBusy(false)
      }
    },
    [load]
  )

  const createSub = useCallback(
    async (payload: { email: string; name: string; password: string }) => {
      const account = accountBridge()

      if (!account?.createSubaccount) {
        return { ok: false, error: '桌面账号服务不可用' }
      }

      setError('')
      setMessage('')

      try {
        await account.createSubaccount(payload)
        setMessage('子账号已创建,已自动归属到你名下')
        await load()

        return { ok: true }
      } catch (err) {
        return { ok: false, error: err instanceof Error ? err.message : String(err) }
      }
    },
    [load]
  )

  const setRoleFor = useCallback(
    async (userId: string, role: string) => {
      const account = accountBridge()

      if (!account?.setRole) {
        return
      }

      try {
        await account.setRole({ user_id: userId, role })
        await load()
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err))
      }
    },
    [load]
  )

  return (
    <div className="max-w-5xl">
      <Panel>
        <SectionHeader
          description="拖动卡片排布;从卡片底部圆点拖线建立协同连接(上级→下级),并在子账号卡上分配角色。角色集在「角色管理」分区定义、给角色授权资源在「权限管理」分区。创建子账号自动归属到你名下。"
          title="团队账号"
        />

        <div className="mt-4 grid gap-3 sm:grid-cols-3">
          <InfoBlock label="共享额度" value={`${money(wallet?.balance)} 积分`} />
          <InfoBlock label="账号数" value={String(nodes.length)} />
          <InfoBlock label="协同连接" value={String(edges.length)} />
        </div>

        <div className="mt-4">
          <TeamCanvas
            busy={busy}
            edges={edges}
            nodes={nodes}
            onConnect={(managerId, memberId) => void connect(managerId, memberId)}
            onCreateSubaccount={createSub}
            onRefresh={() => void load()}
            onSetRole={(userId, role) => void setRoleFor(userId, role)}
            resourcesByNode={resourcesByNode}
            roles={roles}
            rootId={subtree?.root || ''}
          />
        </div>

        {error ? <p className="mt-3 text-xs text-destructive">{error}</p> : null}
        {message ? <p className="mt-3 text-xs text-emerald-400">{message}</p> : null}
      </Panel>
    </div>
  )
}

// 角色管理:定义组织角色集(主账号自定义)。分配给具体子账号在「团队账号」画布卡片上做。
export function RolesSection() {
  const [roles, setRoles] = useState<string[]>([])
  const [error, setError] = useState('')

  const reload = useCallback(async () => {
    const account = accountBridge()

    try {
      const res = account?.roles ? await account.roles() : { roles: [] }
      setRoles(res?.roles ?? [])
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }, [])

  useEffect(() => {
    void reload()
  }, [reload])

  const addRole = useCallback(async (name: string) => {
    const account = accountBridge()

    if (!account?.addRole) {
      return
    }

    try {
      setRoles((await account.addRole({ name })).roles ?? [])
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }, [])

  const removeRole = useCallback(async (name: string) => {
    const account = accountBridge()

    if (!account?.removeRole) {
      return
    }

    try {
      setRoles((await account.removeRole({ name })).roles ?? [])
      // 角色没了 → 清掉它在**主本地**的授权,免得同名新角色继承旧授权(best-effort)。
      await window.hermesDesktop?.api?.({ body: { role: name }, method: 'POST', path: '/api/kari/grants/clear-role' })
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }, [])

  return (
    <div className="max-w-3xl">
      <Panel>
        <SectionHeader
          description="定义组织的角色集(如 财务 / 客服 / 研发)。在「团队账号」的子账号卡片上把角色分配给具体子账号;在「权限管理」给角色勾选可用资源。"
          title="角色管理"
        />
        <div className="mt-4">
          <RolesPanel onAddRole={name => void addRole(name)} onRemoveRole={name => void removeRole(name)} roles={roles} />
        </div>
        {error ? <p className="mt-3 text-xs text-destructive">{error}</p> : null}
      </Panel>
    </div>
  )
}

// 权限管理:点角色 → 勾选它可用的资源(知识库/工作流/智能体),写主本地 grant_policy。
export function PermissionsSection() {
  const [roles, setRoles] = useState<string[]>([])
  const [grants, setGrants] = useState<TeamGrant[]>([])
  const [resourcesByNode, setResourcesByNode] = useState<Record<string, TeamResource[]>>({})
  const [nodes, setNodes] = useState<DesktopAccountTreeNode[]>([])
  const [langflowCapable, setLangflowCapable] = useState(true)
  const [error, setError] = useState('')

  const reload = useCallback(async () => {
    const account = accountBridge()

    try {
      const tree = account?.subtree ? await account.subtree() : null
      setNodes(tree?.nodes ?? [])
      const roleList = account?.roles ? await account.roles() : { roles: [] }
      setRoles(roleList?.roles ?? [])

      // 资源注册表 + 授权策略都从**本地**后端读(authoritative 副本在主本地,非云端)。
      try {
        const [res, grantRes] = await Promise.all([
          window.hermesDesktop?.api?.<{ by_node?: Record<string, TeamResource[]>; langflow_capable?: boolean }>({
            path: '/api/kari/resources'
          }),
          window.hermesDesktop?.api?.<{ grants?: TeamGrant[] }>({ path: '/api/kari/grants' })
        ])

        setResourcesByNode(res?.by_node ?? {})
        setGrants(grantRes?.grants ?? [])
        setLangflowCapable(res?.langflow_capable ?? true)
      } catch {
        setResourcesByNode({})
        setGrants([])
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }, [])

  useEffect(() => {
    void reload()
  }, [reload])

  // 授权策略改在**主本地**(走 api())。乐观更新本地 grants;失败回滚 + 报错。
  const grantResource = useCallback(async (grant: TeamGrant) => {
    setGrants(cur => (cur.some(g => sameGrant(g, grant)) ? cur : [...cur, grant]))

    try {
      await window.hermesDesktop?.api?.({ body: grant, method: 'POST', path: '/api/kari/grants' })
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      setGrants(cur => cur.filter(g => !sameGrant(g, grant)))
    }
  }, [])

  const revokeResource = useCallback(async (grant: TeamGrant) => {
    setGrants(cur => cur.filter(g => !sameGrant(g, grant)))

    try {
      await window.hermesDesktop?.api?.({ body: grant, method: 'POST', path: '/api/kari/grants/delete' })
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      setGrants(cur => (cur.some(g => sameGrant(g, grant)) ? cur : [...cur, grant]))
    }
  }, [])

  return (
    <div className="max-w-3xl">
      <Panel>
        <SectionHeader
          description="点一个角色 → 勾选它可用的工作流(MCP)。下级向上借能力统一走工作流授权;智能体委派默认开、知识库走查询工作流,均不在此单独授权。"
          title="权限管理"
        />
        <div className="mt-4">
          <PermissionsPanel
            grants={grants}
            langflowCapable={langflowCapable}
            nodes={nodes}
            onGrant={grant => void grantResource(grant)}
            onRevoke={grant => void revokeResource(grant)}
            resourcesByNode={resourcesByNode}
            roles={roles}
          />
        </div>
        {error ? <p className="mt-3 text-xs text-destructive">{error}</p> : null}
      </Panel>
    </div>
  )
}

export default AccountManagementView
