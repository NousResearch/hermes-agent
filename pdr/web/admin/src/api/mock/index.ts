/**
 * 示例数据实现：与真实接口同一个 AdminApi，数据保存在浏览器本地（清除站点数据即可还原）。
 * 按登录账号的权限与数据范围过滤，无权限的调用返回 403，与后端约定一致。
 */
import { ApiError } from '../errors'
import type {
  AdminApi, Asset, AuditLog, ClassifyItem, ConfigChangeLog, ConfigSchema, ConfigScope, ConfigValue, Dashboard, Perm,
  ReviewBlock, Role, Stage, StageCount, UserRow
} from '../types'
import * as D from './data'

const KEY = 'pdr-admin-mock-v1'
const STAGES: Stage[] = ['upload', 'parse', 'review', 'label', 'classify', 'store', 'apply']

interface State {
  assets: Asset[]
  blocks: Record<string, ReviewBlock[]>
  segments: typeof D.MEDIA_SEGMENTS
  labels: typeof D.LABEL_ITEMS
  classified: Record<string, string>
  kbEnabled: Record<string, boolean>
  overrides: Record<ConfigScope, Record<string, ConfigValue>>
  configLog: ConfigChangeLog[]
  toolsets: Record<string, boolean>
  skills: Record<string, boolean>
  persona: string
  memory: typeof D.MEMORY_ENTRIES
  cron: typeof D.CRON_JOBS
  roles: Role[]
  userStatus: Record<string, UserRow['status']>
  audit: AuditLog[]
  seq: number
}

const clone = <T>(v: T): T => JSON.parse(JSON.stringify(v))

function fresh(): State {
  return {
    assets: clone(D.ASSETS), blocks: { A20261116001: clone(D.REVIEW_BLOCKS) }, segments: clone(D.MEDIA_SEGMENTS),
    labels: clone(D.LABEL_ITEMS), classified: {}, kbEnabled: {}, overrides: { platform: {}, tenant: {} },
    configLog: [
      { at: '11-15 17:30', user: '张磊', scope: 'tenant', changes: [{ path: 'hermes.memory.write_approval', label: '记忆写入审批', before: false, after: true }] }
    ],
    toolsets: {}, skills: {}, persona: D.PERSONA, memory: clone(D.MEMORY_ENTRIES), cron: clone(D.CRON_JOBS),
    roles: clone(D.ROLES), userStatus: {}, audit: clone(D.AUDIT_LOGS), seq: 14
  }
}

let state: State = (() => {
  try {
    const raw = localStorage.getItem(KEY)
    if (raw) return { ...fresh(), ...JSON.parse(raw) } as State
  } catch {
    // 本地存储不可用时每次刷新从示例数据开始
  }
  return fresh()
})()

function save(): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(state))
  } catch {
    // 只影响刷新后是否保留示例操作
  }
}

let account = ''
export function setMockToken(token: string): void {
  account = token.startsWith('mock.') ? token.slice(5) : ''
}

const wait = (ms = 160) => new Promise((r) => setTimeout(r, ms))

function me() {
  const user = D.USERS[account]
  if (!user) throw new ApiError(401, '登录已过期，请重新登录')
  return user
}

function can(perm: Perm): boolean {
  const { perms } = me()
  return perms === '*' || perms.includes(perm)
}

/** 后端同样会校验：页面入口按权限隐藏，直接调用接口也返回 403 */
function need(...perms: Perm[]): void {
  if (!perms.some(can)) throw new ApiError(403, '没有权限执行此操作')
}

function log(module: string, action: string, target: string, result: AuditLog['result'] = 'ok'): void {
  state.audit.unshift({ id: 'g' + ++state.seq, at: now(), user: me().name, module, action, target, ip: '10.2.1.5', result })
}

function now(): string {
  const d = new Date()
  const p = (n: number) => String(n).padStart(2, '0')
  return `${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`
}

const visibleAssets = () => state.assets.filter(me().visible)

function asset(no: string): Asset {
  const found = visibleAssets().find((a) => a.no === no)
  if (!found) throw new ApiError(404, '资产不存在或不在你的数据范围内')
  return found
}

function advance(a: Asset, stage: Stage, status: Asset['status'], note: string): void {
  Object.assign(a, { stage, status, note, updatedAt: now() })
}

// 1068 项配置的表单数据较大，打开配置页时才加载
const loadSchema = async () => (await import('../../data/config-schema.json')).default as unknown as ConfigSchema

export const mockApi: AdminApi = {
  mode: 'mock',

  async demoAccounts() {
    return D.DEMO_ACCOUNTS
  },

  async login(acc, password) {
    await wait(260)
    const key = acc.trim().toLowerCase()
    if (!key || !password) throw new ApiError(400, '请输入账号和密码')
    if (['chenxiao', 'liuyang', 'zhoujing'].includes(key)) {
      throw new ApiError(403, '这是用户端账号，不能登录管理端')
    }
    const id = D.USERS[key] ? key : 'zhanglei'
    setMockToken('mock.' + id)
    const { bell: _bell, visible: _visible, ...user } = D.USERS[id]
    return { token: 'mock.' + id, user }
  },

  async logout() {
    account = ''
  },

  async notices() {
    // 每条通知对应一个权限，只推送给能处理它的人
    const NEED: Record<string, Perm> = { n1: 'content:retry', n2: 'content:review', n3: 'agent:memory', n4: 'agent:cron', n5: 'content:label' }
    return D.NOTICES.filter((n) => can(NEED[n.id]))
  },

  async dashboard(): Promise<Dashboard> {
    await wait()
    const rows = visibleAssets()
    const mine = rows.filter((a) =>
      (a.stage === 'review' && can('content:review')) || (a.stage === 'label' && can('content:label')) ||
      (a.stage === 'classify' && can('content:classify')) || (a.status === 'failed' && can('content:retry')) || a.status === 'done')
    const handled = rows.filter((a) => ['review', 'label', 'classify'].includes(a.stage)).length
    const scale = rows.length / D.ASSETS.length
    return {
      kpis: [
        { key: 'uploaded', name: '本周上传', value: Math.round(66 * scale), delta: 12, tint: 'purple' },
        { key: 'manual', name: '待人工处理', value: handled, delta: -6, tint: 'amber' },
        { key: 'stored', name: '本周入库', value: Math.round(41 * scale), delta: 8, tint: 'green' },
        { key: 'failed', name: '失败任务', value: rows.filter((a) => a.status === 'failed').length, delta: 0, tint: 'pink' }
      ],
      myTasks: mine,
      trend: [['周一', 18], ['周二', 26], ['周三', 21], ['周四', 34], ['周五', 29], ['周六', 12], ['周日', 23]]
        .map(([day, v]) => ({ day: day as string, value: Math.round((v as number) * Math.max(scale, 0.3)) })),
      distribution: [
        { name: '待人工处理', qty: Math.round(18 * scale), color: '#5443D6' },
        { name: '处理中', qty: Math.round(7 * scale), color: '#E0A12E' },
        { name: '已入库', qty: Math.round(41 * scale), color: '#2FAE6E' }
      ],
      weekPlan: { planned: 24, stored: 16, reviewing: 8 },
      events: [
        { date: '11-17', title: '每日用量同步', tag: '定时任务' },
        { date: '11-18', title: '外滩店酒店合同到期复核', tag: '合同' },
        { date: '11-20', title: '月度内容质量报告', tag: '报告' },
        { date: '11-25', title: 'MinerU 模型升级窗口', tag: '运维' }
      ],
      usage: { storageUsedGb: 38, storageQuotaGb: 100, parsedPages: 12480, mediaMinutes: 640 }
    }
  },

  async assets(query) {
    need('content:asset')
    await wait()
    const rows = visibleAssets()
    const kw = (query.keyword || '').trim().toLowerCase()
    const filtered = rows.filter((a) =>
      (!query.modality || a.modality === query.modality) &&
      (!kw || a.no.toLowerCase().includes(kw) || a.name.toLowerCase().includes(kw)))
    const stages: StageCount[] = [{ stage: 'all', qty: filtered.length },
      ...STAGES.map((s) => ({ stage: s, qty: filtered.filter((a) => a.stage === s).length }))]
    const stage = query.stage || 'all'
    return { rows: filtered.filter((a) => stage === 'all' || a.stage === stage), stages }
  },

  async upload(files) {
    need('content:upload')
    await wait(400)
    const user = me()
    const created = files.map((f) => {
      const ext = f.name.split('.').pop()?.toLowerCase() || ''
      const modality: Asset['modality'] = ext === 'pdf' ? 'PDF' : ['doc', 'docx'].includes(ext) ? 'Word'
        : ['xls', 'xlsx', 'csv'].includes(ext) ? 'Excel' : ['md', 'markdown', 'txt'].includes(ext) ? 'Markdown'
          : ['mp4', 'mov', 'mkv'].includes(ext) ? '视频' : ['mp3', 'wav', 'm4a'].includes(ext) ? '音频' : '图片'
      const no = 'A20261116' + String(100 + ++state.seq).slice(-3)
      const size = f.size > 1 << 20 ? (f.size / (1 << 20)).toFixed(1) + ' MB' : Math.max(1, Math.round(f.size / 1024)) + ' KB'
      return {
        no, name: f.name, modality, region: user.roleCode === 'content_reviewer' ? '黄浦区' : '黄浦区', stage: 'parse', status: 'running',
        note: 'MinerU basic 档位排队中', owner: '系统', sizeText: size, uploadedAt: now(), updatedAt: now(), attempts: 1, maxAttempts: 3
      } satisfies Asset
    })
    state.assets.unshift(...created)
    created.forEach((a) => log('内容任务', '上传文件', `${a.no} ${a.name}`))
    save()
    return created
  },

  async retry(no) {
    need('content:retry')
    await wait(300)
    const a = asset(no)
    advance(a, 'parse', 'running', '人工重新执行，已进入解析队列')
    a.attempts = 1
    log('内容任务', '重新执行失败任务', `${a.no} ${a.name}`)
    save()
    return a
  },

  async contentBoard() {
    need('content:dashboard')
    await wait()
    return {
      funnel: [
        { stage: 'upload', name: '上传', qty: 66 }, { stage: 'parse', name: '解析', qty: 63 }, { stage: 'review', name: '人工核对', qty: 58 },
        { stage: 'label', name: '标注', qty: 50 }, { stage: 'classify', name: '分类', qty: 47 }, { stage: 'store', name: '入库', qty: 43 },
        { stage: 'apply', name: '应用', qty: 41 }
      ],
      byModality: [
        { modality: 'PDF', qty: 24, failed: 1 }, { modality: 'Word', qty: 9, failed: 1 }, { modality: 'Excel', qty: 7, failed: 0 },
        { modality: 'Markdown', qty: 6, failed: 0 }, { modality: '图片', qty: 12, failed: 0 }, { modality: '视频', qty: 5, failed: 0 },
        { modality: '音频', qty: 3, failed: 0 }
      ],
      failures: [
        { code: 'MINERU_TIMEOUT', name: '解析超时', qty: 2, hint: '大文件或扫描件，建议提高档位或拆分后重新上传' },
        { code: 'DOCX_CORRUPTED', name: '文件损坏', qty: 1, hint: '用 Office 另存为后重新上传' },
        { code: 'ASR_LOW_CONF', name: '语音识别置信度低', qty: 1, hint: '已进入人工核对，无需处理' }
      ],
      avgMinutes: [
        { stage: 'parse', name: '解析', minutes: 3.2 }, { stage: 'review', name: '人工核对', minutes: 46 },
        { stage: 'label', name: '标注', minutes: 18 }, { stage: 'classify', name: '分类', minutes: 6 }, { stage: 'store', name: '入库', minutes: 1.4 }
      ]
    }
  },

  async reviewDoc(no) {
    need('content:review')
    await wait()
    const a = asset(no)
    if (!state.blocks[no]) state.blocks[no] = clone(D.REVIEW_BLOCKS)
    return { asset: a, page: 3, pageCount: 12, tier: 'basic', blocks: clone(state.blocks[no]) }
  },

  async submitReview(no, blocks, action, tier) {
    need('content:review')
    await wait(300)
    const a = asset(no)
    state.blocks[no] = clone(blocks)
    if (action === 'submit') {
      advance(a, 'classify', 'confirm', '核对完成，模型建议：产品资料 / 线路手册（0.92）')
      log('人工核对', '提交核对', `${a.no} ${a.name}`)
    } else if (action === 'reject') {
      advance(a, 'parse', 'running', `已驳回，使用 ${tier} 档位重新解析`)
      log('人工核对', '驳回并重新解析', `${a.no} ${a.name}（${tier}）`)
    } else {
      log('人工核对', '保存核对草稿', `${a.no} ${a.name}`)
    }
    save()
  },

  async mediaReview(no) {
    need('content:review')
    await wait()
    return { asset: asset(no), duration: 276, frames: D.MEDIA_FRAMES, segments: clone(state.segments) }
  },

  async submitMedia(no, segments) {
    need('content:review')
    await wait(300)
    const a = asset(no)
    state.segments = clone(segments)
    advance(a, 'classify', 'confirm', '核对完成，模型建议：培训素材 / 视频课程（0.90）')
    log('人工核对', '提交音视频核对', `${a.no} ${a.name}`)
    save()
  },

  async reviewQueue() {
    need('content:label', 'content:classify')
    await wait()
    const rows = visibleAssets()
    const labels = can('content:label') ? state.labels.filter((l) => rows.some((a) => a.no === l.assetNo)) : []
    const classify: ClassifyItem[] = can('content:classify')
      ? rows.filter((a) => a.stage === 'classify' && D.CLASSIFY_SUMMARY[a.no]).map((a) => ({
        asset: a, ...D.CLASSIFY_SUMMARY[a.no], chosen: state.classified[a.no] ?? null
      }))
      : []
    return { label: clone(labels), classify, categories: D.CATEGORIES }
  },

  async saveLabel(id, label) {
    need('content:label')
    await wait(120)
    const item = state.labels.find((l) => l.id === id)
    if (!item) throw new ApiError(404, '标注项不存在')
    Object.assign(item, { label, status: 'done' })
    const left = state.labels.filter((l) => l.assetNo === item.assetNo && l.status === 'pending').length
    const a = state.assets.find((x) => x.no === item.assetNo)
    if (a) {
      if (left === 0) advance(a, 'classify', 'confirm', '标注完成，待分类确认')
      else a.note = `还剩 ${left} 条待确认`
    }
    if (left === 0) log('人工核对', '提交标注', `${item.assetNo} ${item.assetName}`)
    save()
  },

  async confirmCategory(no, code) {
    need('content:classify')
    await wait(200)
    const a = asset(no)
    const name = D.CATEGORIES.flatMap((c) => (c.children || []).map((s) => ({ code: s.code, name: `${c.name} / ${s.name}` })))
      .find((c) => c.code === code)?.name || code
    state.classified[no] = code
    a.category = name
    advance(a, 'store', 'running', `已归类「${name}」，入库中`)
    log('人工核对', '确认分类', `${a.no} ${a.name} → ${name}`)
    save()
  },

  async knowledgeBases() {
    need('knowledge')
    await wait()
    return D.KNOWLEDGE_BASES.map((k) => ({ ...k, enabled: state.kbEnabled[k.code] ?? k.enabled }))
  },

  async knowledgeDocs(code) {
    need('knowledge')
    await wait(120)
    const visible = me().roleCode === 'content_reviewer'
      ? (d: { region: string }) => ['黄浦区', '浦东新区'].includes(d.region) : () => true
    return (D.KNOWLEDGE_DOCS[code] || []).filter(visible)
  },

  async toggleKnowledgeBase(code, enabled) {
    need('knowledge')
    if (!can('sys:role') && me().perms !== '*') throw new ApiError(403, '只有租户管理员可以启用或停用知识库')
    await wait(150)
    state.kbEnabled[code] = enabled
    log('知识库', enabled ? '启用知识库' : '停用知识库', D.KNOWLEDGE_BASES.find((k) => k.code === code)?.name || code)
    save()
  },

  async configSchema(scope) {
    need(scope === 'platform' ? 'sys:config' : 'agent:config', 'sys:config')
    if (scope === 'platform' && !can('sys:config')) throw new ApiError(403, '只有平台超级管理员可以修改 config.yaml')
    const schema = await loadSchema()
    const values = { ...state.overrides.platform, ...(scope === 'tenant' ? state.overrides.tenant : {}) }
    const groups = schema.groups
      .filter((g) => scope === 'platform' || g.scope === 'agent')
      .map((g) => ({ ...g, items: g.items.map((i) => (i.path in values ? { ...i, value: values[i.path] } : i)) }))
    return { ...schema, groups }
  },

  async saveConfig(scope, changes) {
    if (scope === 'platform') need('sys:config')
    else need('agent:config', 'sys:config')
    await wait(300)
    for (const c of changes) state.overrides[scope][c.path] = c.after
    const entry: ConfigChangeLog = {
      at: now(), user: me().name, scope, changes,
      sha256: scope === 'platform' ? Array.from({ length: 64 }, (_, i) => '0123456789abcdef'[(i * 7 + state.seq) % 16]).join('') : undefined
    }
    state.configLog.unshift(entry)
    log('智能体', scope === 'platform' ? '修改平台配置（config.yaml）' : '修改租户覆盖配置', changes.map((c) => c.path).join('、'))
    save()
    return entry
  },

  async configLog() {
    need('agent:config', 'sys:config')
    await wait(100)
    return state.configLog
  },

  async toolsets() {
    need('agent:skill')
    await wait()
    return {
      toolsets: D.TOOLSETS.map((t) => ({ ...t, enabled: state.toolsets[t.name] ?? t.enabled })),
      skills: D.SKILLS.map((s) => ({ ...s, enabled: state.skills[s.name] ?? s.enabled }))
    }
  },

  async toggleToolset(name, enabled) {
    need('agent:skill')
    state.toolsets[name] = enabled
    log('智能体', enabled ? '启用工具集' : '停用工具集', name)
    save()
  },

  async toggleSkill(name, enabled) {
    need('agent:skill')
    state.skills[name] = enabled
    log('智能体', enabled ? '启用技能' : '停用技能', name)
    save()
  },

  async memory() {
    need('agent:memory')
    await wait()
    return { persona: state.persona, entries: clone(state.memory), memoryCharLimit: 2200, userCharLimit: 1375 }
  },

  async savePersona(text) {
    need('agent:memory')
    await wait(200)
    state.persona = text
    log('智能体', '修改人格设定', 'SOUL.md')
    save()
  },

  async decideMemory(id, approve) {
    need('agent:memory')
    await wait(120)
    const entry = state.memory.find((m) => m.id === id)
    if (!entry) return
    if (approve) entry.status = 'active'
    else state.memory = state.memory.filter((m) => m.id !== id)
    log('智能体', approve ? '批准记忆写入' : '拒绝记忆写入', entry.content.slice(0, 24))
    save()
  },

  async deleteMemory(id) {
    need('agent:memory')
    const entry = state.memory.find((m) => m.id === id)
    state.memory = state.memory.filter((m) => m.id !== id)
    if (entry) log('智能体', '删除记忆', entry.content.slice(0, 24))
    save()
  },

  async cronJobs() {
    need('agent:cron')
    await wait()
    return { jobs: clone(state.cron), platforms: D.PLATFORMS }
  },

  async toggleCron(id, enabled) {
    need('agent:cron')
    const job = state.cron.find((j) => j.id === id)
    if (!job) return
    job.enabled = enabled
    log('智能体', enabled ? '启用定时任务' : '暂停定时任务', job.name)
    save()
  },

  async runCron(id) {
    need('agent:cron')
    await wait(500)
    const job = state.cron.find((j) => j.id === id)
    if (!job) throw new ApiError(404, '定时任务不存在')
    Object.assign(job, { lastRun: now(), lastStatus: 'ok' })
    log('智能体', '立即执行定时任务', job.name)
    save()
    return clone(job)
  },

  async qaSessions(keyword) {
    need('agent:audit')
    await wait()
    const kw = keyword.trim()
    return D.QA_SESSIONS.filter((s) => !kw || s.question.includes(kw) || s.user.includes(kw))
  },

  async permCatalog() {
    need('sys:role')
    return D.PERM_CATALOG
  },

  async roles() {
    need('sys:role')
    await wait()
    return clone(state.roles)
  },

  async saveRole(role) {
    need('sys:role')
    await wait(250)
    const i = state.roles.findIndex((r) => r.code === role.code)
    if (i >= 0) state.roles[i] = clone(role)
    else state.roles.push(clone(role))
    log('权限管理', i >= 0 ? '修改角色' : '新建角色', role.name)
    save()
  },

  async regions() {
    need('sys:user', 'sys:role')
    return D.REGIONS
  },

  async users() {
    need('sys:user')
    await wait()
    return D.USER_ROWS.map((u) => ({ ...u, status: state.userStatus[u.account] ?? u.status }))
  },

  async setUserStatus(acc, status) {
    need('sys:user')
    await wait(150)
    state.userStatus[acc] = status
    log('权限管理', status === 'active' ? '启用账号' : '停用账号', acc)
    save()
  },

  async tenants() {
    need('sys:tenant')
    await wait()
    return D.TENANTS
  },

  async auditLogs(module) {
    need('sys:audit')
    await wait()
    return state.audit.filter((g) => !module || g.module === module)
  }
}

/** 恢复示例数据（演示时把操作过的数据还原） */
export function resetMockData(): void {
  state = fresh()
  save()
}
