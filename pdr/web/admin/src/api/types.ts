/**
 * 管理端接口约定。示例数据（mock）与真实接口（http，/api/admin/v1）实现同一个 AdminApi，
 * 页面只依赖这里的类型。统一响应 { code, message, data }，code 非 0 为业务错误；
 * 没有权限的接口返回 403（页面入口本身也按权限隐藏）。
 */

/** 功能权限编码：菜单页面与按钮共用；'*' 表示平台超级管理员 */
export type Perm =
  | 'content:dashboard' | 'content:asset' | 'content:upload' | 'content:retry'
  | 'content:review' | 'content:label' | 'content:classify'
  | 'knowledge'
  | 'agent:config' | 'agent:skill' | 'agent:memory' | 'agent:cron' | 'agent:audit'
  | 'sys:role' | 'sys:user' | 'sys:tenant' | 'sys:config' | 'sys:audit'

export interface AdminUser {
  account: string
  name: string
  roleCode: string
  roleName: string
  tenantCode: string
  tenantName: string
  /** 归属区域与数据范围说明，页面上用于“只显示数据范围内的数据”提示 */
  regionName: string
  dataScopeText: string
  perms: Perm[] | '*'
}

export interface DemoAccount { account: string; name: string; roleName: string; note: string }

// ---------- 内容流水线 ----------

export type Modality = 'PDF' | 'Word' | 'Excel' | 'Markdown' | '图片' | '视频' | '音频'
export type Stage = 'upload' | 'parse' | 'review' | 'label' | 'classify' | 'store' | 'apply'
export type AssetStatus = 'running' | 'pending' | 'confirm' | 'retrying' | 'failed' | 'done'

export interface Asset {
  no: string
  name: string
  modality: Modality
  region: string
  stage: Stage
  status: AssetStatus
  note: string
  owner: string
  sizeText: string
  uploadedAt: string
  updatedAt: string
  attempts: number
  maxAttempts: number
  category?: string
}

export interface AssetQuery { stage?: Stage | 'all'; modality?: Modality | ''; keyword?: string }

export interface StageCount { stage: Stage | 'all'; qty: number }

export interface Kpi { key: string; name: string; value: number; delta: number; unit?: string; tint: 'purple' | 'green' | 'amber' | 'pink' }

export interface TrendPoint { day: string; value: number }
export interface Slice { name: string; qty: number; color: string }
export interface CalendarEvent { date: string; title: string; tag: string }

export interface Dashboard {
  kpis: Kpi[]
  myTasks: Asset[]
  trend: TrendPoint[]
  distribution: Slice[]
  weekPlan: { planned: number; stored: number; reviewing: number }
  events: CalendarEvent[]
  usage: { storageUsedGb: number; storageQuotaGb: number; parsedPages: number; mediaMinutes: number }
}

export interface ContentBoard {
  funnel: { stage: Stage; name: string; qty: number }[]
  byModality: { modality: Modality; qty: number; failed: number }[]
  failures: { code: string; name: string; qty: number; hint: string }[]
  avgMinutes: { stage: Stage; name: string; minutes: number }[]
}

export type BlockType = 'title' | 'text' | 'table' | 'image' | 'footer'
/** 0 未核对 1 已确认 2 已修改 3 标记删除 */
export type BlockStatus = 0 | 1 | 2 | 3

export interface ReviewBlock {
  id: string
  type: BlockType
  conf: number
  origin: string
  /** 修正建议（如 OCR 识别错误），采纳后显示为修改后的文本 */
  fix?: string
  status: BlockStatus
  edited?: string
}

export interface ReviewDoc {
  asset: Asset
  page: number
  pageCount: number
  tier: string
  blocks: ReviewBlock[]
}

export interface MediaFrame { t: number; kind: '场景切换' | '固定间隔'; label: string; ocr: string; seg: number }
export interface MediaSegment { a: number; b: number; speaker: string; text: string; conf: number; status: BlockStatus }
export interface MediaReview { asset: Asset; duration: number; frames: MediaFrame[]; segments: MediaSegment[] }

export interface LabelItem {
  id: string
  assetNo: string
  assetName: string
  modality: Modality
  /** 被标注的片段：音频时间段或图片区域说明 */
  target: string
  content: string
  modelLabel: string
  conf: number
  options: string[]
  label: string | null
  status: 'pending' | 'done'
}

export interface CategoryNode { code: string; name: string; children?: CategoryNode[] }

export interface ClassifyItem {
  asset: Asset
  summary: string
  suggestions: { code: string; name: string; conf: number }[]
  chosen: string | null
}

export interface ReviewQueue { label: LabelItem[]; classify: ClassifyItem[]; categories: CategoryNode[] }

// ---------- 知识库 ----------

export interface KnowledgeBase {
  code: string
  name: string
  desc: string
  docQty: number
  chunkQty: number
  scopeText: string
  updatedAt: string
  enabled: boolean
}

export interface KnowledgeDoc {
  assetNo: string
  name: string
  modality: Modality
  chunkQty: number
  region: string
  storedAt: string
  hitQty: number
}

// ---------- 智能体 ----------

export type ConfigKind = 'bool' | 'enum' | 'number' | 'text' | 'json'
export type ConfigValue = string | number | boolean | null | unknown[] | Record<string, unknown>

export interface ConfigItem {
  path: string
  label: string
  desc: string
  range: string
  /** 生效方式：热加载 / 下一会话 / 重启 */
  reload: string
  kind: ConfigKind
  options?: { value: string; label: string }[]
  min?: number
  max?: number
  nullable?: boolean
  value: ConfigValue
}

export interface ConfigGroup { key: string; name: string; scope: 'pdr' | 'agent'; items: ConfigItem[] }
export interface ConfigSchema { source: string; sha256: string; itemQty: number; groups: ConfigGroup[] }

/** platform：写入 config.yaml（仅平台超级管理员）；tenant：租户覆盖，写入数据库，不改 config.yaml */
export type ConfigScope = 'platform' | 'tenant'

export interface ConfigChange { path: string; label: string; before: ConfigValue; after: ConfigValue }
export interface ConfigChangeLog { at: string; user: string; scope: ConfigScope; changes: ConfigChange[]; sha256?: string }

export interface Toolset { name: string; label: string; desc: string; toolQty: number; enabled: boolean }
export interface Skill { name: string; category: string; desc: string; source: '内置' | '可选' | '租户自建'; enabled: boolean }

export interface MemoryEntry {
  id: string
  kind: 'memory' | 'user'
  content: string
  source: string
  createdAt: string
  status: 'active' | 'pending'
}

export interface MemoryState { persona: string; entries: MemoryEntry[]; memoryCharLimit: number; userCharLimit: number }

export interface CronJob {
  id: string
  name: string
  schedule: string
  scheduleText: string
  deliver: string
  nextRun: string
  lastRun: string
  lastStatus: 'ok' | 'failed' | 'never'
  enabled: boolean
}

export interface GatewayPlatform { name: string; label: string; status: 'online' | 'offline' | 'disabled'; sessions: number; note: string }

export interface QaSession {
  id: string
  user: string
  region: string
  scopeName: string
  question: string
  answer: string
  citations: { title: string; locator: string }[]
  feedback: 'up' | 'down' | null
  turns: number
  createdAt: string
  durationMs: number
}

// ---------- 权限 ----------

export interface PermGroup { name: string; perms: { code: Perm; name: string }[] }

/** 数据范围：1 本租户全部 2 本区域及下级 3 本区域 4 本人创建 5 仅分配给本人 */
export type DataScope = '1' | '2' | '3' | '4' | '5'

export interface RoleDataScope { resource: string; resourceName: string; scope: DataScope; action: string }
/** 列权限：0 不可见 1 脱敏 2 只读 3 可编辑 */
export interface RoleColumn { table: string; col: string; name: string; perm: '0' | '1' | '2' | '3'; mask: string }

export interface Role {
  code: string
  name: string
  builtin: boolean
  userQty: number
  desc: string
  perms: Perm[] | '*'
  dataScopes: RoleDataScope[]
  columns: RoleColumn[]
}

export interface RegionNode { code: string; name: string; children?: RegionNode[] }

export interface UserRow {
  account: string
  name: string
  userType: 'admin' | 'portal'
  roleNames: string[]
  region: string
  phone: string
  status: 'active' | 'disabled'
  lastLoginAt: string
}

export interface Tenant {
  code: string
  name: string
  plan: string
  status: 'active' | 'trial' | 'suspended'
  adminName: string
  userQty: number
  storageUsedGb: number
  storageQuotaGb: number
  createdAt: string
}

export interface AuditLog {
  id: string
  at: string
  user: string
  module: string
  action: string
  target: string
  ip: string
  result: 'ok' | 'denied' | 'failed'
}

export interface Notice { id: string; title: string; at: string; read: boolean; link?: string }

export interface AdminApi {
  readonly mode: 'mock' | 'http'
  demoAccounts(): Promise<DemoAccount[]>
  login(account: string, password: string): Promise<{ token: string; user: AdminUser }>
  logout(): Promise<void>
  notices(): Promise<Notice[]>

  dashboard(): Promise<Dashboard>
  assets(query: AssetQuery): Promise<{ rows: Asset[]; stages: StageCount[] }>
  upload(files: { name: string; size: number }[]): Promise<Asset[]>
  retry(no: string): Promise<Asset>
  contentBoard(): Promise<ContentBoard>
  reviewDoc(no: string): Promise<ReviewDoc>
  submitReview(no: string, blocks: ReviewBlock[], action: 'draft' | 'submit' | 'reject', tier?: string): Promise<void>
  mediaReview(no: string): Promise<MediaReview>
  submitMedia(no: string, segments: MediaSegment[]): Promise<void>
  reviewQueue(): Promise<ReviewQueue>
  saveLabel(id: string, label: string): Promise<void>
  confirmCategory(no: string, code: string): Promise<void>

  knowledgeBases(): Promise<KnowledgeBase[]>
  knowledgeDocs(code: string): Promise<KnowledgeDoc[]>
  toggleKnowledgeBase(code: string, enabled: boolean): Promise<void>

  configSchema(scope: ConfigScope): Promise<ConfigSchema>
  saveConfig(scope: ConfigScope, changes: ConfigChange[]): Promise<ConfigChangeLog>
  configLog(): Promise<ConfigChangeLog[]>
  toolsets(): Promise<{ toolsets: Toolset[]; skills: Skill[] }>
  toggleToolset(name: string, enabled: boolean): Promise<void>
  toggleSkill(name: string, enabled: boolean): Promise<void>
  memory(): Promise<MemoryState>
  savePersona(text: string): Promise<void>
  decideMemory(id: string, approve: boolean): Promise<void>
  deleteMemory(id: string): Promise<void>
  cronJobs(): Promise<{ jobs: CronJob[]; platforms: GatewayPlatform[] }>
  toggleCron(id: string, enabled: boolean): Promise<void>
  runCron(id: string): Promise<CronJob>
  qaSessions(keyword: string): Promise<QaSession[]>

  permCatalog(): Promise<PermGroup[]>
  roles(): Promise<Role[]>
  saveRole(role: Role): Promise<void>
  regions(): Promise<RegionNode[]>
  users(): Promise<UserRow[]>
  setUserStatus(account: string, status: UserRow['status']): Promise<void>
  tenants(): Promise<Tenant[]>
  auditLogs(module: string): Promise<AuditLog[]>
}
