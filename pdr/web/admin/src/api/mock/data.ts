/**
 * 管理端示例数据：一家旅行社租户（示例旅行社）一周的内容处理情况。
 * 与用户端示例数据讲同一个故事（黄山宏村线路、外滩店、导游服务规范……），便于两端对照体验。
 */
import type {
  AdminUser, Asset, AuditLog, CategoryNode, ClassifyItem, CronJob, DemoAccount, GatewayPlatform, KnowledgeBase,
  KnowledgeDoc, LabelItem, MediaFrame, MediaSegment, MemoryEntry, Notice, Perm, PermGroup, QaSession, RegionNode,
  ReviewBlock, Role, Skill, Tenant, Toolset, UserRow
} from '../types'

export const TODAY = '2026-11-16'

/** 示例账号：账号决定角色；任意密码均可登录 */
export const USERS: Record<string, AdminUser & { bell: number; visible: (a: Asset) => boolean }> = {
  admin: {
    account: 'admin', name: '平台管理员', roleCode: 'super_admin', roleName: '平台超级管理员', tenantCode: 'demo_travel',
    tenantName: '示例旅行社', regionName: '全部', dataScopeText: '全部租户', perms: '*', bell: 7, visible: () => true
  },
  zhanglei: {
    account: 'zhanglei', name: '张磊', roleCode: 'tenant_admin', roleName: '租户管理员', tenantCode: 'demo_travel',
    tenantName: '示例旅行社', regionName: '示例旅行社', dataScopeText: '本租户全部',
    perms: ['content:dashboard', 'content:asset', 'content:upload', 'content:retry', 'content:review', 'content:label',
      'content:classify', 'knowledge', 'agent:config', 'agent:skill', 'agent:memory', 'agent:cron', 'agent:audit',
      'sys:role', 'sys:user', 'sys:audit'],
    bell: 7, visible: () => true
  },
  wangmin: {
    account: 'wangmin', name: '王敏', roleCode: 'content_reviewer', roleName: '内容审核员', tenantCode: 'demo_travel',
    tenantName: '示例旅行社', regionName: '上海市', dataScopeText: '上海市及下级',
    perms: ['content:dashboard', 'content:asset', 'content:review', 'content:classify', 'knowledge'],
    bell: 3, visible: (a) => ['黄浦区', '浦东新区'].includes(a.region)
  },
  lihang: {
    account: 'lihang', name: '李航', roleCode: 'content_labeler', roleName: '标注员', tenantCode: 'demo_travel',
    tenantName: '示例旅行社', regionName: '浦东新区', dataScopeText: '仅分配给本人',
    perms: ['content:asset', 'content:label'], bell: 2, visible: (a) => a.owner === '李航'
  }
}

export const DEMO_ACCOUNTS: DemoAccount[] = [
  { account: 'zhanglei', name: '张磊', roleName: '租户管理员', note: '本租户全部功能' },
  { account: 'wangmin', name: '王敏', roleName: '内容审核员', note: '核对、分类、知识库；上海市数据' },
  { account: 'lihang', name: '李航', roleName: '标注员', note: '只有任务列表与标注' },
  { account: 'admin', name: '平台管理员', roleName: '平台超级管理员', note: '含租户管理、系统配置' }
]

export const ASSETS: Asset[] = [
  { no: 'A20261116001', name: '2026春季华东线路产品手册.pdf', modality: 'PDF', region: '黄浦区', stage: 'review', status: 'pending', note: '42 个内容块，3 处低置信度', owner: '王敏', sizeText: '18.4 MB', uploadedAt: '11-16 09:12', updatedAt: '11-16 09:31', attempts: 1, maxAttempts: 3 },
  { no: 'A20261116002', name: '航空行程单_0312.jpg', modality: '图片', region: '黄浦区', stage: 'classify', status: 'confirm', note: '模型建议：发票 / 行程单（0.94）', owner: '王敏', sizeText: '1.2 MB', uploadedAt: '11-16 09:40', updatedAt: '11-16 09:42', attempts: 1, maxAttempts: 3 },
  { no: 'A20261116003', name: '外滩店酒店供应商合同.docx', modality: 'Word', region: '浦东新区', stage: 'parse', status: 'retrying', note: '第 2 次执行失败，60 秒后重试', owner: '系统', sizeText: '846 KB', uploadedAt: '11-16 10:05', updatedAt: '11-16 10:21', attempts: 2, maxAttempts: 3 },
  { no: 'A20261116004', name: '10月结算对账单.xlsx', modality: 'Excel', region: '黄浦区', stage: 'store', status: 'running', note: '回填 ods.ods_fin_invoice_extract_di', owner: '系统', sizeText: '312 KB', uploadedAt: '11-16 10:30', updatedAt: '11-16 10:48', attempts: 1, maxAttempts: 3 },
  { no: 'A20261116005', name: '黄山线路介绍.mp4', modality: '视频', region: '黄浦区', stage: 'review', status: 'pending', note: '语音片段与关键帧已对齐', owner: '王敏', sizeText: '286 MB', uploadedAt: '11-16 11:02', updatedAt: '11-16 11:40', attempts: 1, maxAttempts: 3 },
  { no: 'A20261116006', name: '客服回访录音_1108.mp3', modality: '音频', region: '浦东新区', stage: 'label', status: 'pending', note: '模型预标注 12 条待确认', owner: '李航', sizeText: '24.6 MB', uploadedAt: '11-15 16:20', updatedAt: '11-16 08:55', attempts: 1, maxAttempts: 3 },
  { no: 'A20261116007', name: '导游服务规范.md', modality: 'Markdown', region: '黄浦区', stage: 'apply', status: 'done', note: '已进入知识库「服务规范」', owner: '系统', sizeText: '36 KB', uploadedAt: '11-15 14:10', updatedAt: '11-15 14:26', attempts: 1, maxAttempts: 3, category: '服务规范 / 导游规范' },
  { no: 'A20261116008', name: '签证材料扫描件.pdf', modality: 'PDF', region: '浦东新区', stage: 'parse', status: 'failed', note: 'MINERU_TIMEOUT，已执行 3/3 次', owner: '系统', sizeText: '62.0 MB', uploadedAt: '11-16 08:01', updatedAt: '11-16 09:15', attempts: 3, maxAttempts: 3 },
  { no: 'A20261116009', name: '景区导览图.png', modality: '图片', region: '黄浦区', stage: 'upload', status: 'running', note: '分片上传 3/5', owner: '王敏', sizeText: '9.8 MB', uploadedAt: '11-16 11:52', updatedAt: '11-16 11:53', attempts: 1, maxAttempts: 3 },
  { no: 'A20261115010', name: '景区服务告示牌.png', modality: '图片', region: '浦东新区', stage: 'label', status: 'pending', note: '识别出 4 处文字区域待标注', owner: '李航', sizeText: '3.1 MB', uploadedAt: '11-15 15:02', updatedAt: '11-15 15:20', attempts: 1, maxAttempts: 3 },
  { no: 'A20261115011', name: '宏村民宿合作协议.docx', modality: 'Word', region: '黄山市', stage: 'classify', status: 'confirm', note: '模型建议：合同 / 供应商合同（0.88）', owner: '张磊', sizeText: '520 KB', uploadedAt: '11-15 11:30', updatedAt: '11-15 11:41', attempts: 1, maxAttempts: 3 },
  { no: 'A20261114012', name: '黄山景区门票政策2026.pdf', modality: 'PDF', region: '黄山市', stage: 'apply', status: 'done', note: '已进入知识库「线路产品」', owner: '系统', sizeText: '2.4 MB', uploadedAt: '11-14 10:02', updatedAt: '11-14 10:36', attempts: 1, maxAttempts: 3, category: '产品资料 / 景区资料' },
  { no: 'A20261114013', name: '华东五市团队行程单.xlsx', modality: 'Excel', region: '黄浦区', stage: 'store', status: 'done', note: '已回填 34 行业务数据', owner: '系统', sizeText: '188 KB', uploadedAt: '11-14 09:12', updatedAt: '11-14 09:30', attempts: 1, maxAttempts: 3, category: '产品资料 / 线路手册' }
]

export const REVIEW_BLOCKS: ReviewBlock[] = [
  { id: 'b1', type: 'title', conf: 0.99, origin: '2026 春季华东精品线路', status: 0 },
  { id: 'b2', type: 'text', conf: 0.97, origin: '本手册收录 2026 年 3 月至 5 月出发的华东线路，价格以出团通知为准。', status: 0 },
  { id: 'b3', type: 'table', conf: 0.91, origin: '线路价格表（2 行 × 4 列，已识别为结构化表格）', status: 0 },
  { id: 'b4', type: 'image', conf: 0.95, origin: '图片：宏村南湖（已裁切为派生文件）', status: 0 },
  { id: 'b5', type: 'text', conf: 0.62, origin: '宏村位于安徽省黔县，始建于南宋绍熙年间，村落布局形似牛形。', fix: '宏村位于安徽省黟县，始建于南宋绍熙年间，村落布局形似牛形。', status: 0 },
  { id: 'b6', type: 'footer', conf: 0.98, origin: '第 3 页', status: 0 }
]

export const MEDIA_FRAMES: MediaFrame[] = [
  { t: 3, kind: '场景切换', label: '片头 · 黄山远景', ocr: '黄山 · 宏村 4 日游', seg: 0 },
  { t: 38, kind: '场景切换', label: '迎客松', ocr: '迎客松', seg: 1 },
  { t: 72, kind: '固定间隔', label: '儿童票说明', ocr: '1.2 米以下儿童免门票', seg: 2 },
  { t: 101, kind: '场景切换', label: '宏村南湖', ocr: '（无文字）', seg: 3 },
  { t: 150, kind: '固定间隔', label: '酒店外景', ocr: '[酒店名称]', seg: 4 },
  { t: 158, kind: '场景切换', label: '集合地点地图', ocr: '集合地点：[地点]', seg: 4 }
]

export const MEDIA_SEGMENTS: MediaSegment[] = [
  { a: 0, b: 21, speaker: '旁白', text: '欢迎了解我们的黄山宏村四日游线路，下面介绍行程安排。', conf: 0.93, status: 0 },
  { a: 30, b: 55, speaker: '旁白', text: '第一天上午抵达黄山，乘索道上山，途经迎客松。', conf: 0.9, status: 0 },
  { a: 64, b: 88, speaker: '旁白', text: '身高一米二以下的儿童免门票，索道费用按景区规定另付。', conf: 0.71, status: 0 },
  { a: 95, b: 120, speaker: '导游', text: '第三天前往宏村，游览南湖和月沼。', conf: 0.88, status: 0 },
  { a: 140, b: 166, speaker: '导游', text: '住宿安排在[酒店名称]，第四天上午在[地点]集合返程。', conf: 0.86, status: 0 }
]

const CALL = ['表扬', '投诉', '咨询', '退改', '其他']
const SIGN = ['安全提示', '票务说明', '开放时间', '其他']

export const LABEL_ITEMS: LabelItem[] = [
  { id: 'l1', assetNo: 'A20261116006', assetName: '客服回访录音_1108.mp3', modality: '音频', target: '00:12–00:41', content: '客人反馈导游讲解很细致，孩子也玩得开心，下次还会报名。', modelLabel: '表扬', conf: 0.95, options: CALL, label: null, status: 'pending' },
  { id: 'l2', assetNo: 'A20261116006', assetName: '客服回访录音_1108.mp3', modality: '音频', target: '01:05–01:48', content: '酒店早餐时间太早，六点半就结束了，希望能延长。', modelLabel: '投诉', conf: 0.82, options: CALL, label: null, status: 'pending' },
  { id: 'l3', assetNo: 'A20261116006', assetName: '客服回访录音_1108.mp3', modality: '音频', target: '02:10–02:36', content: '想问一下五一专线什么时候截止报名，老人能不能参团。', modelLabel: '咨询', conf: 0.91, options: CALL, label: null, status: 'pending' },
  { id: 'l4', assetNo: 'A20261116006', assetName: '客服回访录音_1108.mp3', modality: '音频', target: '03:02–03:30', content: '如果出发前三天取消，费用怎么退？', modelLabel: '咨询', conf: 0.58, options: CALL, label: null, status: 'pending' },
  { id: 'l5', assetNo: 'A20261115010', assetName: '景区服务告示牌.png', modality: '图片', target: '区域 1 · 左上', content: '索道运营时间 07:30–16:30，雷雨天气暂停运营', modelLabel: '开放时间', conf: 0.93, options: SIGN, label: null, status: 'pending' },
  { id: 'l6', assetNo: 'A20261115010', assetName: '景区服务告示牌.png', modality: '图片', target: '区域 2 · 右上', content: '1.2 米以下儿童凭有效证件免票', modelLabel: '票务说明', conf: 0.9, options: SIGN, label: null, status: 'pending' },
  { id: 'l7', assetNo: 'A20261115010', assetName: '景区服务告示牌.png', modality: '图片', target: '区域 3 · 中部', content: '山路湿滑，请勿翻越护栏', modelLabel: '安全提示', conf: 0.97, options: SIGN, label: null, status: 'pending' }
]

export const CATEGORIES: CategoryNode[] = [
  { code: 'product', name: '产品资料', children: [{ code: 'product.route', name: '线路手册' }, { code: 'product.scenic', name: '景区资料' }] },
  { code: 'contract', name: '合同', children: [{ code: 'contract.supplier', name: '供应商合同' }, { code: 'contract.customer', name: '客户合同' }] },
  { code: 'finance', name: '财务票据', children: [{ code: 'finance.invoice', name: '发票 / 行程单' }, { code: 'finance.statement', name: '对账单' }] },
  { code: 'service', name: '服务规范', children: [{ code: 'service.guide', name: '导游规范' }, { code: 'service.cs', name: '客服话术' }] },
  { code: 'training', name: '培训素材', children: [{ code: 'training.video', name: '视频课程' }, { code: 'training.audio', name: '录音案例' }] }
]

export const CLASSIFY_SUMMARY: Record<string, { summary: string; suggestions: ClassifyItem['suggestions'] }> = {
  A20261116002: {
    summary: '东方航空电子客票行程单，乘机人 2 名，上海虹桥—黄山屯溪，票价合计 1,360 元，含机场建设费。',
    suggestions: [{ code: 'finance.invoice', name: '财务票据 / 发票 / 行程单', conf: 0.94 }, { code: 'contract.customer', name: '合同 / 客户合同', conf: 0.03 }, { code: 'product.route', name: '产品资料 / 线路手册', conf: 0.02 }]
  },
  A20261115011: {
    summary: '与宏村某民宿签订的 2026 年度房源合作协议，约定团队房价、取消政策与结算周期（月结）。',
    suggestions: [{ code: 'contract.supplier', name: '合同 / 供应商合同', conf: 0.88 }, { code: 'product.scenic', name: '产品资料 / 景区资料', conf: 0.07 }, { code: 'finance.statement', name: '财务票据 / 对账单', conf: 0.03 }]
  }
}

export const KNOWLEDGE_BASES: KnowledgeBase[] = [
  { code: 'kb_route', name: '线路产品', desc: '线路手册、景区资料、价格与出团说明，问答的主要依据', docQty: 128, chunkQty: 5620, scopeText: '全部门店', updatedAt: '11-16 10:36', enabled: true },
  { code: 'kb_service', name: '服务规范', desc: '导游服务规范、客服话术、投诉处理流程', docQty: 34, chunkQty: 980, scopeText: '全部门店', updatedAt: '11-15 14:26', enabled: true },
  { code: 'kb_store', name: '门店运营', desc: '各门店的营业时间、退改与早餐等门店级规定', docQty: 52, chunkQty: 1310, scopeText: '按门店区域隔离', updatedAt: '11-14 18:02', enabled: true },
  { code: 'kb_contract', name: '合同与票据', desc: '供应商合同、发票与对账单；仅供管理端检索，不对问答开放', docQty: 216, chunkQty: 4402, scopeText: '仅管理端', updatedAt: '11-16 10:48', enabled: false }
]

export const KNOWLEDGE_DOCS: Record<string, KnowledgeDoc[]> = {
  kb_route: [
    { assetNo: 'A20261114012', name: '黄山景区门票政策2026.pdf', modality: 'PDF', chunkQty: 46, region: '黄山市', storedAt: '11-14 10:36', hitQty: 312 },
    { assetNo: 'A20261114013', name: '华东五市团队行程单.xlsx', modality: 'Excel', chunkQty: 34, region: '黄浦区', storedAt: '11-14 09:30', hitQty: 88 },
    { assetNo: 'A20261110021', name: '黄山线路介绍（旧版）.mp4', modality: '视频', chunkQty: 22, region: '黄浦区', storedAt: '11-10 16:12', hitQty: 145 },
    { assetNo: 'A20261108017', name: '春季线路出发日期表.xlsx', modality: 'Excel', chunkQty: 18, region: '黄浦区', storedAt: '11-08 11:20', hitQty: 203 }
  ],
  kb_service: [
    { assetNo: 'A20261116007', name: '导游服务规范.md', modality: 'Markdown', chunkQty: 38, region: '黄浦区', storedAt: '11-15 14:26', hitQty: 97 },
    { assetNo: 'A20261102009', name: '投诉处理流程 v3.docx', modality: 'Word', chunkQty: 21, region: '黄浦区', storedAt: '11-02 10:05', hitQty: 64 }
  ],
  kb_store: [
    { assetNo: 'A20261105003', name: '外滩店退改与早餐规定.pdf', modality: 'PDF', chunkQty: 12, region: '黄浦区', storedAt: '11-05 09:40', hitQty: 156 },
    { assetNo: 'A20261105004', name: '浦东店营业时间.md', modality: 'Markdown', chunkQty: 4, region: '浦东新区', storedAt: '11-05 09:44', hitQty: 31 }
  ],
  kb_contract: [
    { assetNo: 'A20261116004', name: '10月结算对账单.xlsx', modality: 'Excel', chunkQty: 120, region: '黄浦区', storedAt: '11-16 10:48', hitQty: 0 }
  ]
}

export const TOOLSETS: Toolset[] = [
  { name: 'web', label: '网页检索与抽取', desc: '联网搜索与网页正文抽取（web_search、web_extract）', toolQty: 2, enabled: true },
  { name: 'vision', label: '图片理解', desc: '分析用户附图或资料中的图片（vision_analyze）', toolQty: 1, enabled: true },
  { name: 'video', label: '视频理解', desc: '分析视频内容（video_analyze），默认不启用', toolQty: 1, enabled: false },
  { name: 'file', label: '文件读写', desc: '读取、写入、修改与搜索文件（read_file、patch 等 4 个）', toolQty: 4, enabled: true },
  { name: 'terminal', label: '命令执行', desc: '执行命令与管理进程；租户环境默认关闭', toolQty: 2, enabled: false },
  { name: 'browser', label: '浏览器自动化', desc: '驱动真实浏览器完成导航、点击、截图等操作', toolQty: 14, enabled: false },
  { name: 'skills', label: '技能', desc: '查看、创建与维护技能文档（skills_list、skill_view、skill_manage）', toolQty: 3, enabled: true },
  { name: 'memory', label: '长期记忆', desc: '跨会话的记忆与用户画像（memory）', toolQty: 1, enabled: true },
  { name: 'session_search', label: '历史会话检索', desc: '搜索并总结过去的对话（session_search）', toolQty: 1, enabled: true },
  { name: 'todo', label: '任务规划', desc: '多步骤任务的计划与跟踪（todo_list）', toolQty: 1, enabled: true },
  { name: 'cronjob', label: '定时任务', desc: '创建、暂停、恢复与触发定时任务（cronjob_manage）', toolQty: 1, enabled: true },
  { name: 'image_gen', label: '图片生成', desc: '根据描述生成图片（image_generate）', toolQty: 1, enabled: false },
  { name: 'tts', label: '语音合成', desc: '把文字转成语音（text_to_speech）', toolQty: 1, enabled: false }
]

export const SKILLS: Skill[] = [
  { name: 'grounded-citations', category: '研究', desc: '回答必须引用已入库资料并标注出处，找不到依据时如实说明', source: '内置', enabled: true },
  { name: 'pdf', category: '办公', desc: '读取、拆分与提取 PDF 中的文字和表格', source: '内置', enabled: true },
  { name: 'microsoft-office', category: '办公', desc: '处理 Word / Excel / PowerPoint 文件', source: '内置', enabled: true },
  { name: 'document-to-action-items', category: '办公', desc: '从会议纪要、通知等文档中整理待办事项', source: '内置', enabled: false },
  { name: 'weekly-review-planning', category: '办公', desc: '按周汇总工作并生成下周计划', source: '内置', enabled: false },
  { name: 'maps', category: '办公', desc: '查询地点、路线与距离', source: '可选', enabled: true },
  { name: 'competitor-news-monitor', category: '研究', desc: '跟踪同行线路与价格动态', source: '可选', enabled: false },
  { name: 'travel-route-faq', category: '业务', desc: '旅行社问答话术：儿童票、老人参团、退改规则的回答口径', source: '租户自建', enabled: true },
  { name: 'complaint-handling', category: '业务', desc: '按《投诉处理流程 v3》引导客服处理投诉', source: '租户自建', enabled: true }
]

export const PERSONA = `你是「示例旅行社」的内部知识助手，服务对象是门店顾问、导游和客服同事。

回答原则：
1. 只依据已审核入库的资料作答，并标注出处；资料里没有的，直接说明“没有找到相关内容”，不要猜测。
2. 涉及价格、退改、证件等事项，提醒以正式出团通知或合同为准。
3. 语气专业、简洁，先给结论，再列要点。
4. 不透露客户个人信息；对外话术与内部规定分开说明。`

export const MEMORY_ENTRIES: MemoryEntry[] = [
  { id: 'm1', kind: 'memory', content: '外滩店的早餐时间 2026 年 11 月起调整为 06:30–09:30。', source: '会话 · 王敏', createdAt: '11-15 17:02', status: 'active' },
  { id: 'm2', kind: 'memory', content: '黄山线路儿童票口径：1.2 米以下免门票，索道另付。', source: '知识库 · 黄山景区门票政策2026.pdf', createdAt: '11-14 10:40', status: 'active' },
  { id: 'm3', kind: 'user', content: '王敏负责上海市门店的内容审核，偏好先看结论再看依据。', source: '会话 · 王敏', createdAt: '11-12 09:20', status: 'active' },
  { id: 'm4', kind: 'memory', content: '五一专线报名截止时间提前到出发前 15 天。', source: '会话 · 陈晓', createdAt: '11-16 10:12', status: 'pending' },
  { id: 'm5', kind: 'user', content: '陈晓在外滩店做门店顾问，常问退改与早餐问题。', source: '会话 · 陈晓', createdAt: '11-16 10:13', status: 'pending' }
]

export const CRON_JOBS: CronJob[] = [
  { id: 'c1', name: '每日用量同步', schedule: '0 2 * * *', scheduleText: '每天 02:00', deliver: '仅记录', nextRun: '11-17 02:00', lastRun: '11-16 02:00', lastStatus: 'ok', enabled: true },
  { id: 'c2', name: '失败任务巡检', schedule: '*/30 * * * *', scheduleText: '每 30 分钟', deliver: '管理端通知', nextRun: '11-16 12:30', lastRun: '11-16 12:00', lastStatus: 'ok', enabled: true },
  { id: 'c3', name: '月度内容质量报告', schedule: '0 9 1 * *', scheduleText: '每月 1 日 09:00', deliver: '邮件 · 租户管理员', nextRun: '12-01 09:00', lastRun: '11-01 09:00', lastStatus: 'ok', enabled: true },
  { id: 'c4', name: '知识库索引重建', schedule: '0 3 * * 0', scheduleText: '每周日 03:00', deliver: '仅记录', nextRun: '11-22 03:00', lastRun: '11-15 03:00', lastStatus: 'failed', enabled: true },
  { id: 'c5', name: '门店晨报', schedule: '30 8 * * 1-5', scheduleText: '工作日 08:30', deliver: '企业微信群', nextRun: '—', lastRun: '—', lastStatus: 'never', enabled: false }
]

export const PLATFORMS: GatewayPlatform[] = [
  { name: 'api_server', label: 'PDR 接入（api_server）', status: 'online', sessions: 23, note: '用户端问答经此进入智能体' },
  { name: 'wecom', label: '企业微信', status: 'disabled', sessions: 0, note: '未配置' },
  { name: 'dingtalk', label: '钉钉', status: 'disabled', sessions: 0, note: '未配置' },
  { name: 'feishu', label: '飞书', status: 'offline', sessions: 0, note: '凭据已过期，需重新授权' },
  { name: 'email', label: '邮件', status: 'online', sessions: 2, note: '仅用于定时报告投递' }
]

export const QA_SESSIONS: QaSession[] = [
  { id: 's1', user: '陈晓', region: '黄浦区', scopeName: '全部可用资料', question: '黄山宏村四日游，1.15 米的孩子要买门票吗？', answer: '不需要买门票。黄山景区对身高 1.2 米以下的儿童免门票，孩子 1.15 米符合条件；索道费用另付，团费中的儿童价以出团通知为准。', citations: [{ title: '黄山线路介绍.mp4', locator: '01:04–01:28' }, { title: '2026春季华东线路产品手册.pdf', locator: '第 3 页' }, { title: '导游服务规范.md', locator: '「儿童与老人」小节' }], feedback: 'up', turns: 2, createdAt: '11-16 11:58', durationMs: 4200 },
  { id: 's2', user: '陈晓', region: '黄浦区', scopeName: '门店运营', question: '外滩店早餐几点结束？', answer: '外滩店早餐时间为 06:30–09:30（11 月起调整）。', citations: [{ title: '外滩店退改与早餐规定.pdf', locator: '第 1 页' }], feedback: null, turns: 1, createdAt: '11-16 10:12', durationMs: 2100 },
  { id: 's3', user: '刘洋', region: '浦东新区', scopeName: '服务规范', question: '客人投诉导游迟到，第一步怎么处理？', answer: '先向客人致歉并记录时间、地点与导游姓名，30 分钟内上报门店负责人；按《投诉处理流程 v3》第 2 节进入核实。', citations: [{ title: '投诉处理流程 v3.docx', locator: '第 2 节' }], feedback: 'up', turns: 3, createdAt: '11-16 09:40', durationMs: 3600 },
  { id: 's4', user: '刘洋', region: '浦东新区', scopeName: '全部可用资料', question: '签证材料要准备哪些？', answer: '没有找到相关内容。已入库资料中暂无签证材料清单，建议联系签证专员确认。', citations: [], feedback: 'down', turns: 1, createdAt: '11-15 16:22', durationMs: 1800 },
  { id: 's5', user: '赵倩', region: '黄浦区', scopeName: '线路产品', question: '春季线路最早哪天出发？', answer: '2026 年春季华东线路最早 3 月 6 日出发，每周五、周六发团。', citations: [{ title: '春季线路出发日期表.xlsx', locator: 'Sheet1 · 第 2 行' }], feedback: null, turns: 1, createdAt: '11-15 14:05', durationMs: 2500 },
  { id: 's6', user: '陈晓', region: '黄浦区', scopeName: '全部可用资料', question: '70 岁老人能参加黄山团吗？', answer: '可以参加，需签署健康声明并由家属陪同；行程含登山项目，建议选择索道上下山。', citations: [{ title: '导游服务规范.md', locator: '「儿童与老人」小节' }], feedback: 'up', turns: 2, createdAt: '11-14 17:30', durationMs: 3900 }
]

export const PERM_CATALOG: PermGroup[] = [
  { name: '内容中心', perms: [
    { code: 'content:dashboard', name: '内容看板' }, { code: 'content:asset', name: '任务列表' }, { code: 'content:upload', name: '上传文件' },
    { code: 'content:retry', name: '重新执行失败任务' }, { code: 'content:review', name: '人工核对（文档 / 音视频）' },
    { code: 'content:label', name: '标注工作台' }, { code: 'content:classify', name: '分类确认' }, { code: 'knowledge', name: '知识库' }
  ] },
  { name: '智能体', perms: [
    { code: 'agent:config', name: '智能体配置' }, { code: 'agent:skill', name: '技能 · 工具集' }, { code: 'agent:memory', name: '记忆 · 人格' },
    { code: 'agent:cron', name: '定时任务 · 消息网关' }, { code: 'agent:audit', name: '问答会话审计' }
  ] },
  { name: '权限与系统', perms: [
    { code: 'sys:role', name: '角色与权限' }, { code: 'sys:user', name: '用户 · 区域' }, { code: 'sys:audit', name: '审计日志' },
    { code: 'sys:tenant', name: '租户管理（平台）' }, { code: 'sys:config', name: '系统配置（平台）' }
  ] }
]

const SCOPES = (asset: string, kb: string, biz: string, report: string) => [
  { resource: 'content_asset', resourceName: '内容资产', scope: asset, action: '读写' },
  { resource: 'knowledge', resourceName: '知识库', scope: kb, action: '读' },
  { resource: 'business_extract', resourceName: '业务回填数据', scope: biz, action: '读' },
  { resource: 'report', resourceName: '报表', scope: report, action: '读' }
] as Role['dataScopes']

const COLUMNS = (amount: '1' | '3', tax: '0' | '3'): Role['columns'] => [
  { table: 'ods.ods_fin_invoice_extract_di', col: 'invoice_no', name: '发票号码', perm: '2', mask: '—' },
  { table: 'ods.ods_fin_invoice_extract_di', col: 'buyer_name', name: '购买方名称', perm: '1', mask: '姓名脱敏（张*）' },
  { table: 'ods.ods_fin_invoice_extract_di', col: 'invoice_amt', name: '价税合计金额', perm: amount, mask: amount === '1' ? '金额区间（1000–5000）' : '—' },
  { table: 'ods.ods_fin_invoice_extract_di', col: 'tax_amt', name: '税额', perm: tax, mask: '—' },
  { table: 'ods.ods_fin_invoice_extract_di', col: 'verify_status', name: '人工核验状态', perm: '3', mask: '—' }
]

export const ROLES: Role[] = [
  { code: 'tenant_admin', name: '租户管理员', builtin: true, userQty: 2, desc: '管理本租户的内容、智能体与人员权限', perms: USERS.zhanglei.perms, dataScopes: SCOPES('1', '1', '1', '1'), columns: COLUMNS('3', '3') },
  { code: 'content_reviewer', name: '内容审核员', builtin: true, userQty: 6, desc: '核对解析结果、确认分类，查看知识库', perms: USERS.wangmin.perms, dataScopes: SCOPES('2', '2', '3', '2'), columns: COLUMNS('1', '0') },
  { code: 'content_labeler', name: '标注员', builtin: true, userQty: 4, desc: '只处理分配给自己的标注任务', perms: USERS.lihang.perms, dataScopes: SCOPES('5', '3', '3', '3'), columns: COLUMNS('1', '0') },
  { code: 'store_viewer', name: '门店只读', builtin: false, userQty: 12, desc: '门店负责人查看看板与知识库', perms: ['content:dashboard', 'knowledge'] as Perm[], dataScopes: SCOPES('4', '4', '4', '4'), columns: COLUMNS('1', '0') }
]

export const REGIONS: RegionNode[] = [
  { code: '310000', name: '上海市', children: [{ code: '310101', name: '黄浦区' }, { code: '310115', name: '浦东新区' }] },
  { code: '340000', name: '安徽省', children: [{ code: '341000', name: '黄山市', children: [{ code: '341003', name: '黄山区（黄山景区）' }, { code: '341023', name: '黟县（宏村）' }] }] }
]

export const USER_ROWS: UserRow[] = [
  { account: 'zhanglei', name: '张磊', userType: 'admin', roleNames: ['租户管理员'], region: '示例旅行社', phone: '138****2041', status: 'active', lastLoginAt: '11-16 08:42' },
  { account: 'wangmin', name: '王敏', userType: 'admin', roleNames: ['内容审核员'], region: '上海市', phone: '139****7730', status: 'active', lastLoginAt: '11-16 09:05' },
  { account: 'lihang', name: '李航', userType: 'admin', roleNames: ['标注员'], region: '浦东新区', phone: '137****1186', status: 'active', lastLoginAt: '11-16 08:58' },
  { account: 'zhaoqian', name: '赵倩', userType: 'admin', roleNames: ['门店只读'], region: '黄浦区', phone: '136****5402', status: 'active', lastLoginAt: '11-15 18:20' },
  { account: 'sunpeng', name: '孙鹏', userType: 'admin', roleNames: ['内容审核员'], region: '黄山市', phone: '135****8809', status: 'disabled', lastLoginAt: '10-28 10:11' },
  { account: 'chenxiao', name: '陈晓', userType: 'portal', roleNames: ['问答用户'], region: '黄浦区', phone: '186****3301', status: 'active', lastLoginAt: '11-16 11:57' },
  { account: 'liuyang', name: '刘洋', userType: 'portal', roleNames: ['问答用户'], region: '浦东新区', phone: '187****6625', status: 'active', lastLoginAt: '11-16 09:38' },
  { account: 'zhoujing', name: '周静', userType: 'portal', roleNames: ['问答用户'], region: '黟县（宏村）', phone: '188****0917', status: 'active', lastLoginAt: '11-13 15:02' }
]

export const TENANTS: Tenant[] = [
  { code: 'demo_travel', name: '示例旅行社', plan: '专业版', status: 'active', adminName: '张磊', userQty: 31, storageUsedGb: 38, storageQuotaGb: 100, createdAt: '2026-09-01' },
  { code: 'east_china_tours', name: '华东国旅', plan: '试用版', status: 'trial', adminName: '何静', userQty: 6, storageUsedGb: 3.2, storageQuotaGb: 10, createdAt: '2026-11-02' },
  { code: 'shanshui_holiday', name: '山水假期', plan: '基础版', status: 'suspended', adminName: '吴峰', userQty: 14, storageUsedGb: 21, storageQuotaGb: 50, createdAt: '2026-06-18' }
]

export const AUDIT_LOGS: AuditLog[] = [
  { id: 'g1', at: '11-16 11:52', user: '王敏', module: '内容任务', action: '上传文件', target: 'A20261116009 景区导览图.png', ip: '10.2.8.14', result: 'ok' },
  { id: 'g2', at: '11-16 11:40', user: '王敏', module: '人工核对', action: '保存核对草稿', target: 'A20261116005 黄山线路介绍.mp4', ip: '10.2.8.14', result: 'ok' },
  { id: 'g3', at: '11-16 10:21', user: '系统', module: '内容任务', action: '自动重试', target: 'A20261116003 外滩店酒店供应商合同.docx', ip: '—', result: 'failed' },
  { id: 'g4', at: '11-16 09:58', user: '李航', module: '智能体', action: '访问智能体配置', target: '/api/admin/v1/config', ip: '10.2.9.31', result: 'denied' },
  { id: 'g5', at: '11-16 09:15', user: '系统', module: '内容任务', action: '解析失败（已达最大次数）', target: 'A20261116008 签证材料扫描件.pdf', ip: '—', result: 'failed' },
  { id: 'g6', at: '11-16 08:42', user: '张磊', module: '登录', action: '登录管理端', target: 'zhanglei', ip: '10.2.1.5', result: 'ok' },
  { id: 'g7', at: '11-15 17:30', user: '张磊', module: '智能体', action: '修改租户覆盖配置', target: 'memory.write_approval：false → true', ip: '10.2.1.5', result: 'ok' },
  { id: 'g8', at: '11-15 16:05', user: '张磊', module: '权限管理', action: '修改角色数据范围', target: '内容审核员 · 内容资产：本区域 → 本区域及下级', ip: '10.2.1.5', result: 'ok' },
  { id: 'g9', at: '11-15 14:26', user: '王敏', module: '人工核对', action: '提交核对', target: 'A20261116007 导游服务规范.md', ip: '10.2.8.14', result: 'ok' },
  { id: 'g10', at: '11-15 11:41', user: '张磊', module: '内容任务', action: '上传文件', target: 'A20261115011 宏村民宿合作协议.docx', ip: '10.2.1.5', result: 'ok' },
  { id: 'g11', at: '11-14 18:02', user: '张磊', module: '知识库', action: '停用知识库', target: '合同与票据', ip: '10.2.1.5', result: 'ok' },
  { id: 'g12', at: '11-14 09:00', user: 'chenxiao', module: '登录', action: '用户端账号登录管理端', target: 'chenxiao', ip: '10.2.8.77', result: 'denied' }
]

export const NOTICES: Notice[] = [
  { id: 'n1', title: '签证材料扫描件.pdf 解析失败，已用尽 3 次自动重试', at: '11-16 09:15', read: false, link: '/content/tasks' },
  { id: 'n2', title: '黄山线路介绍.mp4 待人工核对', at: '11-16 11:40', read: false, link: '/review/media' },
  { id: 'n3', title: '2 条记忆写入等待审批', at: '11-16 10:13', read: false, link: '/agent/memory' },
  { id: 'n4', title: '知识库索引重建任务上次执行失败', at: '11-15 03:12', read: false, link: '/agent/cron' },
  { id: 'n5', title: '客服回访录音_1108.mp3 已分配给你标注', at: '11-16 08:55', read: false, link: '/review/label' }
]
