# Keystat Landing Page Builder Task Breakdown

> 文档状态：Draft v0.1  
> 项目：Keystat Landing Page  
> Source of truth：用户本次指令 + `docs/prd.md` + `docs/technical-solution.md`  
> 当前边界：本文仅拆解 Builder 可执行任务；不在本阶段实现业务代码。  
> 硬约束：不得把项目写成 Hermes Agent 或 Multica；不得引入未确认账号体系/云后端/定价/发布日期/下载承诺。

## Task 0 — Confirm implementation assumptions before coding

**Objective**  
在 Builder 开始写业务代码前确认会影响实现的事实，防止页面出现未授权承诺。

**Files**

- `docs/prd.md`
- `docs/technical-solution.md`
- `docs/task-breakdown.md`

**Steps**

1. 阅读 PRD 和技术方案。
2. 向用户确认至少以下事项：
   - 主 CTA：等待名单 / 下载 / 获取早期访问 / 联系团队。
   - 表单后端：Formspree、Airtable、HubSpot、Buttondown、自建 API、mailto 或暂不提交。
   - 分析工具：PostHog、GA4、Plausible、自建或不接入。
   - Keystat 隐私边界：是否记录具体按键内容、数据存储位置、删除/导出能力。
   - 品牌资产：logo、favicon、OG 图、产品截图/mockup、配色、字体。
   - 域名与语言：最终 URL/canonical、是否中英双语。
3. 若未确认，使用保守文案和 TBD 占位，且表单/下载不得假装可用。

**Verification**

- Builder 输出的实施计划中明确标注已确认项与 TBD 项。
- 无任何未经确认的价格、发布日期、平台支持、隐私承诺、下载承诺。

**Dependencies**

- 无。

**Commit message**

- `docs: confirm keystat landing assumptions`

---

## Task 1 — Create isolated Keystat route and feature skeleton

**Objective**  
建立 `/keystat` 页面入口与 feature 目录骨架，使 Keystat landing page 与现有产品代码隔离。

**Files**

- `multica/apps/web/app/keystat/page.tsx`
- `multica/apps/web/app/keystat/layout.tsx`
- `multica/apps/web/features/keystat-landing/components/KeystatLanding.tsx`
- `multica/apps/web/features/keystat-landing/content/en.ts`
- `multica/apps/web/features/keystat-landing/types.ts`
- `multica/apps/web/features/keystat-landing/lib/constants.ts`

**Steps**

1. 新增 `app/keystat` route，页面入口只渲染 `KeystatLanding`。
2. 新增 `features/keystat-landing`，所有 Keystat 页面组件、内容、类型、工具函数放在该 feature 内。
3. 在 content module 中集中定义区块内容，所有未确认事实使用 `TBD` 或保守文案。
4. 不复用现有 Multica landing 文案；可以参考组件组织方式，但不能复制产品定位。

**Verification**

- `/keystat` 能在 dev server 中访问。
- 搜索页面文案不出现 `Hermes Agent` 或 `Multica` 产品定位。
- TypeScript import 路径清晰，无循环依赖。

**Dependencies**

- Task 0。

**Commit message**

- `feat(keystat): add isolated landing route skeleton`

---

## Task 2 — Implement content-driven landing sections

**Objective**  
按 PRD 信息架构实现页面主体区块，优先保证内容完整、语义清晰、可维护。

**Files**

- `multica/apps/web/features/keystat-landing/components/KeystatLanding.tsx`
- `multica/apps/web/features/keystat-landing/components/HeroSection.tsx`
- `multica/apps/web/features/keystat-landing/components/ProblemSection.tsx`
- `multica/apps/web/features/keystat-landing/components/SolutionSection.tsx`
- `multica/apps/web/features/keystat-landing/components/ValuePropsSection.tsx`
- `multica/apps/web/features/keystat-landing/components/ProductPreviewSection.tsx`
- `multica/apps/web/features/keystat-landing/components/UseCasesSection.tsx`
- `multica/apps/web/features/keystat-landing/components/TrustPrivacySection.tsx`
- `multica/apps/web/features/keystat-landing/components/FaqSection.tsx`
- `multica/apps/web/features/keystat-landing/components/FinalCtaSection.tsx`
- `multica/apps/web/features/keystat-landing/components/KeystatHeader.tsx`
- `multica/apps/web/features/keystat-landing/components/KeystatFooter.tsx`
- `multica/apps/web/features/keystat-landing/content/en.ts`

**Steps**

1. 实现 Header、Hero、Problem、Solution、Value Props、Product Preview、Use Cases、Trust/Privacy、FAQ、Final CTA、Footer。
2. Hero 和 Final CTA 至少各包含一次主 CTA。
3. Privacy/Trust 区块只写已确认内容；未确认数据策略使用保守表达。
4. Product Preview 使用占位 mockup 或用户提供素材，不捏造真实界面能力。
5. FAQ 覆盖隐私、平台、价格/计划、可用时间、联系。

**Verification**

- 页面完整渲染全部必备区块。
- 单个 H1，H2/H3 层级合理。
- 所有文案聚焦 Keystat，不混入现有仓库项目名称。
- 未确认能力均未以确定语气承诺。

**Dependencies**

- Task 1。

**Commit message**

- `feat(keystat): build content-driven landing sections`

---

## Task 3 — Add visual assets and responsive styling

**Objective**  
补齐品牌资产入口、基础视觉系统和响应式布局，确保移动端与桌面端可用。

**Files**

- `multica/apps/web/public/keystat/logo.svg`
- `multica/apps/web/public/keystat/favicon.svg`
- `multica/apps/web/public/keystat/og-image.png`
- `multica/apps/web/public/keystat/product-preview.webp`
- `multica/apps/web/features/keystat-landing/components/*.tsx`
- `multica/apps/web/features/keystat-landing/lib/constants.ts`

**Steps**

1. 放置用户确认的品牌素材；若暂缺，使用明确标注的临时占位资产。
2. 在 constants 中定义 Keystat 品牌色、CTA 文案、锚点 id。
3. 实现 360px、768px、1024px、1440px 关键宽度下的响应式布局。
4. 图片设置 width/height、alt，非首屏图懒加载。
5. 动效只用轻量 CSS；支持 `prefers-reduced-motion`。

**Verification**

- 移动端无横向滚动。
- CTA 点击区域高度建议 ≥ 44px。
- 图片有描述性 alt 或装饰性隐藏。
- reduced motion 下无强制大幅动画。

**Dependencies**

- Task 2。
- 用户提供或确认临时品牌素材策略。

**Commit message**

- `feat(keystat): add responsive visual system and assets`

---

## Task 4 — Implement waitlist/contact CTA flow

**Objective**  
实现可替换的等待名单或联系 CTA 流程，保持低复杂度和隐私友好。

**Files**

- `multica/apps/web/features/keystat-landing/components/WaitlistForm.tsx`
- `multica/apps/web/features/keystat-landing/lib/waitlist.ts`
- `multica/apps/web/features/keystat-landing/types.ts`
- 可选：`multica/apps/web/app/api/keystat/waitlist/route.ts`（仅当用户确认自建 API）

**Steps**

1. 根据 Task 0 确认结果选择 CTA 行为：等待名单、联系、下载或暂不启用。
2. 默认等待名单表单字段：email 必填；role/use_case/company/message 可选。
3. 实现客户端校验、loading、success、error 状态；失败时保留输入。
4. 后端未确认时不提交真实请求，不展示虚假成功；使用 `mailto:` 或禁用态提示。
5. 隐私提示链接使用用户确认 URL；未确认时标记 TBD 或上线前移除。

**Verification**

- 表单空邮箱/非法邮箱有文本错误提示。
- 成功/失败状态可恢复，且不会丢失用户输入。
- 未配置后端时不会发送到未知第三方。
- 不新增账号体系、登录、订阅或用户后台。

**Dependencies**

- Task 0。
- Task 2。

**Commit message**

- `feat(keystat): add waitlist cta flow`

---

## Task 5 — Add analytics adapter and conversion events

**Objective**  
实现隐私友好的埋点抽象，使页面转化事件可接入但默认可 no-op。

**Files**

- `multica/apps/web/features/keystat-landing/lib/analytics.ts`
- `multica/apps/web/features/keystat-landing/components/*.tsx`
- `multica/apps/web/features/keystat-landing/lib/constants.ts`

**Steps**

1. 新增 `track(eventName, properties)`，未配置 provider 时 no-op。
2. 触发 `page_view_landing`、`cta_clicked`、`waitlist_form_started`、`waitlist_form_submitted`、`contact_clicked`。
3. 使用 IntersectionObserver 或轻量 hook 触发 `section_viewed`。
4. 实现滚动深度 25/50/75/100 的一次性事件。
5. 表单事件仅记录必要属性；不把完整邮箱写入 analytics properties。

**Verification**

- analytics provider 未配置时页面无报错。
- CTA、表单、section view、scroll depth 事件可在开发环境观察到。
- 事件属性不含完整邮箱或敏感输入内容。

**Dependencies**

- Task 2。
- Task 4。
- 用户确认是否允许接入分析工具。

**Commit message**

- `feat(keystat): add privacy-safe analytics adapter`

---

## Task 6 — Configure SEO metadata, sitemap, robots, and structured data

**Objective**  
让 `/keystat` 具备基础搜索与分享能力，并避免错误索引私有路径。

**Files**

- `multica/apps/web/app/keystat/page.tsx`
- `multica/apps/web/app/keystat/layout.tsx`
- `multica/apps/web/app/sitemap.ts`
- `multica/apps/web/app/robots.ts`
- `multica/apps/web/public/keystat/og-image.png`

**Steps**

1. 为 `/keystat` 配置唯一 `title`、`description`、OG、Twitter、canonical。
2. 在 `sitemap.ts` 加入 `/keystat`，base URL 使用用户确认域名或环境变量。
3. 在 `robots.ts` allow `/keystat`，继续 disallow 认证和 dashboard 类私有路径。
4. 添加 JSON-LD（推荐 `WebPage` 或保守 `SoftwareApplication`），不写价格、发布日期、未确认平台支持。
5. 确认 OG 图存在且尺寸合适。

**Verification**

- 查看页面 HTML，metadata 齐全。
- `/sitemap.xml` 包含 Keystat URL。
- `/robots.txt` 允许 Keystat 页面且不放开私有路径。
- 分享图和 description 不含 TBD 之外的未确认承诺。

**Dependencies**

- Task 1。
- 用户确认域名/URL 策略；未确认时使用可替换配置。

**Commit message**

- `feat(keystat): configure seo metadata and indexing`

---

## Task 7 — Accessibility and performance hardening

**Objective**  
确保 landing page 达到 PRD 的可访问性与性能基线。

**Files**

- `multica/apps/web/features/keystat-landing/components/*.tsx`
- `multica/apps/web/features/keystat-landing/content/en.ts`
- `multica/apps/web/app/keystat/page.tsx`

**Steps**

1. 检查 heading 层级、landmark、button/link 语义。
2. 为表单 label、error、helper text 补齐 ARIA 关联。
3. 检查 focus ring、键盘导航、跳转锚点。
4. 检查色彩对比、移动端无横向滚动。
5. 优化图片尺寸、懒加载、首屏资源；避免引入大型动画/第三方脚本。
6. 支持 `prefers-reduced-motion`。

**Verification**

- 键盘可从 Header 导航到 CTA、表单、Footer。
- 表单错误可被屏幕阅读器理解。
- Lighthouse Accessibility/SEO/Best Practices/Performance 目标 90+。
- LCP/INP/CLS 目标符合 PRD 建议或给出差距说明。

**Dependencies**

- Task 2。
- Task 3。
- Task 4。

**Commit message**

- `fix(keystat): improve accessibility and performance`

---

## Task 8 — Add validation tests or checks

**Objective**  
用轻量测试/检查降低回归风险，覆盖页面渲染、CTA、无品牌污染和 SEO 基础。

**Files**

- `multica/apps/web/features/keystat-landing/**/*.test.tsx`（如现有 test setup 支持）
- `multica/apps/web/app/keystat/page.test.tsx`（如适用）
- 可选：Playwright smoke test（若 repo 已有 e2e 规范）

**Steps**

1. 添加 render smoke test：页面渲染 Hero、CTA、FAQ、Privacy 区块。
2. 添加表单验证测试：非法邮箱、成功/失败状态。
3. 添加文案 guard：页面可见文案不出现 `Hermes Agent` / `Multica` 产品定位。
4. 添加 SEO smoke：metadata/content 中包含 Keystat title/description。
5. 避免引入新的测试框架；沿用现有 Vitest/Testing Library/Playwright 配置。

**Verification**

- `pnpm test --filter @multica/web` 或仓库既有测试命令通过。
- 若不添加测试，至少提供手动验收 checklist 和原因说明。

**Dependencies**

- Task 2。
- Task 4。
- Task 6。

**Commit message**

- `test(keystat): cover landing page smoke flows`

---

## Task 9 — Final implementation verification and documentation update

**Objective**  
在 Builder 完成实现后做端到端验收，并更新必要文档，不扩大范围。

**Files**

- `docs/technical-solution.md`（如实现偏离方案，更新 trade-off）
- `docs/task-breakdown.md`（如任务状态需要标注）
- 实际实现文件（由前序任务产生）

**Steps**

1. 运行类型检查、lint、测试和生产 build。
2. 手动检查 `/keystat` 桌面/移动端。
3. 检查页面不包含未确认承诺。
4. 检查 sitemap、robots、metadata、OG、表单、analytics no-op/provider 行为。
5. 更新文档中的已确认 trade-off 或上线前阻塞项。

**Verification**

- `pnpm typecheck` / `pnpm lint` / `pnpm test` / `pnpm build` 通过（按 repo 实际命令）。
- Lighthouse 目标 90+ 或记录差距与修复建议。
- 文档列出剩余用户待确认项。

**Dependencies**

- Task 1-8。

**Commit message**

- `docs(keystat): record landing verification notes`

---

## Recommended Builder execution order

1. Task 0：确认实现前提。
2. Task 1：route + feature skeleton。
3. Task 2：内容区块。
4. Task 3：视觉和响应式。
5. Task 4：CTA/表单。
6. Task 5：埋点 adapter。
7. Task 6：SEO/sitemap/robots。
8. Task 7：性能与可访问性 hardening。
9. Task 8：测试/检查。
10. Task 9：最终验证与文档更新。

## Builder guardrails

- 只把 Keystat 定位为 Keystat Landing Page，不使用 Hermes Agent 或 Multica 产品名替代。
- 不实现复杂后端、账号体系、团队管理后台、计费、云同步。
- 不承诺真实价格、发布日期、下载平台、隐私细节，除非用户已确认。
- 不把完整邮箱、具体按键内容或敏感输入写入分析属性。
- 不为了视觉效果引入大体积动画/视频/3D 依赖，除非用户确认性能 trade-off。
