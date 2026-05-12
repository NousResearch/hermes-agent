# Keystat Landing Page Technical Solution

> 文档状态：Draft v0.1  
> 项目：Keystat Landing Page  
> Source of truth：用户本次指令 + `docs/prd.md`  
> 当前边界：仅技术规划与 Builder 任务拆解，不实现业务代码。  
> 重要约束：仓库中的 Hermes Agent / Multica 等命名与功能仅作为 repo 现状参考，不进入 Keystat 产品定位或页面文案。

## 1. 只读仓库现状结论

本次只读检查聚焦前端相关目录，结论如下：

- 根目录是 `hermes-agent` 项目，包含 Python agent/CLI/gateway 主体；根 `package.json` 仅有少量浏览器工具依赖，不适合作为 Keystat landing page 的直接载体。
- `web/` 是 Vite + React + TypeScript + Tailwind CSS 应用，但其 `vite.config.ts` 强绑定 Hermes dashboard dev token、`/api` proxy、生产输出到 `hermes_cli/web_dist`，并包含大量 agent dashboard 页面，不适合作为独立营销页。
- `website/` 是 Docusaurus 文档站，适合产品文档，不适合高定制营销 landing page。
- `multica/apps/web/` 是 Next.js App Router 应用，已有 `(landing)` route group、`sitemap.ts`、`robots.ts`、metadata、i18n、landing 组件组织方式；技术上最接近 landing page 需求。但该目录属于现有 Multica app，包含认证、工作区、dashboard、后端交互等产品功能，不能让 Keystat 继承其产品定位或未确认能力。

## 2. 推荐承载方案

### 2.1 推荐路径

推荐在现有 Next.js monorepo 的 Web app 中新增隔离的 Keystat route/feature：

```text
multica/apps/web/
├── app/
│   ├── keystat/
│   │   ├── layout.tsx
│   │   └── page.tsx
│   ├── keystat-og/route.ts          # 可选：动态 OG，若不使用静态图片
│   ├── robots.ts                    # 更新以允许 /keystat
│   └── sitemap.ts                   # 更新以纳入 /keystat
├── features/
│   └── keystat-landing/
│       ├── components/
│       │   ├── KeystatLanding.tsx
│       │   ├── HeroSection.tsx
│       │   ├── ProblemSection.tsx
│       │   ├── SolutionSection.tsx
│       │   ├── ValuePropsSection.tsx
│       │   ├── ProductPreviewSection.tsx
│       │   ├── UseCasesSection.tsx
│       │   ├── TrustPrivacySection.tsx
│       │   ├── FaqSection.tsx
│       │   ├── FinalCtaSection.tsx
│       │   ├── KeystatHeader.tsx
│       │   ├── KeystatFooter.tsx
│       │   └── WaitlistForm.tsx
│       ├── content/
│       │   └── en.ts                # 或 content.ts，默认单语言
│       ├── lib/
│       │   ├── analytics.ts
│       │   ├── constants.ts
│       │   └── waitlist.ts
│       └── types.ts
└── public/
    └── keystat/
        ├── favicon.svg
        ├── og-image.png
        ├── logo.svg
        └── product-preview.webp
```

### 2.2 备选方案

1. **独立新 app：`apps/keystat-landing/` 或根 `keystat-landing/`**
   - 优点：品牌隔离最彻底，避免现有 Multica/Hermes 依赖污染。
   - 缺点：需要新增 workspace/package 配置，CI/部署/依赖复用成本更高；当前任务边界不宜直接改动。

2. **使用 `web/` Vite 应用新增页面**
   - 优点：React + Vite + Tailwind 已存在。
   - 缺点：强绑定 Hermes dashboard server 与输出路径，不适合独立 SEO landing page；需要拆除很多 Hermes 专用逻辑。

3. **使用 `website/` Docusaurus**
   - 优点：SEO 和静态输出成熟。
   - 缺点：视觉定制和转化表单体验受限，更适合文档而非营销页。

## 3. 技术选型

### 3.1 Framework

- **推荐：Next.js App Router**（沿用 `multica/apps/web` 现有技术栈）
- 理由：
  - 已具备 metadata、robots、sitemap、静态/SSR 页面能力。
  - App Router 对单页营销页、SEO metadata、结构化数据、图片优化支持好。
  - 可通过 route group/feature folder 保持 Keystat 隔离。

### 3.2 UI 与样式

- 使用现有 Tailwind CSS / design token 能力，但 Keystat 需要独立品牌常量：颜色、字体、radius、阴影、图形语言不得直接沿用 Multica/Hermes 文案或视觉识别。
- 推荐先使用语义化组件和纯 CSS/Tailwind 实现，不引入大型动效库。
- 图标可复用现有 `lucide-react`，但需控制 bundle size，仅按需导入。

### 3.3 内容管理

- 初期使用 TypeScript content module：`features/keystat-landing/content/en.ts`。
- 文案中所有未确认事实用 `TBD` 或保守占位表达，避免承诺：价格、发布日期、平台支持、云同步、账号体系、团队版、具体隐私处理方式。
- 若后续需要 CMS/多语言，再升级为 MDX/CMS/i18n 结构。

### 3.4 表单

- 默认推荐等待名单表单：Email 必填；role/use_case/company/message 可选。
- 表单提交后端待用户确认：Formspree / Airtable / HubSpot / Buttondown / 自建 API / 仅 `mailto:`。
- 当前不得擅自引入账号体系、用户后台或复杂后端。
- 未确认前 Builder 应实现可替换的 `waitlist.ts` adapter：
  - `submitWaitlist(payload)`
  - 后端未配置时展示保守 fallback（例如 `mailto:` 或 disabled + “Waitlist backend TBD”），不得假装提交成功。

### 3.5 埋点

- 建议抽象 `track(eventName, properties)`，事件名来自 PRD：
  - `page_view_landing`
  - `section_viewed`
  - `scroll_depth_reached`
  - `cta_clicked`
  - `waitlist_form_started`
  - `waitlist_form_submitted`
  - `download_clicked`（仅下载确认后启用）
  - `contact_clicked`
- 分析工具待确认：PostHog / GA4 / Plausible / 无埋点。
- 隐私要求：不将完整邮箱作为 analytics property；可只记录 `email_domain` 或表单提交结果状态。

## 4. 页面结构规划

页面建议单页 `/keystat`，区块顺序：

1. `KeystatHeader`：品牌、锚点导航、主 CTA。
2. `HeroSection`：H1、一句话定位、主 CTA、辅助 CTA、产品视觉。
3. `ProblemSection`：3-4 个痛点。
4. `SolutionSection`：Keystat 如何把键盘/工作流信号转化为洞察。
5. `ValuePropsSection`：Clarity / Focus / Progress / Lightweight / Privacy-aware。
6. `ProductPreviewSection`：截图或 mockup 占位、关键指标卡片、3 步流程。
7. `UseCasesSection`：个人效率、开发者/创作者、团队生产力（团队能力未确认时使用保守措辞）。
8. `TrustPrivacySection`：隐私问题清单与保守声明，明确待确认项。
9. `FaqSection`：5-7 条 FAQ，覆盖隐私、平台、价格、可用时间、联系。
10. `FinalCtaSection`：重复主 CTA。
11. `KeystatFooter`：品牌、联系、政策链接占位、版权。

## 5. SEO 与元数据策略

- `/keystat/page.tsx` 设置唯一 metadata：
  - `title.absolute`: `Keystat — Keyboard workflow insights`（最终文案待确认）
  - `description`: 120-160 英文字符或等效中文长度。
  - `openGraph`、`twitter`、`alternates.canonical`。
- `app/sitemap.ts` 增加 `/keystat`。
- `app/robots.ts` allow `/keystat`，并继续 disallow 认证/工作区/dashboard 私有路径。
- 结构化数据可使用 `SoftwareApplication` 或 `WebPage`，但不得写价格、正式发布日期、支持平台等未确认字段。
- 图片需要 `alt`，首屏核心信息不得只存在于图片中。

## 6. 性能策略

- 首屏避免大型视频、Three.js、复杂 canvas 或重动画。
- 产品预览图使用现代格式（WebP/AVIF），提供尺寸，非首屏懒加载。
- 动画使用 CSS transform/opacity；支持 `prefers-reduced-motion`。
- 第三方分析脚本异步加载并可通过环境变量关闭。
- 目标：LCP ≤ 2.5s，INP ≤ 200ms，CLS ≤ 0.1；Lighthouse Performance/Accessibility/Best Practices/SEO 建议 90+。

## 7. 可访问性策略

- 目标 WCAG 2.2 AA。
- 语义化 HTML：单个 H1，H2/H3 层级清晰。
- CTA 和表单 input 具备可读 label、错误文本、`aria-describedby`。
- 全站键盘可导航，focus ring 明显。
- 颜色对比度达标；错误状态不只依赖颜色。
- reduced motion 下关闭或简化动画。

## 8. 隐私与合规策略

- 页面必须包含隐私相关区块，但在用户确认前只能保守表达：例如 “Designed with privacy in mind. Details coming soon.”
- 不得承诺：不记录具体按键、仅本地存储、端到端加密、云同步、导出/删除能力，除非用户明确确认。
- 表单需有隐私提示；若隐私政策链接未提供，上线前需补齐或移除链接。
- 分析事件不得采集完整邮箱、具体输入内容、敏感字段。

## 9. 配置文件规划

建议 Builder 在实施阶段仅按需修改：

- `multica/apps/web/app/keystat/page.tsx`：页面入口与 metadata。
- `multica/apps/web/app/keystat/layout.tsx`：Keystat 页面级布局、JSON-LD（可选）。
- `multica/apps/web/app/sitemap.ts`：加入 `/keystat`。
- `multica/apps/web/app/robots.ts`：允许 `/keystat`。
- `multica/apps/web/features/keystat-landing/**`：Keystat 页面组件、内容、表单和埋点 adapter。
- `multica/apps/web/public/keystat/**`：品牌与预览资产。
- `.env` / deployment env（仅文档或部署配置层面）：
  - `NEXT_PUBLIC_KEYSTAT_ANALYTICS_PROVIDER`
  - `NEXT_PUBLIC_KEYSTAT_SITE_URL`
  - `KEYSTAT_WAITLIST_ENDPOINT`（server-only，如使用 API route/server action）

当前文档阶段不修改上述业务代码或配置。

## 10. 部署策略

- 若沿用 `multica/apps/web`：通过现有 Next.js Web app 部署，新增路径 `/keystat`。
- 若用户希望独立域名：可将 `keystat.*` 或主域根路径反向代理到 `/keystat`，但需确认域名、canonical、OG URL、sitemap base URL。
- 上线前需确认：域名、隐私政策/条款链接、表单后端、分析工具、品牌素材、下载/等待名单 CTA。

## 11. 风险与 trade-offs

| 议题 | 推荐 | 风险/代价 | 需确认 |
| --- | --- | --- | --- |
| 承载目录 | `multica/apps/web/app/keystat` | 与现有产品 repo 共存，需严格避免品牌/文案污染 | 是否接受共部署 |
| 独立 app | 暂不新增 | 隔离更好但配置/部署成本高 | 是否需要独立域名/CI |
| CTA | 默认等待名单 | 如已有下载包则路径可能变化 | 主 CTA 类型 |
| 表单后端 | adapter 抽象 | 未确认前不能真实提交 | 使用哪个表单服务 |
| 分析工具 | adapter 抽象 + 可关闭 | 第三方脚本影响隐私/性能 | 是否允许分析，使用哪家 |
| 隐私卖点 | 保守描述 | 转化说服力较弱 | 真实数据采集/存储边界 |
| 多语言 | 默认单语言英文内容模块 | 后续多语言需扩展路由/metadata | 是否需要中英双语 |
| 动效 | 轻量 CSS | 视觉冲击力较弱但性能稳 | 是否有品牌动效需求 |

## 12. 验收标准

### 12.1 目录与边界

- Keystat 代码集中在 `app/keystat` 和 `features/keystat-landing`，不污染现有 Hermes/Multica 产品文案。
- 不新增登录、账号体系、后台、订阅计费、云同步等未确认能力。
- 不承诺价格、发布日期、平台支持、下载链接。

### 12.2 页面与内容

- 页面包含 PRD 要求的 Hero、Problem、Solution、Value Props、Product Preview、Use Cases、Trust/Privacy、FAQ、Final CTA、Footer。
- 主 CTA 至少出现 2 次，移动端可见且可点击。
- 所有 TBD 事实明确标记或使用保守文案。

### 12.3 表单与埋点

- 表单有成功、失败、加载、校验状态；失败不丢失输入。
- 事件埋点通过 adapter 触发，未配置 provider 时 no-op。
- 不把完整邮箱作为分析属性。

### 12.4 SEO/性能/可访问性

- `/keystat` 有 title、description、OG/Twitter、canonical、sitemap/robots。
- Lighthouse Performance、Accessibility、Best Practices、SEO 目标 90+。
- 键盘导航、focus、label、错误文本、alt、reduced motion 通过检查。
