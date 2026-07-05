# WORK NOTES — hc-429 公众号运营 playbook v1(自媒体打穿第一棒)

## Scope

- Repository: `karlligamesvc-spec/hermes-agent` (desktop fork)
- Branch: `feat/hc429-wechat-playbook` (base `origin/main` @ `9790955aa`)
- Seat: W2-3 · hc-429 · P1
- Deliverable: `optional-skills/social-media/wechat-mp-operator/` — 改造 MIT 起号 SOP 为**真数据工作流**的公众号运营技能。
- Ship path: optional 选装技能(不动 S6 seed 名单),optional-skills 目录扫描自动发现;桌面随发版、云端随下批 runtime 镜像生效(见"给 PM 清单")。
- ★施工中途 main 前进两步(rebase 已落):PR #49 平台工具网关三插件迁入 fork(=本技能依赖的 `social_*` 工具**已在 main**);PR #50 hc-406 skill 全集重分级 + China seed 统装(未新建 optional-skills/social-media,本技能新建该分类无冲突)。

## Sources Read (真值核对)

- SOP 基底(MIT):`github.com/chenjin-cmd/agent-skills-launch-pack_` → `skills/wechat-account-launch-expert/`(SKILL.md 78 行 + references/launch-playbook.md 271 行 + agents/openai.yaml)。LICENSE = MIT © 2026 Chen。
- Fork 技能目录惯例:`skills/`(bundled,按分类子目录 apple/social-media/productivity…)、`optional-skills/`(选装,同样分类子目录 + 每类 DESCRIPTION.md)。frontmatter 双范式:bundled(xurl:platforms/prerequisites/metadata.hermes.tags+homepage+upstream_skill)、optional(one-three-one-rule/baoyu-comic/pixel-art:category + metadata.hermes.tags,派生技能配 ATTRIBUTION.md + credits)。
- 技能作者指南:`skills/software-development/hermes-agent-skill-authoring/SKILL.md`(validator 源 = `tools/skill_manager_tool.py::_validate_frontmatter`;name≤64、description≤1024、content≤100k;子目录白名单 references/templates/scripts/assets)。
- 发现机制:`tools/skills_hub.py::OptionalSkillSource`(`rglob("SKILL.md")`,`_optional_dir` 指仓库 optional-skills)—— **无需 index 注册**。`skills-index.yml` / `skills-index-freshness.yml` 均 `if: github.repository == 'NousResearch/hermes-agent'`,fork 上不跑,且是 scheduled/live-probe,非 per-PR 门。
- 云端网关契约:`hermes-cloud/docs/TOOLS-GATEWAY-API.md`(鉴权/错误码/计费/`POST /tools/v1/social/{platform}/{action}`;`wechat_mp` 在 10 平台白名单;action 白名单含 search/profile/posts/content/comments)。
- Fork 网关插件(**施工中途 PR #49 `89aa8d013` 已合 main**,本分支 rebase 后其在 disk 上):`plugins/apexnodes-social-tools/__init__.py` —— tool name 与必给参:`social_search`(platform,query)/`social_profile`(platform,user_id)/`social_posts`(platform,user_id)/`social_content`(platform,+url|item_id)/`social_comments`(platform,item_id|url)。platform 描述含 `wechat_mp`。

## Implementation

技能落 `optional-skills/social-media/wechat-mp-operator/`(social-media 是 bundled 已有分类,optional 侧新建 + DESCRIPTION.md):

- `SKILL.md` — 保留 SOP 骨架(起号模式判定 / 定位句 / 账号地基 / 简报先行 / 发布节奏 / 周复盘 / 风险边界),**重写两节为真数据流**:
  - **第 3 步 对标拆解**:4 步工具流(`social_search` 搜赛道 → `social_posts`/`social_profile` 锁对标号 → `social_content` 抽正文 → `social_comments` 拿真实读者语言),对标记录表每行可追溯工具调用;明确诚实边界(后台阅读/在看/涨粉外部采不到→让用户截图;`vendor_unavailable` 显式降级不自装抓取器)。
  - **第 4 步 选题库**:字段表(选题/痛点/关键词/角度/标题方向/素材源/风险检查),种子来自真实搜索结果;热点暂用 `wechat_mp` search + 留 `<!-- weibo 热搜源上线后接入 -->` 注释。
  - 第 5 步 简报→正文两段式 + 头图一句(调运行时生图/带字封面 flash 默认,无能力诚实跳过)。
  - 第 6 步 周复盘模板:外采(对标动态,重跑 social 工具对比)vs 用户截图(自号后台)两类数据分开,输出有效模式/失败假设/下周实验,明示哪些数据拿不到。
- `references/launch-playbook.md` — prompt-only 的采集表/准备度检查表/标题模式/简报模板/提示词模板/30天日历/指标诊断/合规替代(对标与选题两节移到 SKILL.md,这里留指针)。
- `references/social-tools-cheatsheet.md` — 5 个 action 的入参速查 + 完整错误码分支(api_key/quota/vendor_unavailable/rate_limited/…)+ 采不到的数据清单 + 头图生图边界。
- `ATTRIBUTION.md` — 保留原 MIT LICENSE 全文 + © 2026 Chen 版权声明,注明"骨架复用/两节重写"delta。
- frontmatter credits 指回源仓;`风险边界` 合规段原样保留(含数据诚实铁律:拿不到就说、不承诺涨粉/收益)。

## 与 launchpack 原版 diff 要点(改了哪两节为真数据流)

| 节 | 原版(纯 prompt) | 改造后(真数据) |
| --- | --- | --- |
| 对标系统 / 对标拆解 | "优先选择读者群相同的账号…" + 一张让模型凭空填的对标记录表 | 4 步显式工具调用(search→posts/profile→content→comments,platform=wechat_mp),每行可追溯某次工具返回;加"后台数据外部采不到→截图"诚实边界 + vendor_unavailable 降级纪律 |
| 选题库 | 按"文章任务"分类的静态模式表(搜索/信任/故事…) | 字段表(选题/痛点/关键词/角度/标题方向/素材源/风险检查)种子来自真实 search/comments;≥10 条,≥半数可追溯工具调用;留 weibo 热搜接入注释 |

其余节(模式判定/地基/标题模式/简报模板/日历/诊断/合规替代)= 原版骨架,restructure 进 Hermes SKILL.md 结构 + 少量真数据锚点(如准备度检查表"对标"行改为"已用 social 工具真实拉过")。

## Validation Log

- **frontmatter 校验(权威)**:`tools/skill_manager_tool.py::_validate_frontmatter` 返回 `None`(= valid)。name=`wechat-mp-operator`(18 字符)、description 263 CJK 字符(≤1024)、SKILL.md 6959 字符(≤100k)。references 两文件 11298 / 4317 bytes(远 <1MiB)。
- **目录发现**:`OptionalSkillSource`(`rglob SKILL.md`)在 102 个 optional 技能里发现 `social-media/wechat-mp-operator`,category=social-media。
- **工具姿势交叉核对(mock-key posture)**:把 SKILL.md + cheatsheet 里指示的每个工具调用示例,与 **main 上 `plugins/apexnodes-social-tools/__init__.py`** 真实 SCHEMAS 逐一比对 —— 5 个工具全部命中真实 tool name,示例入参全部满足 required(social_search:platform+query;social_posts:platform+user_id;social_content:platform+url;social_comments:platform+item_id;social_profile:platform+user_id),`platform:"wechat_mp"` 在 9 处示例调用中正确使用。**RESULT: PASS**(技能指示的调用姿势与真实插件 schema 一致)。
- **插件装载 + 工具暴露(rebase 后 main 插件在 disk)**:import `plugins/apexnodes-social-tools/__init__.py` 成功;其 `SCHEMAS` 暴露的正是技能驱动的 5 个工具(+social_trending/social_captions),`social_search` 参数含 platform/query/user_id/item_id/url/count/cursor/params,required=[platform,query],platform 枚举含 `wechat_mp`。设 mock `API_SERVER_KEY` 后 gating 面就绪(不触 vendor)。**PASS**。
- **Skills Hub 装载路径实测(= `hermes skills search/inspect/install` 走的机器)**:
  - `search("公众号"|"wechat"|"对标")` → 各命中 1 结果,含本技能。
  - `inspect("wechat-mp-operator")` → name/path=`optional-skills/social-media/wechat-mp-operator`/source=`official`/trust=`builtin`/description 正确回显。
  - `fetch("wechat-mp-operator")` → 完整 bundle 4 文件(ATTRIBUTION.md + SKILL.md + 两 references);SKILL.md 解码后含全部真数据+诚实标记(social_search/social_posts/social_content/social_comments/wechat_mp/对标拆解/选题库/截图/不承诺/weibo);cheatsheet 含全部 5 tool name + 错误码处理。**DEV-LOAD: PASS**。

### 诚实说明:live 模型 turn 未跑(为什么用上面这套等价证据)

本 seat 环境**无 LLM provider key**(仅 `ANTHROPIC_BASE_URL`,无 key),无法跑 `hermes chat -s wechat-mp-operator -q ...` 让模型真实推理并观察它选哪个工具。因此改用**确定性等价证据**:①技能装载走真实 Skills Hub official source(search/inspect/fetch 全过);②技能指示的工具调用姿势与 main 上真实插件 schema 逐字段交叉核对通过;③插件本体 import + SCHEMAS 暴露核对通过。这三条覆盖了"技能能被装载 + 装载后指示的工具调用姿势正确(mock key 也看得出)"的验收意图。真·端到端"定位→对标真调工具→10 条选题→1 篇简报"需在有 LLM key + 网关 `/tools/v1/social/*` 对 `wechat_mp` 就绪的环境跑(插件代码已在 main,缺的是运行时镜像 + key),列入 PM 清单待桌面发版时由 Kael 真机验。

## 给 PM 清单

- **本 PR 只交付技能(纯 md)**,不合并不出包(seat 铁律)。
- **云端生效依赖**:`social_*` 工具由 `apexnodes-social-tools` 插件提供,该插件 **已随 PR #49 合入 fork main**(施工中途落地)。技能要在云端 agent 真正调通,剩两件:①插件随**下批 runtime 镜像**滚到存量 agent(hc-401 或下批镜像);②云端网关 `/tools/v1/social/*` 对 `wechat_mp` 五件 action 就绪(云端契约已列 wechat_mp 在白名单,需确认 vendor 侧 wechat_mp 端点已配)。
- **桌面生效**:随发版带(desktop bundle 含 optional-skills;用户经 Skills Hub `install` 或 seed 带入)。**注意**:S6 定的 seed 名单未含本技能(它是选装,故意不动 seed);若要桌面默认装,需 PM 决定是否进后续 seed 批次。
- **头图能力**:带字封面走运行时图片生成(caption-cover/flash),属云端能力;桌面无生图时技能会诚实跳过。
- **weibo 热搜源**:另一 seat 在扩;本技能选题库热点行已留接入注释,上线后把热搜词喂 `social_search` query 即可,无需改技能结构。

## 阻塞项

- 无硬阻塞(技能可独立合并、CI 应绿)。
- 软依赖:云端真调通 = 等插件随下批镜像滚存量 agent + 网关 wechat_mp 端点就绪(插件代码已在 main,见 PM 清单);live 端到端模型验收 = 等有 key 的环境/真机(见"诚实说明")。
