# WORK NOTES — hc-428 小红书运营 playbook v1(自媒体打穿第二棒)

## Scope

- Repository: `karlligamesvc-spec/hermes-agent` (desktop fork)
- Branch: `feat/hc428-xhs-playbook` (base `origin/main` @ `28af44c79`)
- Seat: W3-1 · hc-428 · P1
- Deliverable: `optional-skills/social-media/xhs-operator/` — 照抄已合并的 hc-429 `wechat-mp-operator`(#51)结构,把 MIT 小红书起号 SOP 改造为**真数据工作流**的小红书运营技能。
- Ship path: optional 选装技能(不动 S6 seed 名单),`optional-skills/` 目录 rglob 自动发现;桌面随发版、云端随下批 runtime 镜像生效(见"给 PM 清单")。
- ★施工前 rebase:worktree 建于 `origin/main` 后 main 前进到 `28af44c79`(#51 hc-429 wechat-mp-operator 已合),已 `reset --hard origin/main` 落 #51——它就是本技能的结构模板 + 依赖(`apexnodes-social-tools` 插件已在 main)。

## Sources Read (真值核对)

- **结构模板 = 已合并的 #51**(`optional-skills/social-media/wechat-mp-operator/`):SKILL.md(196 行)+ ATTRIBUTION.md(82 行)+ references/launch-playbook.md(222 行)+ references/social-tools-cheatsheet.md(69 行)。逐一读过,照其骨架/诚实边界/ATTRIBUTION 模式落 xhs 版。`optional-skills/social-media/DESCRIPTION.md`(#51 建,泛化文案)已覆盖 social-media 选装类,**未改**。
- SOP 基底(MIT):`github.com/chenjin-cmd/agent-skills-launch-pack_` → `skills/xiaohongshu-account-launch-expert/`。LICENSE = MIT © 2026 Chen(与 #51 同源仓,同一 LICENSE)。
- **插件 SCHEMAS 真值(在 main / disk)**:`plugins/apexnodes-social-tools/__init__.py`——
  - `SCHEMAS`:`social_content`(platform)/`social_search`(platform,query)/`social_profile`(platform,user_id)/`social_comments`(platform,item_id)/`social_trending`(platform)/`social_posts`(platform,user_id)/`social_captions`(platform)。
  - `CREATOR_TOP_POSTS_SCHEMA`:`creator_top_posts`(required=[],需 url 或 user_id;min_likes/min_collects/min_comments/sort_by/top/scan_limit/since/until)。
  - platform 枚举含 `xiaohongshu`;**`creator_top_posts` 的 platform 描述显式列 `xiaohongshu`**(= xhs 支持按互动量枚举,不同于 wechat_mp——#51 cheatsheet 注 wechat_mp 用不上 creator_top_posts,xhs 用得上)。item_id 描述含 `note_id`(xhs 内容 id)。
- 网关白名单:`plugins/apexnodes_gateway.py::SOCIAL_PLATFORMS`——10 平台含 `xiaohongshu`;xhs 链接域名 `xiaohongshu.com` / `xhslink.com`。
- 校验源:`tools/skill_manager_tool.py::_validate_frontmatter`(name≤64、description≤1024、content≤100k;子目录 references/templates/scripts/assets)。
- 发现机制:`tools/skills_hub.py::OptionalSkillSource`(`source_id=official` / `trust=builtin`,`rglob("SKILL.md")`,`HERMES_OPTIONAL_SKILLS` 或仓库 `optional-skills` 目录)——**无需 index 注册**。`skills-index*.yml` 均 `if: github.repository == 'NousResearch/hermes-agent'`,fork 上不跑,非 per-PR 门。

## Implementation

技能落 `optional-skills/social-media/xhs-operator/`(4 文件):

- `SKILL.md` — 保留 SOP 骨架(起号模式判定 / 定位句 / 主页地基 / 简报先行 / 内容日历 / 周复盘 / 风险边界),**改两节为真数据流** + xhs 化(笔记/封面/标签/私信红线):
  - **第 3 步 对标拆解**:5 步工具流(`social_search` 搜赛道 → `social_posts` 拉近期 → **`creator_top_posts` 按互动量枚举爆款**[xhs 支持] → `social_content` 抽正文/封面/标签 → `social_comments` 拿真实读者语言 + 真实标签),对标记录表每行可追溯工具调用;含**封面模式**行(xhs 特有:大字/实拍叠字/前后对比/九宫格…)。诚实边界(后台曝光/点击/涨粉/薯条外部采不到→截图;`vendor_unavailable` 显式降级不自装抓取器)。
  - **第 4 步 选题库**:字段表(选题/痛点/关键词/**标签建议**/角度/标题+封面方向/素材源/风险检查),种子来自真实搜索;热点用 `social_search` 近似 + 留 `<!-- 小红书热门榜数据源上线后接入(hc-426) -->` 注释。
  - **第 5 步 简报→文案+封面两段式**:简报字段含**封面文案 + 标签组**;封面**生成 3 版选 1**,走运行时生图能力(**系统按档位自动选路由**:带字封面默认档 + 免费封面线[平台底图+排版叠字,hc-433]在档位内可选;**不写死 provider/模型**);无生图能力诚实跳过 + 给排版建议。
  - **第 6 步 内容日历(30 天落表)+ 周复盘**:复盘 = 数据回采对比 + **淘汰弱选题**;外采(对标动态,重跑 social 工具对比)vs 用户截图(自号后台)两类数据分开;输出有效模式/失败假设淘汰/下周实验;明示热门榜数据源待接入(hc-426)。
  - **转化路径:只做内容策略层(主页引导 + 评论区口径);私信自动化 = 红线明写不做**(4 处:When-to-use / 主页地基硬边界 / 风险边界列表 / 风险边界硬边界段)。
- `references/launch-playbook.md` — prompt-only:需求采集表/起号模式/主页地基+准备度检查表/**标题钩子模式**/**封面模式清单**/笔记简报模板/**图文+视频笔记提示词**/30天日历/指标诊断/合规替代(含私信红线 + 广告法极限词)。
- `references/social-tools-cheatsheet.md` — 6 个 action 入参速查(含 `creator_top_posts`,注明 xhs 支持)+ 完整错误码分支 + 采不到的数据清单(+ 热门榜 hc-426 注释)+ 封面生成路由边界。
- `ATTRIBUTION.md` — 保留原 MIT LICENSE 全文 + © 2026 Chen 声明,注明"骨架复用/两节重写"delta + xhs 专属诚实注(无热门榜数据源 / 后台指标外采不到)。
- frontmatter credits 指回源仓;`风险边界` 合规段(数据诚实铁律:拿不到就说、不承诺涨粉/爆款/上热门/收益)。

## 与 launchpack 原版 diff 要点(改了哪两节为真数据流 + xhs 专属新增)

| 节 | 原版(纯 prompt) | 改造后(真数据 / xhs 化) |
| --- | --- | --- |
| 对标系统 / 对标拆解 | 让模型凭空列对标账号 + 编爆款封面/标题 + 猜热门标签 | 5 步显式工具调用(search→posts→**creator_top_posts**→content→comments,platform=xiaohongshu),每行可追溯工具返回;新增**封面模式**+**真实标签**抽取;加"后台数据外部采不到→截图"诚实边界 + vendor_unavailable 降级纪律 |
| 选题库 | 静态模式表(按文章任务分类) | 字段表(选题/痛点/关键词/**标签建议**/角度/标题+封面方向/素材源/风险检查)种子来自真实 search/comments;≥10 条,≥半数可追溯;留 hc-426 热门榜接入注释 |
| 封面(xhs 专属) | 无(公众号只提头图) | 简报出封面文案 + **生成 3 版选 1**,走运行时生图(**系统按档位自动选路由**,不写死 provider;带字默认档 + 免费封面线 hc-433);无能力诚实跳过 |
| 转化路径(xhs 专属) | 无明确红线 | **只做内容策略层(主页引导+评论区口径);私信自动化 = 红线明写不做**,留资改官方功能 |

其余节(模式判定/地基/标题钩子/简报模板/30天日历/诊断/合规替代)= 原版骨架,restructure 进 Hermes SKILL.md 结构 + xhs 语汇(笔记/点赞收藏/薯条/蒲公英/广告法极限词)+ 少量真数据锚点。

## SCHEMAS 核对结果(与 main 插件逐字段)

对 SKILL.md + cheatsheet 里 **11 个示例工具调用**,与 main 上 `plugins/apexnodes-social-tools/__init__.py` 真实 `SCHEMAS` / `CREATOR_TOP_POSTS_SCHEMA` 逐一比对(脚本化):

| 工具 | required(真值) | 示例入参 | 结果 |
| --- | --- | --- | --- |
| social_search | platform, query | platform=xiaohongshu, query, count | ✅ |
| social_profile | platform, user_id | platform=xiaohongshu, user_id | ✅ |
| social_posts | platform, user_id | platform=xiaohongshu, user_id, count | ✅ |
| creator_top_posts | [](需 url 或 user_id) | platform=xiaohongshu, user_id, min_likes, sort_by, top | ✅ |
| social_content | platform(+url 或 item_id) | platform=xiaohongshu, url | ✅ |
| social_comments | platform, item_id | platform=xiaohongshu, item_id | ✅ |

**RESULT: PASS**(11/11 调用命中真实 tool name,required 全满足,`platform="xiaohongshu"` 全正确)。`xiaohongshu` 在 platform 枚举 + creator_top_posts platform 描述中均存在;`creator_top_posts` 对 xhs 可用(≠ wechat_mp)。

## Validation Log

- **frontmatter 校验(权威,跑 repo `_validate_frontmatter`)**:返回 `None`(= valid)。name=`xhs-operator`(12 字符 ≤64)、description 332 CJK 字符(≤1024)、SKILL.md 18604 字符(≤100k;`_validate_content_size` 也返回 None)。references 两文件 13589 / 5610 bytes、ATTRIBUTION 4355 bytes(远 <1MiB)。
- **Skills Hub 三连(权威,跑 repo `OptionalSkillSource`,指向 worktree optional-skills)**:
  - `search("小红书"|"xiaohongshu"|"对标"|"封面"|"xhs")` → 5/5 命中,均含本技能。
  - `inspect("xhs-operator")` → name/tags/description 正确回显;identifier=`official/social-media/xhs-operator`;source=`official`。
  - `fetch("xhs-operator")` → 完整 bundle 4 文件(ATTRIBUTION.md + SKILL.md + 两 references);SKILL.md 解码后含全部真数据+诚实标记(social_search/social_posts/creator_top_posts/social_content/social_comments/xiaohongshu/对标拆解/选题库/封面/截图/不承诺/私信/hc-426);cheatsheet 含全部 6 tool name;ATTRIBUTION 含 MIT License 全文。**三连 PASS**。
- **工具姿势交叉核对(mock-key posture)**:见上"SCHEMAS 核对结果"表,**PASS**。

### 诚实说明:live 模型 turn 未跑(为什么用上面这套等价证据)

本 seat 环境**无 LLM provider key**,无法跑 `hermes chat -s xhs-operator -q ...` 让模型真实推理并观察它选哪个工具。因此改用与 #51 同款**确定性等价证据**:①技能装载走真实 Skills Hub official source(search/inspect/fetch 全过);②技能指示的工具调用姿势与 main 上真实插件 schema 逐字段交叉核对通过(11/11);③frontmatter 走 repo 权威校验器通过。这三条覆盖了"技能能被装载 + 装载后指示的工具调用姿势正确(mock key 也看得出)"的验收意图。真·端到端"定位→对标真调工具→10 条选题→1 篇简报出 3 版封面"需在有 LLM key + 网关 `/tools/v1/social/*` 对 `xiaohongshu` 就绪 + 生图能力就绪的环境跑(插件代码已在 main,缺的是运行时镜像 + key + vendor),列入 PM 清单待桌面发版时由 Kael 真机验。

## 给 PM 清单

- **本 PR 只交付技能(纯 md)**,不合并不出包(seat 铁律)。
- **云端生效依赖**:`social_*` / `creator_top_posts` 工具由 `apexnodes-social-tools` 插件提供,该插件 **已在 fork main**(#49/#51 已带)。技能要在云端 agent 真正调通,剩:①插件随**下批 runtime 镜像**滚到存量 agent;②云端网关 `/tools/v1/social/*` 对 `xiaohongshu` 各 action 就绪(白名单已含 xiaohongshu,需确认 vendor 侧 xhs 端点已配)。
- **桌面生效**:随发版带(desktop bundle 含 optional-skills;用户经 Skills Hub `install` 或 seed 带入)。**注意**:S6 seed 名单未含本技能(选装,故意不动 seed);若要桌面默认装,需 PM 决定是否进后续 seed 批次。
- **封面能力**:带字封面走运行时图片生成(**系统按档位自动选路由**,不写死 provider);免费封面线(平台底图+排版叠字)= **hc-433 并行施工中**,上线后自动落入"系统档位内可选",无需改本技能(技能已写成档位路由,不绑具体线路)。桌面无生图时技能诚实跳过 + 给排版建议。
- **热门榜数据源(hc-426)**:小红书当前无公开热门榜端点,技能选题库/复盘热点行已留接入注释 + 用 `social_search` 近似;上线后把热榜话题喂 `social_search` query 即可,无需改技能结构。

## 阻塞项

- 无硬阻塞(技能可独立合并、CI 应绿——纯 md 改动;Docker build 慢检查非门)。
- 软依赖:云端真调通 = 等插件随下批镜像滚存量 agent + 网关 xiaohongshu 端点就绪(插件代码已在 main,见 PM 清单);免费封面线 = hc-433 并行落地(技能已解耦,不阻塞);live 端到端模型验收 = 等有 key 的环境/真机(见"诚实说明")。
