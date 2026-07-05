# WORK NOTES — hc-427 抖音运营 playbook v1(自媒体打穿三平台最后一棒)

## Scope

- Repository: `karlligamesvc-spec/hermes-agent` (desktop fork)
- Branch: `feat/hc427-douyin-playbook` (base `origin/main` @ `3f881e412`)
- Seat: W4-1 · hc-427 · P1
- Deliverable: `optional-skills/social-media/douyin-operator/` — 照抄已合并的 hc-429 `wechat-mp-operator`(#51)+ hc-428 `xhs-operator`(#54)结构,把 MIT 抖音起号 SOP(`douyin-account-launch-expert`)改造为**真数据 + 爆款转写拆解**的抖音运营技能。
- Ship path: optional 选装技能(不动 S6 seed 名单),`optional-skills/` 目录 rglob 自动发现;桌面随发版、云端随下批 runtime 镜像生效(见"给 PM 清单")。
- 三平台样板齐:公众号(#51)、小红书(#54)、抖音(本 PR)= 自媒体打穿三棒收口。

## Sources Read (真值核对)

- **结构模板 = 已合并的 #51 + #54**(`optional-skills/social-media/{wechat-mp-operator,xhs-operator}/`):逐一读过 SKILL.md / ATTRIBUTION.md / references 两文件,照其骨架/诚实边界/ATTRIBUTION 模式落 douyin 版。`optional-skills/social-media/DESCRIPTION.md`(#51 建,泛化文案)已覆盖 social-media 选装类,**未改**。
- SOP 基底(MIT):`github.com/chenjin-cmd/agent-skills-launch-pack_` → `skills/douyin-account-launch-expert/`。LICENSE = MIT © 2026 Chen(与 #51/#54 同源仓,同一 LICENSE)。
- **社媒数据插件 SCHEMAS 真值(在 main / disk)**:`plugins/apexnodes-social-tools/__init__.py`——
  - `SCHEMAS`:`social_content`(platform)/`social_search`(platform,query)/`social_profile`(platform,user_id)/`social_comments`(platform,item_id)/`social_trending`(platform)/`social_posts`(platform,user_id)/`social_captions`(platform)。
  - `CREATOR_TOP_POSTS_SCHEMA`:`creator_top_posts`(required=[],需 url 或 user_id;min_likes/min_collects/min_comments/sort_by/top/scan_limit/since/until);platform 描述**显式列 `douyin`**(抖音是首选枚举平台)。item_id 描述含 `aweme_id`(抖音内容 id)。
  - `social_trending` required 只有 `platform` → **抖音有公开热门榜**(公众号/小红书样板拿不到)。`social_captions` = **YouTube 专用**(官方字幕、无 ASR 成本),抖音不用。
- **媒体插件 SCHEMAS 真值(抖音线独有优势)**:`plugins/apexnodes-douyin-tools/__init__.py`(hc-254:tool name 平台中立,包名保留只为注册稳定)——`social_download`(url)/`media_transcribe`(video_path 或 url,一步下载+转写)/`image_ocr`(url 或 image_urls)/`social_batch_submit`(urls 或 creator_url,异步 job_id)/`social_batch_status`(job_id)。`media_transcribe` 显式禁自装本地 whisper,尊重 `terminal_fallback_allowed=false`。
- 网关白名单:`plugins/apexnodes_gateway.py::SOCIAL_PLATFORMS`——**10 平台含 `douyin`,不含 `weibo`**(实核 grep 全仓 0 weibo 命中;hc-426 weibo 平台在**云 repo** 落地,尚未同步进本 fork)。抖音链接域名 `v.douyin.com` / `douyin.com` / `iesdouyin.com`。
- 校验源:`tools/skill_manager_tool.py::_validate_frontmatter`(name≤64、description≤1024、content≤100k;子目录 references/templates/scripts/assets)。
- 发现机制:`tools/skills_hub.py::OptionalSkillSource`(`source_id=official` / `trust=builtin`,`rglob("SKILL.md")`,`HERMES_OPTIONAL_SKILLS` 或仓库 `optional-skills` 目录)——**无需 index 注册**。`skills-index*.yml` 均 `if: github.repository == 'NousResearch/hermes-agent'`,fork 上不跑,非 per-PR 门。

## Implementation

技能落 `optional-skills/social-media/douyin-operator/`(4 文件):

- `SKILL.md` — 保留 SOP 骨架(起号模式判定 / 定位句 / 主页地基 / 简报先行 / 内容日历 / 周复盘 / 风险边界),**改真数据流 + 加抖音独有步骤**:
  - **第 3 步 对标拆解**:热榜起手(`social_trending platform=douyin`)+ 5 工具流(search → posts → **creator_top_posts** 按互动量枚举[抖音首选] → content → comments),对标记录每行可追溯工具调用。诚实边界(后台播放/完播/涨粉/DOU+外部采不到→截图;vendor_unavailable 显式降级不自装抓取器)。
  - **★第 3.5 步 爆款视频转写拆解(抖音线独有,公众号/小红书没有)**:指示 agent 用 `media_transcribe`(一步下载+转写,传分享链接原文)或 `social_download`+`media_transcribe` 两步,把爆款口播逐字拿下来填拆解表(前 3 秒钩子/内容结构/信息节奏/结尾钩子,均追溯逐字稿);批量拆作者/合集走 `social_batch_submit`+`social_batch_status`(异步只提交/轮询/发产物)。硬禁自装本地 whisper,尊重 terminal_fallback_allowed=false。
  - **第 4 步 选题库**:字段表(选题/痛点/关键词/**参考钩子**/形式/角度/封面方向/素材源/风险检查),种子来自热榜+搜索+转写;热点用 `social_trending`+`social_search`;留跨平台 weibo 接入注释(见诚实说明)。
  - **★第 5 步 9 条视频实验矩阵(3 选题×3 形式)**:落表 brief/前 3 秒钩子(逐字)/发布时间;发布后 `social_posts`+`social_content` 回采**公开信号**对比 + 用户后台截图补精确数据;横向比 + 淘汰弱假设;**明示"样本小(9 条),信号≠定论"**。
  - **第 6 步 口播脚本+封面简报两段式**:简报含前 3 秒开场白(逐字)+ 分镜口播脚本(镜头/口播/字幕/时长)+ 封面文案;封面**生成 3 版选 1**,走运行时生图(**系统按档位自动选路由**,不写死 provider;带字默认档 + 免费封面线 hc-433 档位内可选);无生图能力诚实跳过 + 给排版建议。
  - **第 7 步 内容日历(30 天)+ 周复盘**:复盘 = 数据回采对比 + 下周实验假设 + 淘汰弱选题;外采(公开信号+对标动态,重跑 social 工具)vs 用户截图(自号后台)两类数据分开;四段输出(有效/失败淘汰/下周实验假设/脚本封面产出);明示样本小。
  - **互动设计红线(hc-075)**:评论区分析只产出**人工回复策略 + 置顶评论钩子**;**评论自动回复 / 私信自动化 = 红线明写不做**(4 处:When-to-use / 主页地基硬边界 / 风险边界列表 / 风险边界硬边界段)。
- `references/launch-playbook.md` — prompt-only:需求采集表/起号模式/主页地基+准备度检查表/**前 3 秒钩子模式**/**封面模式清单**/**爆款转写拆解表模板**/**9 条视频实验矩阵模板**/口播脚本简报模板/**口播+剧情+图文提示词**/30天日历/指标诊断/合规替代(含评论自动回复红线 + 广告法极限词 + BGM/肖像版权)。
- `references/social-tools-cheatsheet.md` — A 组数据类 7 工具(含 `social_trending` 注明抖音可用、`creator_top_posts` 注明抖音首选)+ B 组**媒体类 5 工具**(social_download / media_transcribe / image_ocr / social_batch_submit / social_batch_status,抖音独有)入参速查 + 完整错误码分支 + 采不到的数据清单(+ weibo 白名单注释)+ 封面生成路由边界。
- `ATTRIBUTION.md` — 保留原 MIT LICENSE 全文 + © 2026 Chen 声明,注明"骨架复用/真数据重写/**转写拆解+9条矩阵为抖音专属 delta**",+ 诚实注(抖音有热门榜、weibo 未进 fork 白名单、后台指标外采不到、hc-075 评论红线)。
- frontmatter credits 指回源仓;`风险边界` 合规段(数据诚实铁律:拿不到就说、不承诺涨粉/爆款/上热门/上热榜/收益、样本小信号≠定论)。

## 与 launchpack 原版 diff 要点(改了哪些为真数据流 + 抖音专属新增)

| 节 | 原版(纯 prompt) | 改造后(真数据 / 抖音化) |
| --- | --- | --- |
| 对标系统 / 对标拆解 | 让模型凭空列对标账号 + 编爆款钩子/封面 + 猜热门选题 | 热榜起手(`social_trending`)+ 5 步显式工具调用(search→posts→**creator_top_posts**→content→comments,platform=douyin),每行可追溯;加"后台数据外部采不到→截图"诚实边界 + vendor_unavailable 降级纪律 |
| **爆款转写拆解(抖音专属·新增步)** | 无(纯 prompt 包只看标题封面猜) | **★新增第 3.5 步**:`media_transcribe`/`social_download`/`social_batch_submit` 把口播逐字转写再拆(前 3 秒钩子/结构/节奏/结尾),硬禁自装 whisper。**公众号/小红书样板没有这一步。** |
| 选题库 | 静态模式表(按视频任务分类) | 字段表(选题/痛点/关键词/**参考钩子**/形式/角度/封面方向/素材源/风险检查)种子来自热榜/搜索/转写;≥10 条,≥半数可追溯;留 weibo 跨平台接入注释 |
| **9 条视频实验矩阵(抖音专属·新增)** | 无(原版只有 30 天日历粗排) | **★新增第 5 步**:3 选题×3 形式落表(brief/前 3 秒钩子/发布时间);posts+content 回采公开信号横向对比 + 淘汰弱假设;明示"样本小,信号≠定论" |
| 封面/首帧 | 无(原版不涉及生图) | 简报出封面文案 + **生成 3 版选 1**,走运行时生图(**系统按档位自动选路由**,不写死 provider);无能力诚实跳过 |
| 互动设计(hc-075) | 无明确红线 | 评论区分析只产**人工回复策略 + 置顶评论钩子**;**评论自动回复/私信自动化 = 红线明写不做** |

其余节(模式判定/地基/前 3 秒钩子清单/简报模板/30天日历/诊断/合规替代)= 原版骨架,restructure 进 Hermes SKILL.md 结构 + 抖音语汇(完播/前 3 秒/DOU+/星图/BGM 版权/广告法极限词)+ 真数据锚点。

## SCHEMAS 核对结果(与 main 两个插件逐字段,脚本化)

对 SKILL.md + cheatsheet 里 **13 个示例工具调用**,与 main 上 `plugins/apexnodes-social-tools/__init__.py`(`SCHEMAS`+`CREATOR_TOP_POSTS_SCHEMA`)+ `plugins/apexnodes-douyin-tools/__init__.py`(5 媒体 schema)真实定义逐一比对(脚本 import 真模块读 required):

| 工具 | 来源插件 | required(真值) | 示例入参 | 结果 |
| --- | --- | --- | --- | --- |
| social_trending | social-tools | platform | platform=douyin | ✅ |
| social_search | social-tools | platform, query | platform=douyin, query, count | ✅ |
| social_profile | social-tools | platform, user_id | platform=douyin, user_id | ✅ |
| social_posts | social-tools | platform, user_id | platform=douyin, user_id, count | ✅ |
| creator_top_posts | social-tools | [](需 url 或 user_id) | platform=douyin, user_id, min_likes, sort_by, top | ✅ |
| social_content | social-tools | platform(+url 或 item_id) | platform=douyin, url | ✅ |
| social_comments | social-tools | platform, item_id | platform=douyin, item_id | ✅ |
| media_transcribe | douyin-tools | [](需 video_path 或 url) | url / video_path 两式 | ✅✅ |
| social_download | douyin-tools | url | url | ✅ |
| image_ocr | douyin-tools | [](需 url 或 image_urls) | url, prompt | ✅ |
| social_batch_submit | douyin-tools | [](需 urls 或 creator_url) | creator_url, min_likes, top | ✅ |
| social_batch_status | douyin-tools | job_id | job_id | ✅ |

**RESULT: PASS**(13/13 调用命中真实 tool name,required 全满足含 OR-group,`platform="douyin"` 全在白名单)。`douyin` 在 platform 枚举 + creator_top_posts platform 描述中均存在;`social_trending` 抖音可用(公众号/小红书拿不到);媒体三件(download/transcribe/batch)= 抖音线独有。**`weibo` 实核不在 fork 白名单**(grep 全仓 0 命中),技能已避开且留注释。

## Validation Log

- **frontmatter 校验(权威,跑 repo `_validate_frontmatter`)**:返回 `None`(= valid)。name=`douyin-operator`(15 字符 ≤64)、description 459 CJK 字符(≤1024)、SKILL.md 12189 字符(≤100k;`_validate_content_size` 也返回 None)。references 两文件 18204 / 9541 bytes、ATTRIBUTION 6213 bytes(远 <1MiB)。
- **Skills Hub 三连(权威,跑 repo `OptionalSkillSource`,`HERMES_OPTIONAL_SKILLS` 指 worktree optional-skills)**:
  - `search("抖音"|"douyin"|"对标"|"转写"|"运营")` → 5/5 命中,均含本技能("对标"/"运营" 正确返回三 social 技能全集)。
  - `inspect("douyin-operator")` → name/tags(10 个,含 douyin/video-teardown/transcription)/description 正确回显;identifier=`official/social-media/douyin-operator`;source=`official`;trust=`builtin`;path=`optional-skills/social-media/douyin-operator`。
  - `fetch("douyin-operator")` → 完整 bundle 4 文件;SKILL.md 解码后含全部真数据+诚实标记(social_search/social_trending/social_posts/creator_top_posts/social_content/social_comments/media_transcribe/social_download/social_batch_submit/image_ocr/douyin/对标拆解/选题库/逐字稿/转写/不承诺/私信/自动回复/hc-426/样本小 = 20/20);cheatsheet 含全部 **12 tool name**;ATTRIBUTION 含 MIT License 全文。**三连 PASS**。
- **SCHEMAS 交叉核对(脚本 import 真插件模块)**:见上"SCHEMAS 核对结果"表,**13/13 PASS**;网关 `SOCIAL_PLATFORMS` 打印确认含 douyin、不含 weibo。

### 诚实说明:live 模型 turn 未跑(为什么用上面这套等价证据)

本 seat 环境**无 LLM provider key**,无法跑 `hermes chat -s douyin-operator -q ...` 让模型真实推理并观察它选哪个工具。因此改用与 #51/#54 同款**确定性等价证据**:①技能装载走真实 Skills Hub official source(search/inspect/fetch 全过);②技能指示的 13 个工具调用姿势与 main 上两个真实插件 schema 逐字段交叉核对通过(脚本 import 真模块);③frontmatter 走 repo 权威校验器通过。这三条覆盖了"技能能被装载 + 装载后指示的工具调用姿势正确(mock key 也看得出)"的验收意图。真·端到端"定位→对标真调工具→转写拆爆款→10 条选题→9 条实验矩阵→1 篇口播简报出 3 版封面"需在有 LLM key + 网关 `/tools/v1/social/*` 与 `/tools/v1/asr/*` 对 `douyin` 就绪 + 生图能力就绪的环境跑(插件代码已在 main,缺的是运行时镜像 + key + vendor),列入 PM 清单待桌面发版时由 Kael 真机验。

## 给 PM 清单

- **本 PR 只交付技能(纯 md)**,不合并不出包(seat 铁律)。
- **云端生效依赖**:`social_*`/`creator_top_posts` 由 `apexnodes-social-tools` 提供、`social_download`/`media_transcribe`/`image_ocr`/`social_batch_*` 由 `apexnodes-douyin-tools` 提供,**两插件均已在 fork main**(#49/#51 已带)。技能要在云端 agent 真正调通,剩:①两插件随**下批 runtime 镜像**滚到存量 agent;②云端网关 `/tools/v1/social/*`(含 `trending`)+ `/tools/v1/asr/transcribe` 对 `douyin` 就绪(白名单已含 douyin,需确认 vendor 侧抖音数据 + ASR 端点已配)。
- **桌面生效**:随发版带(desktop bundle 含 optional-skills;用户经 Skills Hub `install` 或 seed 带入)。**注意**:S6 seed 名单未含本技能(选装,故意不动 seed);若要桌面默认装,需 PM 决定是否进后续 seed 批次(三 social 技能可一起决定)。
- **封面能力**:带字封面走运行时图片生成(**系统按档位自动选路由**,不写死 provider);免费封面线 = **hc-433**(平台底图+排版叠字),上线后自动落入"系统档位内可选",无需改本技能。桌面无生图时技能诚实跳过 + 给排版建议。
- **★跨平台 weibo 选题源(hc-426)**:云端已上 weibo 平台,但**当前 fork 网关白名单尚未含 weibo**(实核 0 命中)。技能选题库/cheatsheet 已留接入注释 + 明确当前用 douyin `social_trending`+`social_search` 覆盖站内热点、**勿调 weibo(会 platform_not_allowed)**;待 weibo 随下批同步进 fork 白名单后,`social_trending platform=weibo` 即可,无需改技能结构。
- **hc-075 互动红线**:技能评论区**只做人工回复策略 + 置顶评论钩子,不做任何自动回复/私信自动化**——与 hc-075 一致,PM 无需额外动作。

## 阻塞项

- 无硬阻塞(技能可独立合并、CI 应绿——纯 md 改动;Docker build 慢检查非门)。
- 软依赖:云端真调通 = 等两插件随下批镜像滚存量 agent + 网关 douyin 数据/ASR 端点就绪(插件代码已在 main,见 PM 清单);weibo 跨平台选题 = 等 weibo 进 fork 白名单(技能已解耦,不阻塞);免费封面线 = hc-433 并行落地(技能已解耦);live 端到端模型验收 = 等有 key 的环境/真机(见"诚实说明")。
