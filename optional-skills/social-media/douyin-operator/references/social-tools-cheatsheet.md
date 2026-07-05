# 社媒数据 + 媒体工具速查(douyin)

抖音真数据来自平台社媒数据工具(ApexNodes 平台工具网关 `/tools/v1/social/*` 与 `/tools/v1/asr/*`,插件侧 `apexnodes-social-tools` + `apexnodes-douyin-tools`)。这些工具在**桌面(fork 本地跑,配好网关 base + Agent API key)** 和**云端 agent**里都以下列 tool name 暴露给模型。vendor key 不出云,插件只持网关 base + key。

## A. 数据类:抖音可用的 action(全部 `platform="douyin"`)

来自 `plugins/apexnodes-social-tools/__init__.py` 的 `SCHEMAS` + `CREATOR_TOP_POSTS_SCHEMA`:

| tool name | 语义 | 必给参数 | 常用可选 |
| --- | --- | --- | --- |
| `social_search` | 按关键词搜抖音公开视频 | `platform`, `query` | `count`(≤100)、`cursor` |
| `social_trending` | 取抖音**公开热门视频榜(billboard)** | `platform` | — |
| `social_profile` | 取某账号主页/账号记录 | `platform`, `user_id` | — |
| `social_posts` | 列某账号近期视频 | `platform`, `user_id` | `count`、`cursor` |
| `social_content` | 取单条视频详情/文案 | `platform` + (`url` 或 `item_id`) | — |
| `social_comments` | 取单条视频公开评论 | `platform`, `item_id`(或 `url`) | `cursor` |
| `creator_top_posts` | 枚举某创作者作品并按互动量筛选打分 | `url` 或 `platform`+`user_id`(required=[]) | `min_likes`、`min_collects`、`min_comments`、`sort_by`、`top`、`scan_limit`、`since`、`until` |

> **`social_trending` 抖音可用**(`social_trending` 的 required 只有 `platform`,`douyin` 在 10 平台白名单里)——这是抖音的公开热门榜,公众号/小红书样板拿不到。
> **`creator_top_posts` 抖音是首选平台**(其 platform 描述列 `douyin/xiaohongshu/tiktok/kuaishou/bilibili/youtube/instagram`):给对标号主页分享链接(`url`)或 `platform+user_id` + 点赞阈值(`min_likes`),枚举其作品并按点赞/收藏/评论排序打分。也可传**单个抖音合集分享链接**(`douyin.com/collection/…` 或 `mix/…`)只枚举该合集。它返回的是**扫描窗口内命中阈值的样本**(附"扫了 N 条、M 条命中"的诚实注释),不是平台全站热门榜。
> 注:`social_captions`(YouTube 官方字幕、无 ASR 成本)是 **YouTube 专用**,抖音不用它;抖音口播逐字稿走下面 B 组的 `media_transcribe`(下载 + ASR 转写)。

### 数据类调用姿势

```json
// 0) 抖音公开热门榜起手
{ "name": "social_trending",
  "arguments": { "platform": "douyin" } }

// 1) 搜赛道视频
{ "name": "social_search",
  "arguments": { "platform": "douyin", "query": "副业 避坑 新手", "count": 20 } }

// 2) 看对标号主页
{ "name": "social_profile",
  "arguments": { "platform": "douyin", "user_id": "<对标号 sec_user_id>" } }

// 3) 拉对标号近期视频
{ "name": "social_posts",
  "arguments": { "platform": "douyin", "user_id": "<对标号 sec_user_id>", "count": 20 } }

// 4) 按互动量枚举对标号爆款
{ "name": "creator_top_posts",
  "arguments": { "platform": "douyin", "user_id": "<对标号 sec_user_id>", "min_likes": 100000, "sort_by": "likes", "top": 20 } }

// 5) 取单条视频详情(含封面/文案信息)
{ "name": "social_content",
  "arguments": { "platform": "douyin", "url": "https://v.douyin.com/xxxx/" } }

// 6) 取评论区真实观众语言
{ "name": "social_comments",
  "arguments": { "platform": "douyin", "item_id": "<aweme_id>" } }
```

`platform` 白名单里抖音是 `douyin`(链接域名 `v.douyin.com` / `douyin.com` / `iesdouyin.com`)。`user_id` 接受平台各自的 id 拼写(抖音 sec_user_id);`item_id` 是内容 id(抖音 aweme_id);没有 id 时优先传 `url`。

## B. 媒体类:抖音线独有优势——下载 + 转写 + 批量(公众号/小红书没有)

来自 `plugins/apexnodes-douyin-tools/__init__.py`(hc-254:tool name 平台中立,包名保留 `apexnodes-douyin-tools` 只为插件注册稳定)。抖音口播/画面里的信息用这组工具拿:

| tool name | 语义 | 必给参数 | 说明 |
| --- | --- | --- | --- |
| `social_download` | 下载社媒视频到工作区媒体空间 | `url` | 抖音/TikTok/小红书/快手/B站分享链接**第一步优先用它**,别用浏览器打开(反爬又慢又不可靠)。返回 `video_path` + 元数据。 |
| `media_transcribe` | 转写视频(下载好的 `video_path` 或直接传分享链接 `url` 一步下载并转写) | `video_path` **或** `url`(至少一个) | 用户要**视频文案/逐字稿/拆解脚本**时用它。返回完整转写 + `transcript_path`。**禁止用浏览器或自装本地 whisper 替代。** |
| `image_ocr` | 识别社媒图文/图片里的文字 | `url` **或** `image_urls`(至少一个) | 抖音图文/小红书图文/快手图片的**文字**提取;可选 `prompt`。只读字,不理解画面。 |
| `social_batch_submit` | 批量下载+转写一批视频(异步) | `urls` **或** `creator_url`(至少一个) | `creator_url`=作者主页分享链接(枚举全部作品)或某抖音合集分享链接(只枚举该合集);配 `min_likes`/`top` 自动筛选。返回 `job_id`。**只提交,不自己起脚本。** |
| `social_batch_status` | 查批量任务进度与产物 | `job_id` | completed/partial 时产物:file 类(Word/Excel)给 `download_url`,飞书多维表格给 `bitable_url`——发给用户即可。running 时稍后再轮询,别重复提交。 |

### 媒体类调用姿势

```json
// 一步下载 + 转写(拆爆款口播最省事;直接传用户发来的原始分享文本)
{ "name": "media_transcribe",
  "arguments": { "url": "7.99 复制打开抖音，看看【xxx】的作品 https://v.douyin.com/xxxx/" } }

// 分两步:先下载留文件,再转写
{ "name": "social_download",
  "arguments": { "url": "https://v.douyin.com/xxxx/" } }         // → video_path
{ "name": "media_transcribe",
  "arguments": { "video_path": "<上一步返回的 video_path>" } }

// 抖音图文提取文字
{ "name": "image_ocr",
  "arguments": { "url": "https://www.douyin.com/note/xxxx", "prompt": "提取图片里的文案" } }

// 批量转写一个作者点赞≥10万的作品(异步)
{ "name": "social_batch_submit",
  "arguments": { "creator_url": "<对标号主页分享链接>", "min_likes": 100000, "top": 20 } }   // → job_id
{ "name": "social_batch_status",
  "arguments": { "job_id": "<上一步返回的 job_id>" } }
```

## 错误处理(把 message 如实透给用户,别脑补)

网关错误体形如 `{"detail": {"code": "<机器码>", "message": "<中文信息>"}}`。分支处理:

| 情况 | 机器码 | 怎么办 |
| --- | --- | --- |
| key 缺失/无效/过期 | `api_key_required` / `invalid_api_key` / `api_key_expired` | 桌面:引导用户重新登录(key 会随登录重发);别重试 |
| 配额用尽 | `quota_exceeded` | 提示升级套餐,**不要重试** |
| 平台不在白名单 | `platform_not_allowed` | 确认用了 `douyin`(**别用 `weibo`——当前 fork 白名单未含,会命中此错**) |
| 平台侧能力暂不可用 | `vendor_unavailable`(附 `detail.vendor_code`) | **显式告知"抖音数据/转写能力暂不可用",不要静默、不要本地自装替代抓取器**;可改为让用户手动贴链接/发视频文件/贴文字稿 |
| 限流 | `rate_limited`(读 `Retry-After` 头) | 退避后重试 |
| 该平台无此能力映射 | `capability_not_mapped` | 诚实说"该平台暂未配置对应数据能力" |
| 输入类可修错误 | 如 4xx 透传的 message | 原样透给用户,让其修正输入(如换链接) |

`detail.terminal_fallback_allowed: false` 出现时(转写/解析路径的诚实信号,插件会把它拼进错误文案):**禁止模型自装 whisper 等本地替代器**,按失败选项处理(稍后重试 / 发视频文件 / 贴文字稿)。

## 采不到的数据(诚实边界 — 反复强调)

- **抖音创作者中心/巨量后台面板数据**(精确播放量、完播率、5s 完播、平均播放时长、涨粉、转粉率、DOU+/千川投放效果、星图报价)**外部工具采不到**。工具给的是公开可见的标题/文案/封面信息/口播逐字稿/评论 + 平台暴露的公开互动信号(点赞/评论/收藏/转发数)。
- 需要这些精确数据时,明说"这项外部采不到,请从抖音创作者中心/巨量后台截图给我",读图后再录入。
- **跨平台选题(weibo 微博热搜)**:云端 hc-426 已上 weibo 平台,但**当前 fork 的网关白名单(`plugins/apexnodes_gateway.py::SOCIAL_PLATFORMS`,10 平台)尚未含 weibo**;待 weibo 随下批同步进 fork 白名单后,可 `social_trending platform=weibo` 做跨平台选题。当前抖音站内热点用 `douyin` 的 `social_trending` + `social_search` 覆盖,**勿调 weibo(会 `platform_not_allowed`)**。
- 绝不编造对标账号的点赞收藏数,也绝不编造自号后台数据;实验矩阵样本小,信号 ≠ 定论。

## 封面 / 首帧生成(带字封面)

带字封面/首帧走运行时的**图片生成能力(系统按档位自动选路由)**。技能**不写死 provider/模型**:带字封面默认走系统默认档位(适合带标题字的封面);另有免费封面线(平台底图 + 排版叠字,hc-433)在系统档位内可选。用一句话 prompt:`主标题文字 + 画面主体 + 风格 + 排版`,**一次出 3 版不同封面方向让用户选 1**(如:大字纯色底 / 人物出镜叠字 / 悬念遮挡)。

**若当前运行时未装图片生成能力(工具不存在或返回不可用):跳过封面生成,说明"当前环境无生图能力,封面请手动制作(给排版建议即可)",不要编造或谎称已生成。**
