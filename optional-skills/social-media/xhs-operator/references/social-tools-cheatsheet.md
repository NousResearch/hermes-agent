# 社媒数据工具速查(xiaohongshu)

小红书真数据来自平台社媒数据工具(ApexNodes 平台工具网关 `/tools/v1/social/*`,插件侧 `apexnodes-social-tools`)。这些工具在**桌面(fork 本地跑,配好网关 base + Agent API key)** 和**云端 agent**里都以下列 tool name 暴露给模型。vendor key 不出云,插件只持网关 base + key。

## 小红书可用的 action(全部 `platform="xiaohongshu"`)

| tool name | 语义 | 必给参数 | 常用可选 |
| --- | --- | --- | --- |
| `social_search` | 按关键词搜小红书公开笔记 | `platform`, `query` | `count`(≤100)、`cursor` |
| `social_profile` | 取某账号主页/账号记录 | `platform`, `user_id` | — |
| `social_posts` | 列某账号近期笔记 | `platform`, `user_id` | `count`、`cursor` |
| `social_content` | 取单篇笔记详情/正文 | `platform` + (`url` 或 `item_id`) | — |
| `social_comments` | 取单篇笔记公开评论 | `platform`, `item_id`(或 `url`) | `cursor` |
| `creator_top_posts` | 枚举某创作者作品并按互动量筛选打分 | `url` 或 `platform`+`user_id` | `min_likes`、`min_collects`、`min_comments`、`sort_by`、`top`、`scan_limit`、`since`、`until` |

> 小红书是 `creator_top_posts` 的支持平台之一(与 douyin/tiktok/kuaishou/bilibili 等并列):给对标号主页链接或 `platform+user_id` + 点赞阈值,枚举其作品并按点赞/收藏/评论排序打分。它返回的是**扫描窗口内命中阈值的样本**(附"扫了 N 条、M 条命中"的诚实注释),不是平台全站热门榜。**小红书公开热门榜数据源当前没有(hc-426 上线后接入);热点选题暂用 `social_search` 以时效性关键词搜近期高互动笔记近似。**

## 调用姿势(工具入参就是这些键)

```json
// 1) 搜赛道笔记
{ "name": "social_search",
  "arguments": { "platform": "xiaohongshu", "query": "油皮 护肤 平价", "count": 20 } }

// 2) 看对标号主页
{ "name": "social_profile",
  "arguments": { "platform": "xiaohongshu", "user_id": "<对标号 id>" } }

// 3) 拉对标号近期笔记
{ "name": "social_posts",
  "arguments": { "platform": "xiaohongshu", "user_id": "<对标号 id>", "count": 20 } }

// 4) 按互动量枚举对标号爆款笔记
{ "name": "creator_top_posts",
  "arguments": { "platform": "xiaohongshu", "user_id": "<对标号 id>", "min_likes": 10000, "sort_by": "likes", "top": 20 } }

// 5) 取单篇笔记正文(含封面/标签信息)
{ "name": "social_content",
  "arguments": { "platform": "xiaohongshu", "url": "https://www.xiaohongshu.com/explore/xxxx" } }

// 6) 取评论区真实读者语言
{ "name": "social_comments",
  "arguments": { "platform": "xiaohongshu", "item_id": "<note_id>" } }
```

`platform` 白名单里小红书是 `xiaohongshu`(链接域名 `xiaohongshu.com` / `xhslink.com`)。`user_id` 接受平台各自的 id 拼写;`item_id` 是笔记内容 id(note_id);没有 id 时优先传 `url`。

## 错误处理(把 message 如实透给用户,别脑补)

网关错误体形如 `{"detail": {"code": "<机器码>", "message": "<中文信息>"}}`。分支处理:

| 情况 | 机器码 | 怎么办 |
| --- | --- | --- |
| key 缺失/无效/过期 | `api_key_required` / `invalid_api_key` / `api_key_expired` | 桌面:引导用户重新登录(key 会随登录重发);别重试 |
| 配额用尽 | `quota_exceeded` | 提示升级套餐,**不要重试** |
| 平台不在白名单 | `platform_not_allowed` | 确认用了 `xiaohongshu` |
| 平台侧能力暂不可用 | `vendor_unavailable`(附 `detail.vendor_code`) | **显式告知"小红书数据能力暂不可用",不要静默、不要本地自装替代抓取器**;可改为让用户手动贴笔记链接/截图 |
| 限流 | `rate_limited`(读 `Retry-After` 头) | 退避后重试 |
| 该平台无此能力映射 | `capability_not_mapped` | 诚实说"该平台暂未配置对应数据能力" |
| 输入类可修错误 | 如 4xx 透传的 message | 原样透给用户,让其修正输入(如换链接) |

`detail.terminal_fallback_allowed: false` 出现时(解析路径的诚实信号):把它拼进错误文案,**禁止模型自装替代抓取器**。

## 采不到的数据(诚实边界 — 反复强调)

- **小红书创作者后台面板数据**(精确曝光、点击率、完播、涨粉、收藏率、单条数据、薯条/聚光投放效果、蒲公英报价)**外部工具采不到**。工具给的是公开可见的标题/正文/封面信息/评论 + 平台暴露的公开互动信号(点赞/收藏/评论数)。
- 需要这些精确数据时,明说"这项外部采不到,请从小红书创作者后台/投放后台截图给我",读图后再录入。
- **热门榜/热点榜**当前无公开数据源(hc-426 上线后接入);热点判断暂用 `social_search` 以时效性关键词搜近期高互动笔记近似,并注明。
- 绝不编造对标账号的点赞收藏数,也绝不编造自号后台数据。

## 封面生成(带字封面)

带字封面走运行时的**图片生成能力(系统按档位自动选路由)**。技能**不写死 provider/模型**:带字封面默认走系统默认档位(适合带标题字的封面);另有免费封面线(平台底图 + 排版叠字,hc-433)在系统档位内可选。用一句话 prompt:`主标题文字 + 画面主体 + 风格 + 排版`,**一次出 3 版不同封面方向让用户选 1**(如:大字纯色底 / 人物实拍叠字 / 前后对比)。

**若当前运行时未装图片生成能力(工具不存在或返回不可用):跳过封面生成,说明"当前环境无生图能力,封面请手动制作(给排版建议即可)",不要编造或谎称已生成。**
