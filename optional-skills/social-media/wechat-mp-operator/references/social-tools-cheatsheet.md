# 社媒数据工具速查(wechat_mp)

公众号真数据来自平台社媒数据工具(ApexNodes 平台工具网关 `/tools/v1/social/*`,插件侧 `apexnodes-social-tools`)。这些工具在**桌面(fork 本地跑,配好网关 base + Agent API key)** 和**云端 agent**里都以下列 tool name 暴露给模型。vendor key 不出云,插件只持网关 base + key。

## 公众号可用的 5 个 action(全部 `platform="wechat_mp"`)

| tool name | 语义 | 必给参数 | 常用可选 |
| --- | --- | --- | --- |
| `social_search` | 按关键词搜公众号公开文章 | `platform`, `query` | `count`(≤100)、`cursor` |
| `social_profile` | 取某公众号主页/账号记录 | `platform`, `user_id` | — |
| `social_posts` | 列某公众号近期作品 | `platform`, `user_id` | `count`、`cursor` |
| `social_content` | 取单篇文章详情/正文 | `platform` + (`url` 或 `item_id`) | — |
| `social_comments` | 取单篇文章公开评论 | `platform`, `item_id`(或 `url`) | `cursor` |

> `creator_top_posts`(按互动量枚举筛选)当前主要面向 douyin/小红书等有稳定枚举端点的平台,公众号起号用上面 5 件即可覆盖。热点用 `social_search` 以时效性关键词搜近期文章。

## 调用姿势(工具入参就是这些键)

```json
// 1) 搜赛道文章
{ "name": "social_search",
  "arguments": { "platform": "wechat_mp", "query": "副业 避坑", "count": 20 } }

// 2) 看对标号主页
{ "name": "social_profile",
  "arguments": { "platform": "wechat_mp", "user_id": "<公众号 biz/id>" } }

// 3) 拉对标号近期作品
{ "name": "social_posts",
  "arguments": { "platform": "wechat_mp", "user_id": "<公众号 biz/id>", "count": 20 } }

// 4) 取单篇正文
{ "name": "social_content",
  "arguments": { "platform": "wechat_mp", "url": "https://mp.weixin.qq.com/s/xxxx" } }

// 5) 取评论区真实读者语言
{ "name": "social_comments",
  "arguments": { "platform": "wechat_mp", "item_id": "<文章 id>" } }
```

`platform` 白名单里公众号是 `wechat_mp`(视频号是 `wechat_channels`,不是这个)。`user_id` 接受平台各自的 id 拼写(biz / id / username);`item_id` 是文章内容 id;没有 id 时优先传 `url`。

## 错误处理(把 message 如实透给用户,别脑补)

网关错误体形如 `{"detail": {"code": "<机器码>", "message": "<中文信息>"}}`。分支处理:

| 情况 | 机器码 | 怎么办 |
| --- | --- | --- |
| key 缺失/无效/过期 | `api_key_required` / `invalid_api_key` / `api_key_expired` | 桌面:引导用户重新登录(key 会随登录重发);别重试 |
| 配额用尽 | `quota_exceeded` | 提示升级套餐,**不要重试** |
| 平台不在白名单 | `platform_not_allowed` | 确认用了 `wechat_mp` |
| 平台侧能力暂不可用 | `vendor_unavailable`(附 `detail.vendor_code`) | **显式告知"公众号数据能力暂不可用",不要静默、不要本地自装替代抓取器**;可改为让用户手动贴文章链接/截图 |
| 限流 | `rate_limited`(读 `Retry-After` 头) | 退避后重试 |
| 该平台无此能力映射 | `capability_not_mapped` | 诚实说"该平台暂未配置对应数据能力" |
| 输入类可修错误 | 如 4xx 透传的 message | 原样透给用户,让其修正输入(如换链接) |

`detail.terminal_fallback_allowed: false` 出现时(转写/解析路径的诚实信号):把它拼进错误文案,**禁止模型自装 whisper 等本地替代器**。

## 采不到的数据(诚实边界 — 反复强调)

- **公众号后台面板数据**(精确阅读数、在看、涨粉、留存、单篇打开率、流量主收益)**外部工具采不到**。工具给的是公开可见的标题/正文/评论 + 平台暴露的公开信号。
- 需要这些精确数据时,明说"这项外部采不到,请从公众号后台/流量主后台截图给我",读图后再录入。
- 绝不编造对标账号的阅读数,也绝不编造自号后台数据。

## 头图 / 封面图生成

带标题字的封面默认走运行时的图片生成能力(caption-cover / flash 类模型,适合带字头图,属云端能力)。用一句话 prompt:`主标题文字 + 画面主体 + 风格`。

**若当前运行时未装图片生成能力(工具不存在或返回不可用):跳过头图,说明"当前环境无生图能力,头图请手动制作",不要编造或谎称已生成。**
