# QQ（OneBot）

Hermes 通过 **OneBot 11 协议**接入 QQ，兼容 [NapCat](https://napneko.github.io/)、[Lagrange](https://github.com/LagrangeDev/Lagrange.Core)、LLOneBot 和 go-cqhttp。与官方 QQ Bot 平台（需要腾讯审核的应用）不同，OneBot 通过本地桥接驱动一个普通 QQ 号，适合个人机器人和不被官方平台覆盖的群。

```
用户 (QQ) ←→ NapCat ←→ Hermes onebot 适配器 ←→ Hermes agent
                      ├─ 反向 WS 服务端 / 正向 WS 客户端（自动重连）
                      ├─ 入站：CQ 解析、图片下载、语音转写、引用/转发展开
                      └─ 出站：分段发送、markdown 剥离、[[qq_forward]]、图/音/视频/文件
```

> 运行 `hermes gateway setup` 并选择 **QQ (OneBot)** 有引导式配置。

## 功能总览

| 领域 | 能力 |
|---|---|
| 连接 | 反向 WS（NapCat ws-reverse 拨入，默认端口 8643）或正向 WS（主动拨出，默认 `ws://127.0.0.1:3001`）；自动重连 |
| 入站 | 私聊/群聊，段数组优先解析（CQ 字符串回退），CQ 反转义，@/回复触发检测（fail-closed），图片解析（url/base64/file/hash 经 `get_image`），大图压缩，file/voice/video/face/json/poke 段类型，引用取原文（`get_msg`），入站文件走 CDN 直链再 `get_file` 回退（绕过容器路径） |
| 语音 | ffmpeg → 16 kHz 单声道 WAV → Hermes STT 管线；无 URL 的语音先经 `get_record`（base64）取；失败降级为 `[语音]` |
| 文字图 | AstrBot 风格 t2i 卡片渲染器（粗体/斜体/删除线/引用/列表/代码块/表格/行内代码胶囊/emoji/中文标点禁则）；800px 宽 |
| 出站 | 长回复按句子边界分段（默认 ≤100 字），**>150 字渲染文字图卡片**，markdown 剥离为纯文本，`[[qq_forward]]` 合并转发，loop 中间消息合并+撤回，正在输入提示（私聊） |
| 权限 | 管理员白名单（`ONEBOT_ALLOWED_USERS`），dm/group 策略（open/allowlist/disabled），群 @ 门控，受限成员软限制 |
| 运维 | 热加载 `onebot_utils.py` / `t2i_render.py`（免网关重启），临时媒体 TTL 清理 |
| Agent 工具 | 模型侧发送器（`qq_send_image` / `qq_send_voice` / `qq_send_video` / `qq_send_file` / `qq_send_forward`），`qq_napcat_api`（15 个 action 白名单），`qq_group_history`；HTTP `/api/napcat` + `/api/send_media` 端点 |

## 兼容性

| 项目 | 要求 |
|---|---|
| Hermes | 网关启用 onebot 平台 |
| OneBot 11 桥 | NapCat / Lagrange / LLOneBot / go-cqhttp（反向或正向 WebSocket） |
| 可选依赖 | Hermes 主机上装 `ffmpeg` 用于语音转写；`stt:` 下配置 STT 后端（本地 faster-whisper 首次使用自动下载，或 OpenAI 兼容 API）；文字图卡片需要 CJK 字体（见下） |

## 配置 Hermes

在 `~/.hermes/gateway-config.yaml` 的 `gateway` 块中添加平台：

```yaml
gateway:
  platforms:
    onebot:
      enabled: true
      extra:
        mode: reverse              # reverse | forward
        host: "0.0.0.0"            # reverse: 监听地址
        port: 8643                 # reverse: 监听端口
        # url: "ws://127.0.0.1:3001"   # forward: 桥接 ws 端点
        # access_token: ""         # 必须与桥的 token 一致（如有）
        # bot_qq: ""               # 可选；自动从 meta 事件学习
        require_mention: true      # 群聊：仅被 @ 时回复
        dm_policy: open            # open | allowlist | disabled
        allow_from: []             # dm_policy=allowlist 时的用户 id
        group_policy: open         # open | allowlist | disabled
        group_allow_from: []       # group_policy=allowlist 时的群 id
        admin_users: []            # 管理员；回退到 ONEBOT_ALLOWED_USERS
        split_length: 100          # 长回复分段字符数
        text_image_threshold: 150  # 更长回复渲染文字图
        image_max_size: 2048       # 入站图片压缩（0 = 保持原样）
```

### 扩展键（全部可选）

| 键 | 默认值 | 含义 |
|---|---|---|
| `mode` | `reverse` | `reverse`（桥拨入）/ `forward`（适配器拨出） |
| `host` / `port` | `0.0.0.0` / `8643` | 反向监听 |
| `url` | `ws://127.0.0.1:3001` | 正向目标 |
| `access_token` | 空 | OneBot token，必须与桥一致 |
| `bot_qq` | 空 | 机器人 QQ（空 = 从 meta 事件学习） |
| `require_mention` | `true` | 群聊：仅被 @ 或回复时响应 |
| `dm_policy` | `open` | `open`（仅管理员）/ `allowlist` / `disabled` |
| `allow_from` | `[]` | `dm_policy=allowlist` 时的允许用户 id |
| `group_policy` | `open` | `open` / `allowlist` / `disabled` |
| `group_allow_from` | `[]` | `group_policy=allowlist` 时的允许群 id |
| `admin_users` | `[]` | 管理员 QQ id；回退到 `ONEBOT_ALLOWED_USERS` |
| `split_length` | `100` | 长回复分段阈值 |
| `text_image_threshold` | `150` | t2i 卡片阈值；`<=0` 关闭卡片路径 |
| `image_max_size` | `2048` | 入站图片长边上限（px）；`0` 保持原样 |
| `max_inbound_file_bytes` | `20971520`（20 MB） | 入站文件大小上限；超限降级为 `[文件:name]` 标记 |
| `interim_recall_seconds` | `90` | 未结算 interim 消息的自动撤回超时（`0` 关闭） |
| `hot_reload` | `false` | 仅开发用：`onebot_utils.py` / `t2i_render.py` 按 mtime 变化自动 reload（调样式免重启；生产建议关闭，升级时原地写入可能加载半截模块） |

环境变量：`ONEBOT_ALLOWED_USERS`（逗号分隔的管理员 id）、`ONEBOT_ALLOW_ALL_USERS=true`（仅开发），以及全局的 `GATEWAY_ALLOW_ALL_USERS`。

> **`dm_policy: open` + 全放行环境变量**：`ONEBOT_ALLOW_ALL_USERS=true`（或
> `GATEWAY_ALLOW_ALL_USERS=true`）是让 `open` 真正向非管理员开放的显式开关。
> 没有它 `open` 意味着**仅管理员**：适配器在入口就拒绝非管理员私聊，
> 网关的全放行检查根本轮不到。

### 连接模式

| 模式 | 说明 |
|---|---|
| `reverse`（默认） | Hermes 起 WebSocket 服务端；桥的 **ws-reverse** 客户端拨入（`ws://<hermes-host>:8643/ws`）。一条连接同时承载事件和动作。 |
| `forward` | Hermes 主动拨桥的 WebSocket 服务端（NapCat 默认 `ws://<bridge-host>:3001`）。 |

桥若使用 access token，`access_token` 要设同值（reverse 连接上 Hermes 以 `Authorization: Bearer <token>` 发送；forward 模式在握手头里带）。

### NapCat 侧配置（必做）

启用插件后**还必须配置桥**。插件自己连不上：

- **reverse 模式（推荐，NapCat 拨入）**：在 NapCat 网络设置里添加 **WebSocket 客户端**：
  - 上报地址：`ws://<hermes-host-ip>:<port>/ws`（如 `ws://192.168.1.100:8643/ws`；Hermes 和 NapCat 不在同一台机器时用局域网 IP，别用 `127.0.0.1`）
  - token：与 Hermes 侧 `access_token` 一致（都没有就不填）
  - 消息上报格式：推荐 **array**（段数组优先解析，CQ 字符串仅作回退）
- **forward 模式（Hermes 拨出）**：在 NapCat 网络设置里启用 **WebSocket 服务端**（默认 `0.0.0.0:3001`），然后把插件的 `url` 设成 `ws://<napcat-host-ip>:3001`（同机用 `ws://127.0.0.1:3001`），token 保持一致。

桥必须在 Hermes 可达的网络（同一局域网 / 可路由）；WS 连接、图片下载、文件解析都依赖这条通路。本适配器**仅限局域网**，不支持跨公网部署。

> ⚠️ NapCat 与 Hermes **不在同一台机器**时，打开 NapCat 网络设置里的 **「文件转 URL」/ file-to-URL** 开关（`enableLocalFile2Url`）。否则 `get_file` 返回的是容器本地路径，Hermes 够不着，文件消息会退化为 `[文件:name]` 标记而无法下载。

## dm / 群访问策略（配置时选定）

两个策略**首次配置时必须各选一个**。默认值只有在你为每个选了三选一之后才有意义：

| 值 | dm_policy（私聊） | group_policy（群聊） |
|---|---|---|
| `open` | **仅管理员**（非管理员静默拒绝，无配对流程） | 所有群可聊；回复由 `require_mention` 门控 |
| `allowlist` | 仅 `allow_from` 里的 id 可私聊（无需管理员） | 仅 `group_allow_from` 里的群可聊 |
| `disabled` | 拒绝所有私聊 | 忽略所有群消息 |

> ⚠️ **配置时必须至少设一个管理员**（`admin_users` 或 `ONEBOT_ALLOWED_USERS` 环境变量）。`dm_policy: open`（默认）下只有管理员能私聊，斜杠命令也仅限管理员。没配管理员的机器人无人能对话。开发快速测试用 `ONEBOT_ALLOW_ALL_USERS=true`。

怎么选：

- **个人机器人** → `dm_policy: open` + `admin_users: [<你的QQ>]`。只有你能私聊
- **几个朋友** → `dm_policy: allowlist` + `allow_from: [<QQ1>, <QQ2>]`。名单里的非管理员也能私聊
- **群机器人** → `group_policy: open`（默认 `require_mention: true` 保持安静：成员必须 @ 或回复才触发）
- **只服务特定群** → `group_policy: allowlist` + `group_allow_from: [<group_id>]`

### 权限分层（admin / 受限成员）

允许群可以开放给全体成员，特权操作仍限管理员。适配器自管访问策略（`enforces_own_access_policy`），网关信任它的名单判定。

| 角色 | 谁 | 群 @ | 私聊 | 能力 |
|---|---|---|---|---|
| admin | `extra.admin_users`（回退 `ONEBOT_ALLOWED_USERS`） | 全 | 允许 | 全部，含斜杠命令 |
| member | 允许群内其他成员 | 受限 | 拒绝 | 仅限快速问答、看图、群总结 |
| unauthorized | 允许群 / 私聊名单之外 | 拦截 | 拒绝（配对关闭） | — |

执行点：

- member 群消息加 `[受限用户:仅问答]` 文本前缀，让 agent 套用软限制（只做快速问答/看图/群总结；不做文件/终端/配置/HA/跨平台/cron；已在平台提示里声明）
- member 斜杠命令（`/new`、`/model`、`/help` 等）在构造 `MessageEvent` 前被丢弃；路径样文本（`/tmp/x`）不受影响
- member 私聊被拒
- 发给 member 的出站回复按敏感意图关键词扫描并打 WARNING 日志（审计，非硬拦截）

### 管理员本地斜杠命令

管理员有几个在适配器内直接处理的斜杠命令（其余照常流转到网关核心）：

| 命令 | 作用 |
|---|---|
| `/ocr` | 对最近收到的一张入站图片调 NapCat `ocr_image` 做 OCR |
| `/mode interim\|instant` | 按聊天设置 loop 合并模式（`interim` = 中间评论合并转发，`instant` = 原样直发）；仅内存态，重启恢复默认 |
| `/id` | 打印当前 chat id |
| `/ver` | 打印插件版本 |

`/ocr` 要求图片还在临时媒体目录里（6 小时 TTL 清理会删）。

## 群 @ 触发

`require_mention: true`（默认）时机器人在群里只响应显式 @ 或回复已有消息。设为 `false` 则响应每条群消息（很吵，大群不推荐）。未配置 `bot_qq` 时机器人从 OneBot meta 事件学自己的 id，@ 检测开箱即用。

## 长回复（三档，全可配置）

回复长度分三档处理。两个阈值都在平台 `extra` 块里可配（`split_length` 和 `text_image_threshold`）：

- **≤ `split_length`**（默认 100）字：单条文本直接发。
- **`split_length` 到 `text_image_threshold`**（默认 150）：按句子边界分段为多条（在 `。！？!?；;\n` 处断），句子从不被拦腰截断。
- **> `text_image_threshold`**：渲染成黑字白底文字图（800px 宽、CJK 字体回退链）单张图片发送。渲染失败回退分段文本。

`text_image_threshold: 0` 关闭图片路径（全部走分段文本）；调高调低两个值可权衡消息条数与卡片渲染。

文字图渲染器是 AstrBot 风格**元素化 Markdown 渲染器**：粗体 / 斜体 / 删除线 / 行内代码 / 代码块 / 标题 / 引用 / 列表和**表格**（AstrBot 自己没有表格元素）全部原生绘制。遵循中文排版规则：标点不落行首（行首禁则）、行内样式整体换行、纯文本里的字面 `\n` 变成真换行（行内代码里变空格，`\\n` 保留）、行内代码用浅蓝胶囊配等宽字体（拉丁/数字）+ 字形级回退（CJK）。知道回复对象昵称时卡片带 AstrBot 风格蓝色顶栏（`To <昵称>`，克莱因蓝 #002FA7，白字为**正文两倍字号**，约 68px 高），与 AstrBot 卡片头部比例一致。

### 字体依赖（一键安装）

卡片渲染器需要三个字体族（CJK / 等宽 / 彩色 emoji），自动注册自系统；字形缺失会渲染成豆腐块。

| 依赖 | 提供（自动注册路径） | 用途 |
|---|---|---|
| `fonts-noto-cjk` | `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc` | 中文正文/标题（自动选中 ttc SC face，JP/Mono 回退） |
| `fonts-dejavu-core` | `/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf` | 代码块 / 行内代码等宽 |
| `fonts-noto-color-emoji` | `/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf` | 彩色 emoji |
| 可选 `fonts-wqy-zenhei` / `fonts-wqy-microhei` | `/usr/share/fonts/truetype/wqy/*.ttc` | Noto 缺失时的 CJK 回退 |
| 可选 `fonts-unifont` | `/usr/share/fonts/opentype/unifont/*.otf` | 最后兜底 |

Linux（Debian/Ubuntu）一条命令覆盖全部必需字体：

```sh
sudo apt install fonts-noto-cjk fonts-dejavu-core fonts-noto-color-emoji
```

macOS 什么都不用装（系统自带的 Hiragino Sans GB / Songti SC、Menlo 和 Apple Color Emoji 会自动被用上）。渲染器启动时做墨源自检：没有可用字形的字体族会被剔除并静默回退，不会打印豆腐块卡片。

## Markdown 与语音

- **发送前剥离 Markdown**。QQ 不渲染它，所以 `**粗体**`、标题、列表、表格都转成可读纯文本（标题 → `【…】`、列表 → `•`、表格 → 空格分隔单元格、围栏代码块 → 边框框体）。这发生在分段和文字图渲染之前，图片也是干净的。
- **入站语音自动转写**：适配器下载语音片段，`ffmpeg` 转 16 kHz 单声道 WAV 后交给 Hermes STT 管线。NapCat 私聊语音常常只有 file hash（无 URL）：适配器先调 OneBot `get_record` 取 base64 音频再走下载→ffmpeg→STT 管线。

  STT 用全局 `stt:` 配置（与其他平台同一管线）：`provider: local` 在 Hermes 主机跑 faster-whisper（模型 = `stt.local.model`，默认 `small`，首次自动下载）；`provider: openai` 调 OpenAI 兼容端点（配 `stt.openai.*` + API key）。要求：装了 `ffmpeg`；没有它（或 STT 后端不可用）语音降级为 `[语音]` 标记。

## 图片

- 入站消息优先按 **OneBot 段数组**（`message` 字段）解析（可用时）。图片/语音/@/表情/视频/文件段结构化处理；文本格式客户端回退 CQ 码字符串解析。CQ 实体反转义（`&amp;` → `&`、`&#91;` → `[` 等）在任何 URL 使用前完成，带 `&` 参数的 CDN 链接能正确下载。
- 图片下载到临时目录，经 `media_urls` 暴露给视觉工具；下载失败的图片降级为 `[图片]` 占位。图片段只有 `file` hash（无 URL）时，适配器调 OneBot `get_image` 解析真实 URL；`base64://` 和 `file://` 形式直接处理。
- 超过 `image_max_size`（默认长边 **2048** px）的图片在给 LLM 之前用 Pillow 压缩。QQ 高清照片不压缩会让视觉调用变慢或超时。RGBA 保持 PNG，其余转 JPEG（q85）；动图 GIF 折叠为第一帧。`image_max_size: 0` 保持原样。

## 入站文件

入站 `file` 段走**双通道**解析，容器本地路径不会漏给 agent：

1. **CDN 直链优先**：私聊时适配器调 `get_private_file_url`，直接从 QQ CDN 下载。
2. **`get_file` 回退**：无直链时（群文件，或 NapCat 没开 file-to-URL 开关），回退 `get_file`，接受 base64 或 http URL。

下载的文件落在临时媒体目录，消息文本里标注 `[文件:本地路径]` 让 agent 能本地读取，体积受 `max_inbound_file_bytes`（默认 20 MB）限制。超限退化为纯 `[文件:name]` 标记。必须保持路径中立的用户级插件会跳过该标注。

## 出站媒体

适配器把网关原生媒体发送器实现为 OneBot 段，agent 可以通过标准 `MEDIA:` / markdown 图片机制发富媒体：

| 能力 | OneBot 段 | 说明 |
|---|---|---|
| 图片 URL（直发） | `image` + `url` | agent 的 markdown 图片 URL 原样发送；桥自行下载（无需本地文件） |
| 本地图片 | `image` + `base64://` | 上限 8 MB |
| 批量图片 | 多个 `image` 段 | 一条消息，最多 9 张；URL + 本地可混 |
| 语音 | `record` + `base64://` | 上限 20 MB；桥转码为 silk |
| 视频 | `video` + `base64://` | 上限 20 MB |
| 文件 | `file` + `base64://` + `name` | 上限 20 MB |
| 合并转发 | `send_forward_msg` + `node` | 仅群聊 |

**合并转发**由 agent 侧的块触发：

```
[[qq_forward]]
<显示名>
<消息文本>
---
<显示名>
<消息文本>
[[/qq_forward]]
```

每个 `---` 分隔块成为一个转发节点（名字 + 文本，单节点 500 字上限）。私聊中忽略该标记，块退化为纯文本。

## 回复引用消息（quote）

用户回复（引用）一条之前消息时，适配器调 OneBot `get_msg` API 取原消息并：

- 原文本加 `[引用]` 前缀让 agent 看到引用了什么
- 原消息里的图片 / 语音 / 视频作为媒体附加（语音走 STT 管线，视频下载供抽帧）

段数组和 CQ 字符串两种负载都支持。`get_msg` 失败则当前消息原样送达。

## Loop 消息合并（中间评论折叠）

多工具回合里网关会先发中间评论（“正在使用工具 X…”）再发最终回复。为省聊天空间，适配器按聊天缓冲 interim 文本消息，最终消息到达时**先合并成一条 QQ 转发消息，再发最终内容（含可能有的文字图卡片），然后撤回原文**：

- 群聊用 `send_forward_msg`，私聊用 `send_private_forward_msg`
- 缓冲 ≥2 条 interim 才合并
- 撤回（`delete_msg`）只在合并转发成功后执行；失败则保留原文
- 任何新的入站用户消息到达时清空缓冲和待撤回列表

这依赖网关在流消费者元数据里给评论发送打 `interim: True` 标记（见 `gateway/stream_consumer.py`）。

### 单条 interim 自动撤回

每条缓冲的 interim 有独立计时器（`interim_recall_seconds`，默认 **90** 秒）。回合一直不结算（没有 final 到达）时，这条 interim 会单独撤掉，不会无限堆积。缓冲一旦被 final 结算，挂起的任务自我取消，结算过的回合不会被二次撤回。`interim_recall_seconds: 0` 关闭。

### 回合末小结卡

回合结算且缓冲明明有 **≥2** 条 interim 时，把它们渲染成一张 **“本轮进展”文字图卡片** 汇总阶段性进度，撤回原文，最终回复照常继续。渲染或发送失败时回退到普通合并转发路径。

### 撤回限速

批量撤回时 `delete_msg` 调用间隔 **60 ms**，避免触发 NapCat 限流。

## Agent 模型工具

插件注册了模型侧工具，agent 可以直接推媒体、查 NapCat（`provides_tools` 让 CLI/TUI 会话也能用）：

| 工具 | 作用 |
|---|---|
| `qq_send_image` / `qq_send_voice` / `qq_send_video` / `qq_send_file` | 带外发送媒体；chat 从参数或 `HERMES_SESSION_CHAT_ID` 解析 |
| `qq_send_forward` | 文本节点合并转发 |
| `qq_napcat_api` | 白名单 NapCat action：`get_group_member_list`、`get_group_member_info`、`get_stranger_info`、`get_forward_msg`、`get_record`、`get_file`、`upload_group_file`、`upload_private_file`、`get_group_root_files`、`get_group_files_by_folder`、`get_group_file_url`、`ocr_image`、`get_ai_characters`、`send_group_ai_record`、`get_group_msg_history`；白名单外一律 403 |
| `qq_group_history` | 拉取群历史，`message_seq` 翻页（每页 ≤50 条） |

HTTP 等价物：适配器本地 API 的 `GET /api/napcat`（action 代理）和 `POST /api/send_media`。

## 热加载

通过 `extra.hot_reload: true` 启用（默认**关**，仅开发用）。开启后 `onebot_utils.py`（纯函数：CQ 解析、分段、markdown 剥离、表情映射）和 `t2i_render.py`（文字图渲染器）每次使用都会检测：适配器 stat 文件 mtime，变了就 `importlib.reload`，调样式/规则免网关重启。`adapter.py` 自身的改动仍需重启。

## 隐私与数据

- **网络**：与 OneBot 11 桥一条 WebSocket 连接（反向监听或正向拨出）；入站图片/文件从 QQ CDN 下载。
- **文件**：入站媒体缓存在临时目录，6 小时 TTL 清理；顶栏用到的昵称缓存持久化到插件旁的 `nicknames.json`。
- **系统调用**：语音转写调用本地 `ffmpeg`（和配置的 STT 后端）；无其他本地工具依赖。
- **敏感信息**：`access_token` 和管理员白名单只从配置读取，从不写日志；给受限成员聊天的出站回复按敏感意图关键词审计记录。
- **无遥测**：适配器除配置的 OneBot 桥和 QQ CDN 外不做任何第三方调用。

## 故障排查

| 症状 | 原因与修复 |
|---|---|
| 群聊不响应 | `require_mention: true` 需要 @ 或回复；@ 检测 fail-closed；确认 `bot_qq` 已从 meta 事件学到或显式设置 |
| 图片下载 403 | NapCat 把 URL 里的 `&` 转义成 `&amp;`（解析会自动反转义）；还失败就看媒体下载日志 |
| 语音显示 `[语音]` 占位 | `ffmpeg` 不可用，或 `get_record` 失败；装 ffmpeg 重试 |
| 文件消息到达为空 | CQ 字符串桥可能省略 `file` 段名。适配器标记为 `[文件:<name>]`（名字回退到 `file=` 属性）；NapCat 私聊文件只有 hash + 容器路径，名字来自 `file=` 属性 |
| 文字图卡片中文豆腐块 | 缺 CJK 字体：`apt install fonts-noto-cjk` |
| Loop 中间消息没合并 | 网关必须在评论元数据里发 `interim: True`（`gateway/stream_consumer.py` 里打过补丁的 `_send_commentary`）；适配器侧合并只是回退消费者 |

## 备注

- 出站消息用 OneBot 段数组格式（不是 CQ 码字符串），NapCat 处理消息需要这个。
- 回复以纯文本发送，不引用触发消息。
- QQ 表情映射为常见 emoji；未知表情折叠为 `[表情]`。无下载链接的语音降级 `[语音]`；入站视频/文件段降级 `[视频]` / `[文件:name]` 占位；未知段类型（json 卡片、戳一戳、转发 CQ 码）降级 `[卡片]` / `[戳一戳]` / `[合并转发:id]` 占位。
- **私聊**里 agent 生成时机器人显示 QQ 原生“正在输入”气泡（走 NapCat 的 `set_input_status` 扩展）。群聊 QQ 没有正在输入指示。
- 长回复渲染要几秒；网关在支持处显示正在输入指示。
- Cron / 定时投递目前还不能给 OneBot 附带媒体（核心 `send_message_tool` 媒体白名单只覆盖 telegram、discord、matrix、weixin、signal、yuanbao、feishu、whatsapp 和 slack）。交互式回复不受影响。