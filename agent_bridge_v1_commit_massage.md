feat(agent_bridge): 新增本机多 Agent 群聊桥接 v1

## 为什么需要这个 commit

当前存在两个 Hermes Agent，它们部署在同一台 Linux 服务器的不同用户下：

- `/home/azuto` 用户下运行的是“小鸡毛”
- `/home/yixin` 用户下运行的是“猪分身”

这两个 Agent 都可以接入同一个企业微信或 QQ 群聊。人工用户在群里直接 `@小鸡毛`
或 `@猪分身` 时，平台会把人工消息转发到对应 bot，Hermes Gateway 可以正常触发
对应 Agent 回复。

但问题出现在 Agent 之间互相 `@`：

- 小鸡毛在群里发出 `@猪分身 ...`
- 或猪分身在群里发出 `@小鸡毛 ...`

这类消息的作者本身是 bot。企业微信、QQ 或中间 adapter 不一定会把 bot 发出的
`@` 消息再次作为可触发事件投递给另一个 bot。结果就是：从真实群聊界面看，小鸡毛
似乎已经 `@猪分身` 了，但从猪分身的 API/Gateway 视角看，它可能根本没有收到这条
可触发消息。

因此，这不是模型“不想回复”，而是外部平台没有稳定提供“bot 发消息唤醒另一个 bot”
这个事件链路。

同时，不能简单地用“消息文本里出现另一个 Agent 的名字”来触发。比如人工用户发送：

```text
@小鸡毛 你去叫一下猪分身
```

这句话的真实含义是：人工用户在指挥小鸡毛。它虽然出现了“猪分身”三个字，但并不是
人工用户直接要求猪分身回答。如果 bridge 只按关键词匹配，就会导致猪分身也被误触发，
让群聊行为变得很不自然。

这个 commit 的目标是实现一个本机 Agent-to-Agent bridge，让两个 Hermes Gateway
在服务器内部共享群聊事件流：

- 人工用户在外部群里发出的消息可以进入两个 Agent 的上下文
- Agent 在外部群里的可见回复也可以进入另一个 Agent 的上下文
- 只有 Agent 消息中显式包含目标 Agent 的 wake `@mention` 时，才会触发目标 Agent
  回复
- 裸名字只作为上下文观察，不自动唤醒
- 两个 Agent 的对话、人工插话、Agent 回复都尽量保留在同一个 Hermes session 语境里，
  表现得更像真实多人群聊

## 总体框架

这个 commit 引入的 v1 框架分成五层：

1. 外部平台层

   企业微信、QQ 等平台仍然负责真实的人类群聊入口和 Agent 最终可见回复展示。
   人工用户从群里 `@bot` 的消息仍然走原有 Gateway adapter。

2. Hermes Gateway hook 层

   Gateway 在两个关键时机暴露插件 hook：

   - `gateway_startup`
   - `post_gateway_response`

   加上已有的 `pre_gateway_dispatch` 和 `pre_llm_call`，agent_bridge 可以观察 Gateway
   生命周期、真实入站消息、LLM 调用前上下文、以及 Agent 最终回复。

3. agent_bridge 插件层

   `plugins/agent_bridge/__init__.py` 是主要插件逻辑。它负责：

   - 读取当前用户的 `~/.hermes/config.yaml`
   - 在 Gateway 启动时向本机 bridge server 注册当前 Agent
   - 将真实人工消息发布到 bridge
   - 将当前 Agent 的最终回复发布到 bridge
   - 从 bridge 轮询其他 Agent 或人工消息
   - 决定收到的 bridge event 是“只观察”还是“注入 Gateway 触发回复”
   - 给 LLM 注入群聊规则上下文
   - 注册一个格式化 wake mention 的工具

4. 本机 bridge server 层

   `plugins/agent_bridge/server.py` 提供一个只依赖 Python 标准库的本机 HTTP server。
   默认监听：

   ```text
   http://127.0.0.1:8791
   ```

   两个用户下的 Gateway 都连接同一个 server。server 负责 room、participants、
   event queue、thread、去重和 bot-to-bot 轮数限制。

5. Hermes session/transcript 层

   bridge 收到其他参与者消息后，不是简单丢给模型。它会根据消息类型和 wake 规则：

   - 对需要当前 Agent 回复的消息，构造内部 `MessageEvent`，走 Hermes Gateway 的正常
     message handling 流程
   - 对不需要回复的消息，追加到当前 session transcript，作为群聊上下文观察

   这样人工消息、Agent 回复、Agent 之间的交流可以尽量落在同一个 session 语境里，
   而不是分裂成互相看不见的独立对话。

## 本次改动包含哪些文件

### `plugins/agent_bridge/plugin.yaml`

新增插件声明：

- 插件名：`agent_bridge`
- 版本：`0.1.0`
- 描述：本机多 Agent 群聊协调 bridge
- 声明需要的 hooks：
  - `gateway_startup`
  - `pre_gateway_dispatch`
  - `post_gateway_response`
- 声明需要环境变量：
  - `HERMES_AGENT_BRIDGE_TOKEN`

### `plugins/agent_bridge/server.py`

新增本机 bridge server。它提供以下 HTTP 接口：

```text
GET  /health
POST /v1/register
POST /v1/events
GET  /v1/events?agent_id=...
```

server 的核心职责：

- 保存已注册 Agent：
  - `agent_id`
  - `display_name`
  - 更新时间
- 保存 room 配置：
  - `external_targets`
  - `participants`
  - `max_bot_messages`
  - `idle_timeout_seconds`
- 为每个 Agent 维护独立事件队列
- 根据 participants 将事件投递给其他 Agent
- 避免把事件投回作者自身
- 根据 `origin_agent_id` 避免投回事件来源 Agent
- 用 `seen_ids` 做短窗口去重
- 为每条群聊链路维护 `thread_id`
- 用 `max_bot_messages` 限制 bot-to-bot 连续回复轮数，避免无限互相唤醒
- 用 `idle_timeout_seconds` 控制长时间空闲后的 thread 状态重置

server 只使用 Python 标准库，方便在两个 Linux 用户之间共享，不额外引入依赖。

### `plugins/agent_bridge/__init__.py`

新增 agent_bridge 插件核心逻辑。

主要结构包括：

- `RoomConfig`
  描述一个 bridge room。

- `BridgeConfig`
  描述当前 Agent 的 bridge 配置。

- `_load_bridge_config()`
  从当前用户的 Hermes 配置中读取 `agent_bridge` 配置。

- `_Runtime`
  保存当前插件运行时状态，包括：
  - 当前配置
  - Gateway 引用
  - Gateway 所在线程的 asyncio loop
  - bridge poller 线程
  - 已处理 event id
  - session 到 thread 的映射

- `_on_gateway_startup()`
  Gateway 启动后启动 bridge runtime 并注册当前 Agent。

- `_on_pre_gateway_dispatch()`
  观察真实平台进入的人工消息，把它发布到 bridge。

- `_on_post_gateway_response()`
  观察当前 Agent 的最终回复，把它发布到 bridge。

- `_handle_bridge_event()`
  处理从 bridge server 收到的事件，决定注入触发或只追加 transcript。

- `_agent_bridge_llm_context()`
  给 LLM 注入 Agent bridge 群聊规则。

- `agent_bridge_format_wake_message`
  注册给 LLM 的工具，用来生成正确的目标 Agent wake 文本。

### `gateway/run.py`

新增两个插件调用点：

- Gateway 启动后触发 `gateway_startup`
- Agent 生成最终回复后触发 `post_gateway_response`

这让 agent_bridge 可以观察 Gateway 生命周期和 Agent 最终回复，而不需要把自己伪装
成一个平台 adapter。

### `hermes_cli/plugins.py`

在合法 hook 列表里新增：

```text
gateway_startup
post_gateway_response
```

这样 `plugins/agent_bridge/plugin.yaml` 中声明的 hook 可以被插件系统接受。

### `hermes_cli/config.py`

新增默认配置：

```yaml
agent_bridge:
  enabled: false
  agent_id: ""
  display_name: ""
  server_url: "http://127.0.0.1:8791"
  token_env: "HERMES_AGENT_BRIDGE_TOKEN"
  rooms: {}
```

并在可选环境变量中新增：

```text
HERMES_AGENT_BRIDGE_TOKEN
```

注意：这里是默认配置和环境变量说明。实际启用时，不应该直接改
`hermes_cli/config.py`，而是分别改两个用户自己的配置文件：

```text
/home/azuto/.hermes/config.yaml
/home/yixin/.hermes/config.yaml
```

### `tests/gateway/test_agent_bridge.py`

新增测试覆盖：

- 新 hook 是否已声明
- bridge server 是否只把事件投递给其他 Agent
- human event 是否创建新的 thread
- bot-to-bot thread 是否受 `max_bot_messages` 限制
- wake name 与裸名字的触发差异
- slash command 是否跳过 bridge 发布
- wake message tool 是否按当前 Agent 身份生成正确 wake_text
- 工具错误信息是否返回可用 peer Agent
- pre_llm_call 是否注入工具和 peer Agent 上下文
- 显式 wake 的 Agent 消息是否注入内部 Gateway event
- 人工消息是否只观察不触发
- Agent 消息只写裸名字时是否只观察不触发

## 核心语义：`agent_id` 不是 Linux 用户名

`agent_bridge.agent_id` 是 bridge 内部用于识别 Agent 的稳定唯一 ID，不是 Linux 用户名。

也就是说：

- `azuto` 是运行小鸡毛的 Linux 用户
- `yixin` 是运行猪分身的 Linux 用户
- `xiaojimao` 可以是小鸡毛在 bridge 内部的 Agent ID
- `zhufenshen` 可以是猪分身在 bridge 内部的 Agent ID

只要两个用户的配置里对同一个 Agent 使用同一个 `agent_id`，bridge 就能正常工作。

推荐配置：

```yaml
# 小鸡毛
agent_id: "xiaojimao"
display_name: "小鸡毛"

# 猪分身
agent_id: "zhufenshen"
display_name: "猪分身"
```

`agent_id` 的实际用途包括：

1. 向 bridge server 注册当前 Agent

   ```text
   POST /v1/register
   agent_id = xiaojimao
   ```

2. 轮询属于自己的事件队列

   ```text
   GET /v1/events?agent_id=xiaojimao
   ```

3. 判断 participants 中哪一项是自己

   ```text
   participant.agent_id == current_config.agent_id
   ```

4. 发布自己回复时作为 `author_id` 和 `origin_agent_id`

   ```text
   author_id = xiaojimao
   origin_agent_id = xiaojimao
   ```

5. 作为工具 `agent_bridge_format_wake_message` 的目标 ID

   ```json
   {
     "target_agent_id": "zhufenshen",
     "message": "你来接一下这个问题"
   }
   ```

最重要的配置规则：

```text
当前配置里的 agent_bridge.agent_id 必须等于 participants 里代表自己的那一项 agent_id。
```

例如，`/home/azuto/.hermes/config.yaml` 里小鸡毛这样配置：

```yaml
agent_bridge:
  agent_id: "xiaojimao"
  display_name: "小鸡毛"
  rooms:
    home:
      participants:
        - agent_id: "xiaojimao"
          display_name: "小鸡毛"
        - agent_id: "zhufenshen"
          display_name: "猪分身"
```

`/home/yixin/.hermes/config.yaml` 里猪分身这样配置：

```yaml
agent_bridge:
  agent_id: "zhufenshen"
  display_name: "猪分身"
  rooms:
    home:
      participants:
        - agent_id: "xiaojimao"
          display_name: "小鸡毛"
        - agent_id: "zhufenshen"
          display_name: "猪分身"
```

如果把 `agent_id` 写成 Linux 用户名也可以工作，但语义上不如使用稳定的 Agent ID 清晰。
例如 `agent_id: azuto` 能工作，是因为 participants 也写了 `azuto`。但更推荐用
`xiaojimao`，因为这个字段代表的是 Agent 身份，不是操作系统用户。

## 触发规则

agent_bridge v1 的触发规则是：

1. 人工用户直接在外部群聊 `@小鸡毛`

   外部平台正常触发小鸡毛。bridge 会把这条人工消息发布给猪分身，但猪分身只观察，
   不会因为人工消息自动触发。

2. 人工用户直接在外部群聊 `@猪分身`

   外部平台正常触发猪分身。bridge 会把这条人工消息发布给小鸡毛，但小鸡毛只观察，
   不会因为人工消息自动触发。

3. 小鸡毛的最终可见回复里显式包含 `@猪分身`

   bridge 会把这条 Agent 消息投递给猪分身。猪分身检测到自己的 wake mention 后，
   会把这条消息注入为内部 Gateway event，从而触发猪分身回复。

4. 猪分身的最终可见回复里显式包含 `@小鸡毛`

   同理，bridge 会触发小鸡毛。

5. 只出现裸名字，不触发

   例如：

   ```text
   我觉得猪分身之前说得对
   ```

   这只会进入猪分身的 transcript 作为观察上下文，不会触发猪分身回复。

6. 人工命令中提到另一个 Agent，不触发另一个 Agent

   例如：

   ```text
   @小鸡毛 你去叫一下猪分身
   ```

   这只触发小鸡毛。猪分身只观察这条上下文。只有小鸡毛最终回复里显式写出
   `@猪分身 ...`，猪分身才会被 bridge 唤醒。

## 为什么需要 LLM 工具，而不是插件硬编码判断

之前的一个风险是：插件如果用规则判断“这句话是否希望另一个 Agent 回复”，很容易误判。

例如：

```text
我刚才提到了猪分身，但不需要它回复。
```

或者：

```text
猪分身这个名字在这里是讨论对象，不是收信人。
```

插件层很难可靠理解这些语义。如果简单用关键词，会不智能；如果堆复杂规则，又容易在
边界情况出错。

因此这个 commit 采用 LLM + tool 的方式：

- 插件只负责确定机制：
  - 什么文本能唤醒
  - 如何生成 wake 文本
  - 如何投递和注入事件
- LLM 负责判断意图：
  - 当前是否真的想让另一个 Agent 回复
  - 是否需要交接、提问、邀请另一个 Agent
- 工具负责把 LLM 的意图转换成稳定格式：
  - 返回 `wake_text`
  - 确保 wake mention 使用目标 Agent 配置里的 `wake_names`

工具名：

```text
agent_bridge_format_wake_message
```

工具输入示例：

```json
{
  "target_agent_id": "zhufenshen",
  "message": "你来接一下这个问题"
}
```

工具输出示例：

```json
{
  "success": true,
  "wake_text": "@猪分身 你来接一下这个问题",
  "target_agent_id": "zhufenshen",
  "target_display_name": "猪分身",
  "target_wake_name": "@猪分身",
  "room_id": "home",
  "instruction": "Include wake_text verbatim in your final visible reply if you want this agent to respond."
}
```

LLM 最终应该把 `wake_text` 原样放进可见回复中。这样 bridge 不需要猜测复杂语义，只需要
检测最终可见回复里是否有目标 Agent 的 wake mention。

## Hermes session 行为

这个 commit 的核心目标之一是让群聊上下文看起来像真实多人群聊。

因此 bridge event 的处理分成两种：

### 需要当前 Agent 回复

条件：

- 消息作者是另一个 Agent
- bridge server 返回 `allow_auto_reply = true`
- 消息文本包含当前 Agent 的 wake mention

处理方式：

- 构造内部 `MessageEvent`
- 设置 `internal=True`
- `raw_message` 中带上 `agent_bridge` 元数据
- 优先调用对应平台 adapter 的 `handle_message(event)`
- 让 Hermes Gateway 正常处理这条消息

这会让目标 Agent 像收到一条真实群聊消息一样回复。

### 不需要当前 Agent 回复

条件包括：

- 人工消息
- Agent 消息但没有显式 wake 当前 Agent
- 只出现裸名字
- bot-to-bot 轮数超过 `max_bot_messages`

处理方式：

- 不触发 Gateway 回复
- 只追加到当前 session transcript
- 如果当前 session 是 shared multi-user session，则内容会带上说话人前缀，例如：

  ```text
  [小鸡毛] @猪分身 你来接一下这个问题
  ```

这样当前 Agent 后续被真正唤醒时，仍然能看到之前群聊上下文。

## slash command 行为

bridge 会识别直接发给当前 Agent 的 slash command，例如：

```text
/new
@小鸡毛 /new
```

这类命令不会发布到 bridge，也不会触发另一个 Agent。

因此，人工在群里给小鸡毛执行 `/new` 时，默认只影响小鸡毛自己的 Hermes session；
猪分身不会自动同步 `/new`。如果需要两个 Agent 都开启新 session，需要分别对两个
Agent 执行对应命令，或者后续再实现专门的群体 session 管理能力。

同时，bridge 注入内部事件时，如果文本本身以 `/` 开头，会自动加零宽字符转义，
避免被 Gateway 当作真正 slash command 执行。

## 如何配置

实际配置位置是每个 Linux 用户自己的 Hermes 配置文件，而不是源码里的默认配置。

### 1. 配置共享 token

两个 Gateway 和 bridge server 必须使用同一个 token。

azuto 用户：

```text
/home/azuto/.hermes/.env
```

添加：

```env
HERMES_AGENT_BRIDGE_TOKEN=suzijieyixin
```

yixin 用户：

```text
/home/yixin/.hermes/.env
```

添加：

```env
HERMES_AGENT_BRIDGE_TOKEN=suzijieyixin
```

如果当前启动方式不会自动加载 `.env`，启动前手动执行：

```bash
export HERMES_AGENT_BRIDGE_TOKEN='suzijieyixin'
```

### 2. 配置小鸡毛

编辑：

```text
/home/azuto/.hermes/config.yaml
```

添加或修改：

```yaml
agent_bridge:
  enabled: true
  agent_id: "xiaojimao"
  display_name: "小鸡毛"
  server_url: "http://127.0.0.1:8791"
  token_env: "HERMES_AGENT_BRIDGE_TOKEN"
  rooms:
    home:
      external_targets:
        - platform: "wecom"
          chat_id: "<企业微信群 chat_id>"
      participants:
        - agent_id: "xiaojimao"
          display_name: "小鸡毛"
          wake_names: ["@小鸡毛"]
          mention_names: ["小鸡毛", "@小鸡毛"]
        - agent_id: "zhufenshen"
          display_name: "猪分身"
          wake_names: ["@猪分身"]
          mention_names: ["猪分身", "@猪分身"]
      max_bot_messages: 16
      idle_timeout_seconds: 1800
```

注意：

- 这份配置属于 azuto Linux 用户
- 但 `agent_id` 推荐写成 `xiaojimao`
- `display_name` 写成 `小鸡毛`
- participants 中代表小鸡毛的那一项也必须是 `agent_id: "xiaojimao"`

### 3. 配置猪分身

编辑：

```text
/home/yixin/.hermes/config.yaml
```

添加或修改：

```yaml
agent_bridge:
  enabled: true
  agent_id: "zhufenshen"
  display_name: "猪分身"
  server_url: "http://127.0.0.1:8791"
  token_env: "HERMES_AGENT_BRIDGE_TOKEN"
  rooms:
    home:
      external_targets:
        - platform: "wecom"
          chat_id: "<企业微信群 chat_id>"
      participants:
        - agent_id: "xiaojimao"
          display_name: "小鸡毛"
          wake_names: ["@小鸡毛"]
          mention_names: ["小鸡毛", "@小鸡毛"]
        - agent_id: "zhufenshen"
          display_name: "猪分身"
          wake_names: ["@猪分身"]
          mention_names: ["猪分身", "@猪分身"]
      max_bot_messages: 16
      idle_timeout_seconds: 1800
```

注意：

- 这份配置属于 yixin Linux 用户
- 但 `agent_id` 推荐写成 `zhufenshen`
- `display_name` 写成 `猪分身`
- participants 建议和小鸡毛那份保持一致
- `server_url` 必须指向同一个 bridge server
- `token_env` 必须和 `.env` 中的变量名一致

### 4. `wake_names` 和 `mention_names`

`wake_names` 表示真正能唤醒 Agent 的名字：

```yaml
wake_names: ["@猪分身"]
```

bridge 只用 `wake_names` 判断是否要触发回复。

`mention_names` 表示这个 Agent 的可识别名字，主要用于上下文、工具提示和展示：

```yaml
mention_names: ["猪分身", "@猪分身"]
```

推荐规则：

- `wake_names` 只放真正代表“叫你回复”的 `@` 名称
- `mention_names` 可以同时放裸名字和 `@` 名称
- 不要把裸名字放进 `wake_names`，否则会重新引入误触发

## 如何启动

### 1. 启动 bridge server

只需要启动一个 bridge server。推荐放在 azuto 用户下启动：

```bash
cd /home/azuto/.hermes/hermes-agent
export HERMES_AGENT_BRIDGE_TOKEN='suzijieyixin'
./venv/bin/python plugins/agent_bridge/server.py --token-env HERMES_AGENT_BRIDGE_TOKEN
```

如果 8791 端口已被占用：

```bash
ss -ltnp 'sport = :8791'
```

检查健康状态：

```bash
curl -i -H "Authorization: Bearer suzijieyixin" \
  http://127.0.0.1:8791/health
```

预期响应：

```json
{"ok": true}
```

### 2. 启动小鸡毛 Gateway

azuto 用户：

```bash
cd /home/azuto/.hermes/hermes-agent
export HERMES_AGENT_BRIDGE_TOKEN='suzijieyixin'
./venv/bin/hermes gateway
```

如果 Gateway 已经运行：

```bash
./venv/bin/hermes gateway stop
./venv/bin/hermes gateway
```

### 3. 启动猪分身 Gateway

yixin 用户：

```bash
cd /home/yixin/hermes-agent
export HERMES_AGENT_BRIDGE_TOKEN='suzijieyixin'
./venv/bin/hermes gateway
```

## 如何使用

### 人工叫小鸡毛

群里发送：

```text
@小鸡毛 你去问一下猪分身
```

结果：

- 小鸡毛被外部平台正常触发
- 这条人工消息会通过 bridge 进入猪分身上下文
- 猪分身只观察，不自动回复

### 小鸡毛希望猪分身回复

小鸡毛最终可见回复必须包含：

```text
@猪分身 ...
```

例如：

```text
@猪分身 你来接一下这个问题
```

结果：

- bridge 检测到小鸡毛的 Agent 消息显式 wake 猪分身
- 猪分身收到内部 Gateway event
- 猪分身回复

### 猪分身希望小鸡毛回复

猪分身最终可见回复必须包含：

```text
@小鸡毛 ...
```

结果：

- bridge 检测到猪分身的 Agent 消息显式 wake 小鸡毛
- 小鸡毛收到内部 Gateway event
- 小鸡毛回复

### 只提到裸名字

例如：

```text
我觉得小鸡毛刚才说的方向可以继续。
```

结果：

- 小鸡毛只观察
- 不触发回复

## 如何验证

运行测试：

```bash
pytest tests/gateway/test_agent_bridge.py
```

手动验证：

1. 启动 bridge server
2. 启动小鸡毛 Gateway
3. 启动猪分身 Gateway
4. 在群里发送：

   ```text
   @小鸡毛 你去叫一下猪分身
   ```

5. 确认猪分身不会因为裸名字“猪分身”自动回复
6. 让小鸡毛最终回复中包含：

   ```text
   @猪分身 ...
   ```

7. 确认猪分身被 bridge 唤醒并回复
8. 再测试只出现“小鸡毛”或“猪分身”裸名字时，对方只观察不回复

## 这个 commit 的边界

这个 v1 版本解决的是同机多 Hermes Agent 的本机桥接问题，不试图替代外部平台 adapter。

它不做这些事情：

- 不改变企业微信或 QQ 的真实消息投递规则
- 不要求平台必须支持 bot-to-bot mention 转发
- 不把裸名字当成自动触发条件
- 不自动同步所有 Agent 的 `/new` 等 session 管理命令
- 不在插件层用复杂规则猜测 LLM 是否想让另一个 Agent 回复

它提供的是一个稳定机制：

- 外部群聊仍然是人类可见的主界面
- 本机 bridge 负责补齐平台缺失的 bot-to-bot 可触发链路
- LLM 负责判断是否需要交接给另一个 Agent
- 工具负责生成正确 wake mention
- Hermes session transcript 保留群聊上下文

## 使用这个文件提交

可以直接用这个 Markdown 文件作为 commit message：

```bash
git commit -F agent_bridge_v1_commit_massage.md
```

如果想先预览：

```bash
sed -n '1,220p' agent_bridge_v1_commit_massage.md
```
