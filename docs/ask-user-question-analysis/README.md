# AskUserQuestion 能力分析文档集

> 背景：公司「通用层」基于 hermes 的 `api_server`（OpenAI 兼容 Chat Completions + SSE 单向通道）接入，并向上封装一层供业务应用（如 `ins-liexiaoxia-platform`）调用。产品提出新增「AskUserQuestion 卡片」需求，需厘清该需求在现有接入链路上的可行性。

## 文档索引

| 文档 | 面向 | 用途 |
|------|------|------|
| [01 - 不支持 AskUserQuestion 的原因](./01-不支持AskUserQuestion的原因.md) | 技术 + 产品 | 论证当前接入方式为何没有该能力，根因与证据 |
| [02 - 本地 demo 部署与接入方案](./02-本地demo部署与接入方案.md) | 技术（演示用） | 用同样的接入方式本地搭一套轻量 demo，供产品演示 |
| [03 - 接入方式功能差异对比](./03-接入方式功能差异对比.md) | 技术 + 产品 | `api_server` 与 CLI / 消息平台等接入方式的能力矩阵 |

## 一句话结论

当前链路（`api_server` Chat Completions + SSE 单向通道）**底层就没有** AskUserQuestion（hermes 内部名 `clarify`）能力。这是 hermes 在该接入层的设计性裁剪，不是通用层疏漏，也不是技术上不可为。

## 证据速览（详见文档 01）

1. `toolsets.py:397` —— `hermes-api-server` 工具集白名单明确写「no interactive UI tools like clarify」
2. `gateway/platforms/api_server.py:1120` —— 创建 agent 时未注入 `clarify_callback`
3. `agent/agent_init.py:268` —— `clarify_callback` 为 None 时，clarify 工具直接返回错误
4. api_server 给 `approval` 接了完整 SSE+POST 异步通道，**给 clarify 一根线都没接**
5. Chat Completions 是「请求→跑完→响应」的单轮模型，turn 结束通道即拆（`api_server.py:762`）
