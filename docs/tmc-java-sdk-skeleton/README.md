# tmc-java-sdk

面向腾讯企点 TMC OpenAPI 的 Java SDK skeleton，按官方文档能力域拆分模块，并提供统一的 webhook 接入层。

## Features

- 统一签名与 HTTP 调用抽象
- 按能力分类的模块：`customer`、`tag`、`segment`、`analytics`
- webhook 统一入口：验签、解析、标准化、分发、异步执行
- 强类型 DTO 与文档注释，便于 IDE 类型提示
- 可扩展到其他语言的统一设计：模块 API + DTO + webhook pipeline

## API 分类

### Customer
- `POST /cdp-entity/user/create`
- `GET /cdp-entity/user/queryList`

### Tag
- `POST /cdp-tag/inner-api/open/cdp/tag/tagDefine/external/create`
- `GET /cdp-tag/inner-api/open/cdp/tag/tagDefine/queryList`

### Segment
- `POST /cdp-crowd/import/v2`
- `GET /cdp-crowd/queryList`
- `GET /openapi/task/status`

### Analytics
- `GET /apiserver/openapi/panels`
- `GET /apiserver/openapi/analysis/query`

## Quick Start

```java
TmcClientConfig config = TmcClientConfig.builder()
    .baseUrl("https://tmc.qidian.qq.com")
    .corporationId("corp_xxx")
    .secretId("secret_id")
    .secretKey("secret_key")
    .build();

TmcClient client = TmcClient.create(config);
```

## Unified Webhook Endpoint

```java
WebhookEndpoint endpoint = WebhookEndpoint.builder()
    .verifier(new DefaultWebhookVerifier(secretKey, new HmacSha256WebhookSigner()))
    .parser(new JacksonWebhookParser(new ObjectMapper()))
    .normalizer(new DefaultWebhookEventNormalizer())
    .dispatcher(dispatcher)
    .asyncExecutor(new DefaultAsyncEventExecutor(4))
    .build();
```

### Webhook 能力

1. **统一分发**：所有事件先被标准化为 `TmcWebhookEvent`，再按 `eventType` 分发。
2. **标准验签**：`DefaultWebhookVerifier` 基于 `cur_time + sign` 与签名器校验请求身份。
3. **统一解析与格式化**：`DefaultWebhookEventNormalizer` 支持单事件与数组事件，输出统一字段：
   - `eventId`
   - `eventType`
   - `eventTime`
   - `source`
   - `subjectId`
   - `attributes`
   - `rawPayload`
4. **异步处理**：`WebhookEndpoint` 遍历标准化事件列表，通过 `AsyncEventExecutor` 异步执行。

## Multi-language Design Notes

不同语言实现建议保持相同抽象：

- **Java / Kotlin**：接口 + POJO/record + builder
- **TypeScript**：module client + discriminated union + async handler map
- **Python**：service class + pydantic/dataclass models + asyncio/celery dispatch
- **Go**：package client + struct model + worker pool webhook consumer

核心是保证：
- 公开常量暴露官方 path
- webhook 标准事件模型保持一致
- 验签/解析/标准化/分发/异步五段式 pipeline 一致

## Demo

Spring Boot demo classes are under:

- `com.tencent.qidian.tmc.demo.springboot.WebhookDemoApplication`
- `com.tencent.qidian.tmc.demo.springboot.SpringBootWebhookController`
