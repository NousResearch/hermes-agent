# TMC OpenAPI SDK 差距分析（基于 20260414 文档版本）

## 文档抓取结果

- 文档版本入口：`https://tmc.qidian.qq.com/base/console/doc/14154?version=20260414`
- 通过文档站后端接口 `POST /api/operation/server/api/doc/v1/pagetree/toplevel?lang=en_US` 与 `GET /api/operation/server/api/doc/v1/scan/{docId}` 完成抓取
- OpenAPI 根节点：`docId=14381`
- 抓取结果：
  - 叶子页面：244
  - 目录节点：71
  - webhook 文档：9
  - 可提取出请求方法/URL 的页面：213+
- 抓取缓存文件：
  - `~/tmp/tmc_docs_dump/pagetree.json`
  - `~/tmp/tmc_docs_dump/openapi_pages_full.json`
  - `~/tmp/tmc_docs_dump/openapi_folders.json`

## 当前 skeleton 覆盖情况

当前 Java SDK skeleton 仅包含 4 个模块、9 个接口方法，且多数路径为占位符，不匹配真实文档：

- `CustomerModule`
  - `create`
  - `list`
- `TagModule`
  - `create`
  - `list`
- `SegmentModule`
  - `createImportSegment`
  - `list`
  - `queryTask`
- `AnalyticsModule`
  - `listDashboards`
  - `queryAsyncResult`

### 主要问题

1. **API 路径不真实**
   - 例如客户创建 skeleton 使用 `/openapi/customer/create`
   - 文档真实路径为 `/cdp-entity/user/create?bizId={$bizId}`

2. **请求/响应模型严重失真**
   - 当前 `CustomerCreateRequest` 仅有 `customerId/customerName/mobile`
   - 文档要求 `identity + property` 等复杂结构

3. **HTTP 方法与传参方式不完整**
   - 实际接口以 `POST` 为主，也包含 `PUT` / `DELETE`
   - 绝大多数通过 body + `bizId` query 组合传参，不能再用当前的简化 GET 查询模型替代

4. **Webhook 归一化能力不够**
   - 真实 webhook 至少有 9 类，事件载荷形态差异很大：
     - 批量数组事件（行为、实时标签）
     - 单对象事件（积分、成长值、等级、权益、优惠券）
     - 包裹型对象（分群导出完成通知）
   - 当前 `DefaultWebhookEventNormalizer` 只做简单扁平化，无法稳定识别事件类型、批量条目、messageId、sendTime、签名差异

5. **模块边界不足**
   - 文档首批值得封装的高价值模块至少包括：
     - 客户实体（18）
     - 标签（12）
     - 分群（10）
     - 分析（4）
     - 数据实时推送（9）
   - skeleton 尚未覆盖统一的元数据查询、批量操作、异步任务结果等能力

## 各功能域文档规模

| 功能域 | 页面数 | 可解析接口数 | 建议优先级 |
|---|---:|---:|---|
| 客户实体 | 18 | 18 | P0 |
| 标签 | 12 | 11 | P0 |
| 分群 | 10 | 10 | P0 |
| 分析 | 4 | 4 | P1 |
| PA画像分析 | 6 | 5 | P2 |
| 数据实时推送 | 9 | 9 | P0 |
| LM会员忠诚度管理 | 76 | 75 | P3 |
| SCRM企微互动 | 72 | 57 | P3 |

## 建议首批 SDK 封装范围

### 1. 客户实体模块（P0）
建议至少覆盖：

- 客户创建 `POST /cdp-entity/user/create`
- 客户批量创建 `POST /cdp-entity/user/batchCreate`
- 客户批量更新 `POST /cdp-entity/user/batchEdit`
- 客户批量删除 `POST /cdp-entity/user/batchDelete`
- 客户身份绑定 `POST /cdp-entity/user/identity/bind`
- 客户身份解绑 `POST /cdp-entity/user/identity/unbind`
- 客户身份重置 `POST /cdp-entity/user/identity/reset`
- 客户列表查询 `POST /cdp-entity/user/queryList`
- 客户详情查询 `POST /cdp-entity/user/queryDetail`
- 客户轨迹查询 `POST /cdp-entity/user/behaviorTrace`
- 客户元数据查询 `POST /cdp-entity/user/property/list`

需要抽象的公共 DTO：
- `IdentityValue`
- `CustomerQueryCriteria`
- `CustomerFilter`
- `CustomerPropertyItem`
- `CustomerBatchContent`
- `TimeRange`
- `EventCodeCondition`

### 2. 标签模块（P0）
建议至少覆盖：

- 外部标签创建 `POST /cdp-tag/inner-api/open/cdp/tag/tagDefine/external/create`
- 标签列表查询 `POST /cdp-tag/inner-api/open/cdp/tag/tagDefine/list`
- 标签详情查询 `POST /cdp-tag/inner-api/open/cdp/tag/tagDefine/{id}`
- 标签分组查询 `POST /cdp-tag/inner-api/open/cdp/group/tree/list`
- 离散标签值查询 `POST /cdp-tag/inner-api/open/cdp/tag/getTagValList`
- 标签删除 `DELETE /cdp-tag/inner-api/open/cdp/tag/tagDefine/{id}`
- 标签批量删除 `POST /cdp-tag/inner-api/open/cdp/tag/tagDefine/batch/delete`
- 手工/外部标签就绪状态变更通知 `POST /cdp-tag/inner-api/open/cdp/tag/readiness/notify`

### 3. 分群模块（P0）
建议至少覆盖：

- 导入类分群创建V2 `POST /cdp-crowd/import/v2`
- 导入类分群创建 `POST /cdp-crowd/import`
- 导入类分群更新 `POST /cdp-crowd/import/update`
- 分群列表查询 `POST /cdp-crowd/list`
- 分群详情查询 `POST /cdp-crowd/detail`
- 分群分组查询 `POST /cdp-crowd/groupTree/list`
- 分群批量删除 `POST /cdp-crowd/delete`
- 导入类分群就绪状态变更通知 `POST /cdp-crowd/readiness/notify`

### 4. 分析模块（P1）
建议覆盖：

- 空间看板列表查询 `POST /apiserver/openapi/panels`
- 看板图卡列表查询 `POST /apiserver/openapi/cards`
- 图卡分析结果查询 `POST /apiserver/openapi/analysis/card`
- 获取异步图卡分析结果 `POST /apiserver/openapi/analysis/query`

## Webhook 统一封装要求映射

文档中已确认的 webhook 事件类型至少包括：

1. 行为数据实时推送
2. 实时标签实时推送
3. 积分变更实时推送
4. 成长值变更实时推送
5. 等级变更实时推送
6. 会员权益实时推送
7. 会员优惠券实时推送
8. 全类别分群新建/删除实时推送
9. 分群导出完成通知实时推送

### 文档确认到的通用特征

- 多数 webhook 使用 `POST application/json`
- 多数支持以下签名头：
  - `cur_time`
  - `sign = BASE64(HmacSHA256(url + "&" + cur_time, secret))`
- 文档建议可校验时间窗（通常 1 分钟内）
- 部分 webhook body 为数组，部分为对象，部分为 envelope 包装对象

### SDK 统一 webhook 设计建议

#### 统一入口
- `WebhookEndpoint` 保留统一接收入口
- 入参需要升级为：
  - 完整 request URL
  - headers
  - query params
  - raw body

#### 统一分发
- 增加稳定的事件类型枚举，例如：
  - `behavior.event`
  - `realtime.tag`
  - `member.point.changed`
  - `member.growth.changed`
  - `member.level.changed`
  - `member.benefit.changed`
  - `member.coupon.changed`
  - `crowd.changed`
  - `crowd.export.completed`
- 支持一条 HTTP 请求中拆分为多个逻辑事件并逐条异步分发

#### 统一验证
- `WebhookVerifier` 增强为：
  - HMAC SHA256 验签
  - 时间窗校验
  - 可选关闭验签（文档允许后台不配置 secret）

#### 统一解析/格式化
- 设计三层模型：
  - `RawWebhookPayload`
  - `TmcWebhookEnvelope`
  - `TmcWebhookEvent<T>`
- 针对数组型 payload 提供批量展开器
- 将 `messageId/sendTime/event_code/eventName/uniqueCode/...` 标准化到统一事件头

#### 异步处理
- 当前已有 `AsyncEventExecutor`
- 需要保证：
  - 单请求批量事件可并发/串行策略可配置
  - 处理异常隔离
  - 未识别事件有兜底 handler

## 不同语言 SDK 设计差异建议

### Java
- 适合：接口 + builder DTO + 泛型响应模型 + Spring Webhook Demo
- 建议保留 `ObjectMapper` 与强类型 POJO 双轨模型

### TypeScript
- 适合：联合类型、判别式 union、泛型 webhook handler map
- 数组型 webhook 可直接建成 `BehaviorEvent[] | RealtimeTagEvent[]`

### Python
- 适合：dataclass / pydantic 模型
- 更适合提供 `dict` 兼容层，降低动态字段接入成本

### Go
- 适合：按功能域拆 package，webhook 使用 interface + type switch
- 对动态字段建议保留 `map[string]any`

## 下一步实现建议

1. 重构底层 HTTP 客户端，支持：
   - `bizId` 统一拼接
   - POST/PUT/DELETE
   - path template
2. 先落地 P0 功能域：
   - 客户实体
   - 标签
   - 分群
   - 数据实时推送
3. webhook 先实现统一协议层，再补具体事件 DTO
4. 最后补分析模块、README、测试
