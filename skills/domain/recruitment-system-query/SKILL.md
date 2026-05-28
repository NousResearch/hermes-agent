---
name: recruitment-system-query
description: 用于通过 recruitmentSystem 官方 HTTP API 查询招聘岗位和新增招聘岗位。禁止直接查询或写入 MySQL。
version: 1.1.0
required_environment_variables:
  - name: RECRUITMENT_API_BASE_URL
    prompt: recruitmentSystem API base URL
    help: recruitmentSystem HTTP API base URL, for example http://127.0.0.1:8080.
    required_for: 查询或新增 recruitmentSystem 招聘岗位
  - name: RECRUITMENT_API_TENANT_ID
    prompt: recruitmentSystem tenant ID
    help: Tenant ID sent as the X-Tenant-Id HTTP header.
    required_for: 调用 recruitmentSystem /api/v1/jobs 接口
metadata:
  hermes:
    tags: [recruitment, http-api, jobs]
    related_tools: [recruitment_system_query, recruitment_system_create_job, recruitment_system_health]
---

# recruitment-system-query

## 触发场景

当用户询问招聘岗位、正在招聘的职位、岗位列表、招聘状态、岗位要求、岗位职责，或要求新增招聘岗位时使用本 Skill。

## 强制边界

- 必须通过 recruitmentSystem HTTP API 操作业务数据。
- 禁止直接连接 MySQL。
- 禁止手写 `SELECT`、`INSERT`、`UPDATE`、`DELETE` 或通过终端脚本读写招聘表。
- 查询岗位只能调用 `recruitment_system_query`，其底层使用 `GET /api/v1/jobs` 和 `GET /api/v1/jobs/{jobId}`。
- 新增岗位只能调用 `recruitment_system_create_job`，其底层使用 `POST /api/v1/jobs`、`PUT /api/v1/jobs/{jobId}/requirements`，默认再调用 `POST /api/v1/jobs/{jobId}/online`。
- 多租户请求必须传 `tenant_id`，或配置 `RECRUITMENT_API_TENANT_ID`，由工具发送为 `X-Tenant-Id`。

## 查询岗位

调用 `recruitment_system_query`，传入：

- `user_question`：用户原始问题。
- `tenant_id`：租户 ID；如果环境变量已配置可省略。
- `user_id`：当前用户 ID，可选，会作为 `X-User-Id` 透传。
- `status`：岗位状态，查询正在招聘岗位时使用 `ONLINE`。
- `keyword`、`department`、`page_no`、`page_size`：可选过滤条件。

示例：

```json
{
  "user_question": "当前正在招聘的岗位有哪些？",
  "tenant_id": "1001",
  "user_id": "hr_mgr_1001",
  "status": "ONLINE"
}
```

## 新增岗位

调用 `recruitment_system_create_job`，传入：

- `job_name`：岗位名称。
- `tenant_id`：租户 ID；如果环境变量已配置可省略。
- `user_id`：当前用户 ID，可选。
- `department`：部门；默认读取 `RECRUITMENT_DEFAULT_DEPARTMENT`，否则为 `研发中心`。
- `work_location`：工作地点；默认读取 `RECRUITMENT_DEFAULT_WORK_LOCATION`，否则为 `上海`。
- `headcount`：招聘人数，默认 `1`。
- `owner_user_id` / `owner_user_name`：招聘负责人；默认读取对应环境变量。
- `publish`：是否新增后上线，默认 `true`。上线前工具会先通过 API 写入岗位要求。
- `allow_duplicate`：默认 `false`，同名岗位存在时不重复创建。

示例：

```json
{
  "job_name": "高级golang开发工程师",
  "tenant_id": "1001",
  "user_id": "hr_mgr_1001",
  "department": "研发中心",
  "work_location": "上海",
  "headcount": 1,
  "publish": true
}
```

## 输出

工具返回 JSON：

- `success`：是否成功。
- `answer`：自然语言回答。
- `intent`：识别出的业务意图。
- `data`：结构化 API 数据，字段已标准化为 snake_case。
- `api`：实际调用的 API 方法和路径。
- `error_code` / `message`：失败原因。
- `trace_id`：审计追踪 ID。

## 配置

```bash
RECRUITMENT_API_BASE_URL=http://127.0.0.1:8080
RECRUITMENT_API_TENANT_ID=1001
RECRUITMENT_API_USER_ID=hr_mgr_1001
RECRUITMENT_API_TOKEN=
RECRUITMENT_API_TIMEOUT_SECONDS=15
RECRUITMENT_DEFAULT_DEPARTMENT=研发中心
RECRUITMENT_DEFAULT_WORK_LOCATION=上海
RECRUITMENT_DEFAULT_EMPLOYMENT_TYPE=FULL_TIME
RECRUITMENT_DEFAULT_HEADCOUNT=1
RECRUITMENT_DEFAULT_OWNER_USER_ID=hr_mgr_1001
RECRUITMENT_DEFAULT_OWNER_USER_NAME=HR经理
```

## 示例问题

- 当前正在招聘的岗位有哪些？
- 现在有哪些岗位在招？
- AI算法工程师这个岗位要求是什么？
- 帮我新增一个招聘岗位：高级golang开发工程师。
