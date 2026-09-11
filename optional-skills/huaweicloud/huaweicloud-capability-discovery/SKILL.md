---
name: huaweicloud-capability-discovery
description: Discover Huawei Cloud capabilities for an agent task. Use when the user asks what Huawei Cloud service, Skill, CLI command, API, SDK, or document should be used, or when a request is scenario-shaped and needs capability selection before implementation.
---

# Huawei Cloud Capability Discovery

**STOP - Do not answer from general knowledge.** Follow the procedure below.

Use this skill to turn vague developer intent into a precise Huawei Cloud capability path.

## Workflow

1. Classify the goal: application hosting, compute, container, serverless, storage, database, network, security, observability, AI, data, migration, cost, or troubleshooting.
2. Search Huawei Cloud Skills first when the task is scenario-based or operational.
3. Use official docs for exact service limits, API request/response fields, regions, and pricing-sensitive behavior.
4. Prefer KooCLI for local read-only discovery after auth is configured.
5. Before using KooCLI, discover exact operation names with `hcloud <Service> --help`; do not guess from intuition. Examples: list ECS instances with `ECS ListServersDetails`, create ECS with `ECS CreateServers`, inspect images through `IMS GlanceShowImage`.
6. Prefer SDK docs when the deliverable is application code.
7. Prefer MCP only when an approved Huawei Cloud MCP tool exists for the needed operation.
8. Treat Terraform as a secondary V1 path for reviewed IaC, not the default.
9. When no built-in devkit skill matches, browse the community skill marketplace at https://github.com/huaweicloud/huaweicloud-skills. Fetch the index from https://raw.githubusercontent.com/huaweicloud/huaweicloud-skills/master/skills-index/index.json.
10. When the deliverable is a PPT, architecture diagram (draw.io), or frontend page that needs official Huawei Cloud service logos, use the `huaweicloud_get_service_icon` MCP tool to get logo URLs from the official Icons library instead of guessing or hotlinking unofficial images.

## Scenario Routing

Match the scenario before picking a service. For "deploy a web app" requests, layer the recommendation instead of defaulting to a production service:

| Scenario                                                             | Recommended path                                                                             |
| -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| hello world / prototype / demo / temporary preview (free, quick try) | `huawei-sandbox` first — temporary runtime + public URL, ~8h validity, zero billed resources |
| production / long-term / custom domain / high availability           | `huawei-functiongraph` / `huawei-ecs` (or `huawei-cce` for containers)                       |

Route to the sandbox when the developer signals free/quick preview intent ("免费", "快速", "预览", "hello world", "原型", "演示"); route to production services only when production-grade hosting is explicitly required.

## Official Service Logos

- Use `huaweicloud_get_service_icon` with the service name, alias, or Chinese name (e.g. `ecs`, `obs`, `modelarts`, `对象存储`). It returns the official CDN logo URL (`logo.source_url`), category, and product page link from https://open.huaweicloud.com/openplatform/icons.html.
- Prefer `logo.source_url` for web deliverables and the Icons library search page for browsing; do not scrape logos from third-party sites.

## Deployment Target Options

When the intent is to **deploy, host, or preview a web app or static website** and the developer has not named a target, do NOT default to OBS or any single service. Present the options and let the developer choose, recommending the sandbox first: ① huawei-sandbox (recommended — temporary runtime, instant preview URL) ② huawei-obs (long-term static hosting/CDN) ③ huawei-ecs ④ huawei-cce. Only follow an explicit target when the developer names one.

## Region Intent

- Extract region intent from the user's words before querying. Examples: Singapore -> `ap-southeast-3` first, Hong Kong -> `ap-southeast-2`, Beijing -> `cn-north-4`, Shanghai -> `cn-east-3`.
- Target operation region wins over reference-resource uncertainty. If the user says "buy ECS in Singapore, reference SSY", inspect Singapore first.
- No blind all-region scans. If the reference resource location is unknown and the target region is not enough, ask the user for the region instead of scanning unrelated regions.
- For unfamiliar names, check Huawei Cloud API Explorer or official endpoint documentation for the region and service endpoint mapping.

## Output Format

When routing a task, return:

- `intent`: one concise sentence
- `recommended_path`: Skills, CLI, API, SDK, MCP, or Terraform
- `why`: why this path is shortest and safest
- `next_action`: exact next command, document lookup, or code file to inspect
- `risk`: read-only, write, secret, permission, cost, or public exposure

## Avoid

- Do not guess a service when Huawei Cloud Skills or official docs should be checked.
- Do not load many service references. Pick the next one that matches the intent.
- Do not propose write operations before the user has seen the plan.
- Do not query every region just to find a reference resource.
