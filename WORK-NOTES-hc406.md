# WORK-NOTES · hc-406 — v0.18 skill 全集重分级 + China seed 统装

Seat: S6 (P1). Base: `origin/main` @ `9790955aa` (0.16.1 批次,#48 plugins seed 已在)。
只开发 + 推分支 + 开 PR + CI 绿;**不合并、不出包、不 bump pin**(发版批次统一)。

## 背景

`SEED_DISABLED_SKILLS`(fork `apps/desktop/electron/apex-managed.cjs`)在 v0.17 按 73 skill / 49 关拍定。
v0.18 全集实扫(fork worktree)= **173**:`skills/` **72 bundled** + `optional-skills/` **101 optional**
(任务书说 162 = 66+96,数字对不上,以实扫为准)。25 个新 bundled skill 默认全开、未分级。

## 关键架构事实

- **bundled(72)** 安装时自动拷进 `~/.hermes/skills/` 并 ACTIVE,除非进 `skills.disabled` → **seed 只管这 72。**
- **optional(101)** 随仓库发布但**安装时不拷贝**,`hermes skills install` 按需装(`optional-skills/DESCRIPTION.md`)
  → **默认 off,永不进 seed。** 三项已拍决策据此落地:
  - `shopify`/`siyuan`「默认开」= 二者是 optional(不被抑制,用户 install 即用),seed 无需动。
  - `notion` 是 bundled、「灰度开」= 留在 seed OFF、用户按需开(C 级跨境 SaaS)。
- 匹配键 = SKILL.md frontmatter `name:`(缺省回退目录名),**大小写敏感**
  (`agent/skill_utils.py:643` + `_normalize_string_set` 只 strip 不 lower)。
  4 个 name ≠ 目录:serving-llms-vllm / evaluating-llms-harness / segment-anything-model / audiocraft-audio-generation。

## 分级结果(72 bundled)

| 级 | 数 | 处置 |
|---|---|---|
| A 本地纯执行 | 20 | ON |
| A* 矩阵 A 但 v0.17 保留 OFF(创意小众) | 5 | OFF |
| B 国内可达服务 | 2 | ON |
| C 需镜像/国产源 | 18 | OFF |
| D 墙外/geo/竞品 | 7 | OFF |
| DEV-B 能力足但 dev 小众,产品聚焦关 | 19 | OFF |

**有效 22 ON / 50 OFF。`?` 未决 = 0**(全部据 SKILL.md 实证 + 矩阵原则)。
全 72 逐 skill 表见 hermes-cloud `docs/DESKTOP-CHINA-SKILL-MATRIX.md`(单独 cloud 文档 PR)。

### v0.18 delta vs v0.17

- **+3 关**(新 bundled):`huggingface-hub`(C,HF 被墙)、`maps`(C,GCJ-02 偏移)、`plan`(DEV-B)。
- **−2 删**(死 orphan,已非 bundled):`kanban-orchestrator`、`kanban-worker`
  (后继 = opt-in `optional-skills/creative/kanban-video-orchestrator`)。
- **22 新 bundled A/B 保持 ON**:computer-use / apple\* / powerpoint / obsidian / ocr-and-documents /
  nano-pdf / baoyu-infographic / architecture-diagram / excalidraw / sketch / claude-design /
  popular-web-designs / ascii-art / ascii-video / humanizer / llm-wiki / blogwatcher / petdex / yuanbao。

## 改动清单(全在 fork)

1. **`apps/desktop/electron/apex-managed.cjs`**
   - `SEED_DISABLED_SKILLS`:49→**50**,按 D / C / DEV-B 重分组 + 逐组依据注释。
   - 新增 `ensureSkillsDisabledYaml()`:skills.disabled 的 **add-only 并集愈合器**(= 存量机器 v0.17→v0.18
     升级路径:老 config 的 49 名单会漏掉 3 个新关项,靠这个 union 补齐;绝不移除、绝不重加用户已开项)。
   - 把 `ensurePluginsEnabledYaml` 与新 skills 愈合器的公共行手术抽成 `ensureListBlockYaml()`(单一被测算法,
     plugins 语义 byte 不变)。
   - `TODO(hc408/hc414 seed)` 占位:S2/S3 的搜索/抽取网关键值待其 WORK-NOTES 落地后再统装(**不硬编**猜测 shape)。
2. **`apps/desktop/electron/backend-env.cjs`**
   - `buildDesktopBackendEnv` 注入 `HF_ENDPOINT=https://hf-mirror.com`(add-only,不覆盖父 env 已设值)。
     解 STT 首用拉 ~150MB faster-whisper 模型国内裸网必挂(`tools/transcription_tools.py::_transcribe_local`
     → huggingface_hub 读 `HF_ENDPOINT`)。两处 backend spawn 均 `{...process.env, HERMES_HOME, ...backend.env}`,
     backend.env 覆盖继承,故此注入真吃到。顺带覆盖 marker-pdf OCR / `hf` CLI 的 Hub 下载。
3. **`cli-config.yaml.example`** — `skills.disabled` 同步到新 50 名单(纯 CLI 路径;与 seed byte-for-list 一致)。
4. **测试**:`apex-managed.test.cjs`(count 49→50 + 新/删/A-B 保 ON 断言 + `ensureSkillsDisabledYaml` 全场景),
   `backend-env.test.cjs`(HF 注入 + 不覆盖 override + 空值兜底)。
5. **文档**:hermes-cloud `docs/DESKTOP-CHINA-SKILL-MATRIX.md` 追加 v0.18 全集分级(独立 cloud PR)。

## 重 seed / 看门狗

- **新装**:`seedDefaultModelConfig`(`main.cjs`,config.yaml 缺失才写)→ 直接吃新 50 名单。
- **存量机器**:`rm ~/.apexnodes/config.yaml` → 下次启动 `seedDefaultModelConfig` 重 seed 新名单。✅
- **存量机器不删 config**:`guardConfigYamlProductBlocks`(boot + fs.watch)现调 `ensureSkillsDisabledYaml`
  union 补齐 3 个新关项(= v0.18 升级路径),不碰用户已开项。plugins seed(#48)的 `ensurePluginsEnabledYaml`
  语义完全不变,基于 `9790955aa` 未冲掉。

## 未决 / 阻塞

- **hc408/hc414 seed 键值:S2/S3 的 WORK-NOTES 已落地(在其分支上,未并 main),键值已精确记入
  apex-managed.cjs 注释,但 NOT emit 进 seed** —— 两前提未满足,提前 seed 会让桌面搜/抽调到不存在
  的端点而 break:
  - hc-408(`feat/hc408-relay-search` 未并 cloud main)→ `/api/v1/search/searxng/search` 尚未上 prod。
  - hc-414(§4)明说**公网 Firecrawl URL 未定**(云侧仅绑 127.0.0.1;公网走向是 Kael/PM 决策),
    且桌面 web_extract 是 opt-in(「若桌面暂不开,S6 可跳过」)。
  - 精确键(供上线后激活):`web.search_backend=searxng` + `SEARXNG_URL=https://api.apex-nodes.com/api/v1/search/searxng`
    (末尾**不带** `/search`);`web.extract_backend=firecrawl` + `FIRECRAWL_API_URL=<公网走向待定>`。
    激活方式 = SEED_WEB_GATEWAY 常量 + seedWebGatewayBlockYaml(top-level `web:`,同 seedSkillsBlockYaml)
    + backend-env.cjs 注入 env(与 HF_ENDPOINT 同处)+ guard union(同 ensureSkillsDisabledYaml)。
- 桌面 seed 曾缺 `plugins.enabled`(#48 已补,本 PR 未动)。
- CI:fork Tests 带 paths-ignore,纯 md 改动某些 check 会 pending(cloud 文档 PR 尤甚)。
- **cloud 文档 PR #480 的 `backend` check 失败 = 预存 broken test**(`test_hc204_admin_capability_usage.py`
  引用 `app.routers.admin.collect_agent_media_usage`,该符号在 #469 perf 重构 `bb7d8583` 已移除但测试未更;
  main 自身即红,与本纯 md 改动无关。`web` check 绿)。

## 合并后生效(给 PM)

- 随**下一个桌面发版批次**(bump desktop pin + 出签名包)生效,勿单独 bump。
- 存量机器升级到 v0.18 runtime 后:①最简 = `rm ~/.apexnodes/config.yaml` 重启重 seed;
  ②不删 config 也会被 `guardConfigYamlProductBlocks` union 补上 3 个新关项(A/B 新技能已默认 ON,无需动作)。
