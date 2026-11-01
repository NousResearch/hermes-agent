# DECISION-RECORD — DNS 污染检测增强 (PR #81883 迭代)

日期: 2026-08-13
作者: x7peeps (鲸)
PR: NousResearch/hermes-agent#81883 (feat/computer-use-net-fallback 分支迭代)

## 1. Background

`net_download.py`（#81883）为 Hermes 的 cua-driver 安装器等场景提供
proxy-aware + mirror-fallback 的下载策略。在中国大陆网络环境下，除 GitHub
类域名外，**HuggingFace 系域名（huggingface.co / hf-mirror.com）存在系统性
DNS 污染**：系统 DNS、公共 DNS、甚至多个 DoH 服务返回的 A 记录都是垃圾 IP
（实测解析到 Dropbox 162.125.x、Verizon 128.242.x、日本 NTT 160.16.x），
导致 curl 直连必然失败（`Could not resolve host` / 连接超时）。

现有 mirror fallback 只覆盖 GitHub URL（ghfast.top / gh-proxy.com 前缀包装），
对 HF 域名返回空列表——HF 生态（模型元数据、weights、安装脚本）在国内
网络下完全不可下载。

## 2. Existing solutions surveyed

| 方案 | 现状 | 不解决什么 |
|---|---|---|
| 显式代理 (HTTPS_PROXY) | 已有 (explicit_proxy) | 用户 Surge 规则对 HF 域名是 DIRECT（被墙超时），代理帮不上 |
| 系统代理 (scutil --proxy) | 已有 (_macos_system_proxy) | 同上 |
| GitHub 镜像 (ghfast.top/gh-proxy.com) | 已有 (mirror_candidates) | 只覆盖 github.com / raw.githubusercontent.com，对 HF URL 返回 [] |
| hf-mirror.com 文件镜像 | 未接入 | 2026 起只镜像 `resolve/` 文件下载；**/api 与 /models 页面 308 跳回原站**，不能做 API 数据源；且其自身域名也被 DNS 污染 |
| 本机 hosts 手工改 | 用户手工 | 不可自动化、IP 会变、每台机器都要配 |

## 3. Problem decomposition

| 需求 | 显式代理 | 系统代理 | GitHub 镜像 | hosts | **DoH + --resolve** |
|---|---|---|---|---|---|
| 覆盖 HF 域名 | ✗ (规则 DIRECT) | ✗ | ✗ | △ (手工) | **✓** |
| 自动化、零用户配置 | ✓ | ✓ | ✓ | ✗ | **✓** |
| 供应链安全（内容仍来自官方） | ✓ | ✓ | ✗ (第三方) | ✓ | **✓** |
| 对 executed 内容安全 | ✓ | ✓ | ✗ | ✓ | **✓** |
| 国内直连可用 | △ | △ | ✓ | ✓ | **✓** (doh.pub 国内可达) |

## 4. Our approach

在 `curl_download` 失败路径上追加 **DNS 污染 fallback**：

1. 官方 URL 直连失败（保持现有逻辑：proxy → direct）。
2. 判断是否值得尝试 DoH：目标 host 在已知污染域名集合
   (`huggingface.co`, `hf-mirror.com`) **或** 失败信息匹配 DNS/连接错误特征
   （`could not resolve host` / `failed to connect` / `connection refused` …）。
3. 用腾讯 DNSPod DNS-over-HTTPS（`https://doh.pub/dns-query`，国内直连可达）
   查询目标域名的 A 记录，拿到真实 IP 列表。
4. 对每个 IP 用 `curl --resolve <host>:<port>:<ip>` 重试官方 URL。

**明确不做**：
- 不改动成功路径：直连成功时 DoH 零开销、零查询。
- 不引入第三方内容源：`--resolve` 只替换"连接的 IP"，**TLS 证书仍校验
  hostname、请求的 URL 不变**，内容 100% 来自官方服务器。因此对
  `content_class="executed"` 同样安全——与 mirror fallback（第三方内容）
  有本质区别，后者维持 opt-in + executed 永久禁用。
- 不解析失败掩盖原始错误：DoH 查询失败 / IP 重试全失败时返回原始 detail。
- 不注入代理到 DoH 查询：doh.pub 国内直连稳定，避免代理规则干扰 DNS 解析。

## 5. Decision rationale

- **为什么 DoH 而不是 hosts / 其他 DNS**：DoH 是标准协议（RFC 8484），
  doh.pub 是国内可直连的权威解析服务，返回未污染 A 记录；hosts 无法自动化
  且 IP 漂移。实测 doh.pub 对 huggingface.co 返回 CloudFront 真实 IP
  （3.164.110.x），对 hf-mirror.com 返回其真实地址。
- **为什么 DNSPod 而不是 Cloudflare/Google DoH**：后者在国内被墙
  （实测 cloudflare-dns.com / dns.google 连接超时），doh.pub 直连 200。
- **为什么失败才触发**：保持"直连成功零影响"的基线；DoH 查询只在官方
  失败后发生，代价可控（5s 超时兜底）。
- **为什么 --resolve 而非 --resolve-as / SNI 直连**：`--resolve` 是 curl
  原生、跨平台、同时适用于 http/https；只影响本命令的目标解析。

## 6. Capability matrix

| 维度 | 现有 (proxy + mirror) | + DNS 污染 fallback |
|---|---|---|
| GitHub 下载韧性 | ✓ (mirror) | ✓ (不变) |
| HF / 任意域名下载韧性 | ✗ (CN 下必失败) | ✓ (DoH 真实 IP 直连) |
| executed 内容安全 | ✓ (mirror 禁用) | ✓ (等价官方直连) |
| 用户配置负担 | 零 | 零（自动） |
| 失败信息可诊断性 | detail | detail + 是否尝试 DoH fallback 的说明 |

## 7. Tests

- `resolve_dns_doh` 单测：解析 A 记录 / 过滤非 A / curl 失败返回 [] / 超时返回 [] / 非法 JSON 返回 [] / 不注入代理。
- `curl_download` DNS fallback：直连失败 → DoH 成功 → `--resolve` 重试成功；
  直连成功不触发 DoH；非污染域名 + 非 DNS 错误不触发；DoH 失败保留原始错误；`dns_fallback=False` 完全禁用。
- `fetch_with_fallback` 集成：official 失败 → DNS fallback 成功（在 mirror 之前）。
