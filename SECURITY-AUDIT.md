# SECURITY-AUDIT — DNS 污染 fallback (PR #81883 迭代)

日期: 2026-08-13
审计人: x7peeps (鲸)
审计对象: `hermes_cli/net_download.py` 新增的 `resolve_dns_doh` /
`curl_download(dns_fallback=True)` 路径

## 审计结论

**DNS 污染 fallback 不引入新的攻击面。** 它只替换"连接的目标 IP"，
请求的 URL、TLS hostname 校验、内容源全部保持官方不变。供应链安全
等价于官方直连。以下为逐项证据链。

## 1. 威胁模型

| 威胁 | 分析 |
|---|---|
| 第三方内容注入（供应链攻击） | **不适用**。fallback 连接的是官方服务器（DoH 返回的真实 A 记录），URL 不变，TLS 证书仍校验 hostname。与 mirror fallback（第三方前缀包装）本质不同。 |
| DoH 响应伪造 | DNSPod DoH 走 HTTPS（TLS 加密），响应无法被中间人改写。即便 DoH 返回恶意 IP，TLS 证书校验仍会失败（证书不含目标 hostname），连接被 curl 拒绝。**纵深防御：--resolve 只改 IP，不改 TLS 校验。** |
| 数据外泄 | DoH 查询内容（域名）经 TLS 加密；查询目标是公开域名，非敏感信息。 |
| 错误掩盖 | DoH 失败时返回**原始错误 detail**（不替换、不吞掉），fallback 尝试信息附加在 detail 中。 |
| 重试放大 / 循环 | 每个 IP 至多重试一次，IP 列表上限 4；整体在 `fetch_with_fallback` 内有限。DoH 查询 5s 超时兜底。 |

## 2. 安全契约（与 #81883 一致）

- **`content_class="executed"`（默认）**：mirror 永久禁用（现有逻辑）。
  DNS fallback **不触碰此契约**——它不属于"第三方镜像"，而是"官方 IP 直连"，
  对 executed 内容同样允许（等价官方）。
- **`content_class="data"`**：mirror 保持 opt-in；DNS fallback 默认开启
  （对数据内容无额外风险）。
- **直连成功路径零影响**：DoH 查询只在官方失败后触发，且
  `dns_fallback=False` 可完全禁用（调用方可按需关闭）。

## 3. 代码级审计点

| 审计点 | 结论 |
|---|---|
| 无 `shell=True`，全部 argv 列表 | ✓ 与现有代码一致 |
| `--resolve` 仅限本命令，不修改系统 DNS/hosts | ✓ 无持久化副作用 |
| DoH 查询不注入代理 env | ✓ 避免代理规则污染 DNS 解析 |
| IP 白名单过滤（正则 `\d+\.\d+\.\d+\.\d+`，type=1 A 记录） | ✓ 拒绝非 IPv4/恶意 JSON 字段 |
| 失败返回原始 detail | ✓ 不掩盖根因 |
| 端口推导（https→443 / http→80 / 显式端口） | ✓ 防 `--resolve` 格式错误 |

## 4. 验证记录（2026-08-13 本机实测）

| 场景 | 结果 |
|---|---|
| `doh.pub/dns-query?name=huggingface.co` | 200，返回 CloudFront 真实 IP（3.164.110.x），**证明系统 DNS 确实被污染**（系统解析返回 Verizon 128.242.x） |
| `doh.pub/dns-query?name=hf-mirror.com` | 200，返回 160.16.86.14（其真实边缘地址） |
| `curl --resolve hf-mirror.com:443:<cloudflare-ip>` 直连 | 可达（308/200），**证明 --resolve 方案有效** |
| `curl --resolve huggingface.co:443:<cloudfront-ip>` 直连 | 000（IP 被墙），**证明 fallback 失败时正确保留错误**，不会假装成功 |
| `curl -x http://127.0.0.1:6152`（Surge 代理）访问 HF | 超时（规则 DIRECT），**证明代理无法替代 DoH fallback** |

## 5. 测试覆盖（tests/hermes_cli/test_net_download.py 新增）

- DoH 解析成功/失败/超时/非法 JSON/非 A 记录过滤
- 直连成功不触发 DoH（回归保护）
- 直连失败 → DoH → `--resolve` 重试成功（核心路径）
- 非污染域名 + 非 DNS 错误不触发（防误触发）
- DoH 失败保留原始错误（不掩盖）
- `dns_fallback=False` 完全禁用
