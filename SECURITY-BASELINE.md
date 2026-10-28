# SECURITY-BASELINE — net_download DNS 污染 fallback

日期: 2026-08-13
维护者: x7peeps (鲸)
适用范围: `hermes_cli/net_download.py`

## 默认配置基线（测试基准）

| 项 | 默认值 | 说明 |
|---|---|---|
| `curl_download(dns_fallback=...)` | `True` | 官方失败后自动 DoH 解析 + `--resolve` 重试 |
| `fetch_with_fallback(dns_fallback=...)` | `True` | 透传给 `curl_download`，在 mirror 之前尝试 |
| `resolve_dns_doh(timeout=...)` | `5` | DoH 查询超时；失败返回 `[]`（绝不抛异常） |
| DoH 端点 | `https://doh.pub/dns-query` | 腾讯 DNSPod，国内直连可达 |
| 已知污染域名 | `huggingface.co`, `hf-mirror.com` | 失败必触发 DoH fallback |
| 其他域名触发条件 | stderr 含 DNS/连接特征 | `could not resolve host` / `failed to connect` / `connection refused` / `no route to host` / `name or service not known` / `temporary failure in name resolution` |
| IP 重试上限 | 4 个 | 顺序尝试，全失败则返回原始错误 |
| `content_class="executed"` | mirror 永久禁用 | **DNS fallback 不受此限制**（等价官方直连，非第三方） |

## 安全不变式（每次改动必须保持）

1. **内容源不变**：fallback 只换 IP，URL / TLS hostname 校验不变 → 字节
   100% 来自官方服务器。任何"引入第三方内容源"的改动都是违规。
2. **成功路径零查询**：直连成功时不得发起 DoH 查询（0 开销、0 延迟）。
3. **失败不掩盖**：DoH 失败 / IP 全失败 → 返回原始 `detail`，可附加
   fallback 尝试信息但不得替换根因。
4. **无持久化副作用**：`--resolve` 仅作用于单次 curl，不写 hosts、
   不改系统 DNS、不缓存 IP（IP 漂移风险通过每次实时查询规避）。
5. **有限重试**：IP 上限 4、DoH 5s 超时、整体受 `fetch_with_fallback`
   调用边界约束，无死循环路径。

## 回归验证命令

```bash
# worktree 无 venv，用主仓库 venv 解释器（pytest 以当前目录为 rootdir）
~/.hermes/hermes-agent/venv/bin/python3 -m pytest tests/hermes_cli/test_net_download.py -x -q

# 全量相关单测（防 CI 环境探针问题）
~/.hermes/hermes-agent/venv/bin/python3 -m pytest tests/hermes_cli/test_net_download.py tests/tools/test_cua_installer.py -q
```

## 兜底保护

- **DoH 端点不可达（非 CN 网络）**：`resolve_dns_doh` 返回 `[]` →
  fallback 静默跳过，行为退回纯 proxy + mirror 的现状。**非 CN 用户零影响。**
- **已知污染域名但 IP 全被墙**（如 huggingface.co CloudFront）：
  fallback 失败 → 返回原始错误；不会假装成功或产生误导性成功。
- **调用方需要完全禁用**：`dns_fallback=False`（如内网环境禁止出站 DoH）。
