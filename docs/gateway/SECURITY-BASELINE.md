# Federation Cluster Security Baseline

> **Independence Audit Required** — every section in this document must be
> independently verified before merge. The PR includes a signed
> `[SECURITY-AUDIT-CHECKLIST]` comment listing each item's status.

## 1. Threat Model

### 1.1 Assets to Protect

| Asset | Sensitivity | Location |
|---|---|---|
| Cluster auth tokens | **CRITICAL** | Keychain/libsecret + env |
| Node private keys | **CRITICAL** | HSM/Keychain |
| Task payloads (user input/output) | **HIGH** | iCloud + remote nodes |
| Task state metadata | **MEDIUM** | iCloud + remote nodes |
| Node capability info | **LOW** | Local, can be public |
| Cluster topology | **LOW** | Local, can be leaked |

### 1.2 Adversary Model

| Adversary | Capabilities | Defense |
|---|---|---|
| **Eavesdropper** | Read network traffic | TLS 1.3 mandatory |
| **Active MitM** | Inject packets | mTLS + cert pinning |
| **Compromised Node** | Full access to local tokens/keys | Trust 评级 + 隔离 |
| **Cloud Sync Attacker** | Read iCloud Drive | Encryption + HMAC |
| **Protocol Reverse-Engineer** | Static analysis of FedMessage | Version + 协议混淆 |
| **Inside Attacker** | Decompile client | Code obfuscation + integrity checks |
| **0day Exploit** | Unknown CVE in TLS/HMAC/crypto | Defense-in-depth + 多层 |

### 1.3 0day Assumptions

We assume any single cryptographic primitive may be broken:

| Primitive | 0day Scenario | Fallback |
|---|---|---|
| HMAC-SHA256 | Hash collision attack | Migrate to HMAC-SHA3-256 |
| Ed25519 | Discrete log broken | Migrate to Dilithium (post-quantum) |
| TLS 1.3 | Bleichenbacher-style attack | QUIC + cert pinning |
| AES-256 | Side-channel attack | Constant-time impl + noise |
| SQLite | Read-only file access | Encrypted database |
| OpenAI compatible protocol | Reverse-engineered | Schema versioning + custom fields |

## 2. Security Pillar Checklist

### 2.1 Pillar 1: Identity & Authentication

- [ ] **REQ-1.1** Ed25519 公钥每个节点独立生成
- [ ] **REQ-1.2** 节点加入时双向 challenge-response
- [ ] **REQ-1.3** 私钥永不离开本地 (Keychain/Hardware)
- [ ] **REQ-1.4** 节点离开时主动撤销
- [ ] **REQ-1.5** 节点死亡 24h 后自动清理
- [ ] **REQ-1.6** User 通知所有节点 join/leave
- [ ] **REQ-1.7** 节点列表变化触发 review（可选）

### 2.2 Pillar 2: Transport Security

- [ ] **REQ-2.1** TLS 1.3 强制（`require_tls: true` 默认）
- [ ] **REQ-2.2** 拒绝 wss:// 不允许的明文降级
- [ ] **REQ-2.3** 局域网可选 `allow_insecure: true` + 警告
- [ ] **REQ-2.4** HTTP API: Bearer Token 鉴权
- [ ] **REQ-2.5** HTTPS 证书有效期检查
- [ ] **REQ-2.6** 可选证书钉扎防止 CA compromise
- [ ] **REQ-2.7** iCloud 文件加密 (AES-256-GCM)

### 2.3 Pillar 3: Data Integrity

- [ ] **REQ-3.1** Task state HMAC-SHA256 签名
- [ ] **REQ-3.2** 关键字段 (task_id, owner_id, step) 单独签名
- [ ] **REQ-3.3** 接收方强制验证签名（启动即开启）
- [ ] **REQ-3.4** 失败签名计数, 异常阈值报警
- [ ] **REQ-3.5** 篡改检测 = 拒绝 + 报警
- [ ] **REQ-3.6** 共享文件 checksum 强校验
- [ ] **REQ-3.7** Snapshot diff 检测异常修改

### 2.4 Pillar 4: Authorization

- [ ] **REQ-4.1** Trust 评级: `unknown` / `verified` / `trusted` / `admin`
- [ ] **REQ-4.2** Task 敏感度: `low` / `medium` / `high` / `critical`
- [ ] **REQ-4.3** Sensitive task 路由限制 (high → trusted+, critical → admin)
- [ ] **REQ-4.4** 关键操作 (delete task / force reassign) 仅 admin
- [ ] **REQ-4.5** User 显式 approve trust 升级
- [ ] **REQ-4.6** Read-only 节点不能执行 task
- [ ] **REQ-4.7** Cluster admin 操作双重鉴权

### 2.5 Pillar 5: Resilience

- [ ] **REQ-5.1** 死亡检测至少连续 3 次失败
- [ ] **REQ-5.2** 探活 endpoint 限速 60/min per IP
- [ ] **REQ-5.3** 所有 cluster endpoint 限速
- [ ] **REQ-5.4** 异常事件触发 user notification
- [ ] **REQ-5.5** 节点不可达 fallback graceful
- [ ] **REQ-5.6** 状态同步断网后本地缓存 + 恢复
- [ ] **REQ-5.7** Cluster 状态本地备份 + 远程同步

### 2.6 Pillar 6: Audit & Logging

- [ ] **REQ-6.1** 所有节点 join/leave 记录
- [ ] **REQ-6.2** 所有 task 抢领记录
- [ ] **REQ-6.3** User 审批 ask 时记录决策
- [ ] **REQ-6.4** 异常事件报警 (签名失败, 死亡门槛触发)
- [ ] **REQ-6.5** 审计日志加密 + 防篡改
- [ ] **REQ-6.6** 审计日志 90 天保留
- [ ] **REQ-6.7** Token 永远不进日志

### 2.7 Pillar 7: Privacy

- [ ] **REQ-7.1** Task payload 默认不广播
- [ ] **REQ-7.2** Task 接班时上下文按需脱敏
- [ ] **REQ-7.3** 节点 capability 公开, 任务内容不公开
- [ ] **REQ-7.4** User 个人数据不允许跨节点
- [ ] **REQ-7.5** 审计日志脱敏 (无 PII)
- [ ] **REQ-7.6** 死亡节点 24h 后清理相关 task state
- [ ] **REQ-7.7** Task 完成后上下文可清除

### 2.8 Pillar 8: Operational Security

- [ ] **REQ-8.1** Token 存储用 Keychain (macOS) / libsecret (Linux)
- [ ] **REQ-8.2** Token 永远不从环境变量读取到日志
- [ ] **REQ-8.3** Debug 模式显式开启
- [ ] **REQ-8.4** Release 模式关闭所有调试 endpoint
- [ ] **REQ-8.5** Bug report 不包含 token
- [ ] **REQ-8.6** Updates 经过签名验证
- [ ] **REQ-8.7** 配置文件权限 0600

## 3. Mandatory Security Tests

### 3.1 Unit Tests (each REQ above)

```python
# tests/gateway/cluster/test_security_identity.py
def test_node_join_requires_valid_signature(): ...
def test_node_join_rejects_revoked_key(): ...
def test_node_leave_invalidates_session(): ...

# tests/gateway/cluster/test_security_transport.py
def test_tls_required_by_default(): ...
def test_plaintext_http_rejected(): ...
def test_bearer_token_required(): ...

# tests/gateway/cluster/test_security_integrity.py
def test_task_state_signature_verified(): ...
def test_tampered_signature_rejected(): ...
def test_signature_failure_threshold_alerts(): ...

# tests/gateway/cluster/test_security_authorization.py
def test_high_sensitivity_task_requires_trusted(): ...
def test_critical_task_requires_admin(): ...
def test_readonly_node_cannot_claim_task(): ...

# tests/gateway/cluster/test_security_resilience.py
def test_death_detection_requires_3_failures(): ...
def test_rate_limit_blocks_excessive_heartbeat(): ...
def test_fallback_when_peer_unreachable(): ...

# tests/gateway/cluster/test_security_audit.py
def test_node_join_logged(): ...
def test_task_claim_logged(): ...
def test_token_never_logged(): ...

# tests/gateway/cluster/test_security_privacy.py
def test_task_payload_not_broadcast_by_default(): ...
def test_audit_log_no_pii(): ...
```

### 3.2 Penetration Tests

- [ ] **PEN-1** SPDX-1: 节点冒充攻击
- [ ] **PEN-2** 篡改 task state 攻击
- [ ] **PEN-3** 重放攻击 (replay) - 旧消息重用
- [ ] **PEN-4** DoS 攻击 - 探活风暴
- [ ] **PEN-5** Token 泄漏模拟
- [ ] **PEN-6** iCloud 访问权限攻击
- [ ] **PEN-7** Transport 降级攻击
- [ ] **PEN-8** Algorithm 0day 模拟

### 3.3 Fuzz Tests

- [ ] FedMessage parser fuzz
- [ ] Task state 序列化 fuzz
- [ ] Cluster endpoint input fuzz
- [ ] HTTPS handshake fuzz
- [ ] WebSocket frame fuzz

## 4. Security Audit Checklist (Pre-merge)

Before merge, this checklist must be 100% green:

```
□ Pillar 1: Identity ........................ [OK]
□ Pillar 2: Transport ........................ [OK]
□ Pillar 3: Integrity ........................ [OK]
□ Pillar 4: Authorization ................... [OK]
□ Pillar 5: Resilience ...................... [OK]
□ Pillar 6: Audit ............................ [OK]
□ Pillar 7: Privacy .......................... [OK]
□ Pillar 8: Operational ...................... [OK]

□ All unit tests pass (target: 200+)
□ All pen tests pass
□ All fuzz tests pass
□ 0day fallback documented
□ Threat model updated
□ Audit log encrypted
□ Token in Keychain only
□ TLS 1.3 enforced
□ HMAC signature enforced
□ Trust 评级 enforced
□ Rate limiting enforced
□ Death detection enforced
□ Audit log enforced
□ Privacy enforced
□ OpSec enforced
```

## 5. Incident Response Plan

### 5.1 Detected Compromise

```
1. 立即 revoke 节点 (强制)
2. 通知所有 trusted 节点
3. 触发新 trust 评级
4. 审计日志分析
5. Token 轮换
6. User 决定是否继续
```

### 5.2 0day Alerted

```
1. 评估风险范围
2. 临时 fallback 到 known-secure 算法
3. 隔离受影响的 transport
4. Update security baseline
5. 通知用户
```

### 5.3 Reporting

- Security issues: `SECURITY.md` policy
- Disclosure: 90 天 responsible disclosure
- CVE assignment: GitHub Security Advisory

## 6. Update Schedule

- **每月**: 依赖更新 (`safety check`)
- **每季**: 手动审计 + 0day 扫描
- **每半年**: 威胁模型重审
- **每年**: 第三方安全审计
