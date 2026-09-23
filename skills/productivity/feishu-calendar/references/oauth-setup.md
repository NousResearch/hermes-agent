# 飞书 OAuth 配置指南

## 一、核心概念

飞书日历 API 有两种身份验证方式：

| 身份 | Token | 访问的日历 | 适用场景 |
|------|-------|-----------|---------|
| 应用身份 | tenant_access_token | 机器人日历 | 多用户应用 |
| 用户身份 | user_access_token | 用户个人日历 | 个人助理 |

**个人日程管理推荐使用用户身份（user_access_token）**，这样事件会直接出现在你的飞书日历上。

## 二、飞书开放平台配置

### 2.1 创建应用

1. 访问 https://open.feishu.cn/app
2. 点击"创建应用"
3. 选择"自定义应用"
4. 填写应用名称（如"小淡助理"）
5. 创建后记录 `App ID` 和 `App Secret`

### 2.2 开通日历权限

1. 进入应用 → **权限管理**
2. 搜索并添加以下权限：
   - `calendar:calendar:readonly` - 获取日历、日程及忙闲信息
   - `calendar:calendar` - 更新日历及日程信息
3. 点击"申请权限"（如需要管理员审批）

### 2.3 配置重定向 URL

1. 进入应用 → **安全设置**
2. 在"重定向 URL"处添加：
   ```
   http://127.0.0.1:18080/callback
   ```
3. 保存

**注意**：
- 必须是 `http://` 不是 `https://`
- 必须包含 `/callback` 路径
- `127.0.0.1` 和 `localhost` 是不同的，必须用 `127.0.0.1`

### 2.4 配置环境变量

编辑 `~/.hermes/.env` 文件，添加：

```bash
FEISHU_APP_ID=cli_xxxxxxxxxxxxx
FEISHU_APP_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
FEISHU_REDIRECT_URI=http://127.0.0.1:18080/callback
```

## 三、OAuth 授权流程

### 3.1 生成授权链接

```bash
python3 feishu_oauth.py generate_link
```

输出示例：
```
🔐 飞书 OAuth 授权链接

================================================================================
https://open.feishu.cn/open-apis/authen/v1/authorize?app_id=cli_xxx&redirect_uri=http%3A%2F%2F127.0.0.1%3A18080%2Fcallback&state=hermes_oauth_xxx&response_type=code&scope=calendar%3Acalendar
================================================================================
```

### 3.2 浏览器授权

1. 复制链接到浏览器打开
2. 登录飞书账号
3. 查看申请的权限（应包含日历相关权限）
4. 点击"同意授权"

### 3.3 获取授权码

授权成功后，浏览器会跳转到：
```
http://127.0.0.1:18080/callback?code=HERM_xxxxxxxxxxxx&state=hermes_oauth_xxx
```

**复制 `code=` 后面的值**（`HERM_xxxxxxxxxxxx`）

> 注意：页面可能显示"无法连接"，这是正常的（因为本地没有运行服务器）。关键是 URL 中的 code 参数。

### 3.4 换取 Token

```bash
python3 feishu_oauth.py exchange <code>
```

输出示例：
```
🔄 正在换取 token...

✅ 授权成功！
   access_token 有效期：7200 秒（约 2 小时）
   refresh_token 有效期：2592000 秒（约 30 天）

后续创建日历事件将自动使用并刷新此 token
```

### 3.5 验证授权

```bash
python3 setup.py --check
```

应显示：
```
2. OAuth 授权状态：AUTHENTICATED
   已授权（access_token 剩余 119 分钟）

✅ 配置完成，可以使用
```

## 四、Token 管理

### 4.1 Token 有效期

| Token | 有效期 | 刷新方式 |
|-------|--------|---------|
| access_token | 2 小时 | 自动刷新（过期前 5 分钟） |
| refresh_token | 30 天 | 自动刷新（每次刷新获得新的） |

### 4.2 自动刷新机制

每次调用日历 API 时，脚本会：

1. 检查 access_token 是否快过期（<5 分钟）
2. 如果需要刷新，用 refresh_token 获取新的 access_token
3. 同时获得新的 refresh_token（重新计算 30 天）

**理论上可以一直自动续期，无需手动干预。**

### 4.3 查看 Token 状态

```bash
python3 feishu_oauth.py status
```

输出示例：
```
📊 OAuth Token 状态

access_token:
   剩余有效期：115 分钟
   过期时间：2026-04-15 23:00:00
   状态：✅ 有效

refresh_token:
   剩余有效期：29 天
   过期时间：2026-05-14 22:00:00
   状态：✅ 有效
```

### 4.4 手动刷新

```bash
python3 feishu_oauth.py refresh
```

### 4.5 重新授权

如果 refresh_token 也过期了（30 天后），需要重新授权：

```bash
python3 feishu_oauth.py generate_link
# 浏览器授权后
python3 feishu_oauth.py exchange <code>
```

## 五、常见问题

### Q1: 重定向 URL 有误（错误码 20029）

**原因**：飞书开放平台配置的重定向 URL 与授权链接中的不匹配

**解决**：
1. 检查飞书开放平台 → 安全设置 → 重定向 URL
2. 确保精确匹配：`http://127.0.0.1:18080/callback`
3. 常见错误：
   - ❌ `http://localhost:18080/callback`
   - ❌ `http://127.0.0.1:18080`
   - ❌ `https://127.0.0.1:18080/callback`

### Q2: 权限不足（错误码 99991679）

**原因**：应用没有申请日历权限，或授权时没有包含日历 scope

**解决**：
1. 在飞书开放平台 → 权限管理，添加日历权限
2. 重新运行授权流程（旧的 token 需要作废）

### Q3: Token 刷新失败

**原因**：refresh_token 过期或被撤销

**解决**：重新运行授权流程

### Q4: 如何撤销授权？

**方法 1**：在飞书开放平台的应用管理中撤销授权

**方法 2**：删除 token 文件
```bash
rm ~/.hermes/.feishu_user_token.json
```

## 六、安全说明

- Token 文件权限自动设置为 600（仅所有者可读）
- 不要将 token 文件上传到 Git 或分享给他人
- 本技能仅申请日历权限，不访问消息、联系人等其他数据
- 定期查看 token 状态：`python3 feishu_oauth.py status`
