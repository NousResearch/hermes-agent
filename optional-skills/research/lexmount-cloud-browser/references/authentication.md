# 授权说明

推荐使用浏览器授权：

```text
browser-cli auth login [--client-name "NAME"]
```

`NAME` 为当前 Agent 的展示名称，例如 `--client-name "WorkBuddy"`；省略时默认为 `Agent`。

CLI 在 `127.0.0.1` 绑定随机回调端口，生成 PKCE 校验信息和 state，打开 LexMount 授权页，收到一次性 code 后换取凭据，并保存到：

```text
~/.config/lexmount/browser-cli/credentials.json
```

Unix 上文件权限为 `0600`。CLI 对 JSON 输出中的 API Key 做脱敏；不要另外读取并回显凭据文件。

## 已有授权与凭据位置

安装或升级技能不会清空授权。同一系统用户运行的 CLI 可以复用已有凭据，因此未再次出现登录页不等于匿名访问。

- 受管环境的 `LEXMOUNT_API_KEY` 与 `LEXMOUNT_PROJECT_ID` 优先于文件中的对应配置；应成对配置，避免混用来源。
- 凭据文件默认位于当前系统用户的 `~/.config/lexmount/browser-cli/credentials.json`；设置 `LEXMOUNT_BROWSER_CREDENTIALS_FILE` 可以指定独立位置。Windows 的 `~` 表示当前用户主目录。
- `auth status` 的 `data.source` 和 `data.valid` 用于查看本地凭据来源及格式状态；不代表服务端已接受凭据，仍需运行 `doctor`。诊断时只报告状态，不读取或展示密钥。
- 网站 Cookie 属于浏览器 Context，和上述服务访问凭据是两回事。

## 本机与远端环境

此版本的交互授权依赖浏览器回调能够到达 CLI 所在机器。远端沙箱的 `127.0.0.1` 不等于用户电脑；不能假定用户在本机打开授权页就能完成远端 CLI 的回调。

CLI `1.1.15` 的 `--no-open` 仅抑制自动打开浏览器，仍等待回调，也不会输出授权链接；它不是完整的无界面授权方案。若环境无法完成回调，应说明授权尚未完成，并使用环境管理员支持的授权方式；不要重复启动登录或在未授权时声称浏览成功。

受管环境也可通过 SDK 支持的 `LEXMOUNT_API_KEY`、`LEXMOUNT_PROJECT_ID` 提供凭据，另有可选的 `LEXMOUNT_BASE_URL`、`LEXMOUNT_REGION`。由环境管理方安全配置，不让用户在 Agent 聊天中粘贴密钥，不把凭据写入技能包。

`browser-cli auth logout` 只移除本地凭据文件；环境变量由 CLI 外部管理。授权后通过 `browser-cli doctor` 的 `data.ready_for_browser_actions` 核验是否可继续操作。
