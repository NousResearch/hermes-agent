# Windows 杀毒误报与代码签名

桌面端用户（尤其国内）常反馈：网关启动失败、首次安装就被拦、或 exe 被删。根因几乎都是
**未签名的 PyInstaller frozen runtime 被安全软件误杀**。本文给出从「免费立刻能做」到
「花钱根治」的完整清单。

> 适用对象：runtime（`hermes-agent-cn-runtime-*.exe`，本仓库 `release-runtime.yml` 产出）
> 和桌面安装包（`Hermes-CN-Desktop` 的 NSIS 安装器）。**两个都要处理**。

---

## 1. 为什么会被拦

- **未签名**：没有 Authenticode 数字签名的 exe，Windows SmartScreen 与各家杀毒都按
  「未知发布者」对待，启发式打分偏高。
- **PyInstaller 特征**：单文件(onefile)会自解压到 `%TEMP%` 再执行 —— 正好踩行为监控红线；
  无版本信息资源的 exe 也更可疑。
- **国内更复杂**：360 / 火绒 / 腾讯电脑管家跑**自己的云查杀**，和微软 SmartScreen 是
  **两套独立**的信誉系统。签了 Authenticode 微软会认，但国内三家不一定立刻放行。

---

## 2. 免费、立刻能做（不依赖证书）

本仓库 `release-runtime.yml` 已经落地的免费缓解：

| 措施 | 状态 | 说明 |
|------|------|------|
| `--onedir`（非自解压） | ✅ 已用 | 不再自解压到 TEMP，误报大幅下降 |
| `--noupx`（禁 UPX 壳） | ✅ 已加 | UPX 压缩壳是杀毒重灾区；显式禁用 |
| 版本信息资源 | ✅ 已加 | `scripts/gen_win_version_info.py` 给 exe 盖 CompanyName / ProductName / FileVersion，看起来更像正经软件 |

仍可做（免费）：

- **提交误报申诉**到各家厂商（每个新版本可能要重交，但免费）：
  - Microsoft：<https://www.microsoft.com/wdsi/filesubmission>
  - 360：360 安全中心「软件认证 / 误报提交」
  - 火绒：官网「火绒威胁情报 / 误报反馈」
  - 腾讯：腾讯电脑管家 / 哈勃分析「误报申诉」
- **发版说明里写清**：首次运行若被拦，如何加白名单（给用户的话术）。
- 桌面端在网关启动失败时已经会提示「可能被杀毒拦截，请加白名单」（见 Desktop 状态栏重试条）。

---

## 3. 花钱根治：代码签名选型

| 方案 | 价格（约） | 说明 |
|------|-----------|------|
| **Azure Trusted Signing**（首选） | **$9.99/月** | 微软云签名，**无需 USB 硬件盾**、能直接进 CI。OV 级、微软背书。要求**已验证的组织**（一般营业执照、实体成立满 3 年）或个人验证。性价比最高。 |
| **SignPath.io 开源免费档** | **免费** | 面向合格开源项目。本仓库为 public，**值得申请**。 |
| OV 证书（DigiCert / Sectigo / SSL.com） | ¥1500–3000/年 | ⚠️ 2023-06 起私钥**强制存硬件盾 / 云 HSM**，不能再把 `.pfx` 塞进 CI secrets。SmartScreen 信誉随下载量累积，前期仍弹警告。 |
| **EV 证书** | ¥2000–4500/年 + 硬件盾 | 唯一能**立即**消除 SmartScreen「未知发布者」警告的。但硬件盾进 CI 麻烦。 |

> **不要用自签名证书**：对 SmartScreen / 杀毒无任何帮助，只适合内网。

**重要**：Authenticode 只解决微软侧。国内 360 / 火绒 / 腾讯仍需到各家**开发者 / 软件白名单**
登记（通常免费、按版本提交）。把签名 + 三家白名单都做了，才算根治。

---

## 4. 把签名接进 CI（拿到证书后）

### 4a. Azure Trusted Signing（推荐路径）

`release-runtime.yml` 在「Build self-contained executable」之后、打包 zip 之前，对 Windows
产物加一步签名：

```yaml
- name: Sign Windows runtime (Azure Trusted Signing)
  if: matrix.platform == 'win32'
  uses: azure/trusted-signing-action@v0
  with:
    azure-tenant-id: ${{ secrets.AZURE_TENANT_ID }}
    azure-client-id: ${{ secrets.AZURE_CLIENT_ID }}
    azure-client-secret: ${{ secrets.AZURE_CLIENT_SECRET }}
    endpoint: ${{ secrets.TRUSTED_SIGNING_ENDPOINT }}
    trusted-signing-account-name: ${{ secrets.TRUSTED_SIGNING_ACCOUNT }}
    certificate-profile-name: ${{ secrets.TRUSTED_SIGNING_PROFILE }}
    files-folder: dist/hermes-agent-cn-runtime-win32-${{ matrix.arch }}
    files-folder-filter: exe,dll
    file-digest: SHA256
    timestamp-rfc3161: http://timestamp.acs.microsoft.com
    timestamp-digest: SHA256
```

> 要签 **exe 和 dll 两类**（`_internal/` 下有大量 .pyd/.dll；onedir 的二进制都建议签）。

### 4b. 传统证书 + signtool

若用 OV/EV 证书（硬件盾或云 HSM），在自托管 Windows runner 上：

```bash
signtool sign /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 \
  /n "<证书主体名>" "dist/$NAME/$NAME.exe"
```

### 4c. 桌面安装包（NSIS）

桌面端 `release-desktop.yml` 目前只签了 macOS。Windows NSIS 安装包同样要签 ——
Tauri 支持通过 `tauri.conf.json > bundle.windows.certificateThumbprint` + `signCommand`
接 Azure Trusted Signing / signtool。**安装器和内置 runtime 都要签**，否则装的时候照样被拦。

---

## 5. 验收

- 干净 Win10/11 + 启用 Defender 实时保护，下载安装包：不再「未知发布者」硬拦、不被静默删除。
- 360 / 火绒装机环境下安装与首启网关：不被拦（需先过各家白名单）。
- `signtool verify /pa /v <exe>` 通过；exe 属性页能看到正确的版本信息与签名者。

---

## 附：免费项的实现位置

- `--onedir` / `--noupx` / `--version-file`：`.github/workflows/release-runtime.yml`
- 版本信息生成：`scripts/gen_win_version_info.py`（可 `--check` 校验产物语法）
- 桌面端「可能被杀毒拦截」提示与重试按钮：`Hermes-CN-Desktop` 状态栏（`app-status-bar`）
