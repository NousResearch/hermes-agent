# CodexPlusPlus 安装任务包

## 任务：下载并安装 CodexPlusPlus

### 项目信息
- 仓库：https://github.com/BigPizzaV3/CodexPlusPlus
- 最新版本：**v1.1.5**
- 发布日期：2026-05-20

### 下载信息
| 项目 | 值 |
|------|-----|
| 平台 | macOS Apple Silicon (arm64) |
| DMG 文件名 | `CodexPlusPlus-1.1.5-macos-arm64.dmg` |
| 大小 | ~12.3 MB |
| SHA256 | `2172eeed9a5b54a85ce6cae83ec461bd141dbcb0a78becde5444efdff08a9454` |
| 下载 URL | `https://github.com/BigPizzaV3/CodexPlusPlus/releases/download/v1.1.5/CodexPlusPlus-1.1.5-macos-arm64.dmg` |

### 执行步骤（请 Claude Code 依次执行）

#### Step 1：下载 DMG
```bash
cd ~/Downloads
curl -L -o CodexPlusPlus-1.1.5-macos-arm64.dmg \
  "https://github.com/BigPizzaV3/CodexPlusPlus/releases/download/v1.1.5/CodexPlusPlus-1.1.5-macos-arm64.dmg"
```

#### Step 2：验证 SHA256
```bash
echo "2172eeed9a5b54a85ce6cae83ec461bd141dbcb0a78becde5444efdff08a9454  ~/Downloads/CodexPlusPlus-1.1.5-macos-arm64.dmg" | shasum -a 256 -c
```

#### Step 3：挂载 DMG
```bash
hdiutil attach ~/Downloads/CodexPlusPlus-1.1.5-macos-arm64.dmg
```

#### Step 4：安装到 /Applications/
```bash
cp -R "/Volumes/CodexPlusPlus/CodexPlusPlus.app" /Applications/
```

#### Step 5：卸载 DMG
```bash
hdiutil detach "/Volumes/CodexPlusPlus"
```

#### Step 6：验证安装
```bash
ls -la /Applications/CodexPlusPlus.app
# 确认 app bundle 存在
```

#### Step 7：清理下载文件（可选）
```bash
rm ~/Downloads/CodexPlusPlus-1.1.5-macos-arm64.dmg
```

### 验收标准
1. ✅ `/Applications/CodexPlusPlus.app` 存在
2. ✅ 应用可以打开（验证 bundle 完整性）
3. ✅ SHA256 校验通过

### 注意事项
- MacBook Air M1 (Apple Silicon)，务必使用 arm64 版本
- 如果已安装旧版本，直接覆盖即可
- DMG 挂载后卷名通常为 `CodexPlusPlus`，挂载后请确认卷名
