---
name: win-gui-test
title: Windows GUI 自动化测试 — pywinauto + OpenCV 操控原生程序
description: 从 WSL 操控 Windows 原生 GUI 进行自动化测试与视觉分析。
---

# Windows GUI Test Skill

从 WSL 通过 PowerShell 桥接 Windows Python（pywinauto + OpenCV），操控
Windows 原生 GUI 程序 — 截图、控件探测、点击、滚动、视觉分析。

## When to Use

- 需要从 WSL/CI 中自动化测试 Windows 桌面程序
- 验证 Partner GUI 或其他 Qt/Win32 应用的按钮、布局、交互
- 竞品分析：截图微信/QQ/Edge 并提取 CSS 样式参数

## Prerequisites

- **Windows 10/11** + Windows Python 3.8+（非 WSL Python）
- pywinauto、opencv-python、pillow、mss、numpy、pyyaml 安装在 Windows Python
- 从 WSL 调用时使用 PowerShell 桥接（见下方）

```bash
# 在 WSL 中设置路径别名
SKILL_DIR=$(wslpath -w "/mnt/e/work/win-gui-test-skill")
alias winpy="powershell.exe -NoProfile -Command \"python '$SKILL_DIR/scripts/cli.py'\""
```

## How to Run

所有命令通过 `winpy <command> [args]` 执行（从 WSL 通过 PowerShell 桥接
Windows Python）。直接运行需在 Windows CMD/PowerShell 中：

```bash
cd win-gui-test-skill
python scripts/cli.py <command> [args]
```

## Quick Reference

| 命令 | 参数 | 说明 |
|------|------|------|
| `list-all` | — | 列出所有可见窗口 |
| `list-elements` | `<窗口标题>` | 列出窗口内控件 |
| `screenshot` | `[窗口标题] [--out-dir PATH]` | 截图（mss → PIL 降级） |
| `click` | `<窗口标题> <控件名>` | 按控件名点击 |
| `click-coords` | `<窗口标题> <x> <y>` | 按屏幕坐标点击 |
| `sendkeys` | `<窗口标题> <按键>` | 发送键盘按键 |
| `scroll` | `<窗口标题> [--target NAME] [--dy N]` | 滚动 |
| `get-rect` | `<窗口标题> <控件名>` | 获取控件精确矩形 |
| `launch` | `<程序路径> [--wait S]` | 启动程序 |
| `analyze` | `<窗口标题> [--out-dir PATH]` | 全量视觉分析 |

全局选项：`--config PATH`, `--log-dir PATH`, `--timeout N`

## Procedure

### 场景 1：验证 Partner GUI 按钮

```bash
winpy screenshot "Partner"
winpy list-elements "Partner"
winpy get-rect "Partner" "发送"
```

### 场景 2：竞品分析（微信气泡）

```bash
winpy screenshot "微信"
winpy analyze "微信" --out-dir ./reports
```

### 场景 3：自动导航 + 操作

```bash
winpy click "Partner" "实例管理"
winpy click "Partner" "05"
winpy screenshot "Partner"
```

## Pitfalls

| 限制 | 解决 |
|------|------|
| pywinauto 尺寸包含布局间距 | 用 `setFixedHeight` 替代 `setMinimumHeight` |
| QComboBox 在 UIA 中不可检测 | 通过控件间空隙位置推断 |
| 锁屏时 mss 截图失败 | 自动降级到 PIL.ImageGrab（已内置） |
| 大 JSON 管道输出截断 | 用 `> /tmp/file.json` 先存文件再解析 |
| pywinauto 必须用 Windows Python | 从 WSL 必须通过 `powershell.exe` 桥接，不能直接用 WSL python |

## Verification

```bash
# 运行单元测试（在项目根目录）
python -m pytest tests/skills/test_win_gui_test_skill.py -v

# 语法检查
python -m py_compile scripts/core.py scripts/cli.py

# 端到端测试（需要 Windows GUI）
winpy list-all                     # 应列出所有可见窗口
winpy screenshot "任意窗口"        # 应有截图文件生成
```
