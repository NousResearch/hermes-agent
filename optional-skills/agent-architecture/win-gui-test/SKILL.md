---
name: win-gui-test
description: 从 WSL 内部通过 Python（pywinauto + OpenCV）操控 Windows 原生 GUI 程序，进行自动化测试、视觉分析和竞品对比。Windows GUI automation from WSL via pywinauto + OpenCV — screenshots, control inspection, clicking, scrolling, and visual analysis.
version: 1.0.0
author: ZLL (@zty522), Nous Research
license: MIT
platforms: [linux, windows]
metadata:
  hermes:
    tags: [windows, gui-test, pywinauto, opencv, automation, visual-analysis]
    category: agent-architecture
    related_skills: [partner-windows-gui, partner-gui-dialogue]
    requires_toolsets: [terminal]
---

# Windows GUI Test Skill

从 Hermes（WSL 内部）通过 `pywinauto` + `OpenCV` 操控 Windows 原生 GUI 程序。

> **关键要求**：所有 Python 代码必须在 **Windows Python**（C:\Python314\python.exe 或 C:\Users\<user>\AppData\Local\Programs\Python\Python314\python.exe）下运行，**不是 WSL Python**。Hermes 运行在 WSL 中，通过 `powershell.exe -Command ...` 调用 Windows Python。

## 命令列表

| 命令 | 参数 | 说明 |
|------|------|------|
| `list-all` | — | 列出所有可见窗口 |
| `list-elements` | `<窗口标题>` | 列出窗口内所有控件（类型、名称、位置、尺寸） |
| `screenshot` | `<窗口标题>` | 截图到配置目录（mss → PIL 降级） |
| `click` | `<窗口标题> <控件名>` | 按控件名点击 |
| `click-coords` | `<窗口标题> <x> <y>` | 按屏幕坐标点击 |
| `sendkeys` | `<窗口标题> <按键>` | 发送键盘按键 |
| `scroll` | `<窗口标题> --target <控件> --dy <数量>` | 滚动控件或窗口 |
| `get-rect` | `<窗口标题> <控件名>` | 获取控件精确矩形 |
| `launch` | `<程序路径>` | 启动程序 |
| `analyze` | `<窗口标题> --out-dir <目录>` | 全量分析（尺寸+颜色+样式） |

## 目录结构

```
~/.hermes/skills/agent-architecture/win-gui-test/
├── SKILL.md
├── config.example.yaml
├── requirements.txt
├── scripts/
│   ├── core.py                 # 主控制器（窗口操作、重试、结构化输出）
│   ├── cli.py                  # CLI 入口（argparse）
│   ├── analyzers/
│   │   ├── __init__.py
│   │   ├── size_analyzer.py    # 尺寸一致性检测
│   │   ├── color_analyzer.py   # 颜色分析（主色/边缘色）
│   │   └── style_extractor.py  # CSS 属性推断（圆角、字体）
│   └── utils/
│       ├── __init__.py
│       ├── config.py           # 配置加载器（YAML/JSON + 环境变量）
│       ├── logger.py           # 按时轮转的文件+控制台日志
│       └── screenshot.py       # 截图（含自动降级）
├── examples/
│   ├── example1_partner_gui.py
│   ├── example2_wechat_analysis.py
│   └── example3_scroll_click.py
└── tests/
    └── test_core.py
```

## 首次设置

```bash
# 1. 安装 Windows Python 依赖（在 Windows CMD/PowerShell 中运行）
pip install pywinauto opencv-python pillow mss numpy pyyaml

# 2. 复制配置文件
cp ~/.hermes/skills/agent-architecture/win-gui-test/config.example.yaml \
   ~/.hermes/skills/agent-architecture/win-gui-test/config.yaml
```

## 典型工作流

### 场景 1：验证 Partner GUI 按钮

```bash
SCRIPT_DIR=~/.hermes/skills/agent-architecture/win-gui-test/scripts

# 列出控件
python $SCRIPT_DIR/cli.py list-elements "Partner"

# 截图
python $SCRIPT_DIR/cli.py screenshot "Partner"

# 获取精确位置
python $SCRIPT_DIR/cli.py get-rect "Partner" "发送"

# 点击
python $SCRIPT_DIR/cli.py click "Partner" "发送"
```

### 场景 2：竞品分析（微信气泡）

```bash
SCRIPT_DIR=~/.hermes/skills/agent-architecture/win-gui-test/scripts

python $SCRIPT_DIR/cli.py screenshot "微信"
python $SCRIPT_DIR/cli.py analyze "微信" --out-dir ./
# 分析报告在 ./analysis_微信.json
```

### 场景 3：自动导航 + 操作

```bash
SCRIPT_DIR=~/.hermes/skills/agent-architecture/win-gui-test/scripts

python $SCRIPT_DIR/cli.py click "Partner" "实例管理"
python $SCRIPT_DIR/cli.py click "Partner" "05"
python $SCRIPT_DIR/cli.py screenshot "Partner"
```

## 配置

参见 `config.example.yaml`。所有项可通过环境变量 `WG_*` 覆盖：

```bash
export WG_SCREENSHOT_DIR="D:/screenshots"
export WG_RETRY_COUNT=5
export WG_RETRY_DELAY=2.0
```

## 错误处理

| 场景 | 行为 |
|------|------|
| **窗口未找到** | 重试直到超时（默认 10s），返回清晰错误 |
| **点击失败** | 自动重试 3 次（间隔 1s），捕获 `ElementNotFoundError` / `TimeoutError` |
| **截图失败（锁屏）** | 自动降级从 mss → PIL.ImageGrab |
| **控件未找到** | 回退到后代扫描，然后返回描述性错误 |
| **JSON 管道截断** | 始终写文件（`> /tmp/output.json`）后再解析 |

## 已知限制与解决

| 限制 | 解决 |
|------|------|
| pywinauto 报告尺寸包含布局间距 | 用 `setFixedHeight` 替代 `setMinimumHeight` |
| QComboBox 在 UIA 不可检测 | 通过控件间空隙位置推断 |
| 锁屏时 mss 截图失败 | 自动降级到 PIL.ImageGrab（已内置） |
| 大 JSON 管道输出截断 | 用 `> /tmp/file.json` 先存文件再解析 |

## 测试

```bash
# 在 Windows Python 中运行
python ~/.hermes/skills/agent-architecture/win-gui-test/tests/test_core.py -v
```
