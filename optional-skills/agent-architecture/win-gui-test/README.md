# WinGUI Test Skill

Automated Windows GUI testing and visual analysis toolkit. From **WSL** or any Unix shell, drive native Windows applications via Python (pywinauto + OpenCV) — take screenshots, inspect controls, click buttons, scroll lists, and analyze visual styles of any desktop app.

## Features

| Capability | Description |
|-----------|-------------|
| 🖥️ **Window/Control Discovery** | List all visible windows, enumerate the full UIA control tree |
| 📸 **Screenshot (with fallback)** | Primary: `mss` (fast, multi-monitor). Fallback: `PIL.ImageGrab` (works on lock screen) |
| 🎯 **Precision Clicking** | Click by control name or absolute screen coordinates |
| ⌨️ **Keyboard Simulation** | Send any keystrokes to a focused window |
| 🔬 **Visual Analysis** | Detect element size consistency, dominant colors, border radius, font hints |
| 🏭 **Competitor Analysis** | Snapshot WeChat / QQ / Edge and reverse-engineer their CSS parameters |
| 📋 **Structured JSON Output** | Every command returns well-formed JSON — pipe to `jq` or consume from Python |
| ⚙️ **Configurable** | YAML/JSON configuration file, environment variable overrides (`WG_*`), no hardcoded paths |

## Installation

```bash
pip install -r requirements.txt

# Optional: install as a package
pip install -e .
```

### Prerequisites

- **Windows 10/11** with Python 3.8+
- **WSL2** (or any shell that can call `powershell.exe`)
- **Windows Python** with these packages installed:

```bash
# Run on Windows Python (CMD or PowerShell):
pip install pywinauto opencv-python pillow mss numpy pyyaml
```

> **⚠️ pip version note**: pywinauto uses 0.x.y versioning. The requirement `pywinauto>=0.6.8` means **version 0.6.8 or newer**, not `>=6.8`.

## Quick Start

```bash
# 1. Source the tool
cd win-gui-test-skill

# 2. List all visible windows
python scripts/cli.py list-all

# 3. List controls inside a specific window
python scripts/cli.py list-elements "Partner"

# 4. Screenshot
python scripts/cli.py screenshot "Partner"

# 5. Click a button
python scripts/cli.py click "Partner" "Send"

# 6. Get precise control geometry
python scripts/cli.py get-rect "Partner" "Send"

# 7. Full visual analysis (size + color + style)
python scripts/cli.py analyze "Partner" --out-dir ./reports
```

## Command Reference

| Command | Arguments | Description |
|---------|-----------|-------------|
| `list-all` | — | List all visible desktop windows |
| `list-elements` | `<window_title>` | Enumerate all UI controls in a window |
| `screenshot` | `[window_title] [--out-dir PATH]` | Capture screenshot (with optional window focus) |
| `click` | `<window_title> <control_name>` | Click a control by its display name |
| `click-coords` | `<window_title> <x> <y>` | Click at absolute screen coordinates |
| `sendkeys` | `<window_title> <keys>` | Send keystrokes to a window |
| `scroll` | `<window_title> [--target NAME] [--dy N]` | Scroll a control (default: -3 = 3 clicks down) |
| `get-rect` | `<window_title> <control_name>` | Get precise bounding rectangle of a control |
| `launch` | `<app_path> [--wait SECONDS]` | Launch an application |
| `analyze` | `<window_title> [--out-dir PATH]` | Full visual analysis (size + color + style) |

**Global options**:
| Flag | Description |
|------|-------------|
| `--config PATH` | Path to YAML/JSON config file |
| `--log-dir PATH` | Override log output directory |
| `--timeout N` | Window lookup timeout in seconds (default: 10) |

## Examples

### Example 1: Partner GUI Test

```bash
cd examples
python example1_partner_gui.py
```

This script:
1. Launches the Partner desktop application
2. Lists all UI elements and verifies key buttons exist (Send, File, etc.)
3. Takes a screenshot
4. Gets the precise rectangle of the Send button

### Example 2: WeChat Style Analysis

```bash
python example2_wechat_analysis.py
```

Before running, make sure WeChat (微信) is open. The script:
1. Connects to the WeChat window and lists its controls
2. Takes a screenshot
3. Analyzes control size consistency
4. Extracts dominant colors
5. Estimates chat bubble corner radius

### Example 3: Navigation, Scroll & Click

```bash
python example3_scroll_click.py
```

Demonstrates a complete interaction workflow:
1. Clicks a navigation button ("实例管理")
2. Lists the instance table
3. Clicks on a specific instance row
4. Verifies the configuration panel updates
5. Takes a final screenshot

## Configuration

Copy the example config and edit:

```bash
cp config.example.yaml config.yaml
nano config.yaml
```

All settings can be overridden via environment variables with the `WG_` prefix:

```bash
export WG_SCREENSHOT_DIR="D:/screenshots"
export WG_RETRY_COUNT=5
export WG_RETRY_DELAY=2.0
```

## Error Handling & Resilience

| Scenario | Behavior |
|----------|----------|
| **Window not found** | Retries for `timeout` seconds (default 10), then returns clear error |
| **Click fails** | Retries 3 times (configurable) with 1s delay |
| **Screenshot fails (lock screen)** | Auto-fallback from `mss` → `PIL.ImageGrab` |
| **Control not found** | Falls back to descendant scan, then returns descriptive error |
| **JSON pipeline truncated** | Always write to file (`> /tmp/output.json`) before parsing |

## Project Structure

```
win-gui-test-skill/
├── README.md
├── SKILL.md                    # Hermes skill definition
├── config.example.yaml         # Sample configuration
├── requirements.txt
├── setup.py
├── .gitignore
├── LICENSE                     # Apache 2.0
├── scripts/
│   ├── core.py                 # Main controller (window ops, retry, structured output)
│   ├── cli.py                  # CLI entry point (argparse)
│   ├── analyzers/
│   │   ├── size_analyzer.py    # Size consistency detection
│   │   ├── color_analyzer.py   # Color analysis (dominant/edge)
│   │   └── style_extractor.py # CSS property inference (radius, font)
│   └── utils/
│       ├── config.py           # Config loader (YAML/JSON + env vars)
│       ├── logger.py           # Timed rotating file + console logger
│       └── screenshot.py       # Screenshot with automatic fallback
├── examples/
│   ├── example1_partner_gui.py
│   ├── example2_wechat_analysis.py
│   └── example3_scroll_click.py
├── tests/
│   └── test_core.py            # Unit tests (config, screenshot, analyzers)
├── logs/                       # Runtime logs (auto-created)
└── screenshots/                # Default screenshot directory (auto-created)
```

## Development

```bash
# Run unit tests
python -m pytest tests/ -v
# or directly:
python tests/test_core.py -v
```

### Adding a New Analyzer

1. Create a module in `scripts/analyzers/` (e.g., `my_analyzer.py`)
2. Implement a function that takes `elements: list[dict]` and returns a dict
3. The `analyze` CLI command will automatically pick it up

## Known Limitations & Workarounds

| Limitation | Workaround |
|-----------|------------|
| pywinauto reported sizes include layout margins | Use `setFixedHeight()` instead of `setMinimumHeight()` in Qt apps |
| QComboBox is undetectable via UIA backend | Infer combo box position from gaps between adjacent button rects |
| Large JSON output gets truncated in PowerShell pipe | Always redirect to file: `> /tmp/output.json` |
| Lock screen blocks mss BitBlt capture | Built-in fallback to `PIL.ImageGrab` handles this automatically |

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Commit your changes (`git commit -am 'Add my feature'`)
4. Push (`git push origin feat/my-feature`)
5. Open a Pull Request

## License

Apache 2.0. See [LICENSE](LICENSE).
