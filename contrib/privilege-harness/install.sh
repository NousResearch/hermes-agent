#!/bin/bash
# Hermes VIP — 跨平台安装入口
# 自动检测系统类型，调用对应平台的安装脚本
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Hermes VIP Installer"
echo "===================="

case "$(uname)" in
    Darwin)
        echo "Detected macOS"
        exec sudo bash "$DIR/examples/install-macos.sh"
        ;;
    Linux)
        echo "Detected Linux"
        exec sudo bash "$DIR/examples/install-linux.sh"
        ;;
    *)
        echo "❌ Unsupported platform: $(uname)"
        echo "   VIP daemon supports macOS and Linux only."
        exit 1
        ;;
esac
