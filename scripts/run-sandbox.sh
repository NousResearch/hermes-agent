#!/bin/bash
# =============================================================================
# Hermes Agent 沙箱环境启动脚本
# =============================================================================
# 一键构建并启动隔离的 Docker 沙箱（hermes-agent + Redis + MySQL + PostgreSQL）
#
# 用法：
#   ./scripts/run-sandbox.sh [up | down | logs | shell | chat]
#
# 环境变量：
#   HERMES_UID    — 映射到容器内 hermes 用户的 UID（默认：当前用户 UID）
#   HERMES_GID    — 映射到容器内 hermes 用户的 GID（默认：当前用户 GID）
#   COMPOSE_FILE  — compose 文件路径（默认：docker-compose.sandbox.yml）
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${COMPOSE_FILE:-${PROJECT_ROOT}/docker-compose.sandbox.yml}"

export HERMES_UID="${HERMES_UID:-$(id -u)}"
export HERMES_GID="${HERMES_GID:-$(id -g)}"

CMD="${1:-up}"

cd "${PROJECT_ROOT}"

case "${CMD}" in
  up)
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Hermes Agent Sandbox"
    echo "  UID=${HERMES_UID}  GID=${HERMES_GID}"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""

    # 检测宿主机是否已有容器占用常用端口，给出友好提示
    local_ports=(6379 3306 5432)
    for port in "${local_ports[@]}"; do
      if lsof -Pi ":${port}" -sTCP:LISTEN -t >/dev/null 2>&1 || \
         netstat -an 2>/dev/null | grep -q ":${port} .*LISTEN"; then
        echo "⚠️  警告：宿主机端口 ${port} 已被占用。"
        echo "   docker-compose.sandbox.yml 中的数据库默认不映射宿主机端口，"
        echo "   因此不会冲突。如果你手动启用了端口映射，请检查配置。"
        echo ""
      fi
    done

    # 如果宿主机 ~/.hermes/.env 存在，提示如何迁移
    if [ -f "${HOME}/.hermes/.env" ]; then
      echo "ℹ️  检测到宿主机配置：${HOME}/.hermes/.env"
      echo "   沙箱使用独立的数据卷，不会自动复用宿主机配置。"
      echo "   启动后可通过以下命令复制 API keys："
      echo "   docker cp ${HOME}/.hermes/.env hermes-agent:/opt/data/.env"
      echo "   docker cp ${HOME}/.hermes/config.yaml hermes-agent:/opt/data/config.yaml 2>/dev/null || true"
      echo ""
    fi

    echo "🔨 构建镜像并启动服务..."
    docker compose -f "${COMPOSE_FILE}" up -d --build

    echo ""
    echo "✅ 沙箱启动完成！"
    echo ""
    echo "服务状态："
    docker compose -f "${COMPOSE_FILE}" ps
    echo ""
    echo "常用命令："
    echo "  ./scripts/run-sandbox.sh chat    # 进入交互式 CLI"
    echo "  ./scripts/run-sandbox.sh shell   # 进入容器 Shell"
    echo "  ./scripts/run-sandbox.sh logs    # 查看日志"
    echo "  ./scripts/run-sandbox.sh down    # 停止并移除沙箱"
    ;;

  down)
    echo "🛑 停止并移除沙箱..."
    docker compose -f "${COMPOSE_FILE}" down
    echo "✅ 沙箱已停止。数据卷保留，下次启动可复用。"
    ;;

  logs)
    shift || true
    docker compose -f "${COMPOSE_FILE}" logs -f "$@"
    ;;

  shell)
    echo "🐚 进入 hermes-agent 容器（bash）..."
    docker exec -it hermes-agent bash
    ;;

  chat)
    echo "🤖 启动 Hermes 交互式 CLI..."
    docker exec -it hermes-agent hermes chat
    ;;

  *)
    echo "用法: $(basename "$0") [up | down | logs | shell | chat]"
    echo ""
    echo "  up     — 构建并启动沙箱（默认）"
    echo "  down   — 停止并移除沙箱容器"
    echo "  logs   — 跟踪查看所有服务日志"
    echo "  shell  — 进入 hermes-agent 容器的 bash"
    echo "  chat   — 直接进入 hermes 交互式对话"
    exit 1
    ;;
esac
