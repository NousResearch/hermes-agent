#!/usr/bin/env bash
# =============================================================================
# Hermes Agent 中国区一键安装脚本（Linux 版）
# =============================================================================
set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${BLUE}[信息]${NC} $1"; }
ok()    { echo -e "${GREEN}[成功]${NC} $1"; }
warn()  { echo -e "${YELLOW}[注意]${NC} $1"; }
err()   { echo -e "${RED}[错误]${NC} $1"; }

if [ "$(id -u)" != "0" ]; then
    err "请用 root 用户运行这个脚本！"; err "在终端里输入：sudo bash install.sh"; exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/config.env" ]; then
    info "找到配置文件 config.env，自动加载"
    set -a; source "$SCRIPT_DIR/config.env"; set +a
fi

echo ""
info "=============================================="
info "  Hermes Agent 中国区一键安装"
info "=============================================="
echo ""
info "你要用哪种方式安装？"
info "  1) 全新安装（从头开始，推荐）"
info "  2) 我已经有 VPS/Docker 环境了，只配置"
read -p "请选择 [1/2]（默认 1）：" INSTALL_MODE
INSTALL_MODE="${INSTALL_MODE:-1}"
echo ""

# ---- 第1步：换阿里云源 ----
info "【第1步/共7步】换阿里云源，让下载飞起来..."
if [ "$INSTALL_MODE" = "1" ]; then
    cp /etc/apt/sources.list /etc/apt/sources.list.bak 2>/dev/null || true
    sed -i 's|//archive.ubuntu.com|//mirrors.aliyun.com|g; s|//security.ubuntu.com|//mirrors.aliyun.com|g' /etc/apt/sources.list 2>/dev/null || true
    if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then
        sed -i 's|//archive.ubuntu.com|//mirrors.aliyun.com|g; s|//security.ubuntu.com|//mirrors.aliyun.com|g' /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null || true
    fi
    apt-get update -qq 2>/dev/null
    ok "阿里云源已就绪"
else
    info "你选的是已有环境模式，跳过换源"
fi

# ---- 第2步：安装基础工具 ----
info "【第2步/共7步】安装基础工具..."
if [ "$INSTALL_MODE" = "1" ]; then
    apt-get install -y -qq curl wget apt-transport-https ca-certificates 2>/dev/null
    ok "基础工具已装好"
fi

# ---- 第3步：安装 Docker ----
info "【第3步/共7步】安装 Docker（容器引擎）..."
if [ "$INSTALL_MODE" = "1" ]; then
    if command -v docker &>/dev/null; then
        ok "Docker 已经装好了，跳过"
    else
        curl -fsSL https://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg 2>/dev/null | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg 2>/dev/null
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://mirrors.aliyun.com/docker-ce/linux/ubuntu $(lsb_release -cs) stable" > /etc/apt/sources.list.d/docker.list
        apt-get update -qq 2>/dev/null && apt-get install -y -qq docker-ce docker-ce-cli containerd.io 2>/dev/null
        systemctl start docker 2>/dev/null || service docker start 2>/dev/null || true
        ok "Docker 安装成功！"
    fi
else
    if ! command -v docker &>/dev/null; then
        warn "没检测到 Docker，请先装好 Docker 再运行"; exit 1
    fi
    ok "Docker 已就绪"
fi

# ---- 第4步：从魔塔下载最新镜像 ----
info "【第4步/共7步】从魔塔下载最新 Hermes 镜像..."
echo ""
info "正在查询最新版本号..."
VERSION_LIST=$(curl -s --connect-timeout 10 "https://modelscope.cn/api/v1/models/aifengheguai/hermes-agent/repo/files?Revision=master&Root=" 2>/dev/null | grep -o '"Name":"[^"]*\\.tar"' | sed 's/"Name":"//;s/"//' 2>/dev/null)
TAR_FILE=$(echo "$VERSION_LIST" | head -1 2>/dev/null)
if [ -z "$TAR_FILE" ]; then
    warn "从魔塔查询版本失败，改用默认文件名"
    TAR_FILE="nousresearch_hermes-agent_latest.tar"
fi
DOWNLOAD_URL="https://modelscope.cn/models/aifengheguai/hermes-agent/resolve/master/${TAR_FILE}"
info "下载地址：$DOWNLOAD_URL"
echo ""
if command -v aria2c &>/dev/null; then
    info "用 aria2（多线程下载器），速度更快"
    aria2c -s 16 -x 16 -k 1M "$DOWNLOAD_URL" 2>&1 | tail -5
else
    info "用 curl 下载（耐心等待）..."
    curl -L -o "$TAR_FILE" "$DOWNLOAD_URL" 2>&1
fi
if [ ! -f "$TAR_FILE" ]; then
    err "下载失败了！请检查网络或手动去魔塔下载"
    err "魔塔地址：https://modelscope.cn/datasets/aifengheguai/hermes-agent-v0.18.2"
    exit 1
fi
ok "镜像下载完成！大小：$(du -h "$TAR_FILE" | cut -f1)"

# ---- 第5步：加载镜像到 Docker ----
info "【第5步/共7步】把镜像加载到 Docker..."
if [[ "$TAR_FILE" == *.tar.gz ]] || [[ "$TAR_FILE" == *.tgz ]]; then
    info "检测到压缩包，先解压..."
    tar -zxf "$TAR_FILE" 2>&1 | tail -2
    EXTRACTED=$(ls -t *.tar 2>/dev/null | head -1)
    [ -n "$EXTRACTED" ] && TAR_FILE="$EXTRACTED"
fi
LOAD_OUTPUT=$(docker load -i "$TAR_FILE" 2>&1)
echo "$LOAD_OUTPUT" | tail -2
IMAGE_NAME=$(echo "$LOAD_OUTPUT" | grep "Loaded image:" | sed 's/.*Loaded image: //')
[ -z "$IMAGE_NAME" ] && IMAGE_NAME="nousresearch/hermes-agent:latest"
rm -f "$TAR_FILE" *.tar.gz *.tgz 2>/dev/null
ok "镜像加载成功！镜像名：$IMAGE_NAME"

# ---- 第6步：启动容器 ----
info "【第6步/共7步】启动 Hermes Agent 容器..."
docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q '^hermes-agent$' && docker rm -f hermes-agent >/dev/null 2>&1
mkdir -p "$HOME/.hermes" 2>/dev/null
docker run -d -it --name hermes-agent --restart unless-stopped -v "$HOME/.hermes:/opt/data" -p 8080:8080 "$IMAGE_NAME" 2>&1
sleep 3
CONTAINER_STATUS=$(docker ps --filter name=hermes-agent --format '{{.Status}}' 2>/dev/null)
if [ -n "$CONTAINER_STATUS" ]; then
    ok "容器已启动！状态：$CONTAINER_STATUS"
else
    err "容器启动失败了，请检查 Docker 日志"
    docker logs hermes-agent 2>/dev/null | tail -10; exit 1
fi

# ---- 第7步：配置 API Key ----
info "【第7步/共7步】配置 API Key 和聊天通道..."
echo ""
info "现在需要配置 API Key"
echo ""
read -p "请输入你的 API Key（输入后按回车）：" USER_API_KEY
if [ -n "$USER_API_KEY" ]; then
    docker exec hermes-agent hermes config set api_key "$USER_API_KEY" 2>/dev/null || true
    ok "API Key 已配置"
fi

echo ""
info "要不要配置聊天通道？"
read -p "配置飞书？[y/N]：" SETUP_FEISHU
if [ "$SETUP_FEISHU" = "y" ] || [ "$SETUP_FEISHU" = "Y" ]; then
    info "正在配置飞书通道..."
    docker exec hermes-agent hermes setup gateway feishu 2>&1 | head -10 || true
    ok "飞书通道配置完成，请按照提示扫码配对"
fi

read -p "配置钉钉？[y/N]：" SETUP_DINGTALK
if [ "$SETUP_DINGTALK" = "y" ] || [ "$SETUP_DINGTALK" = "Y" ]; then
    info "正在配置钉钉通道..."
    docker exec hermes-agent hermes setup gateway dingtalk 2>&1 | head -10 || true
fi

# ---- 完成 ----
echo ""
echo -e "${GREEN}==============================================${NC}"
echo -e "${GREEN}  🎉 Hermes Agent 安装完成！${NC}"
echo -e "${GREEN}==============================================${NC}"
echo ""
info "你可以用下面的命令和 Hermes 聊天："
echo "  docker exec -it hermes-agent hermes"
echo ""
info "网页端管理界面：http://你的服务器IP:8080"
echo ""
info "Telegram：https://t.me/aifengheguai"
info "网站：http://gdibao.com"
