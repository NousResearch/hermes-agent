#!/usr/bin/env bash
# =============================================================================
# Hermes Agent 中国区一键安装脚本（Linux 版）
# =============================================================================
# 使用方法（任选其一）：
#   bash <(curl -sL https://gdibao.com/deploy-china/install.sh)
#   bash <(wget -qO- https://gdibao.com/deploy-china/install.sh)
#   wget https://gdibao.com/deploy-china/install.sh && sudo bash install.sh
# =============================================================================
set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${BLUE}[信息]${NC} $1"; }
ok()    { echo -e "${GREEN}[成功]${NC} $1"; }
warn()  { echo -e "${YELLOW}[注意]${NC} $1"; }
err()   { echo -e "${RED}[错误]${NC} $1"; }

# ===== 检查 root =====
if [ "$(id -u)" != "0" ]; then
    err "请用 root 用户运行这个脚本！"
    err "试试：sudo bash install.sh"
    exit 1
fi

# ===== 加载 config.env（如果有）=====
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
[ -f "$SCRIPT_DIR/config.env" ] && set -a && source "$SCRIPT_DIR/config.env" && set +a

# ===== 检测操作系统 =====
detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS_ID="$ID"
        OS_LIKE="$ID_LIKE"
        OS_VERSION_ID="$VERSION_ID"
    elif [ -f /etc/redhat-release ]; then
        OS_ID="rhel"
        OS_LIKE="rhel"
    elif [ -f /etc/debian_version ]; then
        OS_ID="debian"
        OS_LIKE="debian"
    else
        OS_ID="unknown"
    fi
    info "检测到系统：${OS_ID} ${OS_VERSION_ID:-}"
}
detect_os

# ===== 包管理器工具 =====
PKG_MGR=""
PKG_UPDATE=""
PKG_INSTALL=""
case "$OS_ID" in
    ubuntu|debian|kali|linuxmint|deepin|uos)
        PKG_MGR="apt-get"
        PKG_UPDATE="apt-get update -qq"
        PKG_INSTALL="apt-get install -y -qq"
        MIRROR_SRC="apt"
        ;;
    centos|rhel|almalinux|rocky|ol|tencentos)
        PKG_MGR="yum"
        PKG_UPDATE="yum makecache -q"
        PKG_INSTALL="yum install -y -q"
        MIRROR_SRC="yum"
        # CentOS 8+ 用 dnf
        which dnf &>/dev/null && PKG_MGR="dnf" && PKG_UPDATE="dnf makecache -q" && PKG_INSTALL="dnf install -y -q"
        ;;
    fedora)
        PKG_MGR="dnf"
        PKG_UPDATE="dnf makecache -q"
        PKG_INSTALL="dnf install -y -q"
        MIRROR_SRC="dnf"
        ;;
    alpine)
        PKG_MGR="apk"
        PKG_UPDATE="apk update -q"
        PKG_INSTALL="apk add -q"
        MIRROR_SRC="apk"
        ;;
    opensuse*|suse|sles)
        PKG_MGR="zypper"
        PKG_UPDATE="zypper refresh -q"
        PKG_INSTALL="zypper install -y -q"
        MIRROR_SRC="zypper"
        ;;
    arch|manjaro|endeavouros)
        PKG_MGR="pacman"
        PKG_UPDATE="pacman -Sy --noconfirm"
        PKG_INSTALL="pacman -S --noconfirm --needed"
        MIRROR_SRC="pacman"
        ;;
    *)
        err "不支持的系统：$OS_ID"
        err "目前支持：Ubuntu/Debian、CentOS/RHEL、Fedora、Alpine、Arch、openSUSE"
        err "你也可以手动安装 Docker 后再运行本脚本（选模式2）"
        exit 1
        ;;
esac
info "包管理器：$PKG_MGR"

# ===== 确保 curl/wget 已安装（脚本本身需要）=====
info "检查基础工具..."
NEED_INSTALL=""
command -v curl &>/dev/null || NEED_INSTALL="$NEED_INSTALL curl"
command -v wget &>/dev/null || NEED_INSTALL="$NEED_INSTALL wget"
if [ -n "$NEED_INSTALL" ]; then
    info "正在安装：$NEED_INSTALL"
    $PKG_UPDATE 2>/dev/null || true
    $PKG_INSTALL $NEED_INSTALL 2>/dev/null || true
fi
ok "基础工具就绪"

# ===== 选择安装模式 =====
echo ""
info "=============================================="
info "  Hermes Agent 中国区一键安装"
info "  系统：$OS_ID $OS_VERSION_ID"
info "=============================================="
echo ""
info "你要用哪种方式安装？"
info "  1) 全新安装（从头开始，推荐）"
info "  2) 我已经有 Docker 环境了，只配置"
read -p "请选择 [1/2]（默认 1）：" INSTALL_MODE
INSTALL_MODE="${INSTALL_MODE:-1}"
echo ""

# ===== 第1步：换国内源 =====
info "【第1步/共7步】换国内源，让下载飞起来..."
if [ "$INSTALL_MODE" = "1" ]; then
    case "$MIRROR_SRC" in
        apt)
            cp /etc/apt/sources.list /etc/apt/sources.list.bak 2>/dev/null || true
            sed -i 's|//archive.ubuntu.com|//mirrors.aliyun.com|g; s|//security.ubuntu.com|//mirrors.aliyun.com|g' /etc/apt/sources.list 2>/dev/null || true
            if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then
                sed -i 's|//archive.ubuntu.com|//mirrors.aliyun.com|g; s|//security.ubuntu.com|//mirrors.aliyun.com|g' /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null || true
            fi
            if [ "$OS_ID" = "debian" ]; then
                sed -i 's|//deb.debian.org|//mirrors.aliyun.com|g; s|//security.debian.org|//mirrors.aliyun.com/debian-security|g' /etc/apt/sources.list 2>/dev/null || true
            fi
            $PKG_UPDATE 2>/dev/null || true
            ;;
        yum|dnf)
            # CentOS/Fedora 换阿里云源
            if [ -f /etc/yum.repos.d/CentOS-Base.repo ]; then
                cp /etc/yum.repos.d/CentOS-Base.repo /etc/yum.repos.d/CentOS-Base.repo.bak 2>/dev/null || true
                curl -sL -o /etc/yum.repos.d/CentOS-Base.repo "https://mirrors.aliyun.com/repo/Centos-${OS_VERSION_ID%.*}.repo" 2>/dev/null || true
            fi
            $PKG_UPDATE 2>/dev/null || true
            ;;
        *) $PKG_UPDATE 2>/dev/null || true ;;
    esac
    ok "国内源已就绪"
else
    info "已有环境模式，跳过换源"
fi

# ===== 第2步：安装 Docker =====
info "【第2步/共7步】安装 Docker..."
if [ "$INSTALL_MODE" = "1" ]; then
    if command -v docker &>/dev/null; then
        ok "Docker 已经装好了，跳过"
    else
        case "$PKG_MGR" in
            apt-get)
                # 阿里云 Docker 镜像（Ubuntu/Debian）
                curl -fsSL https://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg 2>/dev/null | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg 2>/dev/null || true
                echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://mirrors.aliyun.com/docker-ce/linux/ubuntu $(lsb_release -cs) stable" 2>/dev/null > /etc/apt/sources.list.d/docker.list || true
                $PKG_UPDATE 2>/dev/null || true
                $PKG_INSTALL docker-ce docker-ce-cli containerd.io 2>/dev/null || true
                ;;
            yum|dnf)
                $PKG_INSTALL yum-utils 2>/dev/null || true
                $PKG_MGR config-manager --add-repo https://mirrors.aliyun.com/docker-ce/linux/centos/docker-ce.repo 2>/dev/null || true
                sed -i 's|download.docker.com|mirrors.aliyun.com/docker-ce|g' /etc/yum.repos.d/docker-ce.repo 2>/dev/null || true
                $PKG_INSTALL docker-ce docker-ce-cli containerd.io 2>/dev/null || true
                ;;
            apk)
                $PKG_INSTALL docker 2>/dev/null || true
                rc-update add docker boot 2>/dev/null || true
                ;;
            pacman)
                $PKG_INSTALL docker 2>/dev/null || true
                systemctl enable docker 2>/dev/null || true
                ;;
            *)
                err "Docker 自动安装暂不支持 $OS_ID，请手动安装后再运行"
                exit 1
                ;;
        esac
        # 启动 Docker
        systemctl start docker 2>/dev/null || service docker start 2>/dev/null || rc-service docker start 2>/dev/null || true
        ok "Docker 安装成功！"
    fi
else
    command -v docker &>/dev/null || { warn "没检测到 Docker，请先装好 Docker 再运行"; exit 1; }
    ok "Docker 已就绪"
fi

# ===== 第3步：从魔塔下载最新镜像 =====
info "【第3步/共7步】从魔塔下载最新 Hermes 镜像..."
echo ""
info "正在查询最新版本号..."
DL_CMD=""
command -v curl &>/dev/null && DL_CMD="curl -sL --connect-timeout 10" || DL_CMD="wget -qO- --timeout=10"
VERSION_LIST=$($DL_CMD "https://modelscope.cn/api/v1/models/aifengheguai/hermes-agent/repo/files?Revision=master&Root=" 2>/dev/null | grep -o '"Name":"[^"]*\.tar"' | sed 's/"Name":"//;s/"//' 2>/dev/null)
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
elif command -v curl &>/dev/null; then
    info "用 curl 下载..."
    curl -L -o "$TAR_FILE" "$DOWNLOAD_URL" 2>&1
elif command -v wget &>/dev/null; then
    info "用 wget 下载..."
    wget -O "$TAR_FILE" "$DOWNLOAD_URL" 2>&1
else
    err "没有 curl 也没有 wget，无法下载！"
    err "请手动下载：$DOWNLOAD_URL"
    exit 1
fi
if [ ! -f "$TAR_FILE" ]; then
    err "下载失败了！手动下载地址："
    err "  https://modelscope.cn/datasets/aifengheguai/hermes-agent-v0.18.2"
    exit 1
fi
ok "镜像下载完成！大小：$(du -h "$TAR_FILE" | cut -f1)"

# ===== 第4步：加载镜像到 Docker =====
info "【第4步/共7步】把镜像加载到 Docker..."
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

# ===== 第5步：启动容器 =====
info "【第5步/共7步】启动 Hermes Agent 容器..."
docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q '^hermes-agent$' && docker rm -f hermes-agent >/dev/null 2>&1
mkdir -p "$HOME/.hermes" 2>/dev/null
docker run -d -it --name hermes-agent --restart unless-stopped -v "$HOME/.hermes:/opt/data" -p 8080:8080 "$IMAGE_NAME" 2>&1
sleep 3
docker ps --filter name=hermes-agent --format '{{.Status}}' 2>/dev/null | grep -q . && ok "容器已启动！" || { err "容器启动失败，检查日志：docker logs hermes-agent"; exit 1; }

# ===== 第6步：配置 API Key =====
info "【第6步/共7步】配置 API Key..."
echo ""
read -p "请输入你的 API Key（直接回车跳过）：" USER_API_KEY
if [ -n "$USER_API_KEY" ]; then
    docker exec hermes-agent hermes config set api_key "$USER_API_KEY" 2>/dev/null || true
    ok "API Key 已配置"
fi

# ===== 第7步：配置聊天通道 =====
info "【第7步/共7步】配置聊天通道..."
read -p "配置飞书？[y/N]：" SETUP_FEISHU
if [ "$SETUP_FEISHU" = "y" ] || [ "$SETUP_FEISHU" = "Y" ]; then
    docker exec hermes-agent hermes setup gateway feishu 2>&1 | head -10 || true
fi
read -p "配置钉钉？[y/N]：" SETUP_DINGTALK
if [ "$SETUP_DINGTALK" = "y" ] || [ "$SETUP_DINGTALK" = "Y" ]; then
    docker exec hermes-agent hermes setup gateway dingtalk 2>&1 | head -10 || true
fi

# ===== 完成 =====
echo ""
echo -e "${GREEN}==============================================${NC}"
echo -e "${GREEN}  🎉 Hermes Agent 安装完成！${NC}"
echo -e "${GREEN}==============================================${NC}"
echo ""
info "聊天：docker exec -it hermes-agent hermes"
info "网页：http://你的服务器IP:8080"
info "反馈：https://t.me/aifengheguai"
info "网站：http://gdibao.com"
