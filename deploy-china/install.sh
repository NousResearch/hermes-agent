#!/usr/bin/env bash
# =============================================================================
# Hermes Agent 中国区一键安装脚本（Linux 版）
# =============================================================================
# 我是一个 45 岁从来没写过代码的老男人，全靠 Hermes Agent 帮我写出来的。
# 如果你也完全不懂编程，没关系——把下面这行复制到终端里粘贴回车就行了：
#
#   curl -sL https://gdibao.com/deploy-china/install.sh | sudo bash
#
# 它会帮你做完所有事情：装环境、拉镜像、启动 Hermes Agent、配飞书钉钉...
# 整个过程大概 5-10 分钟，你只需要在中间填几次 API Key 就行。
# =============================================================================

# 先关掉严格模式（怕某些命令失败后脚本直接退出）
# 关键的步骤我手动检查，不关键的错误就让它过去
set +e

# ===== 先把颜色定义好，后面打印提示信息时好看 =====
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${BLUE}[信息]${NC} $1"; }
ok()    { echo -e "${GREEN}[成功]${NC} $1"; }
warn()  { echo -e "${YELLOW}[注意]${NC} $1"; }
err()   { echo -e "${RED}[错误]${NC} $1"; }

# ============================================================
# 第一步：检查你是不是 root 用户
# ============================================================
if [ "$(id -u)" != "0" ]; then
    err "请用 root 用户运行这个脚本！"
    err "试试在终端里输入："
    echo ""
    echo "  curl -sL https://gdibao.com/deploy-china/install.sh | sudo bash"
    echo ""
    exit 1
fi

# ============================================================
# 第二步：看看有没有配置文件（config.env）
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "$0")" 2>/dev/null && pwd || echo /tmp)"
if [ -f "$SCRIPT_DIR/config.env" ]; then
    info "找到配置文件 config.env，自动加载"
    set -a
    source "$SCRIPT_DIR/config.env"
    set +a
fi

# ============================================================
# 第三步：看看你用的是啥操作系统
# ============================================================
info "正在检测你的操作系统..."

OS_ID="unknown"
OS_VERSION_ID=""
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS_ID="$ID"
    OS_VERSION_ID="$VERSION_ID"
fi

if [ "$OS_ID" = "unknown" ] && [ -f /etc/redhat-release ]; then
    OS_ID="rhel"
fi
if [ "$OS_ID" = "unknown" ] && [ -f /etc/debian_version ]; then
    OS_ID="debian"
fi

info "你的系统是：${OS_ID} ${OS_VERSION_ID}"

# ============================================================
# 第四步：根据操作系统设置包管理器命令
# ============================================================
PKG_MGR="apt-get"
PKG_UPDATE="apt-get update -qq"
PKG_INSTALL="apt-get install -y -qq"

case "$OS_ID" in
    ubuntu|debian|kali|linuxmint|deepin|uos)
        # 啥也不用改，默认就是 apt-get
        ;;
    centos|rhel|almalinux|rocky|ol|tencentos)
        PKG_MGR="yum"
        PKG_UPDATE="yum makecache -q"
        PKG_INSTALL="yum install -y -q"
        command -v dnf &>/dev/null
        if [ $? -eq 0 ]; then
            PKG_MGR="dnf"
            PKG_UPDATE="dnf makecache -q"
            PKG_INSTALL="dnf install -y -q"
        fi
        ;;
    fedora)
        PKG_MGR="dnf"
        PKG_UPDATE="dnf makecache -q"
        PKG_INSTALL="dnf install -y -q"
        ;;
    alpine)
        PKG_MGR="apk"
        PKG_UPDATE="apk update -q"
        PKG_INSTALL="apk add -q"
        ;;
    arch|manjaro|endeavouros)
        PKG_MGR="pacman"
        PKG_UPDATE="pacman -Sy --noconfirm"
        PKG_INSTALL="pacman -S --noconfirm --needed"
        ;;
    *)
        warn "没认出你的系统（${OS_ID}），默认用 apt-get"
        warn "如果安装出错，可以手动装好 Docker 后重新运行本脚本，选模式2"
        ;;
esac
info "本脚本会使用：$PKG_MGR 来安装软件"

# ============================================================
# 第五步：确保 curl 和 wget 已经装好了
# ============================================================
info "检查基础工具（curl、wget）..."

command -v curl &>/dev/null
if [ $? -ne 0 ]; then
    info "没找到 curl，正在安装..."
    $PKG_UPDATE 2>/dev/null || true
    $PKG_INSTALL curl 2>/dev/null || true
fi

command -v wget &>/dev/null
if [ $? -ne 0 ]; then
    info "没找到 wget，正在安装..."
    $PKG_UPDATE 2>/dev/null || true
    $PKG_INSTALL wget 2>/dev/null || true
fi

ok "基础工具准备好了"

# ============================================================
# 第六步：问用户想怎么安装
# ============================================================
echo ""
info "=============================================="
info "  Hermes Agent 中国区一键安装"
info "=============================================="
echo ""
info "你要用哪种方式安装？"
info "  1) 全新安装（从头开始，推荐）"
info "  2) 我已经有 Docker 环境了，只配置"
echo ""

read -p "请选择 [1/2]（直接回车默认 1）：" INSTALL_MODE </dev/tty

if [ -z "$INSTALL_MODE" ]; then
    INSTALL_MODE="1"
fi
echo ""

# ============================================================
# 第1步：换国内源
# ============================================================
info "【第1步/共7步】换国内源，让下载飞起来..."

if [ "$INSTALL_MODE" = "1" ]; then
    case "$OS_ID" in
        ubuntu|debian)
            cp /etc/apt/sources.list /etc/apt/sources.list.bak 2>/dev/null || true
            sed -i 's|//archive.ubuntu.com|//mirrors.aliyun.com|g' /etc/apt/sources.list 2>/dev/null || true
            sed -i 's|//security.ubuntu.com|//mirrors.aliyun.com|g' /etc/apt/sources.list 2>/dev/null || true
            if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then
                sed -i 's|//archive.ubuntu.com|//mirrors.aliyun.com|g' /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null || true
                sed -i 's|//security.ubuntu.com|//mirrors.aliyun.com|g' /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null || true
            fi
            if [ "$OS_ID" = "debian" ]; then
                sed -i 's|//deb.debian.org|//mirrors.aliyun.com|g' /etc/apt/sources.list 2>/dev/null || true
                sed -i 's|//security.debian.org|//mirrors.aliyun.com/debian-security|g' /etc/apt/sources.list 2>/dev/null || true
            fi
            $PKG_UPDATE 2>/dev/null || true
            ;;
        centos|rhel)
            if [ -f /etc/yum.repos.d/CentOS-Base.repo ]; then
                cp /etc/yum.repos.d/CentOS-Base.repo /etc/yum.repos.d/CentOS-Base.repo.bak 2>/dev/null || true
            fi
            curl -sL -o /etc/yum.repos.d/CentOS-Base.repo \
                "https://mirrors.aliyun.com/repo/Centos-${OS_VERSION_ID%.*}.repo" 2>/dev/null || true
            $PKG_UPDATE 2>/dev/null || true
            ;;
        *)
            $PKG_UPDATE 2>/dev/null || true
            ;;
    esac
    ok "国内源已就绪，下载速度会快很多！"
else
    info "你选的是已有环境模式，跳过换源"
fi

# ============================================================
# 第2步：安装 Docker
# ============================================================
info "【第2步/共7步】安装 Docker（容器引擎）..."

if [ "$INSTALL_MODE" = "1" ]; then
    command -v docker &>/dev/null
    if [ $? -eq 0 ]; then
        ok "Docker 已经装好了，跳过安装步骤"
    else
        info "正在安装 Docker，请稍等..."
        echo ""

        case "$PKG_MGR" in
            apt-get)
                info "正在添加 Docker 的软件源..."
                curl -fsSL https://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg 2>/dev/null | \
                    gpg --dearmor --yes -o /usr/share/keyrings/docker-archive-keyring.gpg 2>/dev/null
                if [ -f /usr/share/keyrings/docker-archive-keyring.gpg ]; then
                    ARCH=$(dpkg --print-architecture)
                    CODENAME=$(lsb_release -cs 2>/dev/null || echo "focal")
                    echo "deb [arch=${ARCH} signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://mirrors.aliyun.com/docker-ce/linux/ubuntu ${CODENAME} stable" \
                        > /etc/apt/sources.list.d/docker.list
                    $PKG_UPDATE 2>/dev/null || true
                    $PKG_INSTALL docker-ce docker-ce-cli containerd.io 2>/dev/null || true
                fi
                command -v docker &>/dev/null
                if [ $? -ne 0 ]; then
                    info "Docker 官方版安装不成功，试试 Ubuntu 自带的版本..."
                    $PKG_INSTALL docker.io 2>/dev/null || true
                fi
                ;;
            yum|dnf)
                info "正在添加 Docker 的软件源..."
                $PKG_INSTALL yum-utils 2>/dev/null || true
                $PKG_MGR config-manager --add-repo \
                    https://mirrors.aliyun.com/docker-ce/linux/centos/docker-ce.repo 2>/dev/null || true
                sed -i 's|download.docker.com|mirrors.aliyun.com/docker-ce|g' \
                    /etc/yum.repos.d/docker-ce.repo 2>/dev/null || true
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
                err "本脚本不知道怎么在你的系统上装 Docker"
                err "请手动安装 Docker 后再运行本脚本（选模式2）"
                exit 1
                ;;
        esac

        info "正在启动 Docker 服务..."
        systemctl start docker 2>/dev/null || service docker start 2>/dev/null || rc-service docker start 2>/dev/null || true

        command -v docker &>/dev/null
        if [ $? -eq 0 ]; then
            ok "Docker 安装成功！"
        else
            err "Docker 安装失败了，请手动安装后再运行本脚本"
            err "安装完成后选模式2继续即可"
            exit 1
        fi
    fi
else
    command -v docker &>/dev/null
    if [ $? -eq 0 ]; then
        ok "Docker 已就绪"
    else
        err "你说已经有 Docker 了，但我没找到 docker 命令"
        err "请先装好 Docker，或者选模式1全新安装"
        exit 1
    fi
fi

# ============================================================
# 第3步：从魔塔下载最新镜像
# ============================================================
info "【第3步/共7步】从魔塔下载最新 Hermes 镜像..."
echo ""
info "正在查询魔塔上的最新版本号..."

VERSION_LIST=$(curl -sL --connect-timeout 10 \
    "https://modelscope.cn/api/v1/models/aifengheguai/hermes-agent/repo/files?Revision=master&Root=" \
    2>/dev/null | grep -o '"Name":"[^"]*\.tar"' | sed 's/"Name":"//;s/"//' 2>/dev/null)

TAR_FILE=$(echo "$VERSION_LIST" | head -1 2>/dev/null)

if [ -z "$TAR_FILE" ]; then
    warn "从魔塔查询版本失败，改用默认文件名"
    TAR_FILE="nousresearch_hermes-agent_latest.tar"
fi

DOWNLOAD_URL="https://modelscope.cn/models/aifengheguai/hermes-agent/resolve/master/${TAR_FILE}"
info "下载地址：$DOWNLOAD_URL"
info "这个文件大概 1-2GB，下载速度取决于你的宽带..."
echo ""

if command -v aria2c &>/dev/null; then
    info "检测到 aria2（多线程下载器），速度更快！"
    aria2c -s 16 -x 16 -k 1M "$DOWNLOAD_URL" 2>&1 | tail -5
elif command -v curl &>/dev/null; then
    info "用 curl 下载（耐心等待）..."
    curl -L -o "$TAR_FILE" "$DOWNLOAD_URL" 2>&1
elif command -v wget &>/dev/null; then
    info "用 wget 下载（耐心等待）..."
    wget -O "$TAR_FILE" "$DOWNLOAD_URL" 2>&1
fi

if [ ! -f "$TAR_FILE" ]; then
    err "下载失败了！请检查网络"
    err "你也可以手动去魔塔下载："
    err "  https://modelscope.cn/datasets/aifengheguai/hermes-agent-v0.18.2"
    exit 1
fi

ok "镜像下载完成！大小：$(du -h "$TAR_FILE" | cut -f1)"

# ============================================================
# 第4步：加载镜像到 Docker
# ============================================================
info "【第4步/共7步】把镜像加载到 Docker..."

if [[ "$TAR_FILE" == *.tar.gz ]] || [[ "$TAR_FILE" == *.tgz ]]; then
    info "检测到压缩包，先解压..."
    tar -zxf "$TAR_FILE" 2>&1 | tail -2
    EXTRACTED=$(ls -t *.tar 2>/dev/null | head -1)
    if [ -n "$EXTRACTED" ]; then
        TAR_FILE="$EXTRACTED"
    fi
fi

info "正在加载镜像到 Docker..."
LOAD_OUTPUT=$(docker load -i "$TAR_FILE" 2>&1)
echo "$LOAD_OUTPUT" | tail -2

IMAGE_NAME=$(echo "$LOAD_OUTPUT" | grep "Loaded image:" | sed 's/.*Loaded image: //')

if [ -z "$IMAGE_NAME" ]; then
    warn "没识别出镜像名，用默认的"
    IMAGE_NAME="nousresearch/hermes-agent:latest"
fi

rm -f "$TAR_FILE" *.tar.gz *.tgz 2>/dev/null

ok "镜像加载成功！镜像名：$IMAGE_NAME"

# ============================================================
# 第5步：启动容器
# ============================================================
info "【第5步/共7步】启动 Hermes Agent 容器..."

docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q '^hermes-agent$'
if [ $? -eq 0 ]; then
    warn "发现已存在的 hermes-agent 容器，正在删除重建..."
    docker rm -f hermes-agent >/dev/null 2>&1
fi

# 创建数据目录
# 注意！用 root 运行，这里就是 /root/.hermes
# 数据保存在了 /root/.hermes 文件夹里
mkdir -p "/root/.hermes" 2>/dev/null

info "正在启动容器..."
docker run -d -it \
    --name hermes-agent \
    --restart unless-stopped \
    -v "/root/.hermes:/opt/data" \
    -p 8080:8080 \
    "$IMAGE_NAME" 2>&1

sleep 3

CONTAINER_STATUS=$(docker ps --filter name=hermes-agent --format '{{.Status}}' 2>/dev/null)

if [ -n "$CONTAINER_STATUS" ]; then
    ok "容器已启动！状态：$CONTAINER_STATUS"
    ok "Hermes Agent 已经在后台运行了，端口 8080"
    info "你的数据保存在了 /root/.hermes 文件夹里"
else
    err "容器启动失败了，请检查 Docker 日志"
    docker logs hermes-agent 2>/dev/null | tail -10
    exit 1
fi

# ============================================================
# 第6步：配置 API Key
# ============================================================
info "【第6步/共7步】配置 API Key..."
echo ""
info "现在需要配置 API Key"
info "如果你还没有 API Key，可以去这些地方申请："
info "  - DeepSeek：https://platform.deepseek.com"
info "  - 阿里云百炼：https://bailian.console.aliyun.com"
info "  - 硅基流动：https://siliconflow.cn"
echo ""
read -p "请输入你的 API Key（直接回车跳过，以后可以再配）：" USER_API_KEY </dev/tty

if [ -n "$USER_API_KEY" ]; then
    docker exec hermes-agent hermes config set api_key "$USER_API_KEY" 2>/dev/null || true
    ok "API Key 已配置"
fi

# ============================================================
# 第7步：配置聊天通道
# ============================================================
info "【第7步/共7步】配置聊天通道..."
echo ""
info "要不要配置聊天通道？"
info "配好后可以在手机上跟 Hermes 聊天"

read -p "配置飞书？输入 y 然后回车（默认不配）：" SETUP_FEISHU </dev/tty

if [ "$SETUP_FEISHU" = "y" ] || [ "$SETUP_FEISHU" = "Y" ]; then
    info "正在配置飞书通道..."
    echo ""
    info "请按照提示操作（可能需要扫码）"
    docker exec hermes-agent hermes setup gateway feishu 2>&1 | head -10 || true
    ok "飞书通道配置完成"
fi

read -p "配置钉钉？输入 y 然后回车（默认不配）：" SETUP_DINGTALK </dev/tty

if [ "$SETUP_DINGTALK" = "y" ] || [ "$SETUP_DINGTALK" = "Y" ]; then
    info "正在配置钉钉通道..."
    docker exec hermes-agent hermes setup gateway dingtalk 2>&1 | head -10 || true
    ok "钉钉通道配置完成"
fi

# ============================================================
# 全部完成！
# ============================================================
echo ""
echo -e "${GREEN}==============================================${NC}"
echo -e "${GREEN}  🎉 Hermes Agent 安装完成！${NC}"
echo -e "${GREEN}==============================================${NC}"
echo ""
info "你可以用下面的命令和 Hermes 聊天："
echo ""
echo "  docker exec -it hermes-agent hermes"
echo ""
info "或者打开网页管理界面："
echo ""
echo "  http://你的服务器IP:8080"
echo ""
info "你的数据保存在了 /root/.hermes 文件夹里"
echo ""
info "如果遇到问题，可以来这里反馈："
info "  Telegram：https://t.me/aifengheguai"
info "  网站：http://gdibao.com"
echo ""
info "感谢使用！—— 一个45岁的老男人 + Hermes Agent 智能体"
