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

set +e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${BLUE}[信息]${NC} $1"; }
ok()    { echo -e "${GREEN}[成功]${NC} $1"; }
warn()  { echo -e "${YELLOW}[注意]${NC} $1"; }
err()   { echo -e "${RED}[错误]${NC} $1"; }

HERMES_DATA_DIR="/home/hermes/.hermes"
HERMES_USER="hermes"

# ===== 检查 root =====
if [ "$(id -u)" != "0" ]; then
    err "请用 root 运行：curl -sL https://gdibao.com/deploy-china/install.sh | sudo bash"
    exit 1
fi

# ===== 加载 config.env =====
SCRIPT_DIR="$(cd "$(dirname "$0")" 2>/dev/null && pwd || echo /tmp)"
[ -f "$SCRIPT_DIR/config.env" ] && set -a && source "$SCRIPT_DIR/config.env" && set +a

# ===== 创建 hermes 用户 =====
info "检查 hermes 用户..."
id "$HERMES_USER" &>/dev/null
if [ $? -ne 0 ]; then
    info "没找到 hermes 用户，正在创建..."
    useradd -m -s /bin/bash "$HERMES_USER"
    if [ $? -eq 0 ]; then
        ok "用户 hermes 创建成功！home 目录：/home/hermes"
    else
        err "创建 hermes 用户失败了！"; exit 1
    fi
else
    ok "hermes 用户已经存在"
fi

# ===== sudo 免密 =====
info "配置 hermes 用户的 sudo 免密权限..."
echo "$HERMES_USER ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$HERMES_USER
chmod 440 /etc/sudoers.d/$HERMES_USER
ok "sudo 免密已配置"

# ===== docker 组 =====
info "把 hermes 用户加入 docker 组..."
usermod -aG docker "$HERMES_USER"
ok "已加入 docker 组"

# ===== 检测操作系统 =====
info "正在检测你的操作系统..."
OS_ID="unknown"; OS_VERSION_ID=""
[ -f /etc/os-release ] && . /etc/os-release && OS_ID="$ID" && OS_VERSION_ID="$VERSION_ID"
[ "$OS_ID" = "unknown" ] && [ -f /etc/redhat-release ] && OS_ID="rhel"
[ "$OS_ID" = "unknown" ] && [ -f /etc/debian_version ] && OS_ID="debian"
info "你的系统是：${OS_ID} ${OS_VERSION_ID}"

# ===== 包管理器 =====
PKG_MGR="apt-get"; PKG_UPDATE="apt-get update -qq"; PKG_INSTALL="apt-get install -y -qq"
case "$OS_ID" in
    ubuntu|debian|kali|linuxmint|deepin|uos) ;;
    centos|rhel|almalinux|rocky|ol|tencentos)
        PKG_MGR="yum"; PKG_UPDATE="yum makecache -q"; PKG_INSTALL="yum install -y -q"
        command -v dnf &>/dev/null && [ $? -eq 0 ] && PKG_MGR="dnf" && PKG_UPDATE="dnf makecache -q" && PKG_INSTALL="dnf install -y -q" ;;
    fedora) PKG_MGR="dnf"; PKG_UPDATE="dnf makecache -q"; PKG_INSTALL="dnf install -y -q" ;;
    alpine) PKG_MGR="apk"; PKG_UPDATE="apk update -q"; PKG_INSTALL="apk add -q" ;;
    arch|manjaro|endeavouros) PKG_MGR="pacman"; PKG_UPDATE="pacman -Sy --noconfirm"; PKG_INSTALL="pacman -S --noconfirm --needed" ;;
    *) warn "没认出你的系统（${OS_ID}），默认用 apt-get" ;;
esac
info "本脚本会使用：$PKG_MGR 来安装软件"

# ===== 确保 curl/wget =====
info "检查基础工具（curl、wget）..."
command -v curl &>/dev/null || { $PKG_UPDATE 2>/dev/null || true; $PKG_INSTALL curl 2>/dev/null || true; }
command -v wget &>/dev/null || { $PKG_UPDATE 2>/dev/null || true; $PKG_INSTALL wget 2>/dev/null || true; }
ok "基础工具准备好了"

# ===== 选择安装模式 =====
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
[ -z "$INSTALL_MODE" ] && INSTALL_MODE="1"
echo ""

# ===== 第1步：换源 =====
info "【第1步/共7步】换国内源，让下载飞起来..."
if [ "$INSTALL_MODE" = "1" ]; then
    case "$OS_ID" in
        ubuntu|debian)
            cp /etc/apt/sources.list /etc/apt/sources.list.bak 2>/dev/null || true
            sed -i 's|//archive.ubuntu.com|//mirrors.aliyun.com|g; s|//security.ubuntu.com|//mirrors.aliyun.com|g' /etc/apt/sources.list 2>/dev/null || true
            [ -f /etc/apt/sources.list.d/ubuntu.sources ] && sed -i 's|//archive.ubuntu.com|//mirrors.aliyun.com|g; s|//security.ubuntu.com|//mirrors.aliyun.com|g' /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null || true
            [ "$OS_ID" = "debian" ] && sed -i 's|//deb.debian.org|//mirrors.aliyun.com|g; s|//security.debian.org|//mirrors.aliyun.com/debian-security|g' /etc/apt/sources.list 2>/dev/null || true
            $PKG_UPDATE 2>/dev/null || true ;;
        centos|rhel)
            [ -f /etc/yum.repos.d/CentOS-Base.repo ] && cp /etc/yum.repos.d/CentOS-Base.repo{,.bak} 2>/dev/null || true
            curl -sL -o /etc/yum.repos.d/CentOS-Base.repo "https://mirrors.aliyun.com/repo/Centos-${OS_VERSION_ID%.*}.repo" 2>/dev/null || true
            $PKG_UPDATE 2>/dev/null || true ;;
        *) $PKG_UPDATE 2>/dev/null || true ;;
    esac
    ok "国内源已就绪，下载速度会快很多！"
else
    info "已有环境模式，跳过换源"
fi

# ===== 第2步：安装 Docker =====
info "【第2步/共7步】安装 Docker（容器引擎）..."
if [ "$INSTALL_MODE" = "1" ]; then
    command -v docker &>/dev/null
    if [ $? -eq 0 ]; then
        ok "Docker 已经装好了，跳过安装步骤"
    else
        info "正在安装 Docker，请稍等..."
        case "$PKG_MGR" in
            apt-get)
                info "正在添加 Docker 的软件源..."
                curl -fsSL https://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg 2>/dev/null | gpg --dearmor --yes -o /usr/share/keyrings/docker-archive-keyring.gpg 2>/dev/null
                if [ -f /usr/share/keyrings/docker-archive-keyring.gpg ]; then
                    ARCH=$(dpkg --print-architecture)
                    CODENAME=$(lsb_release -cs 2>/dev/null || echo "focal")
                    echo "deb [arch=${ARCH} signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://mirrors.aliyun.com/docker-ce/linux/ubuntu ${CODENAME} stable" > /etc/apt/sources.list.d/docker.list
                    $PKG_UPDATE 2>/dev/null || true
                    $PKG_INSTALL docker-ce docker-ce-cli containerd.io 2>/dev/null || true
                fi
                command -v docker &>/dev/null || { info "Docker CE 不行，改装 docker.io..."; $PKG_INSTALL docker.io 2>/dev/null || true; } ;;
            yum|dnf)
                $PKG_INSTALL yum-utils 2>/dev/null || true
                $PKG_MGR config-manager --add-repo https://mirrors.aliyun.com/docker-ce/linux/centos/docker-ce.repo 2>/dev/null || true
                sed -i 's|download.docker.com|mirrors.aliyun.com/docker-ce|g' /etc/yum.repos.d/docker-ce.repo 2>/dev/null || true
                $PKG_INSTALL docker-ce docker-ce-cli containerd.io 2>/dev/null || true ;;
            apk) $PKG_INSTALL docker 2>/dev/null || true; rc-update add docker boot 2>/dev/null || true ;;
            pacman) $PKG_INSTALL docker 2>/dev/null || true; systemctl enable docker 2>/dev/null || true ;;
            *) err "不知道怎么装 Docker，请手动装好后再运行（选模式2）"; exit 1 ;;
        esac
        systemctl start docker 2>/dev/null || service docker start 2>/dev/null || rc-service docker start 2>/dev/null || true
        command -v docker &>/dev/null && ok "Docker 安装成功！" || { err "Docker 安装失败，请手动安装"; exit 1; }
    fi
else
    command -v docker &>/dev/null || { err "请先装好 Docker"; exit 1; }
    ok "Docker 已就绪"
fi

# ===== 第3步：下载镜像 =====
info "【第3步/共7步】从魔塔下载最新 Hermes 镜像..."
echo ""
info "正在查询魔塔上的最新版本号..."
DL_CMD="curl -sL --connect-timeout 10"; command -v curl &>/dev/null || DL_CMD="wget -qO- --timeout=10"
VERSION_LIST=$($DL_CMD "https://modelscope.cn/api/v1/models/aifengheguai/hermes-agent/repo/files?Revision=master&Root=" 2>/dev/null | grep -o '"Name":"[^"]*\.tar"' | sed 's/"Name":"//;s/"//' 2>/dev/null)
TAR_FILE=$(echo "$VERSION_LIST" | head -1 2>/dev/null)
[ -z "$TAR_FILE" ] && TAR_FILE="nousresearch_hermes-agent_latest.tar"
DOWNLOAD_URL="https://modelscope.cn/models/aifengheguai/hermes-agent/resolve/master/${TAR_FILE}"
info "下载地址：$DOWNLOAD_URL"
info "这个文件大概 1-2GB，下载速度取决于你的宽带..."
if command -v aria2c &>/dev/null; then
    info "用 aria2，更快！"; aria2c -s 16 -x 16 -k 1M "$DOWNLOAD_URL" 2>&1 | tail -5
elif command -v curl &>/dev/null; then
    info "用 curl 下载..."; curl -L -o "$TAR_FILE" "$DOWNLOAD_URL" 2>&1
else
    info "用 wget 下载..."; wget -O "$TAR_FILE" "$DOWNLOAD_URL" 2>&1
fi
[ ! -f "$TAR_FILE" ] && { err "下载失败！"; exit 1; }
ok "镜像下载完成！大小：$(du -h "$TAR_FILE" | cut -f1)"

# ===== 第4步：加载镜像 =====
info "【第4步/共7步】把镜像加载到 Docker..."
[[ "$TAR_FILE" == *.tar.gz ]] || [[ "$TAR_FILE" == *.tgz ]] && { tar -zxf "$TAR_FILE" 2>&1 | tail -2; EXTRACTED=$(ls -t *.tar 2>/dev/null | head -1); [ -n "$EXTRACTED" ] && TAR_FILE="$EXTRACTED"; }
LOAD_OUTPUT=$(docker load -i "$TAR_FILE" 2>&1)
echo "$LOAD_OUTPUT" | tail -2
IMAGE_NAME=$(echo "$LOAD_OUTPUT" | grep "Loaded image:" | sed 's/.*Loaded image: //')
[ -z "$IMAGE_NAME" ] && IMAGE_NAME="nousresearch/hermes-agent:latest"
rm -f "$TAR_FILE" *.tar.gz *.tgz 2>/dev/null
ok "镜像加载成功！镜像名：$IMAGE_NAME"

# ===== 第5步：启动容器 =====
info "【第5步/共7步】启动 Hermes Agent 容器..."
docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q '^hermes-agent$' && docker rm -f hermes-agent >/dev/null 2>&1
info "创建数据目录：$HERMES_DATA_DIR"
mkdir -p "$HERMES_DATA_DIR" 2>/dev/null
chown -R "$HERMES_USER:$HERMES_USER" "$HERMES_DATA_DIR"
info "正在启动容器..."
docker run -d -it --name hermes-agent --restart unless-stopped -v "$HERMES_DATA_DIR:/opt/data" -p 8080:8080 "$IMAGE_NAME" 2>&1
sleep 3
docker ps --filter name=hermes-agent --format '{{.Status}}' 2>/dev/null | grep -q . && ok "容器已启动！" || { err "启动失败"; docker logs hermes-agent 2>/dev/null | tail -10; exit 1; }
ok "Hermes Agent 已经在后台运行了，端口 8080"
info "你的数据保存在了 /home/hermes/.hermes 文件夹里"

# ===== 第6步：API Key =====
info "【第6步/共7步】配置 API Key..."
read -p "请输入你的 API Key（直接回车跳过）：" USER_API_KEY </dev/tty
[ -n "$USER_API_KEY" ] && docker exec hermes-agent hermes config set api_key "$USER_API_KEY" 2>/dev/null || true && ok "API Key 已配置"

# ===== 第7步：聊天通道 =====
info "【第7步/共7步】配置聊天通道..."
read -p "配置飞书？输入 y 然后回车（默认不配）：" SETUP_FEISHU </dev/tty
[ "$SETUP_FEISHU" = "y" ] || [ "$SETUP_FEISHU" = "Y" ] && docker exec hermes-agent hermes setup gateway feishu 2>&1 | head -10 || true && ok "飞书通道配置完成"
read -p "配置钉钉？输入 y 然后回车（默认不配）：" SETUP_DINGTALK </dev/tty
[ "$SETUP_DINGTALK" = "y" ] || [ "$SETUP_DINGTALK" = "Y" ] && docker exec hermes-agent hermes setup gateway dingtalk 2>&1 | head -10 || true && ok "钉钉通道配置完成"

# ===== 完成 =====
echo ""
echo -e "${GREEN}==============================================${NC}"
echo -e "${GREEN}  🎉 Hermes Agent 安装完成！${NC}"
echo -e "${GREEN}==============================================${NC}"
echo ""
info "你的数据保存在了 /home/hermes/.hermes 文件夹里"
echo ""
info "以后操作 Hermes，请先切换用户："
echo ""
echo "  su - hermes"
echo ""
info "然后就可以用 docker 命令了（已配好免密和 docker 组）："
echo ""
echo "  docker exec -it hermes-agent hermes"
echo ""
info "或者打开网页管理界面："
echo ""
echo "  http://你的服务器IP:8080"
echo ""
info "反馈：Telegram https://t.me/aifengheguai"
info "网站：http://gdibao.com"
