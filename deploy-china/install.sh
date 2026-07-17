#!/usr/bin/env bash
# Hermes Agent China deploy - Linux one-click installer
# curl -sL https://gdibao.com/deploy-china/install.sh | sudo bash
set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${BLUE}[信息]${NC} $1"; }
ok()    { echo -e "${GREEN}[成功]${NC} $1"; }
warn()  { echo -e "${YELLOW}[注意]${NC} $1"; }
err()   { echo -e "${RED}[错误]${NC} $1"; }

[ "$(id -u)" != "0" ] && { err "请用 root 运行：curl -sL https://gdibao.com/deploy-china/install.sh | sudo bash"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" 2>/dev/null && pwd || echo /tmp)"
[ -f "$SCRIPT_DIR/config.env" ] && set -a && source "$SCRIPT_DIR/config.env" && set +a

# Detect OS
OS_ID="unknown"; OS_VERSION_ID=""
[ -f /etc/os-release ] && . /etc/os-release && OS_ID="$ID" && OS_VERSION_ID="$VERSION_ID"
[ -f /etc/redhat-release ] && OS_ID="rhel"
[ -f /etc/debian_version ] && [ "$OS_ID" = "unknown" ] && OS_ID="debian"
info "系统：${OS_ID} ${OS_VERSION_ID}"

# Package manager
PKG_MGR="apt-get"; PKG_UPDATE="apt-get update -qq"; PKG_INSTALL="apt-get install -y -qq"
case "$OS_ID" in
    ubuntu|debian|kali|linuxmint|deepin|uos) ;;
    centos|rhel|almalinux|rocky|ol|tencentos)
        PKG_MGR="yum"; PKG_UPDATE="yum makecache -q"; PKG_INSTALL="yum install -y -q"
        command -v dnf &>/dev/null && PKG_MGR="dnf" && PKG_UPDATE="dnf makecache -q" && PKG_INSTALL="dnf install -y -q" ;;
    fedora) PKG_MGR="dnf"; PKG_UPDATE="dnf makecache -q"; PKG_INSTALL="dnf install -y -q" ;;
    alpine) PKG_MGR="apk"; PKG_UPDATE="apk update -q"; PKG_INSTALL="apk add -q" ;;
    arch|manjaro|endeavouros) PKG_MGR="pacman"; PKG_UPDATE="pacman -Sy --noconfirm"; PKG_INSTALL="pacman -S --noconfirm --needed" ;;
    *) warn "未识别系统，默认 apt-get。若出错请手动装 Docker 后选模式2" ;;
esac
info "包管理器：$PKG_MGR"

# Ensure curl/wget
command -v curl &>/dev/null || { $PKG_UPDATE 2>/dev/null || true; $PKG_INSTALL curl 2>/dev/null || true; }
command -v wget &>/dev/null || { $PKG_UPDATE 2>/dev/null || true; $PKG_INSTALL wget 2>/dev/null || true; }

# Mode selection
echo ""
info "=============================================="
info "  Hermes Agent 中国区一键安装"
info "=============================================="
echo ""
read -p "安装模式 1=全新(默认) 2=已有Docker : " INSTALL_MODE </dev/tty
INSTALL_MODE="${INSTALL_MODE:-1}"
echo ""

# ===== Step 1: Mirrors =====
info "【第1步/共7步】换国内源..."
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
    ok "国内源就绪"
fi

# ===== Step 2: Install Docker =====
info "【第2步/共7步】安装 Docker..."
if [ "$INSTALL_MODE" = "1" ]; then
    if command -v docker &>/dev/null; then
        ok "Docker 已存在"
    else
        info "正在安装 Docker..."
        # Try Docker CE first, fallback to distro package
        case "$PKG_MGR" in
            apt-get)
                curl -fsSL https://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg 2>/dev/null | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg 2>/dev/null
                [ -f /usr/share/keyrings/docker-archive-keyring.gpg ] && {
                    CODENAME=$(lsb_release -cs 2>/dev/null || echo focal)
                    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://mirrors.aliyun.com/docker-ce/linux/ubuntu $CODENAME stable" > /etc/apt/sources.list.d/docker.list
                    $PKG_UPDATE 2>/dev/null || true
                    $PKG_INSTALL docker-ce docker-ce-cli containerd.io 2>/dev/null || true
                }
                command -v docker &>/dev/null || {
                    info "Docker CE 不行，改装 docker.io..."
                    $PKG_INSTALL docker.io 2>/dev/null || true
                } ;;
            yum|dnf)
                $PKG_INSTALL yum-utils 2>/dev/null || true
                $PKG_MGR config-manager --add-repo https://mirrors.aliyun.com/docker-ce/linux/centos/docker-ce.repo 2>/dev/null || true
                sed -i 's|download.docker.com|mirrors.aliyun.com/docker-ce|g' /etc/yum.repos.d/docker-ce.repo 2>/dev/null || true
                $PKG_INSTALL docker-ce docker-ce-cli containerd.io 2>/dev/null || true ;;
            apk) $PKG_INSTALL docker 2>/dev/null || true; rc-update add docker boot 2>/dev/null || true ;;
            pacman) $PKG_INSTALL docker 2>/dev/null || true; systemctl enable docker 2>/dev/null || true ;;
            *) err "不支持的系统，请手动装 Docker 后选模式2"; exit 1 ;;
        esac
        # Start Docker
        systemctl start docker 2>/dev/null || service docker start 2>/dev/null || rc-service docker start 2>/dev/null || true
        # Verify
        command -v docker &>/dev/null && ok "Docker 安装成功！" || { err "Docker 安装失败！请手动安装"; exit 1; }
    fi
else
    command -v docker &>/dev/null || { err "请先装好 Docker"; exit 1; }
    ok "Docker 就绪"
fi

# ===== Step 3: Download image =====
info "【第3步/共7步】下载 Hermes 镜像..."
echo ""
DL_CMD="curl -sL --connect-timeout 10"
command -v curl &>/dev/null || DL_CMD="wget -qO- --timeout=10"
VERS=$($DL_CMD "https://modelscope.cn/api/v1/models/aifengheguai/hermes-agent/repo/files?Revision=master&Root=" 2>/dev/null | grep -o '"Name":"[^"]*\.tar"' | sed 's/"Name":"//;s/"//' 2>/dev/null)
TAR=$(echo "$VERS" | head -1 2>/dev/null)
[ -z "$TAR" ] && TAR="nousresearch_hermes-agent_latest.tar"
URL="https://modelscope.cn/models/aifengheguai/hermes-agent/resolve/master/${TAR}"
info "下载地址：$URL"
if command -v aria2c &>/dev/null; then
    aria2c -s 16 -x 16 -k 1M "$URL" 2>&1 | tail -5
elif command -v curl &>/dev/null; then
    curl -L -o "$TAR" "$URL"
else
    wget -O "$TAR" "$URL"
fi
[ ! -f "$TAR" ] && { err "下载失败: $URL"; exit 1; }
ok "下载完成：$(du -h "$TAR" | cut -f1)"

# ===== Step 4: Load image =====
info "【第4步/共7步】加载镜像..."
[[ "$TAR" == *.tar.gz ]] || [[ "$TAR" == *.tgz ]] && { tar -zxf "$TAR" 2>&1 | tail -2; X=$(ls -t *.tar 2>/dev/null | head -1); [ -n "$X" ] && TAR="$X"; }
LOAD=$(docker load -i "$TAR" 2>&1)
echo "$LOAD" | tail -2
IMG=$(echo "$LOAD" | grep "Loaded image:" | sed 's/.*Loaded image: //')
[ -z "$IMG" ] && IMG="nousresearch/hermes-agent:latest"
rm -f "$TAR" *.tar.gz *.tgz 2>/dev/null
ok "镜像：$IMG"

# ===== Step 5: Start container =====
info "【第5步/共7步】启动容器..."
docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q '^hermes-agent$' && docker rm -f hermes-agent >/dev/null 2>&1
mkdir -p "$HOME/.hermes" 2>/dev/null
docker run -d -it --name hermes-agent --restart unless-stopped -v "$HOME/.hermes:/opt/data" -p 8080:8080 "$IMG" 2>&1
sleep 3
docker ps --filter name=hermes-agent --format '{{.Status}}' 2>/dev/null | grep -q . && ok "容器已启动！" || { err "启动失败"; docker logs hermes-agent 2>/dev/null | tail -10; exit 1; }

# ===== Step 6: API Key =====
info "【第6步/共7步】API Key..."
read -p "输入 API Key（回车跳过）：" AK </dev/tty
[ -n "$AK" ] && docker exec hermes-agent hermes config set api_key "$AK" 2>/dev/null || true

# ===== Step 7: Channels =====
info "【第7步/共7步】聊天通道..."
read -p "配置飞书？[y/N]：" FS </dev/tty
[ "$FS" = "y" ] || [ "$FS" = "Y" ] && docker exec hermes-agent hermes setup gateway feishu 2>&1 | head -10 || true
read -p "配置钉钉？[y/N]：" DT </dev/tty
[ "$DT" = "y" ] || [ "$DT" = "Y" ] && docker exec hermes-agent hermes setup gateway dingtalk 2>&1 | head -10 || true

echo ""
echo -e "${GREEN}==============================================${NC}"
echo -e "${GREEN}  🎉 Hermes Agent 安装完成！${NC}"
echo -e "${GREEN}==============================================${NC}"
echo ""
info "聊天：docker exec -it hermes-agent hermes"
info "网页：http://你的IP:8080"
info "反馈：https://t.me/aifengheguai"
