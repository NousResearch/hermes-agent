#!/usr/bin/env bash
# =============================================================================
# Hermes Agent 中国区一键安装脚本（Linux 版）
# =============================================================================
# 我是一个 45 岁从来没写过代码的老男人，全靠 Hermes Agent 帮我写出来的。
#
#    curl -sL https://gdibao.com/deploy-china/install.sh | sudo bash
#    curl -sL https://gdibao.com/deploy-china/install.sh | sudo bash -s -- -y   # 静默安装
#
# 装完后在终端输入 hermes 就能跟它聊天了。
# =============================================================================
set +e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${BLUE}[信息]${NC} $1"; }
ok()    { echo -e "${GREEN}[成功]${NC} $1"; }
warn()  { echo -e "${YELLOW}[注意]${NC} $1"; }
err()   { echo -e "${RED}[错误]${NC} $1"; }

HERMES_DATA_DIR="/home/hermes/.hermes"
HERMES_USER="hermes"
TMP_FILES=""   # 记录临时文件，中断时清理用

# ============================================================
# 中断清理：用户按 Ctrl+C 时删除没下载完的文件
# ============================================================
cleanup() {
    echo ""
    warn "安装被中断，正在清理临时文件..."
    for f in $TMP_FILES; do
        [ -f "$f" ] && rm -f "$f" && info "已删除：$f"
    done
    ok "清理完成"
    exit 1
}
trap cleanup EXIT INT TERM

# ============================================================
# 静默模式判断：传 -y 参数则跳过交互选择
# ============================================================
SILENT_MODE=0
for arg in "$@"; do
    [ "$arg" = "-y" ] && SILENT_MODE=1
done

# ============================================================
# 检查 root
# ============================================================
if [ "$(id -u)" != "0" ]; then
    err "请用 root 运行：curl -sL https://gdibao.com/deploy-china/install.sh | sudo bash"
    exit 1
fi

# ============================================================
# 创建 hermes 用户（仅创建用户，不给 sudo 免密）
# ============================================================
info "检查 hermes 用户..."
id "$HERMES_USER" &>/dev/null
if [ $? -ne 0 ]; then
    info "没找到 hermes 用户，正在创建..."
    useradd -m -s /bin/bash "$HERMES_USER"
    if [ $? -eq 0 ]; then
        ok "用户 hermes 创建成功"
    else
        err "创建 hermes 用户失败了！"
        exit 1
    fi
else
    ok "hermes 用户已经存在"
fi

# ============================================================
# 检测操作系统
# ============================================================
info "正在检测你的操作系统..."
OS_ID="unknown"; OS_VERSION_ID=""
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
# 包管理器
# ============================================================
PKG_MGR="apt-get"
PKG_UPDATE="apt-get update -qq"
PKG_INSTALL="apt-get install -y -qq"

case "$OS_ID" in
    ubuntu|debian|kali|linuxmint|deepin|uos) ;;
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
        ;;
esac
info "包管理器：$PKG_MGR"

# ============================================================
# 确保 curl / wget
# ============================================================
info "检查基础工具..."
$PKG_UPDATE 2>/dev/null || true

command -v curl &>/dev/null
if [ $? -ne 0 ]; then
    info "没找到 curl，正在安装..."
    $PKG_INSTALL curl 2>/dev/null || true
fi

command -v wget &>/dev/null
if [ $? -ne 0 ]; then
    info "没找到 wget，正在安装..."
    $PKG_INSTALL wget 2>/dev/null || true
fi

ok "基础工具准备好了"

# ============================================================
# 安装模式选择（静默模式跳过交互）
# ============================================================
echo ""
info "=============================================="
info "  Hermes Agent 中国区一键安装"
info "=============================================="
echo ""

if [ $SILENT_MODE -eq 1 ]; then
    INSTALL_MODE="1"
    info "静默模式：全新安装"
else
    info "你要用哪种方式安装？"
    info "  1) 全新安装（从头开始，推荐）"
    info "  2) 我已经有 Docker 环境了，只配置"
    echo ""
    read -p "请选择 [1/2]（直接回车默认 1）：" INSTALL_MODE </dev/tty
    if [ -z "$INSTALL_MODE" ]; then
        INSTALL_MODE="1"
    fi
fi
echo ""

# ============================================================
# 第1步：换国内源
# ============================================================
info "【第1步】换国内源，让下载飞起来..."

if [ "$INSTALL_MODE" = "1" ]; then
    case "$OS_ID" in
        ubuntu|debian)
            # 备份旧源
            cp /etc/apt/sources.list /etc/apt/sources.list.bak 2>/dev/null || true
            if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then
                cp /etc/apt/sources.list.d/ubuntu.sources /etc/apt/sources.list.d/ubuntu.sources.bak 2>/dev/null || true
            fi
            # Ubuntu 新版 ubuntu.sources 格式（24.04+）
            if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then
                sed -i 's|http://archive.ubuntu.com|http://mirrors.aliyun.com|g' /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null || true
                sed -i 's|http://security.ubuntu.com|http://mirrors.aliyun.com|g' /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null || true
                sed -i 's|https://archive.ubuntu.com|http://mirrors.aliyun.com|g' /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null || true
                sed -i 's|https://security.ubuntu.com|http://mirrors.aliyun.com|g' /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null || true
            fi
            # 传统 sources.list 格式
            sed -i 's|//archive.ubuntu.com|//mirrors.aliyun.com|g' /etc/apt/sources.list 2>/dev/null || true
            sed -i 's|//security.ubuntu.com|//mirrors.aliyun.com|g' /etc/apt/sources.list 2>/dev/null || true
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
    ok "国内源已就绪"
else
    info "已有环境模式，跳过换源"
fi

# ============================================================
# 第2步：安装 Docker
# ============================================================
info "【第2步】安装 Docker..."

if [ "$INSTALL_MODE" = "1" ]; then
    command -v docker &>/dev/null
    if [ $? -eq 0 ]; then
        ok "Docker 已经装好了"
    else
        echo ""
        info "正在安装 Docker，这个过程可能需要一两分钟，请耐心等待..."

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
                    info "正在下载并安装 Docker..."
                    apt-get install -y docker-ce docker-ce-cli containerd.io
                fi
                command -v docker &>/dev/null
                if [ $? -ne 0 ]; then
                    info "Docker 官方版没装上，改用系统自带的 docker.io..."
                    apt-get install -y docker.io
                fi
                ;;
            yum|dnf)
                info "正在添加 Docker 的软件源..."
                $PKG_INSTALL yum-utils 2>/dev/null || true
                $PKG_MGR config-manager --add-repo \
                    https://mirrors.aliyun.com/docker-ce/linux/centos/docker-ce.repo 2>/dev/null || true
                sed -i 's|download.docker.com|mirrors.aliyun.com/docker-ce|g' \
                    /etc/yum.repos.d/docker-ce.repo 2>/dev/null || true
                info "正在下载并安装 Docker..."
                $PKG_INSTALL docker-ce docker-ce-cli containerd.io
                ;;
            apk)
                info "正在安装 Docker..."
                $PKG_INSTALL docker
                rc-update add docker boot 2>/dev/null || true
                ;;
            pacman)
                info "正在安装 Docker..."
                $PKG_INSTALL docker
                systemctl enable docker 2>/dev/null || true
                ;;
            *)
                err "不知道怎么装 Docker，请手动安装后再运行（选模式2）"
                exit 1
                ;;
        esac

        info "正在启动 Docker 服务..."
        systemctl start docker 2>/dev/null || service docker start 2>/dev/null || rc-service docker start 2>/dev/null || true

        command -v docker &>/dev/null
        if [ $? -eq 0 ]; then
            ok "Docker 安装成功！"
        else
            err "Docker 安装失败了，请手动安装后再运行"
            exit 1
        fi
    fi

    # 把 hermes 用户加入 docker 组（这是唯一给的权限）
    info "把 hermes 用户加入 docker 组..."
    groupadd docker 2>/dev/null || true
    usermod -aG docker "$HERMES_USER"
    ok "hermes 用户已加入 docker 组"
else
    command -v docker &>/dev/null
    if [ $? -ne 0 ]; then
        err "请先装好 Docker 再选模式2"
        exit 1
    fi
    ok "Docker 已就绪"
fi

# ============================================================
# 第3步：选择镜像来源 + 版本 + 下载
# ============================================================
echo ""
info "【第3步】获取 Hermes 镜像..."
echo ""
info "请选择镜像来源："
info "  1) 魔塔 ModelScope（国内下载快，推荐）"
info "  2) Docker Hub（从国内镜像拉取）"
echo ""
read -p "请选择 [1/2]（直接回车默认 1）：" IMAGE_SOURCE </dev/tty
echo ""

# ---------- 确保 curl 或 wget 可用（获取版本列表用）----------
DL_QUERY=""
if command -v curl &>/dev/null; then
    DL_QUERY="curl -sL --connect-timeout 10"
elif command -v wget &>/dev/null; then
    DL_QUERY="wget -qO- --timeout=10"
else
    err "既没有 curl 也没有 wget，无法查询版本列表"
    err "请手动选择版本后重试"
    exit 1
fi

# ---------- 方案 A：从魔塔下载 ----------
if [ "$IMAGE_SOURCE" != "2" ]; then
    info "正在查询魔塔上的可用版本..."
    echo ""

    VERSION_LIST=$($DL_QUERY \
        "https://modelscope.cn/api/v1/models/aifengheguai/hermes-agent/repo/files?Revision=master&Root=" \
        2>/dev/null | grep -o '"Name":"[^"]*\.tar"' | sed 's/"Name":"//;s/"//' 2>/dev/null)

    if [ -z "$VERSION_LIST" ]; then
        warn "查询魔塔失败，使用默认版本"
        VERSION_LIST="nousresearch_hermes-agent_latest.tar"
    fi

    echo ""
    info "魔塔上可用的版本："
    echo ""
    echo "$VERSION_LIST" | nl -w2 -s') '
    echo ""
    read -p "请输入版本编号（直接回车选第1个）：" VER_CHOICE </dev/tty
    TAR_FILE=$(echo "$VERSION_LIST" | sed -n "${VER_CHOICE:-1}p" 2>/dev/null)
    if [ -z "$TAR_FILE" ]; then
        TAR_FILE=$(echo "$VERSION_LIST" | head -1)
    fi

    DOWNLOAD_URL="https://modelscope.cn/models/aifengheguai/hermes-agent/resolve/master/${TAR_FILE}"
    info "下载版本：$TAR_FILE"
    info "文件大概 1-2GB，根据网速可能需要几分钟，请耐心等待..."
    echo ""

    TMP_FILES="$TMP_FILES $TAR_FILE"

    if command -v aria2c &>/dev/null; then
        info "检测到 aria2（多线程下载器），速度更快！"
        aria2c -s 16 -x 16 -k 1M "$DOWNLOAD_URL" 2>&1 | tail -5
    elif command -v curl &>/dev/null; then
        info "正在用 curl 下载（看到进度条就说明在跑）..."
        curl -L -o "$TAR_FILE" "$DOWNLOAD_URL"
    else
        info "正在用 wget 下载..."
        wget -O "$TAR_FILE" "$DOWNLOAD_URL"
    fi

    if [ ! -f "$TAR_FILE" ]; then
        err "下载失败！"
        cleanup
        exit 1
    fi

    TMP_FILES=""
    ok "下载完成：$(du -h "$TAR_FILE" | cut -f1)"

    # 加载镜像到 Docker
    info "正在加载镜像到 Docker..."
    if [[ "$TAR_FILE" == *.tar.gz ]] || [[ "$TAR_FILE" == *.tgz ]]; then
        info "正在解压，请耐心等待..."
        echo ""
        gzip -d "$TAR_FILE"
        TAR_FILE="${TAR_FILE%.gz}"
        TAR_FILE="${TAR_FILE%.tgz}.tar"
    fi
    echo ""
    info "正在加载镜像，请耐心等待..."
    echo ""
    docker load -i "$TAR_FILE" 2>&1 | tee /tmp/docker_load_output.txt
    LOAD_EXIT_CODE=${PIPESTATUS[0]}
    if [ $LOAD_EXIT_CODE -ne 0 ]; then
        err "docker load 失败"
        exit 1
    fi
    IMAGE_NAME=$(grep "Loaded image:" /tmp/docker_load_output.txt | sed 's/.*Loaded image: //')
    if [ -z "$IMAGE_NAME" ]; then
        warn "未识别镜像名，用默认值"
        IMAGE_NAME="nousresearch/hermes-agent:main"
    fi
    rm -f "$TAR_FILE" /tmp/docker_load_output.txt 2>/dev/null
    ok "镜像加载成功：$IMAGE_NAME"

# ---------- 方案 B：从 Docker Hub 国内镜像拉取 ----------
else
    info "从 Docker Hub 国内镜像拉取..."
    echo ""
    info "请选择 Docker Hub 国内镜像源："
    info "  1) 阿里云（registry.cn-hangzhou.aliyuncs.com）"
    info "  2) 腾讯云（mirror.ccs.tencentyun.com）"
    info "  3) 网易（hub-mirror.c.163.com）"
    info "  4) DaoCloud（docker.m.daocloud.io）"
    echo ""
    read -p "请选择 [1-4]（直接回车默认 1）：" MIRROR_CHOICE </dev/tty

    case "$MIRROR_CHOICE" in
        2) DH_MIRROR="mirror.ccs.tencentyun.com" ;;
        3) DH_MIRROR="hub-mirror.c.163.com" ;;
        4) DH_MIRROR="docker.m.daocloud.io" ;;
        *) DH_MIRROR="registry.cn-hangzhou.aliyuncs.com" ;;
    esac
    info "使用镜像：$DH_MIRROR"
    echo ""

    # 查询 Docker Hub 可用标签
    info "正在查询 Docker Hub 上的可用版本..."
    DH_TAGS=$($DL_QUERY \
        "https://hub.docker.com/v2/repositories/nousresearch/hermes-agent/tags?page_size=20" \
        2>/dev/null | grep -o '"name":"[^"]*"' | sed 's/"name":"//;s/"//' 2>/dev/null)

    if [ -z "$DH_TAGS" ]; then
        warn "查询 Docker Hub 失败，使用默认标签 latest"
        DH_TAGS="latest"
    fi

    echo ""
    info "Docker Hub 上可用的版本标签："
    echo ""
    echo "$DH_TAGS" | nl -w2 -s') '
    echo ""
    read -p "请输入版本编号（直接回车选第1个）：" TAG_CHOICE </dev/tty
    DH_TAG=$(echo "$DH_TAGS" | sed -n "${TAG_CHOICE:-1}p" 2>/dev/null)
    if [ -z "$DH_TAG" ]; then
        DH_TAG=$(echo "$DH_TAGS" | head -1)
    fi

    # 从国内镜像拉取
    # Docker Hub 镜像的国内加速格式：mirror/library/image:tag
    FULL_IMAGE="${DH_MIRROR}/nousresearch/hermes-agent:${DH_TAG}"
    info "拉取版本：$DH_TAG"
    info "拉取地址：$FULL_IMAGE"
    info "镜像大概 2-3GB，根据网速可能需要几分钟..."
    echo ""
    docker pull "$FULL_IMAGE" 2>&1
    PULL_EXIT=$?
    if [ $PULL_EXIT -ne 0 ]; then
        err "Docker pull 失败，请换一个镜像源试试"
        exit 1
    fi

    # 打回原名，方便后续启动
    IMAGE_NAME="nousresearch/hermes-agent:${DH_TAG}"
    docker tag "$FULL_IMAGE" "$IMAGE_NAME" 2>/dev/null || true
    ok "镜像拉取成功：$IMAGE_NAME"
fi

# ============================================================
# 第5步：启动容器
# 端口映射：127.0.0.1:8787 → 容器内 8787
# ============================================================
info "【第4步】启动 Hermes Agent 容器..."

docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q '^hermes-agent$'
if [ $? -eq 0 ]; then
    warn "发现以前装过的 hermes-agent 容器，正在删除重建..."
    docker rm -f hermes-agent >/dev/null 2>&1
fi

info "创建数据目录：$HERMES_DATA_DIR"
mkdir -p "$HERMES_DATA_DIR" 2>/dev/null

# 把数据目录的所有权彻底锁定为 hermes 用户
chown -R "$HERMES_USER:$HERMES_USER" "$HERMES_DATA_DIR"

info "正在启动容器..."
docker run -d -it \
    --name hermes-agent \
    --restart unless-stopped \
    -v "$HERMES_DATA_DIR:/opt/data" \
    -p 127.0.0.1:8787:8787 \
    -p 127.0.0.1:9119:9119 \
    "$IMAGE_NAME" 2>&1

sleep 3

docker ps --filter name=hermes-agent --format '{{.Status}}' 2>/dev/null | grep -q .
if [ $? -eq 0 ]; then
    ok "容器已启动！端口 8787"
else
    err "容器启动失败了"
    docker logs hermes-agent 2>/dev/null | tail -10
    exit 1
fi

# ============================================================
# Nginx 反向代理 + 密码保护
# 不直接把容器端口暴露到局域网，一律走 Nginx 加密码
# ============================================================
info "配置 Nginx 反向代理..."
echo ""
info "是否配置 Nginx 反代 + 密码登录？（推荐）"
info "配好后局域网其他电脑输入密码才能访问 Hermes"
read -p "配置 Nginx 反代？[Y/n]：" SETUP_NGINX </dev/tty

if [ "$SETUP_NGINX" != "n" ] && [ "$SETUP_NGINX" != "N" ]; then
    info "正在安装 Nginx 和密码工具..."
    $PKG_INSTALL nginx apache2-utils 2>/dev/null || true

    # 创建登录用户名和密码
    echo ""
    info "请设置反代登录的用户名和密码"
    read -p "用户名（默认 hermesadmin）：" NGINX_USER </dev/tty
    [ -z "$NGINX_USER" ] && NGINX_USER="hermesadmin"

    info "请输入密码（输入时不会显示）："
    htpasswd -c /etc/nginx/.htpasswd "$NGINX_USER"

    # 写入 Nginx 配置
    cat > /etc/nginx/sites-enabled/hermes-proxy << NGINXEOF
server {
    listen 8789;
    server_name _;
    auth_basic "Hermes Agent - 请输入用户名和密码";
    auth_basic_user_file /etc/nginx/.htpasswd;
    location / {
        proxy_pass http://127.0.0.1:8787;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
}
server {
    listen 9118;
    server_name _;
    auth_basic "Hermes Dashboard - 请输入用户名和密码";
    auth_basic_user_file /etc/nginx/.htpasswd;
    location / {
        proxy_pass http://127.0.0.1:9119;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
}
NGINXEOF

    # 移除默认站点（避免冲突）
    rm -f /etc/nginx/sites-enabled/default 2>/dev/null

    # 检查配置并重载
    nginx -t 2>/dev/null
    if [ $? -eq 0 ]; then
        systemctl reload nginx 2>/dev/null || nginx -s reload 2>/dev/null || true
        ok "Nginx 反代配置成功！"
    else
        warn "Nginx 配置有误，请手动检查 /etc/nginx/sites-enabled/hermes-proxy"
    fi

    # 放行防火墙（如果有 UFW）
    if command -v ufw &>/dev/null; then
        ufw allow 8789/tcp 2>/dev/null || true
        ufw allow 9118/tcp 2>/dev/null || true
    fi
else
    info "跳过 Nginx 配置，端口直接暴露到局域网（不推荐）"
    # 不改端口映射，保持默认
fi

# ============================================================
# 添加快捷命令（防重复写入）
# ============================================================
info "添加快捷命令..."
ALIAS_LINE="alias hermes='docker exec -it hermes-agent hermes'"

if grep -q "$ALIAS_LINE" /root/.bashrc 2>/dev/null; then
    ok "root 用户的快捷命令已存在，跳过"
else
    echo "$ALIAS_LINE" >> /root/.bashrc
    ok "已为 root 用户添加 hermes 快捷命令"
fi

if grep -q "$ALIAS_LINE" /home/$HERMES_USER/.bashrc 2>/dev/null; then
    ok "hermes 用户的快捷命令已存在，跳过"
else
    echo "$ALIAS_LINE" >> /home/$HERMES_USER/.bashrc 2>/dev/null || true
    ok "已为 hermes 用户添加 hermes 快捷命令"
fi

# ============================================================
# 取消 trap（安装顺利完成，不需要 cleanup 了）
# ============================================================
trap - EXIT INT TERM

# ============================================================
# 完成！
# ============================================================
echo ""
echo -e "${GREEN}==============================================${NC}"
echo -e "${GREEN}  🎉 Hermes Agent 安装完成！${NC}"
echo -e "${GREEN}==============================================${NC}"
echo ""
echo -e "${GREEN}安装完成！请输入 hermes setup 来手动配置 API 和聊天网关。${NC}"
echo ""
info "你的数据保存在：/home/hermes/.hermes"
echo ""
info "安装后请执行 source ~/.bashrc 以立即使用 hermes 命令，"
info "或退出当前终端重新登录。"
echo ""
info "常用命令："
echo ""
echo "  hermes             跟她聊天"
echo "  hermes dashboard   启动网页管理界面（端口 9119）"
echo "  hermes setup       配置 API Key 和聊天通道"
echo ""
info "局域网访问（如果配了 Nginx 反代）："
echo "  http://你的服务器内网IP:8789    Hermes WebSocket"
echo "  http://你的服务器内网IP:9118    Dashboard 网页"
echo ""
info "如果没配 Nginx，或已在服务器本机操作，直接用："
echo ""
echo "  su - hermes"
echo ""
info "然后直接输入 hermes 即可开始聊天。"
echo ""
info "网页管理界面：http://你的服务器内网IP:8787"
echo ""
info "反馈：https://t.me/aifengheguai"
info "网站：http://gdibao.com"
echo ""
