# hermes-agent-cn-deploy

**45岁码盲老男人用 Hermes Agent 智能体缝合的国内一键部署工具**

我是一个45岁从未接触过编程的老男人，靠 **Hermes Agent** 智能体帮忙，一步步写出了这个让国内用户免翻墙安装 Hermes Agent 的自动化部署方案。

## 使用方式

### 一键脚本（Linux / VPS）

把下面任意一行复制到终端里粘贴回车就行：

```
# 有 curl 用这个（推荐）
curl -sL https://gdibao.com/deploy-china/install.sh | sudo bash

# 有 wget 用这个
wget -qO- https://gdibao.com/deploy-china/install.sh | sudo bash

# 没 curl 也没 wget，先手动装 curl
apt-get update && apt-get install -y curl
```

脚本会引导你完成：
- 换国内源
- 安装 Docker
- 选镜像来源（魔塔下载 / Docker Hub 国内镜像）
- 选版本
- 启动容器
- 配置 Dashboard 密码（可选，配好后局域网直接浏览器访问）

### 安装完成后

浏览器访问：`http://你的服务器内网IP:9119`（需要先配置 Dashboard 密码）

终端输入 `hermes` 即可开始聊天。

### 网页端安装

访问 https://gdibao.com 使用网页端部署（无需命令行）。

### Windows 用户

Windows 一键脚本（install.ps1）正在开发中。

## 更多信息

- **魔塔数据集**（镜像下载地址）：https://modelscope.cn/datasets/aifengheguai/hermes-agent-v0.18.2
- **脚本源码**：https://github.com/aifengheguai/hermes-agent/tree/add-china-deploy-scripts/deploy-china
- **反馈**：Telegram https://t.me/aifengheguai
- **网站**：https://gdibao.com

Made with ❤️ by a 45-year-old coding newbie powered by Hermes Agent