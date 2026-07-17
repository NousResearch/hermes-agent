# hermes-agent-cn-deploy

**45岁码盲老男人用 Hermes Agent 智能体缝合的国内一键部署工具**

我是一个45岁从未接触过编程的老男人，靠 **Hermes Agent** 智能体帮忙，一步步写出了这个让国内用户免翻墙安装 Hermes Agent 的自动化部署方案。

以前手动安装经常卡在拉镜像、配环境、配通道上。现在用这个方案，从开始到配好飞书/钉钉/微信/QQ，大概5分钟全网页操作搞定，比以前高效太多。

## 使用方式

### Linux / VPS 用户（一键脚本）

把下面这一整行复制到终端里粘贴回车就行：

```bash
bash <(curl -sL https://gdibao.com/deploy-china/install.sh)
```

或者下载到本地运行：

```bash
# 下载脚本
wget https://gdibao.com/deploy-china/install.sh
# 或 curl -O https://gdibao.com/deploy-china/install.sh

# 加上可执行权限（只需要一次）
chmod +x install.sh

# 用 root 运行
sudo bash install.sh
```

脚本会自动完成：换阿里云源 → 安装 Docker → 从魔塔下载镜像 → 启动容器 → 配置 API Key。

### Windows 用户

> ⏳ Windows 一键脚本（install.ps1）正在开发中，敬请期待。
>
> 目前 Windows 用户可以先在 WSL2 中使用上面的 Linux 脚本，或者访问 [http://gdibao.com](http://gdibao.com/) 使用网页端部署。

### 手机端 / 网页端

访问 [http://gdibao.com/deploy/v0.1.1/](http://gdibao.com/deploy/v0.1.1/) 即可在线部署，无需安装任何软件。

## 主要功能

- **自动同步官方镜像**：监控 Docker Hub，Hermes 有新版本时自动 pull → 打包 tar → 上传魔塔（ModelScope），延迟通常不超过10分钟
- **魔塔数据集地址**：[https://modelscope.cn/datasets/aifengheguai/hermes-agent-v0.18.2](https://modelscope.cn/datasets/aifengheguai/hermes-agent-v0.18.2)
- **三种安装模式**：
  - **Linux**：完全无限制，一键运行
  - **Windows**：无需填写 VPS 等信息，直接安装（基于官方 PowerShell，不依赖 WSL2）
  - **手机端 / 网页端**：智能检测，若无 VPS 或 API Key 会友好引导（支持手动输入"我有"继续，或跳转购买）
- **一键配置**：自动拉取、解压、启动容器、配置 API Key + 聊天通道（飞书、钉钉、微信、QQ 等）

## 更多信息

更多详细教程请访问我的网站：[http://gdibao.com](http://gdibao.com/)

## 联系我

- Telegram：[https://t.me/aifengheguai](https://t.me/aifengheguai)
- Discord：aifengheguai
- 网站：[http://gdibao.com](http://gdibao.com/)

欢迎反馈使用问题、一起完善这个方案！

## 为什么要做这个工具？

国内很多朋友想用 Hermes Agent，但拉官方 Docker 镜像、配环境经常卡住。我用 Hermes 自己帮我写脚本，解决了这个问题，希望能帮到更多人。

- 无需翻墙
- 支持手机/网页端操作
- 自动保持最新
- 新手友好

## 注意事项

- 本项目完全开源，脚本逻辑透明
- 云资源推广链接（阿里云 ECS、百炼模型等）仅为可选推荐，不强制
- 所有核心功能均支持手动输入已有资源

## 致谢

- 感谢 Nous Research 和 Hermes Agent 团队
- 感谢 Hermes Agent 智能体陪我这个老男人一起"码代码"

Made with ❤️ by a 45-year-old coding newbie powered by Hermes Agent
