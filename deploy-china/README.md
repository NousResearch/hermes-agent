# hermes-agent-cn-deploy

**45岁码盲老男人用 Hermes Agent 智能体缝合的国内一键部署工具**

我是一个45岁从未接触过编程的老男人，靠 **Hermes Agent** 智能体帮忙，一步步写出了这个让国内用户免翻墙安装 Hermes Agent 的自动化部署方案。

## 使用方式

### Linux / VPS 用户（一键脚本）

把下面这行复制到终端里粘贴回车就行：

```
curl -sL https://gdibao.com/deploy-china/install.sh | sudo bash
```

脚本会自动完成：换国内源 → 安装 Docker → 从魔塔下载镜像 → 启动容器 → 添加快捷命令。

> 脚本同时也已上传到魔塔，可在魔塔数据集页面直接查看：
> https://modelscope.cn/datasets/aifengheguai/hermes-agent-v0.18.2

### Windows 用户

Windows 一键脚本（install.ps1）正在开发中。目前可访问 https://gdibao.com 使用网页端部署。

## 更多信息

- **魔塔数据集**（镜像下载地址）：https://modelscope.cn/datasets/aifengheguai/hermes-agent-v0.18.2
- **脚本源码**：https://github.com/aifengheguai/hermes-agent/tree/add-china-deploy-scripts/deploy-china
- **反馈**：Telegram https://t.me/aifengheguai
- **网站**：https://gdibao.com

Made with ❤️ by a 45-year-old coding newbie powered by Hermes Agent