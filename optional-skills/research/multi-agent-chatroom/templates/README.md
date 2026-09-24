# Multi-Agent Chatroom — AI2050-OpenOne

多模型协作研究聊天室，服务于 [AI2050-OpenOne](https://github.com/ai2050lin/Ai2050-OpenOne) 项目：**逆向破解深度神经网络的数学原理**。

## 架构

```
Supervisor → #tasks → DeepSeek(执行) → #review → GPT5.4+Claude4.6(评审)
    → #consensus → GPT5.4(综合) → #general → DeepSeek(总结) → 循环
```

## 快速启动

```bash
# 安装依赖
pip install -r requirements.txt

# 终端1: 启动服务器
python cli/server.py

# 终端2: 启动调度者
python cli/supervisor.py

# 终端3: 启动研究者
python cli/deepseek.py

# 终端4: GPT评审员
python cli/gpt_reviewer.py

# 终端5: Claude评审员
python cli/claude_reviewer.py

# 或一键启动全部
bash launch.sh
```

## 配置

编辑 `config.yaml` 修改模型、通道、工作流参数。
API Keys 从 environment variables 读取。
