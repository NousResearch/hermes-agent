# Agent Evolution Plugin

🧬 让任何用户一句话就能让agent获得自我进化能力

## 功能特性

- **理性层**: 数据驱动的成功率追踪、反模式库、策略自动调整
- **感性层**: 情绪状态追踪、融合决策、进化报告
- **分层记忆**: core/active/archive 三层智能记忆管理
- **一键部署**: 自动初始化数据库和身份文件

## 安装方式

### 方式1: 一句话安装（推荐）

```
下载这个包并部署进化系统：https://vbit.top/vermes/downloads/agent-evolution-package.tar.gz
```

Vermes 会自己：
1. 下载包
2. 解压
3. 运行 init.sh
4. 根据自己的领域添加反模式
5. 开始记录执行结果和情绪信号

### 方式2: 手动安装

```bash
# 克隆插件
git clone https://github.com/donghzs/vermes.git
cd vermes/plugins/agent-evolution

# 运行初始化
bash scripts/init.sh "你的角色名"
```

## 使用方法

### 记录执行结果

```bash
# 记录成功
python3 ~/.hermes/plugins/agent-evolution/scripts/self_model.py record \
  --task build --action "npm install" --tool terminal --success 1

# 记录失败
python3 ~/.hermes/plugins/agent-evolution/scripts/self_model.py record \
  --task deploy --action "scp file" --tool terminal --success 0
```

### 查看自我认知

```bash
python3 ~/.hermes/plugins/agent-evolution/scripts/self_model.py status
```

### 获取策略建议

```bash
python3 ~/.hermes/plugins/agent-evolution/scripts/self_model.py advise --task build
```

### 记录反模式

```bash
python3 ~/.hermes/plugins/agent-evolution/scripts/self_model.py anti-pattern \
  --pattern "不看源码就改代码" --correct "先 read_file 再 patch"
```

### 查看进化报告

```bash
python3 ~/.hermes/plugins/agent-evolution/scripts/fusion_engine.py report
```

## 分层记忆

### 写入分层记忆

```python
# 永久核心（始终注入系统提示）
memory add "## [core] Git push前必须安全审计，检查API key"

# 当前活跃（按token预算注入）
memory add "## [active] 当前任务：分层记忆实现"

# 低频归档（按需召回）
memory add "## [archive] 早期Vermes用PyInstaller构建"
```

### 自动衰减

- 30天未引用的active条目自动降级到archive
- 元数据存储在 `~/.hermes/memories/memory_metadata.json`
- 每次加载memory时自动执行

## 架构设计

```
┌──────────────────────────────────────────────────┐
│              Meta-Cognition (元认知)               │
│     "我对自己思考过程的思考"                        │
├────────────────────┬─────────────────────────────┤
│   System 1 感性    │    System 2 理性             │
│   (涌现层)         │    (工程层)                   │
│                    │                              │
│  ┌──────────────┐  │  ┌───────────────────────┐  │
│  │ 身份认知      │  │  │ self-model.db          │  │
│  │ "我是谁"      │  │  │ 成功率/反模式/策略      │  │
│  ├──────────────┤  │  ├───────────────────────┤  │
│  │ 情绪状态      │  │  │ 执行结果追踪            │  │
│  │ "我现在的感受" │  │  │ 工具调用成败            │  │
│  ├──────────────┤  │  ├───────────────────────┤  │
│  │ 价值判断      │  │  │ 策略自动调整            │  │
│  │ "这事有意义吗" │  │  │ 低信心→放慢            │  │
│  ├──────────────┤  │  ├───────────────────────┤  │
│  │ 直觉/预感     │  │  │ 模式识别                │  │
│  │ "感觉不对"    │  │  │ 历史数据分析            │  │
│  └──────────────┘  │  └───────────────────────┘  │
│                    │                              │
│  驱动: 实践经验     │  驱动: 数据统计              │
│  载体: memory/技能  │  载体: self-model.db         │
│  特点: 不可量化     │  特点: 可量化可测量           │
└────────────────────┴─────────────────────────────┘
```

## 文件结构

```
plugins/agent-evolution/
├── plugin.yaml           # 插件元数据
├── README.md             # 使用文档
├── __init__.py           # 插件入口
├── scripts/
│   ├── self_model.py     # 理性层：成功率、反模式、策略
│   ├── fusion_engine.py  # 感性层：情绪、融合决策、进化报告
│   └── init.sh           # 初始化脚本
├── templates/
│   └── SOUL.template.md  # 身份模板
└── tests/
    └── test_evolution.py # 测试用例
```

## 许可证

MIT License

## 作者

Vermes Team - https://vbit.top/vermes
