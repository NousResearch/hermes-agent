---
name: tradingagents-setup
description: "Install and configure TradingAgents (multi-agent financial trading framework) with OpenRouter + free models. Includes troubleshooting for Yahoo Finance rate limits, terminal compatibility issues, and data source switching. Based on real-world testing and successful deployment."
version: "2.0"
author: Judy (朱迪) / Hermes adaptation - Updated 2026-05-05
license: MIT
---

# TradingAgents 完整安装与配置指南

基于实际测试经验整理的完整指南。涵盖安装、配置、问题排查和成功运行的全流程。

---

## 触发条件

- 用户要求安装/运行 TradingAgents
- 用户提到多智能体金融交易框架
- 需要使用 OpenRouter + 免费模型
- 终端不支持交互式 CLI（无箭头键）

---

## 环境要求

- **Python**: 3.11+ （检查：`python3 --version`）
- **uv**: 包管理器（安装：`curl -LsSf https://astral.sh/uv/install.sh | sh`）
- **Git**: 版本控制
- **OpenRouter API Key**: 从 https://openrouter.ai/keys 获取
- **Alpha Vantage API Key**: 免费申请 https://www.alphavantage.co/support/#api-key

---

## 完整安装流程

### Step 1: 克隆仓库

```bash
cd /tmp  # 或你喜欢的目录
git clone https://github.com/TauricResearch/TradingAgents.git
cd TradingAgents
```

### Step 2: 创建虚拟环境

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### Step 3: 安装依赖

**推荐：使用 uv（解决 SSL 问题）**

```bash
uv sync
```

**备选：使用 pip（可能遇到 SSL 错误）**

```bash
pip install . -i https://pypi.org/simple/
```

**验证安装：**

```bash
uv run tradingagents --help
# 或
python -c "import tradingagents; print('安装成功')"
```

### Step 4: 配置环境变量

创建 `/tmp/TradingAgents/.env` 文件：

```bash
cat > .env << 'EOF'
OPENAI_API_KEY=[YOUR_OPENROUTER_KEY]
OPENAI_BASE_URL=https://openrouter.ai/api/v1
ALPHA_VANTAGE_API_KEY=[YOUR_ALPHA_VANTAGE_KEY]
EOF
```

**重要：** 替换 `[YOUR_OPENROUTER_KEY]` 和 `[YOUR_ALPHA_VANTAGE_KEY]` 为实际密钥。

### Step 5: 创建运行脚本

创建 `/tmp/TradingAgents/run_analysis.py`：

```python
#!/usr/bin/env python3
"""Run TradingAgents with OpenRouter - using free model"""
import os

# Set environment variables for OpenRouter and Alpha Vantage
os.environ['OPENAI_API_KEY'] = '[YOUR_OPENROUTER_KEY]'
os.environ['OPENAI_BASE_URL'] = 'https://openrouter.ai/api/v1'
os.environ['ALPHA_VANTAGE_API_KEY'] = '[YOUR_ALPHA_VANTAGE_KEY]'

# Import after setting env
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

# Create config - use openrouter provider
config = DEFAULT_CONFIG.copy()
config['llm_provider'] = 'openrouter'
config['deep_think_llm'] = 'tencent/hy3-preview:free'
config['quick_think_llm'] = 'tencent/hy3-preview:free'
config['output_language'] = 'English'

# Switch to Alpha Vantage to avoid Yahoo Finance rate limit
config['data_vendors'] = {
    'core_stock_apis': 'alpha_vantage',
    'technical_indicators': 'alpha_vantage',
    'fundamental_data': 'alpha_vantage',
    'news_data': 'alpha_vantage',
}

# Initialize TradingAgents
ta = TradingAgentsGraph(debug=True, config=config)

# Run analysis
print("Starting TradingAgents analysis for SPY...")
print("Using OpenRouter with tencent/hy3-preview:free (free model)")
print("Analysis date: 2024-12-18")
print("-" * 50)

result = ta.propagate(company_name="SPY", trade_date="2024-12-18")

print("\n" + "=" * 50)
print("FINAL TRADE DECISION:")
print("=" * 50)
if result:
    print(result)
else:
    print("No result returned")
```

**记得替换脚本中的 API keys！**

### Step 6: 运行分析

```bash
cd /tmp/TradingAgents
source venv/bin/activate
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
timeout 300 python run_analysis.py
```

**注意：** 首次运行可能需要 5-10 分钟（LLM API 调用 + 数据获取）。

---

## 问题排查指南

### 问题 1: pip 安装时出现 SSL 证书错误

**症状：**
```
pip._vendor.urllib3.exceptions.MaxRetryError: 
SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED]
```

**解决方案：** 使用 uv 替代 pip：
```bash
uv sync
```

---

### 问题 2: 终端不支持箭头键（无法使用交互式 CLI）

**症状：**
- 运行 `tradingagents` 交互式命令时，无法选择选项
- 箭头键在 Hermes 终端中不起作用

**解决方案：** 使用 Python 脚本直接调用 API（参见上面的 `run_analysis.py`）

---

### 问题 3: Yahoo Finance 限流

**症状：**
```
yfinance.exceptions.YFRateLimitError: Too Many Requests. Rate limited.
```

**原因：** Yahoo Finance 对 IP 地址进行限流，不是针对特定日期或股票代码。

**解决方案：** 切换到 Alpha Vantage：

1. **申请免费 API Key**：访问 https://www.alphavantage.co/support/#api-key
   - 选择用户类型："I am from the Trading Agents project on Github"
   - 填写邮箱（会收到验证邮件）
   - 点击验证链接，获得 API Key（格式类似：`KB4Z7J3TXJZMV1ZR`）

2. **添加到 `.env`**：
   ```bash
   echo "ALPHA_VANTAGE_API_KEY=你的key" >> .env
   ```

3. **更新脚本配置**：
   ```python
   config['data_vendors'] = {
       'core_stock_apis': 'alpha_vantage',
       'technical_indicators': 'alpha_vantage',
       'fundamental_data': 'alpha_vantage',
       'news_data': 'alpha_vantage',
   }
   ```

**Alpha Vantage 免费版限制：**
- 每分钟 5 次请求 ✅（够用）
- 每天 500 次请求 ✅（够用）

---

### 问题 4: OpenRouter 模型地区不可用

**症状：**
```
Error: This model is not available in your region (openai/gpt-4o)
```

**解决方案：** 使用无地区限制的免费模型：
- `tencent/hy3-preview:free` （推荐，全球可用）
- `google/gemma-3-27b-it:free`
- `meta-llama/llama-3.3-70b-instruct:free`

---

### 问题 5: 进程超时

**症状：**
```
exit_code: 124  (timeout)
```

**解决方案：** 增加超时时间：
```bash
timeout 600 python run_analysis.py  # 10分钟
# 或去掉超时限制（用于复杂分析）
python run_analysis.py
```

---

### 问题 6: SOCKS 代理缺少 socksio

**症状：**
```
Missing optional dependency 'socksio'
```

**解决方案：**
```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
```

---

## 配置选项详解

### LLM 模型配置（config 字典）

```python
config['llm_provider'] = 'openrouter'  # 使用 OpenRouter
config['deep_think_llm'] = 'tencent/hy3-preview:free'  # 分析师模型
config['quick_think_llm'] = 'tencent/hy3-preview:free'  # 快速任务模型
```

### 数据源配置（config 字典）

```python
config['data_vendors'] = {
    'core_stock_apis': 'alpha_vantage',  # 或 'yfinance'
    'technical_indicators': 'alpha_vantage',
    'fundamental_data': 'alpha_vantage',
    'news_data': 'alpha_vantage',
}
```

### 分析参数（propagate 调用）

```python
result = ta.propagate(
    company_name="SPY",      # 股票代码
    trade_date="2024-12-18"  # 分析日期
)
```

---

## OpenRouter 已验证免费模型

| 模型 | ID | 说明 |
|-------|-----|------|
| 腾讯 Hy3 Preview 免费版 | `tencent/hy3-preview:free` | 推荐，全球可用 ✅ |
| Google Gemma 3 27B | `google/gemma-3-27b-it:free` | 适合分析 |
| Llama 3.3 70B | `meta-llama/llama-3.3-70b-instruct:free` | 强推理能力 |

---

## 实测不可用模型

| 模型 | 问题 |
|-------|------|
| `openai/gpt-4o` | 403 地区限制 |
| `gpt-5.4` / `gpt-5.5` | bobdong.cn key 无访问权限 |
| Yahoo Finance | IP 限流（需切换 Alpha Vantage） |

---

## 成功运行输出示例

正常运行时应该看到：

```
Starting TradingAgents analysis for SPY...
Using OpenRouter with tencent/hy3-preview:free (free model)
Analysis date: 2024-12-18
--------------------------------------------------

================================ Human Message =================================

SPY

================================== Ai Message ==================================

I'll analyze SPY (SPDR S&P 500 ETF Trust)...

Tool Calls:
  get_stock_data (chatcmpl-tool-xxx)
  get_indicators (chatcmpl-tool-yyy)
  ...
```

**如果看到 LLM 响应，说明配置正确！**

---

## 完整工作流总结

```
1. 克隆仓库 → 2. 创建虚拟环境 → 3. uv sync 安装依赖
   ↓
4. 配置 .env（OpenRouter + Alpha Vantage API Keys）
   ↓
5. 创建 run_analysis.py 脚本
   ↓
6. 取消代理变量 → 7. 运行（带超时）
   ↓
8. 查看结果（技术指标 + 新闻分析 + 交易建议）
```

---

## 核心约束（用户要求）

1. **不要随意修改源码**：严格按照官方文档操作。如需修改，仅修改 `default_config.py` 作为最后手段，然后用 `git checkout` 恢复。
2. **使用 Python 脚本，不用 CLI**：终端不支持箭头键，无法交互式选择提供商。
3. **始终取消代理**：防止 SOCKS 代理错误。
4. **Alpha Vantage 需要 API Key**：免费版限制：5次/分钟，500次/天。

---

## 文件结构

```
/tmp/TradingAgents/
├── .env                    # API keys 配置
├── run_analysis.py         # 主运行脚本
├── venv/                  # 虚拟环境
├── tradingagents/          # 源代码
│   ├── graph/
│   │   └── trading_graph.py
│   ├── default_config.py  # 默认配置
│   └── dataflows/
│       ├── alpha_vantage_*.py
│       └── y_finance.py
└── README.md
```

---

## 实战经验总结

### ✅ 成功要点

1. **使用 uv，不用 pip** - 避免 SSL 问题
2. **始终取消代理变量** - 再运行
3. **Alpha Vantage 免费版够用** - 5次/分钟，500次/天
4. **从短时间窗口开始测试** - 例如 30 天
5. **API keys 同时写入 `.env` 和脚本** - 提高可靠性

### 📊 实测结果（2024-12-18，SPY）

- **LLM 调用**：✅ 成功（`tencent/hy3-preview:free`）
- **技术指标分析**：✅ 完成（8个指标：EMA、SMA、MACD、RSI、布林带、ATR）
- **新闻分析**：✅ 完成（获取11-12月新闻，分析宏观经济趋势）
- **最终交易建议**：✅ 生成 **HOLD（持有）**

**关键支撑/阻力位：**
- 支撑：579-580（50日均线 + 布林带下轨）
- 阻力：590-595（10日EMA + 布林带中轨）

---

## 参考资源

- **官方仓库**: https://github.com/TauricResearch/TradingAgents
- **OpenRouter 模型列表**: https://openrouter.ai/models?max_price=free
- **Alpha Vantage 文档**: https://www.alphavantage.co/documentation/
- **申请 Alpha Vantage Key**: https://www.alphavantage.co/support/#api-key

---

## 更新日志

**v2.0 (2026-05-05)**：
- 添加完整的 Alpha Vantage 申请流程
- 补充 Yahoo Finance 限流问题的完整解决方案
- 添加实测成功运行的输出示例
- 补充实战经验总结和数据源切换详解
- 添加文件结构说明和配置选项详解

**v1.0 (初始版本)**：
- 基础安装流程
- 交互式 CLI 绕过方案
- 基础问题排查
