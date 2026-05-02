# Hermes Agent 集成 Vibe-Trading：A 股分析方案

记录日期：2026-05-02  
目标：通过飞书与 Hermes Agent 对话时，让 Hermes 调用 Vibe-Trading 完成 A 股研究、买卖建议辅助、策略回测和风险检查。

实施状态：第一阶段 Hermes 插件已完成并部署到 `192.168.1.63:/root/.hermes/plugins/vibe-trading/`，已在 Hermes 配置中启用。当前阶段使用免费的 AKShare 优先方案，不默认使用 Tushare Pro 或 QVerisAI。

## 1. 当前部署事实

Vibe-Trading 已部署在：

```text
主机：toddsun@192.168.1.58
目录：~/aiproxy/Vibe-Trading
服务：Docker Compose
API：http://192.168.1.58:8899
健康检查：GET /health
```

已验证服务状态：

```json
{"status":"healthy","service":"Vibe-Trading API"}
```

Hermes Agent 已部署在：

```text
主机：root@192.168.1.63
容器：hermes
镜像：nousresearch/hermes-agent:latest
版本：Hermes Agent v0.12.0
入口：飞书 gateway
```

Hermes 容器内可以访问 Vibe-Trading API：

```text
http://192.168.1.58:8899/health
```

当前 Hermes 配置中已启用插件：

```yaml
plugins:
  enabled:
    - tencent-news
    - vibe-trading
```

尚未配置 `mcp_servers`。第一阶段集成已沿用现有插件模式，没有在 Hermes 容器内安装 Vibe-Trading 的 MCP 依赖。

## 2. 推荐集成方式

推荐使用 Hermes 插件封装 Vibe-Trading REST API：

```text
飞书 -> Hermes Gateway -> Hermes Agent -> vibe-trading Hermes 插件 -> Vibe-Trading API -> Hermes 总结 -> 飞书
```

理由：

- Vibe-Trading 已经作为独立 Docker 服务稳定运行，不需要把它的 Python 依赖安装进 Hermes 容器。
- Hermes v0.12.0 的插件系统已经在当前部署中使用过，`tencent-news` 插件可作为实现参考。
- 插件可以只暴露经过筛选的金融工具，避免一次性把所有文件读写、搜索、回测和 swarm 工具暴露给模型。
- Vibe-Trading 后续升级可以独立进行，Hermes 只需要保持 HTTP 调用兼容。

备选方案是把 `vibe-trading-mcp` 配成 Hermes MCP server。该方式工具覆盖更完整，但需要在 Hermes 容器内安装 `vibe-trading-ai`，依赖更重，升级和数据目录隔离也更复杂。

## 3. Vibe-Trading 已识别能力

Vibe-Trading 仓库中包含：

- `agent/api_server.py`：FastAPI 服务。
- `agent/mcp_server.py`：MCP server，暴露金融研究工具。
- `agent/SKILL.md`：说明 PyPI 包、MCP 命令和工具清单。
- `docker-compose.yml`：当前 Docker 部署入口。

已识别 REST API 路由包括：

- `GET /health`
- `GET /skills`
- `GET /api`
- `GET /runs`
- `GET /runs/{run_id}`
- `GET /runs/{run_id}/code`
- `GET /runs/{run_id}/pine`
- `GET /correlation`
- `POST /sessions`
- `GET /sessions`
- `POST /sessions/{session_id}/messages`
- `GET /sessions/{session_id}/messages`
- `POST /sessions/{session_id}/cancel`
- `GET /sessions/{session_id}/events`
- `POST /upload`
- `GET /shadow-reports/{shadow_id}`
- `GET /swarm/presets`
- `POST /swarm/runs`
- `GET /swarm/runs`
- `GET /swarm/runs/{run_id}`
- `GET /swarm/runs/{run_id}/events`
- `POST /swarm/runs/{run_id}/cancel`

已识别 MCP 工具包括：

- `list_skills`
- `load_skill`
- `backtest`
- `factor_analysis`
- `analyze_options`
- `pattern_recognition`
- `read_url`
- `read_document`
- `web_search`
- `write_file`
- `read_file`
- `list_swarm_presets`
- `run_swarm`
- `get_market_data`
- `get_swarm_status`
- `get_run_result`
- `list_runs`
- `analyze_trade_journal`
- `extract_shadow_strategy`
- `run_shadow_backtest`
- `render_shadow_report`
- `scan_shadow_signals`

## 4. A 股相关功能

### 4.1 个股买卖建议辅助

适用问题：

```text
分析 600519.SH 当前是否适合买入，给出买入、观望、减仓或卖出的倾向。
```

建议输出格式：

- 结论：买入、观望、减仓或卖出。
- 核心理由：趋势、估值、资金面、基本面、技术面、事件风险。
- 反向风险：什么情况会推翻当前判断。
- 入场条件：需要满足哪些价格、量能或事件条件。
- 止损条件：跌破什么位置或出现什么风险信号应退出。
- 观察周期：短线、中线或长期。
- 数据限制：本次分析依赖的数据源和可能缺口。

注意：该能力应定位为研究辅助，不应在飞书中表述为确定性投资指令。

### 4.2 A 股行情获取

Vibe-Trading 支持 A 股行情数据源：

- `akshare`：免费数据源，第一阶段默认优先使用。
- `tushare`：适合更稳定的 A 股数据，需要 `TUSHARE_TOKEN`；第一阶段不默认使用。
- `auto`：根据证券代码自动判断并选择数据源。

适用问题：

```text
获取 000001.SZ 过去一年的日线行情，分析趋势、波动率和支撑阻力。
```

可用于：

- 个股 OHLCV 行情。
- 指数或板块相关行情。
- 回测前的数据准备。
- 趋势、波动率、支撑阻力和量价关系分析。

### 4.3 策略生成与回测

Vibe-Trading 支持通过自然语言生成策略并运行回测。A 股场景中可用于把“买入逻辑”变成可验证规则。

适用问题：

```text
用 600519.SH 回测一个均线突破策略，时间从 2021-01-01 到 2025-12-31，输出收益、最大回撤、夏普和交易次数。
```

典型输出：

- 策略逻辑。
- 回测区间。
- 数据源。
- 总收益。
- 年化收益。
- 最大回撤。
- 夏普比率。
- 胜率。
- 交易次数。
- 结果局限。

### 4.4 因子分析

MCP 工具 `factor_analysis` 可用于 A 股横截面研究，例如估值、换手、动量或财务因子有效性检查。

适用问题：

```text
对沪深 300 成分股做 pe_ttm 因子分析，时间 2023-01-01 到 2025-12-31，判断低 PE 因子是否有效。
```

可回答：

- 某个因子最近是否有效。
- 因子 IC/IR 表现如何。
- 多头组合和空头组合收益差异。
- 因子是否适合进入后续选股框架。

### 4.5 技术分析和形态识别

Vibe-Trading 的技能体系覆盖多类技术分析方法：

- K 线形态。
- 均线和常见技术指标。
- 一目均衡。
- Elliott wave。
- SMC。
- 缠论或技术基础方法。
- 图形识别，如头肩顶、双底、三角形等。

适用问题：

```text
用技术分析框架评估 601318.SH，重点看趋势、支撑阻力、量价关系和短线交易风险。
```

### 4.6 ST 和 pre-ST 风险筛查

Vibe-Trading README 记录了 `ashare-pre-st-filter` 技能，用于 A 股 ST 或 pre-ST 风险筛查。

适用问题：

```text
检查 002XXX.SZ 是否存在 ST 或 pre-ST 风险，关注财务异常、处罚公告、连续亏损和交易所风险提示。
```

该能力适合在买入建议前作为强制风控步骤。

### 4.7 板块和行业轮动分析

Vibe-Trading 的技能体系包含 sector rotation、asset allocation、macro-analysis 等能力，可用于 A 股板块比较。

适用问题：

```text
分析当前 A 股新能源、半导体、白酒、银行几个板块的相对强弱，给出更适合关注的方向。
```

可输出：

- 板块强弱排序。
- 近期驱动因素。
- 风险事件。
- 适合关注的标的类型。
- 不适合追涨的条件。

### 4.8 多代理投资委员会

Vibe-Trading 提供 29 个 swarm preset。A 股买卖建议相关场景中优先考虑：

- `investment_committee`：多空辩论后给出投资建议。
- `risk_committee`：专门审查风险。
- `quant_strategy_desk`：偏量化和回测验证。
- `global_equities_desk`：跨 A 股、港美股和加密市场视角。

适用问题：

```text
用 Vibe-Trading 的 investment_committee 分析 600519.SH，输出买入、观望或卖出建议、核心分歧、风险点和触发条件。
```

### 4.9 交易日志和影子账户

如果用户上传券商交割单，Vibe-Trading 可用于交易复盘：

- 分析持仓周期。
- 计算胜率和盈亏比。
- 识别处置效应。
- 识别过度交易。
- 识别追涨杀跌。
- 识别锚定效应。
- 从历史盈利交易中提取规则。
- 回测“如果严格执行这些规则会怎样”。

适用问题：

```text
分析我的同花顺交割单，提取影子账户策略，并回测过去一年如果按这些规则交易会有什么差异。
```

第一阶段 Hermes 集成暂不开放文件上传，等基础查询和分析稳定后再处理飞书附件到 Vibe `/upload` 的链路。

## 5. 第一阶段插件工具

第一阶段已暴露高价值且可控的工具。其中飞书里的股票自然语言问题应优先走 `vibe_ask_ashare`。

| Hermes 工具名 | 调用目标 | 用途 |
| --- | --- | --- |
| `vibe_ask_ashare` | `POST /sessions` + `POST /sessions/{session_id}/messages` + 轮询 `GET /sessions/{session_id}/messages` | A 股或股票问题的自然语言入口，默认要求 Vibe Agent 优先使用免费 AKShare |
| `vibe_ask` | 同上 | 通用 Vibe-Trading Agent 自然语言转发 |
| `vibe_health` | `GET /health` | 检查 Vibe-Trading 服务是否可用 |
| `vibe_list_skills` | `GET /skills` | 查看可用金融技能 |
| `vibe_list_swarm_presets` | `GET /swarm/presets` | 查看可用多代理团队 |
| `vibe_run_swarm` | `POST /swarm/runs` | 运行投资委员会或风控委员会分析 |
| `vibe_get_swarm_run` | `GET /swarm/runs/{run_id}` | 查询 swarm 结果 |
| `vibe_create_session` | `POST /sessions` | 创建 Vibe 会话 |
| `vibe_send_message` | `POST /sessions/{session_id}/messages` | 向 Vibe 会话发送自然语言请求 |
| `vibe_get_run_result` | `GET /runs/{run_id}` | 查询历史回测或分析结果 |
| `vibe_list_runs` | `GET /runs` | 列出近期运行记录 |

暂缓开放：

- 文件写入类工具。
- 任意网页搜索类工具。
- 任意文件读取类工具。
- 飞书附件上传到 Vibe 的自动处理。

这些能力有用，但需要额外做路径、权限、文件大小、敏感信息和金融风险边界设计。

## 6. 飞书使用示例

### 自然语言 A 股分析

用户在飞书中不需要记工具名，也不需要记参数。直接问股票问题即可：

```text
我想看看南山铝业600219的投资策略，在什么位置买入或者卖出比较好
```

Hermes 应选择 `vibe_ask_ashare`，把用户原始问题转发给 Vibe-Trading Agent，并附加第一阶段约束：

- 优先使用免费 AKShare 数据源和 Vibe-Trading 内置 A 股能力。
- 不默认调用 Tushare 或 QVeris。
- 输出中文结构化报告。
- 明确说明数据时效和限制。
- 保留“仅供研究参考，不构成投资建议”的免责声明。

### 个股买卖建议

```text
用 Vibe-Trading 分析 600519.SH 当前是否适合买入。
请输出：结论、核心理由、反向风险、入场条件、止损条件、观察周期。
```

### 风险优先分析

```text
用 Vibe-Trading 的 risk_committee 分析 300750.SZ。
重点检查下跌风险、估值风险、行业风险和需要回避的触发条件。
```

### 投资委员会分析

```text
用 Vibe-Trading 的 investment_committee 分析 601318.SH。
请给出多方观点、空方观点、最终建议和关键验证信号。
```

### 策略回测

```text
用 Vibe-Trading 为 000001.SZ 设计一个均线突破策略，并回测 2021-01-01 到 2025-12-31。
输出收益、最大回撤、夏普、胜率、交易次数和策略缺陷。
```

### ST 风险检查

```text
用 Vibe-Trading 检查 002XXX.SZ 是否有 ST 或 pre-ST 风险。
请关注财务异常、监管处罚、连续亏损和公告风险。
```

## 7. 风险和边界

### 金融建议边界

Hermes 输出应使用“研究结论”“交易辅助判断”“风险提示”等措辞，不应承诺收益，也不应把模型输出表述为确定性投资建议。

推荐固定免责声明：

```text
以下内容仅用于研究辅助，不构成投资建议。A 股交易存在本金亏损风险，请结合自己的风险承受能力和实时行情独立决策。
```

### 数据时效边界

A 股买卖建议高度依赖实时或准实时数据。每次输出应说明：

- 使用的数据源。
- 数据截止日期或时间。
- 是否成功获取最新行情。
- 哪些结论依赖模型推断而不是实时数据。

### 工具权限边界

第一版插件不应开放 Vibe 的任意文件读写能力。涉及交割单、报告、附件上传时，应单独设计文件路径和权限。

### 操作边界

第一阶段只做分析和研究，不连接券商交易接口，不自动下单。

### 响应耗时边界

完整 Vibe Agent 分析通常需要几十秒到两分钟以上。插件默认等待 `180` 秒，可以通过环境变量调整：

```text
VIBE_TRADING_AGENT_TIMEOUT_SECONDS=180
VIBE_TRADING_AGENT_POLL_SECONDS=3
```

## 8. 第一阶段完成情况

第一阶段已完成：

1. 已在 `192.168.1.63:/root/.hermes/plugins/vibe-trading/` 创建 Hermes 插件。
2. 插件默认连接 `http://192.168.1.58:8899`，也支持通过 `VIBE_TRADING_BASE_URL` 覆盖。
3. 已注册第一阶段工具：自然语言 A 股转发、通用 Vibe Agent 转发、健康检查、技能列表、swarm preset、运行 swarm、查询 run、会话发送消息。
4. 已更新 `/root/.hermes/config.yaml`：

```yaml
plugins:
  enabled:
    - tencent-news
    - vibe-trading
```

5. 已重启 Hermes 容器。
6. 已验证 `vibe_health` 到 Vibe-Trading API 的连通性。
7. 已验证 `vibe_ask_ashare` 可以创建 Vibe session、发送股票自然语言问题、等待 Agent 完成并返回中文报告。

后续可以在飞书中测试：

```text
检查 Vibe-Trading 是否可用
我想看看南山铝业600219的投资策略，在什么位置买入或者卖出比较好
用 Vibe-Trading 的 investment_committee 分析 600519.SH
```

根据实际使用结果，再决定是否增加文件上传、影子账户和完整回测自动化。
