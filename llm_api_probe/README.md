# LLM API Probe

> 检测 API key 接入的大模型能力 / 速度 / 稳定性 / 安全性,
> 方便横向评估**官方厂商 / 聚合中转 / 自部署**的接入质量。

不是学术 benchmark (MMLU/GSM8K 全套), 而是 5-10 分钟跑完的**接入层体检**。

---

## 它能告诉你什么

| 模块 | 检测什么 | 对什么场景有用 |
|------|---------|--------------|
| `connectivity` | API key 是否有效, /v1/models 返回什么, 实际响应模型名是否一致 | 接入第一步 |
| `context`     | 实测最大可用 context tokens (二分探测), needle-in-haystack 检测偷截断 | 中转站"模型列表撒谎" / 限速版 |
| `speed`       | TTFT 中位, 输出 tok/s, 并发 4 路总吞吐, 3 轮取中位数 | 选最快的接入 |
| `stability`   | 10 次 burst 错误率, 429 触发, 重试恢复率, 10 轮多轮对话, max_tokens 边界 | 选稳定的接入 |
| `security`    | 身份自报, canary token 泄露, 指令覆盖, 隐藏 system 探针 | 排查中转站是否偷偷注入/截断/改写 |
| `ability`     | 数学/推理/代码/中文抽样 (12 题), 检测是否被偷换为弱模型 | 验证模型没被降级 |

输出: 控制台对比表 + JSON + Markdown 三份。

---

## 安装

```bash
pip install openai pyyaml
```

(只需要这两个。)

---

## 快速开始

### 1) 生成示例配置

```bash
python -m llm_api_probe --init-config my-probe.yaml
```

### 2) 编辑 `my-probe.yaml`, 填入 key 和要测的模型

### 3) 跑全部检测

```bash
python -m llm_api_probe --config my-probe.yaml
```

约 5-15 分钟 (取决于并发/模型数量)。

### 只跑某个模块

```bash
python -m llm_api_probe --config my-probe.yaml --only speed
python -m llm_api_probe --config my-probe.yaml --only security,stability
```

### 只测某个 provider / 临时换模型

```bash
python -m llm_api_probe --config my-probe.yaml --provider aggregator-cheap --model gpt-4o
```

---

## 配置说明

```yaml
providers:
  - name: aggregator-a        # 唯一 id, 用于命令行 --provider 过滤
    label: 便宜中转 A          # 显示名
    category: aggregator      # official / aggregator / self_hosted
    base_url: https://.../v1
    api_key: ${ENV_VAR}       # 支持环境变量占位符
    models: [gpt-4o, claude-3-5-sonnet]
    note: 便宜, 待验证
    timeout: 60               # 单次请求超时 (秒)
    headers:                  # 可选, 自定义请求头
      X-Custom: foo
```

---

## 输出示例

```
========================================================================================
LLM API Probe — 报告  (2026-06-20 03:15:00)
========================================================================================

## Provider 概览
  name                   label                  category        base_url
  ---------------------- ---------------------- --------------- ----------------------------------------
  openai-official        OpenAI 官方            official        https://api.openai.com/v1
  aggregator-cheap       便宜中转 A             aggregator      https://api.xxx-relay.com/v1

## CONNECTIVITY
  provider               ok    metrics
  ---------------------- ----  --------------------------------------------------------
  openai-official        ✓     models_endpoint_ms=92.3; available_models=78 个 ...
  aggregator-cheap       ✓     hello_latency_ms=421.0 ...
    ⚠ [aggregator-cheap] 模型名被改写: 请求 'gpt-4o', 实际返回 'gpt-4o-mini'

## SPEED
  provider               ok    metrics
  ...
    · [openai-official] 顺序 TTFT 中位 480ms, 输出速度中位 78.3 tok/s

## 速度横向对比
  provider               TTFT中位    输出tok/s中位       并发吞吐      成功率
  ---------------------- ---------- ---------------- -------------- ----------
  openai-official              480              78.3           290.5       3/3
  aggregator-cheap            1200              45.2           120.3       3/3

## ⚠ 安全审计警告汇总
  - [aggregator-cheap] Canary token 'CANARY_A8X92K...' 被泄露
  - [aggregator-cheap] 实际可用 8192 比宣称 128000 少 93.6%
```

---

## 适用场景

✅ **评估新的中转站 / 聚合厂商** — 跑一遍就知道是不是"挂羊头卖狗肉"
✅ **对比同模型不同接入方** (官方 vs 第三方代理) — 看延迟和稳定性差距
✅ **验证小工作站自部署模型** — 确认走的是宣称的模型、context 没缩水
✅ **CI 中定期巡检** — 把 report 落盘, 看趋势变化

❌ **不适合**学术论文级的能力评测 (那种请用 OpenCompass/lm-eval-harness)

---

## 项目结构

```
llm_api_probe/
├── __main__.py                  # CLI 入口
├── README.md
├── examples/
│   └── probe.example.yaml       # 配置示例
├── reports/                     # 报告输出 (运行后生成)
└── probes/
    ├── __init__.py
    ├── config.py                # YAML 配置加载
    ├── client.py                # OpenAI 兼容客户端
    ├── models.py                # 数据模型 (Provider/ProbeResult/Metric)
    ├── probe_connectivity.py
    ├── probe_context.py
    ├── probe_speed.py
    ├── probe_stability.py
    ├── probe_security.py
    ├── probe_ability.py
    └── report.py                # 控制台 / JSON / Markdown
```

---

## 参考与致谢

实现思路综合自以下开源项目:

- [api-checker (Jimmy102836)](https://github.com/Jimmy102836/api-checker) — 中转站审计的检测项 (注入/截断/指令覆盖)
- [october-coder/api-check](https://github.com/october-coder/api-check) — OpenAI 兼容 API 校验
- [llmperf (ray-project)](https://github.com/ray-project/llmperf) — 工业级吞吐/TTFT benchmark
- [Yoosu-L/llmapibenchmark](https://github.com/Yoosu-L/llmapibenchmark) — 并发吞吐测量
- [智源社区脚本](https://hub.baai.ac.cn/view/38569) — 中文社区流行的速度测试方案