"""LLM API Probe — 检测 API key 接入的大模型能力 / 速度 / 稳定性 / 安全性。

子模块:
    config       — provider 配置加载与校验
    client       — OpenAI 兼容客户端封装
    models       — 数据模型 (ProbeResult / Provider / Metric 等)
    probes       — 各个检测模块
    report       — 报告生成
"""
__version__ = "0.1.0"