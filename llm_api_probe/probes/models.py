"""数据模型：所有检测模块共用的结构。"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class Provider:
    """一个接入点 (官方厂商 / 聚合中转 / 自部署)。"""
    name: str                              # 简称, e.g. "openai-official"
    label: str                             # 显示名, e.g. "OpenAI 官方"
    category: str                          # official / aggregator / self_hosted
    base_url: str                          # OpenAI 兼容 API 根地址
    api_key: str                           # API key
    models: list[str] = field(default_factory=list)   # 待测模型列表
    headers: dict[str, str] = field(default_factory=dict)  # 额外请求头
    timeout: float = 60.0                  # 单次请求超时
    note: str = ""                         # 备注 (例如 "便宜但疑似降级")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # api_key 脱敏后再输出
        if self.api_key:
            k = self.api_key
            d["api_key"] = (k[:4] + "***" + k[-4:]) if len(k) > 8 else "***"
        return d


@dataclass
class Metric:
    """单个数值型指标。"""
    name: str
    value: Any
    unit: str = ""
    note: str = ""

    def fmt(self) -> str:
        if self.value is None:
            return "—"
        if isinstance(self.value, float):
            return f"{self.value:.2f}{self.unit}"
        return f"{self.value}{self.unit}"


@dataclass
class ProbeResult:
    """一个 probe 模块对一家 provider 的检测结果。"""
    probe: str                             # 模块名
    provider: str                          # provider.name
    ok: bool = True                        # 模块是否完成
    error: Optional[str] = None            # 顶层错误
    metrics: list[Metric] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)   # 文本结论
    warnings: list[str] = field(default_factory=list)   # 告警 (例如发现注入)
    raw: dict[str, Any] = field(default_factory=dict)   # 原始数据

    def add(self, name: str, value: Any, unit: str = "", note: str = "") -> None:
        self.metrics.append(Metric(name, value, unit, note))

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def find(self, msg: str) -> None:
        self.findings.append(msg)

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe": self.probe,
            "provider": self.provider,
            "ok": self.ok,
            "error": self.error,
            "metrics": [asdict(m) for m in self.metrics],
            "findings": self.findings,
            "warnings": self.warnings,
            "raw": self.raw,
        }