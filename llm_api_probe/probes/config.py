"""配置文件加载。

配置文件格式 (YAML):
    providers:
      - name: openai-official
        label: OpenAI 官方
        category: official                # official / aggregator / self_hosted
        base_url: https://api.openai.com/v1
        api_key: sk-xxx
        models: [gpt-4o, gpt-4o-mini]
      - name: aggregator-a
        label: 某中转
        category: aggregator
        base_url: https://api.xxx.com/v1
        api_key: sk-xxx
        models: [gpt-4o, claude-3-5-sonnet]
        note: 便宜, 待验证真伪
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

import yaml

from .models import Provider


REQUIRED_FIELDS = ("name", "label", "category", "base_url", "api_key")


def load_config(path: str | Path, *, api_key_override: Optional[dict[str, str]] = None) -> list[Provider]:
    """加载 YAML 配置文件, 返回 Provider 列表。

    api_key_override: 可选, {provider_name: api_key}, 用于命令行 --api-key 覆盖。
    """
    api_key_override = api_key_override or {}
    path = Path(path)
    if not path.exists():
        print(f"[!] 配置文件不存在: {path}", file=sys.stderr)
        sys.exit(2)

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    raw_providers = data.get("providers", [])
    if not raw_providers:
        print("[!] 配置文件中没有 providers 字段或为空。", file=sys.stderr)
        sys.exit(2)

    providers: list[Provider] = []
    for i, raw in enumerate(raw_providers):
        miss = [f for f in REQUIRED_FIELDS if not raw.get(f)]
        if miss:
            print(f"[!] 第 {i + 1} 个 provider 缺少字段: {miss}", file=sys.stderr)
            sys.exit(2)
        # api_key 解析顺序: 命令行覆盖 > 环境变量 > YAML 字面量
        if raw["name"] in api_key_override:
            api_key = api_key_override[raw["name"]]
        else:
            api_key = os.path.expandvars(str(raw["api_key"]))

        providers.append(Provider(
            name=raw["name"],
            label=raw["label"],
            category=raw["category"],
            base_url=raw["base_url"].rstrip("/"),
            api_key=api_key,
            models=list(raw.get("models") or []),
            headers=dict(raw.get("headers") or {}),
            timeout=float(raw.get("timeout", 60.0)),
            note=raw.get("note", ""),
        ))
    return providers


def write_example_config(path: str | Path) -> None:
    """生成示例配置文件。"""
    example = """# LLM API Probe 示例配置
# api_key 支持 ${ENV_VAR} 占位符

providers:
  # === 官方厂商 ===
  - name: openai-official
    label: OpenAI 官方
    category: official
    base_url: https://api.openai.com/v1
    api_key: ${OPENAI_API_KEY}
    models: [gpt-4o-mini]
    note: 基准对照

  # === 聚合中转 ===
  - name: aggregator-a
    label: 某聚合 A
    category: aggregator
    base_url: https://api.aggregator-a.com/v1
    api_key: sk-xxxxxxxxxxxx
    models: [gpt-4o, claude-3-5-sonnet]
    note: 价格低, 待验证是否真模型

  # === 自部署 (OpenAI 兼容服务) ===
  - name: self-llm
    label: 机房自部署 vLLM
    category: self_hosted
    base_url: http://10.0.0.5:8000/v1
    api_key: EMPTY
    models: [Qwen2.5-72B-Instruct]
    note: 内部机房
"""
    Path(path).write_text(example, encoding="utf-8")