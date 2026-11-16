"""PDR 配置加载：唯一入口是 pdr/config.yaml（可用 PDR_CONFIG 指向其它文件）。"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PDR_ROOT = Path(__file__).resolve().parents[2]  # pdr/


class Cfg(dict):
    """dict 子类，支持 cfg.a.b 属性访问，缺失键返回空 Cfg 便于链式读取。"""

    def __getattr__(self, item: str) -> Any:
        value = self.get(item)
        if isinstance(value, dict) and not isinstance(value, Cfg):
            value = Cfg(value)
            self[item] = value
        return Cfg() if value is None else value


def config_path() -> Path:
    return Path(os.environ.get("PDR_CONFIG") or PDR_ROOT / "config.yaml").resolve()


def resolve_path(raw: str) -> Path:
    """配置中的相对路径以 config.yaml 所在目录为基准。"""
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (config_path().parent / path).resolve()


@lru_cache(maxsize=1)
def get_settings() -> Cfg:
    with open(config_path(), encoding="utf-8") as fh:
        return Cfg(yaml.safe_load(fh) or {})


def reload_settings() -> Cfg:
    get_settings.cache_clear()
    return get_settings()


def data_dir() -> Path:
    path = resolve_path(get_settings().app.get("data_dir", "./.pdr/data"))
    path.mkdir(parents=True, exist_ok=True)
    return path
