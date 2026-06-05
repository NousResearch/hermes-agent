"""MES 巡检配置管理器。"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


# 默认配置文件搜索路径
_CONFIG_SEARCH_PATHS = [
    Path(os.getenv("MES_INSPECTION_HOME", "")) / "config" / "mes_inspection.yaml",
    Path.home() / ".mes-inspection" / "config.yaml",
    Path(__file__).parent / "mes_inspection.yaml",
]


def _deep_merge(base: dict, override: dict) -> dict:
    """深度合并两个字典，override 覆盖 base。"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class ConfigManager:
    """MES 巡检配置管理器。

    搜索顺序：
    1. $MES_INSPECTION_HOME/config/mes_inspection.yaml
    2. ~/.mes-inspection/config.yaml
    3. 本目录/mes_inspection.yaml
    """

    def __init__(self, config_path: Optional[str] = None):
        self._config: Dict[str, Any] = {}
        self._config_path: Optional[Path] = None
        if config_path:
            self._config_path = Path(config_path)
        else:
            for p in _CONFIG_SEARCH_PATHS:
                if p.exists():
                    self._config_path = p
                    break

    def load(self) -> Dict[str, Any]:
        """加载配置文件，与默认阈值深度合并。"""
        from config.default_thresholds import DEFAULT_THRESHOLDS
        config = DEFAULT_THRESHOLDS.copy()
        if self._config_path and self._config_path.exists():
            with open(self._config_path, "r", encoding="utf-8") as f:
                user_config = yaml.safe_load(f) or {}
            config = _deep_merge(config, user_config)
        self._config = config
        return self._config

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持点号分隔的路径（如 'nginx.error_5xx_threshold'）。"""
        obj = self._config
        for part in key.split("."):
            if isinstance(obj, dict):
                obj = obj.get(part)
                if obj is None:
                    return default
            else:
                return default
        return obj

    @property
    def config(self) -> Dict[str, Any]:
        if not self._config:
            self.load()
        return self._config

    def get_component_config(self, component: str) -> Dict[str, Any]:
        """获取指定组件的配置。"""
        return self.config.get(component, {})
