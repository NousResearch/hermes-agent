"""配置管理器单元测试。"""

import json
import os
import tempfile
from pathlib import Path
import pytest

from config.config_manager import ConfigManager, _deep_merge
from config.default_thresholds import DEFAULT_THRESHOLDS


class TestDeepMerge:
    def test_simple_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"a": {"x": 1, "y": 2}, "b": 1}
        override = {"a": {"y": 3, "z": 4}}
        result = _deep_merge(base, override)
        assert result == {"a": {"x": 1, "y": 3, "z": 4}, "b": 1}

    def test_override_dict_with_non_dict(self):
        base = {"a": {"x": 1}}
        override = {"a": "string"}
        result = _deep_merge(base, override)
        assert result == {"a": "string"}


class TestConfigManager:
    def test_load_defaults(self):
        cm = ConfigManager(config_path="/nonexistent/path.yaml")
        config = cm.load()
        assert "nginx" in config
        assert "jvm" in config
        assert config["nginx"]["error_5xx_threshold_warn"] == 1.0

    def test_get_nested(self):
        cm = ConfigManager(config_path="/nonexistent/path.yaml")
        cm.load()
        assert cm.get("nginx.error_5xx_threshold_warn") == 1.0
        assert cm.get("nonexistent.key", "default") == "default"

    def test_get_component_config(self):
        cm = ConfigManager(config_path="/nonexistent/path.yaml")
        cm.load()
        nginx_cfg = cm.get_component_config("nginx")
        assert "error_5xx_threshold_warn" in nginx_cfg
        assert "process_check" in nginx_cfg

    def test_load_user_config_override(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            f.write("nginx:\n  error_5xx_threshold_warn: 2.5\n")
            f.flush()
            cm = ConfigManager(config_path=f.name)
            config = cm.load()
            assert config["nginx"]["error_5xx_threshold_warn"] == 2.5
            # 其他默认值保留
            assert config["nginx"]["error_5xx_threshold_critical"] == 5.0
        os.unlink(f.name)


class TestDefaultThresholds:
    def test_all_components_present(self):
        components = ["nginx", "jvm", "rabbitmq", "oracle", "elk", "skywalking", "alerter", "self_healing", "memory", "gitlab", "model"]
        for comp in components:
            assert comp in DEFAULT_THRESHOLDS, f"Missing component: {comp}"

    def test_threshold_types(self):
        assert isinstance(DEFAULT_THRESHOLDS["nginx"]["error_5xx_threshold_warn"], float)
        assert isinstance(DEFAULT_THRESHOLDS["jvm"]["heap_usage_warn"], float)
        assert isinstance(DEFAULT_THRESHOLDS["rabbitmq"]["queue_depth_warn"], int)
