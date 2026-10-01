"""Profile-scoped scalar configuration writes."""

import logging
import os
from typing import Any

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


def save_config_value(key_path: str, value: Any) -> bool:
    """Persist a dot-separated setting in the active profile's config.yaml."""
    config_path = get_hermes_home() / 'config.yaml'

    try:
        from hermes_constants import mkdir_under_hermes_home
        mkdir_under_hermes_home(config_path.parent)
        from utils import atomic_roundtrip_yaml_update
        atomic_roundtrip_yaml_update(config_path, key_path, value)
        try:
            os.chmod(config_path, 0o600)
        except (OSError, NotImplementedError):
            pass
        return True
    except Exception as e:
        logger.error("Failed to save config: %s", e)
        return False
