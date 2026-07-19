"""Compatibility shim for the migrated Feishu platform plugin.

Upstream moved the adapter to ``plugins/platforms/feishu/``. Tests and a few
legacy import sites still use ``gateway.platforms.feishu``; re-export the public
surface so both paths keep working after the merge.
"""

from plugins.platforms.feishu.adapter import *  # noqa: F403
from plugins.platforms.feishu.adapter import FeishuAdapter, normalize_feishu_message

__all__ = ["FeishuAdapter", "normalize_feishu_message"]
