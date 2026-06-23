import sys

from plugins.platforms.wecom import adapter as _adapter

sys.modules[__name__] = _adapter
