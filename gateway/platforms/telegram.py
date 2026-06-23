import sys

from plugins.platforms.telegram import adapter as _adapter

sys.modules[__name__] = _adapter
