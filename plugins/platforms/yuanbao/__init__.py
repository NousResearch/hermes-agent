# Absolute import (not ``from .adapter``): the plugin loader executes this package as
# ``hermes_plugins.platforms__yuanbao``; importing the adapter by its canonical module name keeps
# ONE ``YuanbaoAdapter`` class object, so ``YuanbaoAdapter.get_active()`` from tools/yuanbao_tools.py
# and the standalone sender see the instance the gateway registry created.
from plugins.platforms.yuanbao.register import register

__all__ = ["register"]
