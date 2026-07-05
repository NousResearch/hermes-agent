# Bundled web search providers — plugins/web/.
#
# Each subdirectory follows the image_gen plugin layout:
#   plugins/web/<name>/{plugin.yaml, __init__.py, provider.py}
#
# They auto-load via kind: backend and register via
# ctx.register_web_search_provider() into agent.web_search_registry.

from . import brave_free  # noqa: F401
from . import crawl4ai  # noqa: F401
from . import ddgs  # noqa: F401
from . import exa  # noqa: F401
from . import firecrawl  # noqa: F401
from . import parallel  # noqa: F401
from . import searxng  # noqa: F401
from . import tavily  # noqa: F401
from . import xai  # noqa: F401