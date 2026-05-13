# Bundled web search providers — plugins/web/.
#
# Each subdirectory follows the image_gen plugin layout:
#   plugins/web/<name>/{plugin.yaml, __init__.py, provider.py}
#
# They auto-load via kind: backend and register via
# ctx.register_web_search_provider() into agent.web_search_registry.
