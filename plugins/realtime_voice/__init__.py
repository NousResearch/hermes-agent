# Bundled realtime (speech-to-speech) voice providers — plugins/realtime_voice/.
#
# Each subdirectory follows the image_gen / web plugin layout:
#   plugins/realtime_voice/<name>/{plugin.yaml, __init__.py, provider.py}
#
# ``kind: backend`` manifests auto-load and register through
# ``ctx.register_realtime_voice_provider()`` — the same hook a user plugin
# in ~/.hermes/plugins/ uses, so nothing here is privileged.
