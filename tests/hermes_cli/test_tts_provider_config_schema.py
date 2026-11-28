"""The config UI offers every TTS engine the runtime speaks with.

``tts.provider`` lacked ``deepinfra``, a built-in engine, so no Settings screen built
from the schema could pick it.
"""

from hermes_cli.web_server_config import CONFIG_SCHEMA
from tools.tts_tool import BUILTIN_TTS_PROVIDERS


def test_tts_provider_choices_include_every_builtin_engine():
    field = CONFIG_SCHEMA["tts.provider"]
    assert field["type"] == "select"
    assert sorted(BUILTIN_TTS_PROVIDERS - set(field["options"])) == []
