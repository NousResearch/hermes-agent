"""Frozen native plugin for the plugin-surface conformance suite.

Written the way the plugin guide tells an external author to write one: ``register(ctx)`` plus the
documented ABCs, importing only names on the published plugin API. Keep it frozen. When Hermes grows
a field, a parameter or a provider method, this file must keep loading and working unchanged; a
change here is a change to what existing plugins are promised.

Every callback appends one JSON line to ``<HERMES_HOME>/conformance-events.jsonl`` so the tests read
outcomes, not plugin internals.
"""

import json

from agent.browser_provider import BrowserProvider
from agent.delegation_provider import DelegationProvider
from agent.image_gen_provider import ImageGenProvider
from agent.secret_sources.base import SecretSource
from agent.terminal_env_provider import TerminalEnvironmentProvider
from agent.transcription_provider import TranscriptionProvider
from agent.tts_provider import TTSProvider
from agent.video_gen_provider import VideoGenProvider
from agent.web_search_provider import WebSearchProvider
from hermes_constants import get_hermes_home

PROVIDER_NAME = "conformance"


def _record(event, **fields):
    with open(get_hermes_home() / "conformance-events.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": event, **fields}, default=repr) + "\n")


# -- tool ------------------------------------------------------------------------------------------

ECHO_SCHEMA = {
    "name": "conformance_echo",
    "description": "Echo the arguments it was called with.",
    "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
}


def _echo(args, **kwargs):
    _record("tool", args=args)
    return json.dumps({"echo": args})


# -- hooks (one **kwargs form, one narrow legacy form) ---------------------------------------------

def _pre_tool_call(**kwargs):
    _record("pre_tool_call", keys=sorted(kwargs), tool_name=kwargs.get("tool_name"), args=kwargs.get("args"))


def _post_tool_call(**kwargs):
    _record("post_tool_call", keys=sorted(kwargs), tool_name=kwargs.get("tool_name"),
            status=kwargs.get("status"), result=kwargs.get("result"))


def _transform_tool_result(tool_name, result):
    if tool_name == "conformance_echo":
        return "transformed:" + result
    return None


def _pre_command(**kwargs):
    _record("pre_command", keys=sorted(kwargs), command=kwargs.get("command"), args_raw=kwargs.get("args_raw"))


# -- middleware ------------------------------------------------------------------------------------

def _tool_request(tool_name, args):
    if tool_name != "conformance_echo":
        return None
    return {"args": {**args, "text": args["text"] + "+mw"}, "source": "conformance-surface"}


# -- commands --------------------------------------------------------------------------------------

def _ping(raw_args):
    return "pong:" + raw_args


def _cli_setup(parser):
    parser.add_argument("--word", default="none")


def _cli_handler(args):
    print("conformance-cli:" + args.word)
    return 0


# -- providers (only the documented abstract methods are implemented) -------------------------------

class _Named:
    @property
    def name(self):
        return PROVIDER_NAME


class ConformanceImageGen(_Named, ImageGenProvider):
    def generate(self, prompt, *args, **kwargs):
        return {"success": False, "error": "conformance fixture"}


class ConformanceVideoGen(_Named, VideoGenProvider):
    def generate(self, prompt, *args, **kwargs):
        return {"success": False, "error": "conformance fixture"}


class ConformanceWebSearch(_Named, WebSearchProvider):
    def is_available(self):
        return False


class ConformanceBrowser(_Named, BrowserProvider):
    def is_available(self):
        return False

    def create_session(self, task_id):
        return {}

    def close_session(self, session_id):
        return True

    def emergency_cleanup(self, session_id):
        return None


class ConformanceTerminal(_Named, TerminalEnvironmentProvider):
    def is_available(self):
        return False

    def create_environment(self, *args, **kwargs):
        raise RuntimeError("conformance fixture")


class ConformanceTTS(_Named, TTSProvider):
    def synthesize(self, text, output_path, *args, **kwargs):
        raise RuntimeError("conformance fixture")


class ConformanceTranscription(_Named, TranscriptionProvider):
    def transcribe(self, file_path, *args, **kwargs):
        return {"success": False, "error": "conformance fixture"}


class ConformanceDelegation(_Named, DelegationProvider):
    def route_task(self, task, config):
        return {}


class ConformanceSecretSource(SecretSource):
    name = PROVIDER_NAME

    def fetch(self, cfg, home_path):
        raise RuntimeError("conformance fixture")


def _on_unload():
    _record("on_unload")


def register(ctx):
    ctx.register_tool(name="conformance_echo", toolset="conformance", schema=ECHO_SCHEMA, handler=_echo,
                      description="Echo the arguments it was called with.")
    ctx.register_hook("pre_tool_call", _pre_tool_call)
    ctx.register_hook("post_tool_call", _post_tool_call)
    ctx.register_hook("transform_tool_result", _transform_tool_result)
    ctx.register_hook("pre_command", _pre_command)
    ctx.register_middleware("tool_request", _tool_request)
    ctx.register_command("conformance-ping", _ping, description="Conformance ping")
    ctx.register_cli_command("conformance-cli", "Conformance CLI command", _cli_setup, _cli_handler)
    ctx.register_image_gen_provider(ConformanceImageGen())
    ctx.register_video_gen_provider(ConformanceVideoGen())
    ctx.register_web_search_provider(ConformanceWebSearch())
    ctx.register_browser_provider(ConformanceBrowser())
    ctx.register_terminal_environment_provider(ConformanceTerminal())
    ctx.register_tts_provider(ConformanceTTS())
    ctx.register_transcription_provider(ConformanceTranscription())
    ctx.register_delegation_provider(ConformanceDelegation())
    ctx.register_secret_source(ConformanceSecretSource())
    ctx.on_unload(_on_unload)
