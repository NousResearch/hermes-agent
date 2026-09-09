"""Catalog capacity must agree with the actual native --model selection."""
import json
import os
import sys

from tests.providers.test_claude_oauth_directsdk import FAKE
from tests.providers.test_claude_oauth_directsdk_discovery import profile  # noqa: F401

EXPECTED = {
    'claude-sonnet-5[1m]': 1_000_000,
    'claude-haiku-4-5-20251001': 200_000,
    'claude-opus-5[1m]': 1_000_000,
    'claude-opus-4-8[1m]': 1_000_000,
    'claude-fable-5-1[1m]': 1_000_000,
}


def test_catalog_windows_match_explicit_native_routes(profile):
    from agent.model_metadata import get_model_context_length
    assert set(profile.fallback_models) == set(EXPECTED)
    assert profile.default_aux_model == 'claude-sonnet-5[1m]'
    for model, window in EXPECTED.items():
        assert profile.get_model_context_length(model) == window
        assert get_model_context_length(model, provider=profile.name) == window
        assert get_model_context_length(model, provider=profile.name, config_context_length=200000) == 200000
    assert profile.get_model_context_length('unqualified-future-model') is None


def test_native_argv_enables_only_known_long_context_models(profile, tmp_path):
    capture = tmp_path / 'argv.json'
    native = tmp_path / 'native.py'
    native.write_text(FAKE.replace('rows=[]', "pathlib.Path(os.environ['ARGV_CAPTURE']).write_text(json.dumps(sys.argv))\nrows=[]"))
    aliases = {'sonnet':'claude-sonnet-5[1m]', 'opus':'claude-opus-5[1m]',
               'haiku':'claude-haiku-4-5-20251001', 'fable':'claude-fable-5-1[1m]',
               'unqualified-future-model':'unqualified-future-model'}
    with_client = profile.create_client(command=[sys.executable,str(native)], env={'PATH':os.defpath,'HOME':str(tmp_path),'ARGV_CAPTURE':str(capture)})
    try:
        for requested, expected in {**{m:m for m in EXPECTED}, **aliases}.items():
            with_client.create(model=requested, messages=[{'role':'user','content':'fixture'}],
                               tools=[{'type':'function','function':{'name':'probe','description':'TAIL','parameters':{'type':'object','properties':{'value':{'type':'string'}}}}}])
            argv = json.loads(capture.read_text())
            assert argv[argv.index('--model')+1] == expected
    finally:
        with_client.close()
