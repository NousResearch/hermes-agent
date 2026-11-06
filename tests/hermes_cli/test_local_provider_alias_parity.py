"""Parity contract for local OpenAI-compatible provider aliases (issue #62213).

The CLI, config, and credential layers each normalise provider names through a
different table:

* ``hermes_cli.providers.normalize_provider``  (routing / api-mode)
* ``hermes_cli.models.normalize_provider``     (model picker / web server)
* ``hermes_cli.auth.resolve_provider``         (credential resolution)

Historically these disagreed for the local self-hosted server aliases: bare
``local`` stayed ``"local"`` in ``providers`` and ``models`` while ``auth`` mapped
it to ``"custom"``, and ``vllm`` got three answers (``local`` / ``vllm`` /
``custom``) — the "custom, local, custom:local" confusion the bug report
describes. Every alias the ``custom`` provider profile declares must land on
``custom``; the one exception is the model table's managed llama.cpp runtime id,
which the picker's Local row and the staged-library validator key on.
"""

import pytest

from hermes_cli.auth import resolve_provider
from hermes_cli.models import normalize_provider as models_normalize
from hermes_cli.providers import LLAMACPP_ALIASES, normalize_provider as providers_normalize
from providers import get_provider_profile

_CUSTOM_ALIASES = tuple(get_provider_profile("custom").aliases)


@pytest.mark.parametrize("alias", _CUSTOM_ALIASES)
def test_custom_profile_aliases_normalize_to_custom_in_every_table(alias):
    assert providers_normalize(alias) == "custom"
    assert resolve_provider(alias) == "custom"
    if alias in LLAMACPP_ALIASES:
        assert models_normalize(alias) in LLAMACPP_ALIASES
    else:
        assert models_normalize(alias) == "custom"
