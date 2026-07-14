"""Parity contract for local OpenAI-compatible provider aliases (issue #62213).

The CLI, config, and credential layers each normalise provider names through a
different table:

* ``hermes_cli.providers.normalize_provider``  (routing / api-mode)
* ``hermes_cli.models.normalize_provider``     (model picker / web server)
* ``hermes_cli.auth.resolve_provider``         (credential resolution)

Historically these disagreed for the local self-hosted server aliases: bare
``vllm`` / ``llamacpp`` resolved to ``"local"`` in ``providers`` (an orphan id
with no ``ProviderDef``), stayed ``"vllm"`` in ``models`` (unknown), yet mapped
to ``"custom"`` in ``auth`` — the "custom, local, custom:local" confusion the
bug report describes. They must all agree on the generic ``"custom"`` provider.
"""

import pytest

from hermes_cli.auth import resolve_provider
from hermes_cli.models import normalize_provider as models_normalize
from hermes_cli.providers import normalize_provider as providers_normalize

# Local OpenAI-compatible server aliases users are told to configure.
#
# ``local`` is included: it is declared a ``custom`` alias in
# ``plugins/model-providers/custom/__init__.py`` and ``auth.resolve_provider``
# already mapped it to ``"custom"`` (statically and via the plugin import), so
# leaving it as the orphan ``"local"`` id (no ``ProviderDef``) in the providers
# and models tables was exactly the cross-table disagreement this contract
# guards against. Routing code already treats ``{"custom", "local"}`` as
# equivalent (e.g. ``model_switch``/``web_server``), so unifying to ``custom``
# is behaviour-preserving.
_LOCAL_ALIASES = ("local", "ollama", "vllm", "llamacpp", "llama.cpp", "llama-cpp")


@pytest.mark.parametrize("alias", _LOCAL_ALIASES)
def test_local_aliases_normalize_to_custom_in_every_table(alias):
    assert providers_normalize(alias) == "custom"
    assert models_normalize(alias) == "custom"
    assert resolve_provider(alias) == "custom"
