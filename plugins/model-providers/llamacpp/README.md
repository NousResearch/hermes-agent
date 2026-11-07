# llamacpp model provider plugin for Hermes

Serves llama.cpp endpoints - a bare llama-server or a llama-swap proxy -
as a first-class Hermes provider named "llamacpp" (aliases "llama.cpp",
"llama-cpp" and "llama-swap").
Local endpoint, no API key.

What it adds over the generic custom provider:

- Endpoint detection and a read-only server probe (probe.py). On
  llama-swap the probe never touches a model-scoped route for a model
  that is not resident: llama-swap starts models on those routes, and a
  metadata probe must never load a model as a side effect.

Each optional feature has its own page under docs/, with the config
keys it reads.

## Setup

The plugin ships bundled with hermes-agent and registers on start; there
is nothing to install. A user plugin of the same name at
~/.hermes/plugins/model-providers/llamacpp overrides this bundled copy
(last-writer-wins), so the profile can be replaced without editing the
checkout.

The plugin depends on provider-profile hook points in core that ship
in the same change. Installing only the plugin directory as a user
plugin on a release without those hooks is inert: the provider registers
but requests keep the stock custom-provider shape, so nothing changes
on the wire.

Point it at your server and pick a model:

    model:
      default: <model-id>
      provider: llamacpp
    providers:
      llamacpp:
        api: http://<host>:<port>/v1

Or configure only the providers block and select interactively:

    hermes model

The llamacpp provider row lists the server's live model catalog;
picking a model writes the selection to your config. No API key is
needed.

## Config surface

    providers.llamacpp.api              endpoint base URL (.../v1)
