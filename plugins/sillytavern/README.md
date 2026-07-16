# SillyTavern Hermes plugin

This plugin pins SillyTavern at vendor/SillyTavern and exposes a narrow Hermes
bridge. The upstream checkout remains untouched. Hermes can inspect the
revision, start or stop an isolated local server, and send one non-streaming
chat-completions request through the SillyTavern provider configuration.

The upstream source is the AGPL-3.0 project at:

https://github.com/SillyTavern/SillyTavern

## Enable

Run:

~~~text
hermes plugins enable sillytavern
~~~

Add the non-secret behaviour settings to the active Hermes config.yaml:

~~~yaml
plugins:
  enabled:
    - sillytavern
  entries:
    sillytavern:
      port: 8000
      allow_process_control: true
      allow_network: true
      model: "your-configured-model"
      chat_completion_source: "openai"
      startup_timeout_seconds: 90
      chat_timeout_seconds: 600
~~~

Provider credentials belong in SillyTavern's own data directory and must be
configured through its UI. The bridge never accepts or writes provider API keys
through Hermes config.yaml.

## Install the upstream runtime

The submodule is intentionally kept as a source pin. Install its production
dependencies once:

~~~text
cd vendor/SillyTavern
npm ci --omit=dev
~~~

The bridge stores runtime state under:

~~~text
~/.hermes/sillytavern/
~~~

That directory contains the isolated SillyTavern data root, generated config,
server log, and managed-process metadata. It is not part of the repository.

## Use from Hermes

Inspect readiness:

~~~text
hermes sillytavern status
~~~

Start the local server:

~~~text
hermes sillytavern start --acknowledge-side-effects
~~~

Send a request after configuring a provider and model:

~~~text
hermes sillytavern generate --prompt "Write a short greeting." --acknowledge-side-effects
~~~

The AI-facing tool is sillytavern_generate. It accepts either prompt or
messages, requires explicit acknowledgement, and returns a normalized reply.
The start and stop tools also require acknowledgement. Unmanaged processes are
never terminated by the stop path.

## Update the pinned source

From the parent repository:

~~~text
git -C vendor/SillyTavern fetch origin release
git -C vendor/SillyTavern checkout origin/release
git add .gitmodules vendor/SillyTavern
~~~

Review the resulting submodule commit before publishing it.
