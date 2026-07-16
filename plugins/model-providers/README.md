# Model Provider Plugins

Each subdirectory is a self-contained provider profile plugin. The
directory layout mirrors `plugins/platforms/`:

```
plugins/model-providers/
├── openrouter/
│   ├── __init__.py      # registers the ProviderProfile
│   └── plugin.yaml      # manifest: name, kind, version, description
├── anthropic/
│   ├── __init__.py
│   └── plugin.yaml
└── ...
```

## How discovery works

`providers/__init__.py._discover_providers()` loads bundled profiles, enabled
Python-package entry points from `hermes_agent.model_providers`, and then
`$HERMES_HOME/plugins/model-providers/` the first time anything calls
`get_provider_profile()` or `list_providers()`. Each directory `__init__.py`
is imported and expected to call `providers.register_provider(profile)`.

A user-directory leaf reserves its canonical `model-providers/<leaf>` key, so a
same-named package entry point is skipped without invoking its registrar. User
directories can still override bundled registrations through last-writer-wins
registration. Legacy `<repo>/providers/<name>.py` modules load last and can
replace earlier registrations by the same rule. Package providers must be
explicitly enabled. Manually dropped directories remain active by default, but
an explicit `plugins.disabled` entry always wins.

## Adding a new provider

1. Create `plugins/model-providers/<your_provider>/__init__.py`:

   ```python
   from providers import register_provider
   from providers.base import ProviderProfile

   my_provider = ProviderProfile(
       name="your-provider",
       aliases=("alias1", "alias2"),
       display_name="Your Provider",
       description="One-line description shown in the setup picker",
       signup_url="https://your-provider.example.com/keys",
       env_vars=("YOUR_PROVIDER_API_KEY", "YOUR_PROVIDER_BASE_URL"),
       base_url="https://api.your-provider.example.com/v1",
       default_aux_model="your-cheap-model",
   )

   register_provider(my_provider)
   ```

2. Create `plugins/model-providers/<your_provider>/plugin.yaml`:

   ```yaml
   name: your-provider-profile
   kind: model-provider
   version: 1.0.0
   description: Short sentence about the provider
   author: Your Name
   ```

Nothing else needs to change. `auth.py`, `config.py`, `models.py`,
`doctor.py`, `model_metadata.py`, `runtime_provider.py`, and the
chat_completions transport all auto-wire from the registry.

## Distribution and lifecycle

Python packages publish a zero-argument registration callable in the dedicated
entry-point group, then require explicit activation:

```toml
[project.entry-points."hermes_agent.model_providers"]
your-provider = "your_provider_package:register"
```

```bash
pip install <distribution>
hermes plugins enable model-providers/your-provider
```

Git repositories with `kind: model-provider` install under
`$HERMES_HOME/plugins/model-providers/<name>/`. See the
[model-provider plugin guide](../../website/docs/developer-guide/model-provider-plugin.md)
for package, Git, activation, ownership, and restart details.

A repository supporting both pip and Git/manual installation also needs a root
`__init__.py` shim that imports and calls the same registrar exposed by the
package entry point.

## Non-trivial profiles

Override the `ProviderProfile` hooks in a subclass for per-provider
quirks — see `plugins/model-providers/openrouter/__init__.py` for
`build_extra_body` and `build_api_kwargs_extras` examples, and
`plugins/model-providers/gemini/__init__.py` for `thinking_config`
translation.
