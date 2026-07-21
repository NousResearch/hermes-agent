# Future standalone repository

`projects/Zai-router/` is structured so its documentation can later become the root of a dedicated repository. The current implementation remains integrated into Hermes Agent because it modifies shared provider and Anthropic transport code.

## Before extraction

A standalone package would need stable extension hooks for all of the following:

1. Registering a model provider profile.
2. Selecting `anthropic_messages` without modifying Hermes core.
3. Installing endpoint-specific request adaptation and sanitisation.
4. Preserving behaviour during provider switching and fallback activation.
5. Adding provider tests without copying Hermes internals.

## Proposed standalone layout

```text
Zai-router/
├── README.md
├── pyproject.toml
├── src/zai_router/
│   ├── provider.py
│   ├── adapter.py
│   └── plugin.py
├── tests/
├── docs/
└── LICENSE
```

## Migration rule

Do not delete the integrated Hermes files until the standalone package is installable, auto-discovered, and passes the same provider, transport, picker, and live smoke tests. The extraction should be a separate pull request so rollback remains straightforward.
