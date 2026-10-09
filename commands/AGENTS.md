# commands/ — shared slash command definitions and execution

`COMMAND_REGISTRY` is the single built-in registry. Preserve aliases, ordering, execution keys,
busy policies, and Desktop wire metadata. `resolve_command`, `available_commands`, and
`command_desktop_meta` are the shared query interface.

Keep this package import-light. Configuration loaders, persistence, localized help rendering,
terminal dependencies, plugin discovery/lifecycle, and authorization belong to their consumers.
Gateway availability takes supplied canonical gate names; discovery never grants authority.
Dynamic metadata uses lazy `plugin_runtime.api` reads without registering another built-in index.

CLI presentation lives in `hermes_cli/commands_presentation.py`; Gateway help and configuration
gate loading live in `gateway/command_presentation.py`. `hermes_cli/commands.py` contains only
manifest-listed external plugin compatibility entries and must never be imported internally.

`commands.execution` owns the existing frozen execution contracts and the single executor map.
Application boundaries supply version/egress callbacks, resolved profile display inputs, and
catalog/translation callbacks through the existing context options. Shared execution must not
import CLI or Gateway implementation modules. Existing skill/bundle subsystem APIs retain
discovery and lifecycle ownership. Do not restore `hermes_cli/slash_exec.py` or add a forwarder.
Preserve Gateway authorization and approval enforcement independently of discovery/execution.

Tests belong in `tests/commands/` and run through `scripts/run_tests.sh`.

Run `python scripts/check_phase7_boundaries.py` for the blocking ownership gate.
Its contract tests live in `tests/scripts/test_check_phase7_boundaries.py`;
retired imports, forwarding paths and duplicate command owners must stay absent.
