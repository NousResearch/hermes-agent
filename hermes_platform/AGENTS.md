# hermes_platform/ — machine facts and resource lookup

Applies on top of the root `AGENTS.md`.

**Machine facts and resource lookup go through `hermes_platform`.** `hermes_platform.host` is the
one answer for OS family, native architecture (`IsWow64Process2` → `platform.machine()`; never
`PROCESSOR_ARCHITECTURE` alone, it reads AMD64 under x64-on-ARM64 emulation), CPU identity, and
WSL/container/Termux. Facts are cached per process and take **no environment-variable input**, so
a hardware recognizer (`host/products.py`) cannot be set from a shell. Distinguish the control
host (where this Python runs) from the terminal execution target (SSH/container) and the Desktop
client (another machine): `host.*` answers only the first. A new bare `shutil.which` or a
hand-written known-path table outside `hermes_platform/` fails
`tests/test_managed_runtime_resolution.py` unless allowlisted with a reason; resolvers land in
`hermes_platform/resolver/`. Lookup never installs, downloads, or starts anything.
