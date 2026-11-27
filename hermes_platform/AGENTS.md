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
`hermes_platform/resolver/`. Lookup never installs, downloads, or starts anything, with one
exception: `resolver/git.py` may run `/usr/bin/xcode-select -p` (macOS only, when the git found is
`/usr/bin/git` and `DEVELOPER_DIR` is unset; bounded at 2 s; a positive answer cached for the
process, a negative one for 5 s). Only Apple's own lookup says whether that stub would open the
Command Line Tools dialog, and the link files it reads are undocumented and change between releases.
