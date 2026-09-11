# Locked CI toolchains

Check out this repository, then use `./.github/actions/setup-pm`. The runner's
preinstalled Python bootstraps PM with the standard library only. PM downloads
and verifies the exact native artifacts from `pm/lock.json`. The action does
not resolve a version range, install another setup action, or modify the lock.

```yaml
- uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd # v6.0.2
- uses: ./.github/actions/setup-pm
  with:
    toolchain: all
    extras: '["dev"]'
- run: python --version && node --version && npm --version
```

`toolchain` defaults to `python` (Python and PM's private installer). `node` installs Node and npm;
`all` installs both pairs. There are no version overrides. `extras` is a JSON
array because GitHub action inputs are strings. Omit it for tools only; `[]`
installs the core Python dependencies, and `["dev"]` adds the dev extra. PM
checks `uv.lock`, installs the requested dependencies, and validates the environment.
The dev extra uses an independent test environment including the test dependency
group; only non-test installs publish an application selection. It does not enable plugins.

Subsequent steps get `python`, `python3`, `node`, `npm` and `npx`
for the selected toolchain on PATH. The pinned npm precedes Node's bundled npm.
On Windows, PM selects the host architecture even if the bootstrap interpreter
runs under x64 emulation. A disposable command environment supplies the missing
`python3.exe` alias without changing the verified interpreter store.

For Python dependencies, the action exports `HERMES_PYTHON` and `VIRTUAL_ENV`.
Use `scripts/run_tests.sh`; do not activate `.venv`. The environment belongs to
PM under the runner's temporary home, not the checkout. No installer executable
or `UV_*` policy variables are exposed. Tool-only jobs can prepare small isolated
CI environments with `python -m scripts.ci.python_packages package==version`.
This exports the selected Python and tool entrypoints for subsequent steps;
append `-- -m module ...` to run a Python command directly and preserve its exit
status. Neither route modifies the cached tool store. Native builds still need their system libraries
and compiler, such as OpenSSL for Windows ARM64 cryptography.

## Caches

The official, SHA-pinned `actions/cache` transports three independent caches:

| Cache | Contents | Identity |
| --- | --- | --- |
| Tools | PM store and installed facts | Native target, toolchain and PM lock hash |
| Python | PM's actual uv download/build cache | Native target, OS version, Python version, prune policy and dependency-file hash |
| Node | `npm config get cache` | Runner OS, native architecture and npm dependency-lock hash |

All restores use the exact primary key, without fallback prefixes, matching
setup-uv and setup-node's npm behavior. Successful jobs save at teardown;
exact hits are not saved again. PM re-verifies restored tools before use.
Dependency caches never contain `node_modules` or virtual environments.
Keep an installed-tree cache in the caller if that job needs one.

`cache`, `cache-python` and `cache-node` independently disable the store, uv,
and npm caches. Each defaults to `true`; language-specific caches run only for
that toolchain. `python-cache-dependency-glob` defaults to `pyproject.toml` and
`uv.lock`. `node-cache-dependency-path` defaults to `package-lock.json`; use
`website/package-lock.json` for site jobs. Both accept multiline glob strings.
An npm cache without a matching lockfile fails, rather than caching an
unversioned dependency set.

`prune-python-cache: true` registers PM's CI cache-pruning operation at teardown,
after the caller's installs and before the cache save. It is skipped on an
exact hit. The default is `false`, as in setup-uv v9; migrated v8 callers opt in
to retain their former policy. The small nested JavaScript action exists only
because GitHub composite actions cannot declare their own post step. It uses
Node's standard library and has no bundled dependencies.

Outputs include `python-version`, `uv-version`, `node-version`, `npm-version`,
`python-path`, `venv`, `target`, and the three `*-cache-hit` flags.
Use the version outputs in installed-tree cache keys instead of repeating pins.

`.github/workflows/pm-toolchain.yml` exercises cold setup and a separate warm
runner for Linux, macOS and Windows on both architectures. Its optional cache
suffix isolates cache lookup, not the repository's storage budget. The trusted
`pm-toolchain-cache-cleanup.yml` completion workflow deletes only that run's
smoke keys after all cache saves finish, including failed or cancelled runs.
It must be on the default branch for GitHub to run it.
