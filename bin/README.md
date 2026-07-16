# bin/hermes — in-repo launcher

This is the ejected-mode launcher stub. It's a committed shell script (not
the native Rust binary) that runs Hermes from a source checkout.

## How it works

1. If `.hermes-launcher/hermes` exists (installed by `hermes dev sync`),
   exec the native launcher (which handles env setup + venv self-check).
2. Otherwise, fall back to `exec .venv/bin/python -m hermes_cli.main`
   with inline env hygiene (unsets `PYTHONPATH`/`PYTHONHOME`, sets
   `UV_NO_CONFIG=1`, `VIRTUAL_ENV`).

In managed bundles, `bin/hermes` is the real native binary (phase 1).

## Usage

```sh
./bin/hermes --version      # run from the checkout
./bin/hermes doctor         # any hermes subcommand
```

If `.venv` is missing, prints a clear error and exits 3:

```
hermes: this tree's virtualenv is missing or broken.
  tree: /path/to/checkout
  fix:  hermes dev sync        (source checkout)
```

See `docs/updater-world.md` §2.5.1 for the full design.
