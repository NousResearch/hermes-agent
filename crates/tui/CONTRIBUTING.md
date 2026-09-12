# Contributing

This crate is the opt-in native (ratatui) Hermes TUI. Python `tui_gateway` still owns the agent. Ink `ui-tui/` stays the default `hermes --tui`.

## Test

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --locked
```

Dogfood a real session before a PR:

```bash
cargo install --path crates/tui
hermes --tui --native
```

Or from this directory:

```bash
export HERMES_PYTHON_SRC_ROOT=/absolute/path/to/hermes-agent
export HERMES_PYTHON=$HERMES_PYTHON_SRC_ROOT/.venv/bin/python
export HERMES_HOME=~/.hermes
cargo run --release
```

Do not replace Ink or flip `hermes --tui` without soak time on macOS, Linux, and WSL2.

Keep this tree in sync with the standalone crate via `scripts/sync-landing.sh crates/tui` from https://github.com/0xNyk/hermes-tui. That script copies `src/` and `Cargo.lock` and keeps this package's `repository` URL.
