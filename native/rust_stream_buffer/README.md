# Rust stream buffer

This crate is the first bounded native slice of the gateway refactor. It is an
optional `cdylib` used by `gateway.stream_consumer` for append-heavy storage of
the active stream text and complete turn ledger.

The Python gateway remains authoritative for session/profile identity,
authorization, queueing, rendering, transport delivery, and final-response
reconciliation. When the library is absent or rejects a write, the wrapper
continues with the equivalent Python backend.

Build and verify from the repository root:

```bash
cargo fmt --manifest-path native/rust_stream_buffer/Cargo.toml -- --check
cargo test --locked --manifest-path native/rust_stream_buffer/Cargo.toml
cargo clippy --locked --manifest-path native/rust_stream_buffer/Cargo.toml --all-targets -- -D warnings
cargo build --release --locked --manifest-path native/rust_stream_buffer/Cargo.toml
```

Source checkouts load the platform library from the crate's `target/release`
or `target/debug` directories. Packaged-runtime discovery is intentionally
not enabled by this first slice; a future packaging change must add an explicit,
profile-safe candidate path and an artifact smoke test.
