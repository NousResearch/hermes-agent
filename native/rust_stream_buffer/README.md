# Rust stream buffer

Small optional `cdylib` used by `gateway.stream_consumer` for high-frequency
stream accumulation. Python remains the compatibility fallback.

Build from the repository root:

```bash
cargo test --manifest-path native/rust_stream_buffer/Cargo.toml
cargo build --release --manifest-path native/rust_stream_buffer/Cargo.toml
```

The Python loader discovers the release/debug library automatically from the
crate target directory. Deployed bundles should place the library alongside the
native payload and update the loader's candidate path for that bundle.
