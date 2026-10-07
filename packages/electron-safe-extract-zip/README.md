# Electron safe ZIP extractor

This private workspace replaces Electron's unsigned native ZIP addon while preserving its `extract(archive, { dir })` contract. Electron's installer downloads the release archive through `@electron/get` and verifies the pinned checksum before this adapter runs.

The adapter launches only an absolute operating-system tool with `shell: false`:

- Windows: `%SystemRoot%\System32\tar.exe`
- macOS: `/usr/bin/tar`
- Linux and other Unix: `/usr/bin/unzip`, `/bin/unzip`, or an absolute system Python running the bundled, path-confined `extract.py` helper. The helper preserves executable mode bits required by the Electron runtime.

It never resolves an extractor through `PATH`, carries no native `.node` binary, validates absolute archive/output paths, bounds captured stderr, and fails closed when no trusted extractor exists.
