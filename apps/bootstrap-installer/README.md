# Hermes Bootstrap Installer

Tauri-based installer application for Hermes Agent. Provides a GUI-driven setup experience for macOS and Windows users.

## Features

- One-click installation of Hermes Agent
- Automatic dependency detection and installation (Python, Node.js, etc.)
- Progress tracking and error handling
- Native look-and-feel on macOS and Windows

## Architecture Support

The macOS installer is built as a **universal binary** supporting both Intel (x86_64) and Apple Silicon (arm64) Macs.

- **Minimum macOS version**: 12.0 (Monterey)
- **Supported architectures**: x86_64, arm64 (universal binary)

## Development

### Prerequisites

- Node.js 22+
- Rust toolchain (rustup recommended)
- For universal macOS builds:
  ```bash
  rustup target add aarch64-apple-darwin
  rustup target add x86_64-apple-darwin
  ```

### Building

```bash
cd apps/bootstrap-installer

# Install dependencies (run from monorepo root)
cd ../..
npm ci

# Build for current architecture
cd apps/bootstrap-installer
npm run tauri:build

# Build universal macOS binary (contains both x64 and arm64)
npm run tauri:build -- --target universal-apple-darwin

# Build for specific architecture
npm run tauri:build -- --target x86_64-apple-darwin   # Intel only
npm run tauri:build -- --target aarch64-apple-darwin  # Apple Silicon only
```

### Output

Bundles are created in `src-tauri/target/<target>/release/bundle/`:
- **DMG**: `dmg/Hermes_<version>_<arch>.dmg`
- **App bundle**: `macos/Hermes.app`

### Verifying Universal Binary

```bash
DMG="src-tauri/target/universal-apple-darwin/release/bundle/dmg/Hermes_0.0.1_universal.dmg"
hdiutil attach "$DMG" -mountpoint /tmp/verify -readonly
lipo -info "/tmp/verify/Hermes.app/Contents/MacOS/Hermes-Setup"
# Expected: Architectures in the fat file: ... are: x86_64 arm64
hdiutil detach /tmp/verify
```

## CI/CD

The `.github/workflows/build-desktop-installer.yml` workflow builds universal macOS DMG on push of `installer-v*` tags or manual dispatch.

### Creating a Release

```bash
git tag installer-v0.1.0
git push origin installer-v0.1.0
```

The workflow will:
1. Build a universal DMG for macOS
2. Verify the binary contains both x86_64 and arm64
3. Upload as a GitHub Actions artifact
4. Create a draft GitHub Release with the DMG attached

## Project Structure

- `src/` — React frontend (Vite + TypeScript)
- `src-tauri/` — Tauri backend (Rust)
  - `src/main.rs` — Main Tauri entry point
  - `src/install_script.rs` — Installation logic (wraps `scripts/install.sh` / `install.ps1`)
  - `Cargo.toml` — Rust dependencies
  - `tauri.conf.json` — Tauri configuration (bundle settings, entitlements, etc.)

## Why Universal Binary?

Apple [recommends](https://developer.apple.com/documentation/apple-silicon/building-a-universal-macos-binary) universal binaries for app distribution because:
- Single download for all Mac users
- Automatic architecture selection at runtime
- Better user experience (no "Which Mac do I have?" confusion)

Universal binaries are slightly larger (~1.5× size) but avoid the maintenance overhead of dual release channels.
