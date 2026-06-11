# Bootstrap Tool Archives

Place release-time portable tool archives here before building the Tauri bundle.

The Rust bootstrapper checks this bundled resource directory before downloading
Node.js, uv, and Git for Windows into `HERMES_HOME/bootstrap-cache`.

Expected Windows archive names:

- `node-v22.*-win-x64.zip`, `node-v22.*-win-arm64.zip`, or `node-v22.*-win-x86.zip`
- `uv-x86_64-pc-windows-msvc.zip`, `uv-aarch64-pc-windows-msvc.zip`, or `uv-i686-pc-windows-msvc.zip`
- `PortableGit-2.54.0-64-bit.7z.exe`, `PortableGit-2.54.0-arm64.7z.exe`, or `MinGit-2.54.0-32-bit.zip`
