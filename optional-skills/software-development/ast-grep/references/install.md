# Install ast-grep

The skill ships an `install.sh` (POSIX) and `install.ps1` (Windows) that try every reasonable method in priority order and fall back to a GitHub release download as a last resort. **You usually do not need to read this page.** Run the installer:

```bash
bash install.sh                        # macOS / Linux / WSL / Git Bash
pwsh -File install.ps1                 # Windows PowerShell
```

This page exists for the (rare) case the installer cannot find a working method, or you want to install ast-grep manually.

---

## Per-OS install commands (verbatim, copy-paste)

### macOS

```bash
brew install ast-grep                  # Homebrew - the primary path
sudo port install ast-grep             # MacPorts
npm install -g @ast-grep/cli           # if you have Node already
cargo install ast-grep --locked        # if you have Rust already
```

### Linux

```bash
# Universal (works on every distro)
npm install -g @ast-grep/cli
cargo install ast-grep --locked
pip install ast-grep-cli

# Distro-specific
nix-env -iA nixpkgs.ast-grep           # NixOS / Nix
brew install ast-grep                  # Linuxbrew

# NixOS shell.nix
nix-shell -p ast-grep
```

> **Linux gotcha**: the binary is named `sg`, but on most Linux systems `sg` is also the **`setgroups` command** from `util-linux`. The shell sees `setgroups` first and ignores ast-grep. Two options:
>
> 1. Always invoke `ast-grep` (full name).
> 2. Add an alias: `alias sg=ast-grep` in your `~/.bashrc` / `~/.zshrc`.
>
> The `ast_grep_helper.py` script in `scripts/` already handles this — when it sees `sg` on PATH on Linux, it runs `--version` and rejects the binary if it isn't ast-grep.

### Windows

```powershell
scoop install main/ast-grep            # Scoop (most common on dev machines)
winget install --id ast-grep.ast-grep  # Winget (Microsoft built-in)
choco install ast-grep                 # Chocolatey
npm install -g @ast-grep/cli           # any OS with Node
cargo install ast-grep --locked        # any OS with Rust
```

### WSL / Git Bash on Windows

Treat as Linux. Use `npm`, `cargo`, `pip`, or `bash install.sh`.

---

## Cross-platform / language-ecosystem methods

These work on every OS:

| Method | Command | Pros | Cons |
|---|---|---|---|
| **npm** | `npm install -g @ast-grep/cli` | Fast, prebuilt platform binaries | Needs Node 18+ |
| **cargo** | `cargo install ast-grep --locked` | Always builds latest from source | Slow (~3-5 min compile) |
| **cargo binstall** | `cargo binstall ast-grep` | Fast (downloads release binary) | Needs `cargo-binstall` first |
| **pip** | `pip install ast-grep-cli` | Works in any Python venv | Needs Python 3.8+ |
| **pipx** | `pipx install ast-grep-cli` | Isolated install | Needs pipx |
| **mise** | `mise use -g ast-grep` | asdf successor, version-pinning | Needs mise |
| **GitHub release** | manual download | Pure binary, no toolchain | Manual PATH setup |

---

## GitHub release manual install

If every package manager fails:

```bash
# 1. Pick the right asset for your OS+arch from the latest release:
#    https://github.com/ast-grep/ast-grep/releases/latest
#
#    Naming pattern:
#      app-aarch64-apple-darwin.zip          macOS Apple Silicon
#      app-x86_64-apple-darwin.zip           macOS Intel
#      app-aarch64-unknown-linux-gnu.zip     Linux ARM64 (glibc)
#      app-x86_64-unknown-linux-gnu.zip      Linux x86_64 (glibc)
#      app-x86_64-pc-windows-msvc.zip        Windows x86_64
#      app-aarch64-pc-windows-msvc.zip       Windows ARM64

# 2. Download and extract:
VERSION=0.45.0
TRIPLE=aarch64-apple-darwin
curl -fsSL "https://github.com/ast-grep/ast-grep/releases/download/${VERSION}/app-${TRIPLE}.zip" -o /tmp/ast-grep.zip
unzip /tmp/ast-grep.zip -d /tmp/ast-grep
sudo mv /tmp/ast-grep/ast-grep /usr/local/bin/sg
sudo chmod +x /usr/local/bin/sg

# 3. Verify:
sg --version
```

The skill's `install.sh` does steps 1-3 automatically and drops the binary in `<skill_root>/bin/sg` so you can use it without sudo.

---

## Build from source

```bash
git clone https://github.com/ast-grep/ast-grep.git
cd ast-grep
cargo install --path ./crates/cli --locked
```

Requires Rust 1.74+. Slowest path; only useful when you need a specific commit or unreleased fix.

---

## Verifying the install

```bash
ast-grep --version            # or `sg --version`
# ast-grep 0.45.0
```

Then sanity-check a real query:

```bash
echo 'console.log("hello")' | sg run -p 'console.log($MSG)' --lang js --stdin
```

Expected: a single match with the `console.log("hello")` call highlighted.

---

## Editor integration

After installing the CLI, set up your editor:

- **VS Code**: install the [`ast-grep`](https://marketplace.visualstudio.com/items?itemName=ast-grep.ast-grep-vscode) extension. Requires `sgconfig.yml` in workspace root for live diagnostics.
- **Neovim**: configure `nvim-lspconfig` with `ast_grep` server, or install [`telescope-ast-grep.nvim`](https://github.com/ray-x/telescope-ast-grep.nvim).
- **Helix**: add `ast-grep lsp` as a language server in `languages.toml`.
- **Emacs**: install [`ast-grep.el`](https://github.com/SunskyXH/ast-grep.el).

See `references/cli.md` for `ast-grep lsp` flags.

---

## Uninstall

| Method | Command |
|---|---|
| brew | `brew uninstall ast-grep` |
| npm | `npm uninstall -g @ast-grep/cli` |
| cargo | `cargo uninstall ast-grep` |
| pip | `pip uninstall ast-grep-cli` |
| pipx | `pipx uninstall ast-grep-cli` |
| scoop | `scoop uninstall ast-grep` |
| winget | `winget uninstall --id ast-grep.ast-grep` |
| choco | `choco uninstall ast-grep` |
| GitHub binary | `rm <skill_root>/bin/sg` |
