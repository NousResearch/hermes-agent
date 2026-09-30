# Termux Tensor — Complete Installation Guide

## Scope

This is the Termux-only V1 deployment. Hermes remains the agent layer. LiteRT-LM is the preferred local .litertlm runtime. llama.cpp is the persistent OpenAI-compatible localhost fallback.

## 1. Termux prerequisites

Use the normal Termux application with an ARM64 Android device.

```bash
pkg update
pkg upgrade -y
pkg install -y git curl wget ca-certificates openssl clang make cmake pkg-config jq unzip zip
termux-setup-storage
```

Verify:

```bash
uname -m
getprop ro.product.cpu.abi
getprop ro.soc.manufacturer
getprop ro.soc.model
getprop ro.build.version.sdk
```

Expected CPU architecture is aarch64 / arm64-v8a.

## 2. Hermes installation

The official Hermes documentation currently provides a signed APT repository for aarch64 Termux, but its dedicated Termux page currently warns that the package is temporarily broken. Do not disable signature verification or use an unsigned package.

When the official package is working, the documented installation is:

```bash
pkg install curl gnupg
mkdir -p "$PREFIX/etc/apt/keyrings"
curl -fsSL https://hermes-assets.nousresearch.com/releases/termux/stable/key.asc -o "$PREFIX/etc/apt/keyrings/hermes-agent.asc"
gpg --show-keys --with-fingerprint "$PREFIX/etc/apt/keyrings/hermes-agent.asc"
```

Expected fingerprint:

```text
C572 B5FD D1A2 9CCF A9A9 12B6 840B 0848 E139 156D
```

Then:

```bash
printf '%s\n' 'deb [signed-by=$PREFIX/etc/apt/keyrings/hermes-agent.asc] https://hermes-assets.nousresearch.com/releases/termux/stable hermes-stable main' > "$PREFIX/etc/apt/sources.list.d/hermes-agent.list"
pkg update
pkg install hermes-agent
hermes --version
hermes doctor
```

If your existing Hermes source installation already works, keep it and continue.

## 3. Clone this project

```bash
git clone -b feature/termux-google-tensor https://github.com/abdulraheemnohri/hermes-agent.git
cd hermes-agent
bash termux-tensor/install.sh
```

## 4. Verify the installation

```bash
tensor doctor
tensor status
```

## 5. Directory layout

```text
~/.hermes/
├── models/
│   ├── litertlm/
│   └── gguf/
└── tensor/
    ├── config/
    ├── logs/
    ├── runtime/
    ├── state/
    └── cache/
```

## 6. LiteRT-LM setup

The official LiteRT-LM native documentation uses the Android ARM64 native executable named litert_lm_main. Do not assume that every build provides a litert-lm serve command.

Check:

```bash
command -v litert_lm_main
litert_lm_main --help
```

If the binary is stored elsewhere:

```bash
export TENSOR_LITERT_BIN=/path/to/litert_lm_main
```

Persist the setting:

```bash
echo 'export TENSOR_LITERT_BIN=/path/to/litert_lm_main' >> ~/.bashrc
source ~/.bashrc
```

LiteRT-LM documents Android ARM64 native builds using NDK r28b or newer. Building the complete native runtime directly inside Termux is experimental; a compatible ARM64 Android binary is the preferred V1 deployment.

## 7. Model installation

Copy a compatible .litertlm model:

```bash
cp ~/storage/downloads/model.litertlm ~/.hermes/models/litertlm/
tensor model list
tensor model select model
tensor model current
```

## 8. Direct LiteRT-LM test

```bash
tensor lm run
```

Benchmark:

```bash
tensor lm benchmark
```

Run direct LiteRT-LM successfully before connecting Hermes.

## 9. Hermes local-server mode

Hermes custom-provider mode requires an OpenAI-compatible HTTP endpoint. For Termux V1, use llama.cpp when you need a persistent localhost server.

```bash
tensor server start
tensor server status
tensor server test
```

Default endpoint:

```text
http://127.0.0.1:9379/v1
```

Then:

```bash
tensor hermes configure
hermes
```

## 10. Recommended phone settings

```text
Backend              CPU
Active models        1
Parallel inference   OFF
Context              conservative
Background mode      OFF
LAN exposure         OFF
Auto start            OFF
Thermal mode         BALANCED
Logging              INFO
```

For a 6 GB phone, keep one model loaded, use a conservative context, and avoid sustained background generation.

## 11. Commands

```text
tensor doctor
tensor status
tensor start
tensor stop
tensor model list
tensor model import <file>
tensor model select <name>
tensor model current
tensor model remove <name>
tensor lm run
tensor lm benchmark
tensor lm status
tensor server start
tensor server stop
tensor server restart
tensor server status
tensor server test
tensor hermes configure
tensor hermes chat
tensor hermes gateway
tensor thermal status
tensor thermal cool
tensor thermal balanced
```

## 12. Environment variables

```text
HERMES_HOME
TENSOR_LITERT_BIN
TENSOR_MODEL
TENSOR_MODEL_DIR
TENSOR_HOST
TENSOR_PORT
TENSOR_SERVER_BIN
TENSOR_SERVER_ARGS
TENSOR_THERMAL_MODE
TENSOR_LOG_LEVEL
```

Example:

```bash
export TENSOR_MODEL_DIR="$HOME/.hermes/models"
export TENSOR_HOST=127.0.0.1
export TENSOR_PORT=9379
export TENSOR_THERMAL_MODE=balanced
```

## 13. Background execution

Android can suspend or terminate Termux processes. A wake lock can help but is not a guarantee.

```bash
termux-wake-lock
tensor server start
hermes gateway run
```

Release it:

```bash
termux-wake-unlock
```

## 14. Security

Keep the local server on 127.0.0.1. Do not expose port 9379 to Wi-Fi or the Internet without authentication and access control. Never commit API keys.

## 15. Troubleshooting

### Hermes
```bash
command -v hermes
hermes --version
hermes doctor
```

### LiteRT-LM
```bash
command -v litert_lm_main
echo "$TENSOR_LITERT_BIN"
tensor doctor
```

### Model
Check the file path, model compatibility, free RAM, storage permissions, and selected backend.

### Hermes endpoint
First make direct inference work, then make the local HTTP server work, then configure Hermes. This isolates runtime problems from agent problems.

## 16. V2

True Android Tensor accelerator integration belongs in the native Android LiteRT-LM service. Termux remains the Hermes control layer and communicates with that service over localhost.