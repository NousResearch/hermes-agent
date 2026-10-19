# Hermes Agent for Android - APK Installation

This guide explains how to install and use Hermes Agent on Android devices.

## \ud83d\udcf1 Quick Start

### Option 1: Download Pre-built APK from GitHub Actions

1. Go to: [GitHub Actions](https://github.com/luciano2-web/hermes-agent/actions)
2. Find the latest successful run of "Android APK Build" workflow
3. Click on the workflow run
4. Scroll down to "Artifacts" section
5. Download `hermes-agent-apk` artifact
6. Extract the APK file from the zip archive

### Option 2: Build the APK Yourself

#### 1. Download or Build the APK

Use the build script:
```bash
./scripts/build_android_apk.sh --version 1.0.0 --debug
```

The APK will be created as `Hermes-Agent-1.0.0-debug.apk`

---

## \ud83c\udfaf Features

\u2705 **Full AI Agent** - Same powerful agent as desktop version
\u2705 **Continuous Learning** - Skills improve with use
\u2705 **Multi-Model Support** - Use OpenRouter, Nous Research, and more
\u2705 **Background Execution** - Keep agent running
\u2705 **Offline Mode** - Works without internet (with local models)
\u2705 **Easy Updates** - Download new versions as they're released

## \ud83d\udccb Requirements

- **Android 7.0 (Nougat) or later**
- **2GB RAM minimum** (4GB recommended)
- **500MB+ free storage**
- **Internet connection** for first setup

## \ud83d\ude80 First Launch

On first launch, Hermes Agent will:

1. Request necessary permissions
2. Initialize Python runtime (bundled with the APK)
3. Initialize your agent

## \ud83d\udcac Usage

### Starting a Conversation

Just type your message and press Enter. Hermes will:
- Understand your request
- Use tools when appropriate
- Maintain conversation context
- Learn from interactions

### Commands

```bash
# List available commands
hermes --help

# Change model
hermes model

# Start a new chat
hermes chat

# Run in gateway mode
hermes gateway
```

### Model Providers

Configure your preferred AI model provider:

```bash
# Set OpenRouter (default)
hermes config set model_provider openrouter
hermes config set openrouter_api_key your_key

# Set Nous Research
hermes config set model_provider nous
hermes config set nous_api_key your_key

# Use local model
hermes config set model_provider local
```

## \u2699\ufe0f Configuration

Edit `~/.hermes/config.yaml` or use the `hermes config` command.

### Example Configuration

```yaml
model_provider: openrouter
openrouter_api_key: sk-or-v1-your-key
model: openai/gpt-4o-mini
```

## \ud83d\udd27 Troubleshooting

### "App not installed" Error

- **Solution 1**: Uninstall any previous version first
- **Solution 2**: Check storage space (need 500MB+ free)
- **Solution 3**: Enable "Install unknown apps" for your browser

### "Parse error" When Installing

- **Solution 1**: Download the APK again (may be corrupted)
- **Solution 2**: Check you're downloading the arm64 version
- **Solution 3**: Try a different browser

### App Crashes on Startup

- **Solution 1**: Clear app data (Settings > Apps > Hermes Agent > Storage > Clear Data)
- **Solution 2**: Reinstall the APK
- **Solution 3**: Check logs in `/data/data/com.nousresearch.hermes/files/logs/`

### Python Initialization Fails

- **Solution 1**: Ensure you have internet connection (only needed for first setup with Termux, not for APK)
- **Solution 2**: Wait and retry
- **Solution 3**: Try on a different network

### Out of Memory Errors

- **Solution 1**: Use a smaller model (e.g., `mistralai/mistral-tiny`)
- **Solution 2**: Close other apps to free memory
- **Solution 3**: Use Termux version for better memory management

## \ud83c\udfd7\ufe0f Building from Source

### Prerequisites

- [Android Studio](https://developer.android.com/studio) or [Android SDK](https://developer.android.com/studio#command-tools)
- [JDK 17](https://adoptium.net/) or later
- Internet connection

### Build Steps

```bash
# Clone repository
git clone https://github.com/luciano2-web/hermes-agent.git
cd hermes-agent

# Make build script executable
chmod +x scripts/build_android_apk.sh

# Build debug APK
./scripts/build_android_apk.sh --version 1.0.0 --debug

# APK will be at: Hermes-Agent-1.0.0-debug.apk
```

### Release Build

```bash
# Build signed APK
./scripts/build_android_apk.sh --version 1.0.0 --release
```

\u26a0\ufe0f **Note**: Release builds require a signing keystore. Place it at `android/app/keystore/release.keystore`

## \ud83d\udcda Alternative: Termux Installation

For advanced users who prefer terminal access:

```bash
# In Termux:
pkg update && pkg upgrade
pkg install wget
wget https://github.com/NousResearch/hermes-agent/releases/latest/download/hermes-agent_<version>_aarch64.deb
dpkg -i hermes-agent_<version>_aarch64.deb
hermes
```

## \ud83d\udd12 Security

\u26a0\ufe0f **Important**:

- Only install APKs from official sources
- Review permissions before installing
- Your conversation data stays on your device
- Model interactions go through your configured provider

## \ud83d\udcca Comparison: APK vs Termux

| Feature | APK | Termux |
|---------|-----|--------|
| Easy Installation | \u2705 Yes | \u274c Manual |
| GUI Interface | \u2705 Yes | \u274c CLI only |
| Background Execution | \u2705 Yes | \u2705 Yes |
| Notifications | \u2705 Yes | \u274c No |
| Storage Access | \u2705 Yes | \u2705 Yes |
| Updates | \u2705 Automatic | \u274c Manual |
| Performance | \u26a0\ufe0f Good | \u2705 Better |
| Memory Usage | \u26a0\ufe0f Higher | \u2705 Lower |
| Offline Mode | \u2705 Yes | \u2705 Yes |

## \ud83d\udcde Support

- **Issues**: [GitHub Issues](https://github.com/NousResearch/hermes-agent/issues)
- **Discord**: [Join our server](https://discord.gg/NousResearch)
- **Docs**: [Official Documentation](https://hermes-agent.nousresearch.com/docs/)

## \ud83d\udcdc License

MIT License - See [LICENSE](LICENSE) for details.

---

**Hermes Agent - AI with Continuous Learning**

*Created by [Nous Research](https://nousresearch.com)*
