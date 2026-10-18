# Hermes Agent for Android - APK Installation

[![Download APK](https://img.shields.io/badge/Download-APK-green?style=for-the-badge)](https://github.com/NousResearch/hermes-agent/releases/latest/download/Hermes-Agent-latest.apk)

This guide explains how to install and use Hermes Agent on Android devices.

## 📱 Quick Start

### 1. Download the APK

Click the button above or visit: [Latest Release](https://github.com/NousResearch/hermes-agent/releases/latest)

### 2. Enable Unknown Sources

Before installing the APK, enable installation from unknown sources:

**Android 8.0+:**
- Open **Settings**
- Go to **Apps & notifications** > **Special app access** > **Install unknown apps**
- Select your browser or file manager and enable it

**Older Android:**
- Open **Settings**
- Go to **Security**
- Enable **Unknown sources**

### 3. Install the APK

- Open the downloaded file (usually in your **Downloads** folder)
- Tap **Install**
- Wait for installation to complete
- Tap **Open** to launch Hermes Agent

## 🎯 Features

✅ **Full AI Agent** - Same powerful agent as desktop version
✅ **Continuous Learning** - Skills improve with use
✅ **Multi-Model Support** - Use OpenRouter, Nous Research, and more
✅ **Background Execution** - Keep agent running
✅ **Offline Mode** - Works without internet (with local models)
✅ **Easy Updates** - Download new versions as they're released

## 📋 Requirements

- **Android 7.0 (Nougat) or later**
- **2GB RAM minimum** (4GB recommended)
- **500MB+ free storage**
- **Internet connection** for first setup

## 🚀 First Launch

On first launch, Hermes Agent will:

1. Request necessary permissions
2. Download Python runtime (if needed)
3. Download latest Hermes package
4. Initialize your agent

This may take 2-5 minutes depending on your connection speed.

## 💬 Usage

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

## ⚙️ Configuration

Edit `~/.hermes/config.yaml` or use the `hermes config` command.

### Example Configuration

```yaml
model_provider: openrouter
openrouter_api_key: sk-or-v1-your-key
model: openai/gpt-4o-mini
```

## 🔧 Troubleshooting

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

- **Solution 1**: Ensure you have internet connection
- **Solution 2**: Wait and retry (first launch downloads dependencies)
- **Solution 3**: Try on a different network

### Out of Memory Errors

- **Solution 1**: Use a smaller model (e.g., `mistralai/mistral-tiny`)
- **Solution 2**: Close other apps to free memory
- **Solution 3**: Use Termux version for better memory management

## 🏗️ Building from Source

### Prerequisites

- [Android Studio](https://developer.android.com/studio) or [Android SDK](https://developer.android.com/studio#command-tools)
- [JDK 17](https://adoptium.net/) or later
- Internet connection

### Build Steps

```bash
# Clone repository
git clone https://github.com/NousResearch/hermes-agent.git
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

⚠️ **Note**: Release builds require a signing keystore. Place it at `android/app/keystore/release.keystore`

## 📚 Alternative: Termux Installation

For advanced users who prefer terminal access:

```bash
# In Termux:
pkg update && pkg upgrade
pkg install wget
wget https://github.com/NousResearch/hermes-agent/releases/latest/download/hermes-agent_<version>_aarch64.deb
dpkg -i hermes-agent_<version>_aarch64.deb
hermes
```

## 🔒 Security

⚠️ **Important**:

- Only install APKs from official releases
- Review permissions before installing
- Your conversation data stays on your device
- Model interactions go through your configured provider

## 📊 Comparison: APK vs Termux

| Feature | APK | Termux |
|---------|-----|--------|
| Easy Installation | ✅ Yes | ❌ Manual |
| GUI Interface | ✅ Yes | ❌ CLI only |
| Background Execution | ✅ Yes | ✅ Yes |
| Notifications | ✅ Yes | ❌ No |
| Storage Access | ✅ Yes | ✅ Yes |
| Updates | ✅ Automatic | ❌ Manual |
| Performance | ⚠️ Good | ✅ Better |
| Memory Usage | ⚠️ Higher | ✅ Lower |
| Offline Mode | ✅ Yes | ✅ Yes |

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/NousResearch/hermes-agent/issues)
- **Discord**: [Join our server](https://discord.gg/NousResearch)
- **Docs**: [Official Documentation](https://hermes-agent.nousresearch.com/docs/)

## 📜 License

MIT License - See [LICENSE](LICENSE) for details.

---

**Hermes Agent - AI with Continuous Learning**

*Created by [Nous Research](https://nousresearch.com)*
