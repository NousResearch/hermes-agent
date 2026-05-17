# Android SDK Setup

Step-by-step Android SDK installation on Ubuntu Linux for Meta Quest development. No Android Studio required.

## Prerequisites

```bash
sudo apt update
sudo apt install -y openjdk-17-jdk wget unzip
```

## 1. Download Command-Line Tools

```bash
mkdir -p "$HOME/android-sdk/cmdline-tools"
cd "$HOME/android-sdk/cmdline-tools"

wget https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip
unzip -q commandlinetools-linux-*.zip
mv cmdline-tools latest
```

## 2. Add to PATH

```bash
echo 'export ANDROID_HOME="$HOME/android-sdk"' >> ~/.bashrc
echo 'export ANDROID_SDK_ROOT="$ANDROID_HOME"' >> ~/.bashrc
echo 'export PATH="$ANDROID_HOME/cmdline-tools/latest/bin:$PATH"' >> ~/.bashrc
echo 'export PATH="$ANDROID_HOME/platform-tools:$PATH"' >> ~/.bashrc
echo 'export PATH="$ANDROID_HOME/build-tools/34.0.0:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

## 3. Accept Licenses

This is mandatory before installing packages:

```bash
sdkmanager --licenses
```

You will be prompted multiple times. Type `y` and press Enter for each license.

To accept all licenses non-interactively (useful for CI):

```bash
yes | sdkmanager --licenses
```

## 4. Install Core Packages

```bash
sdkmanager "platform-tools"
sdkmanager "build-tools;34.0.0"
sdkmanager "platforms;android-29"
sdkmanager "platforms;android-34"
sdkmanager "ndk;25.2.9519653"
```

## 5. Verify Installation

```bash
# Platform tools
adb version
# Example output: Android Debug Bridge version 1.0.41

# Build tools
apksigner --version
# Example output: 0.9

# SDK manager list
sdkmanager --list_installed
```

## 6. Create Debug Keystore

Godot and manual signing require a debug keystore:

```bash
mkdir -p "$HOME/.android"
keytool -keyalg RSA -genkeypair -alias androiddebugkey \
  -keypass android -keystore "$HOME/.android/debug.keystore" \
  -storepass android -dname "CN=Android Debug,O=Android,C=US" \
  -validity 9999
```

## 7. Godot Editor Settings

Tell Godot where the SDK and keystore are:

```bash
# In Godot editor (GUI):
# Editor -> Editor Settings -> Export -> Android
#   Android SDK Path: /home/USER/android-sdk
#   Java SDK Path: /usr/lib/jvm/java-17-openjdk-amd64
#   Debug Keystore: /home/USER/.android/debug.keystore
#   Debug Keystore Pass: android
```

Or edit `~/.config/godot/editor_settings-4.5.tres`:

```tres
export/android/android_sdk_path = "/home/USER/android-sdk"
export/android/java_sdk_path = "/usr/lib/jvm/java-17-openjdk-amd64"
export/android/debug_keystore = "/home/USER/.android/debug.keystore"
export/android/debug_keystore_pass = "android"
```

Replace `USER` with your actual username.

## Common Issues

### `sdkmanager: command not found`

The `latest` directory is not on PATH. Ensure:

```bash
ls "$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager"
```

If it exists but is not found, re-source your profile:

```bash
source ~/.bashrc
```

### `Warning: Could not create settings`

The cmdline-tools directory structure is incorrect. It must be:

```
$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager
```

Not:

```
$ANDROID_HOME/cmdline-tools/bin/sdkmanager
```

### ADB permissions (no devices found)

Add udev rules for Oculus/Meta devices:

```bash
sudo tee /etc/udev/rules.d/51-android.rules << 'EOF'
SUBSYSTEM=="usb", ATTR{idVendor}=="2833", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTR{idVendor}=="18d1", MODE="0666", GROUP="plugdev"
EOF

sudo udevadm control --reload-rules
sudo udevadm trigger
```

Then add your user to the `plugdev` group:

```bash
sudo usermod -aG plugdev $USER
```

Log out and back in for group changes to take effect.

### NDK version mismatch

Quest native builds typically use NDK 25. If you install a different version, update `ANDROID_NDK_HOME` accordingly:

```bash
export ANDROID_NDK_HOME="$ANDROID_HOME/ndk/XX.X.XXXXXXXX"
```

Godot's export system also checks `ANDROID_NDK_HOME` during Gradle builds.
