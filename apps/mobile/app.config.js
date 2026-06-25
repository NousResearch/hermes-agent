/** @type {import('expo/config').ExpoConfig} */
const config = {
  name: 'Hermes',
  slug: 'hermes-iphone',
  version: '0.1.0',
  orientation: 'portrait',
  icon: './assets/icon.png',
  userInterfaceStyle: 'automatic',
  ios: {
    supportsTablet: true,
    bundleIdentifier: 'com.nousresearch.hermes',
    infoPlist: {
      NSMicrophoneUsageDescription:
        'Hermes records audio you explicitly start so it can transcribe, summarize, and extract actions.',
    },
  },
  android: {
    package: 'com.nousresearch.hermes',
    adaptiveIcon: {
      foregroundImage: './assets/android-icon-foreground.png',
      backgroundImage: './assets/android-icon-background.png',
      monochromeImage: './assets/android-icon-monochrome.png',
      backgroundColor: '#F6F7FB',
    },
    predictiveBackGestureEnabled: false,
  },
  web: {
    favicon: './assets/favicon.png',
  },
  assetBundlePatterns: ['assets/**/*'],
  plugins: [
    [
      'expo-audio',
      {
        microphonePermission:
          'Hermes records audio you explicitly start so it can transcribe, summarize, and extract actions.',
        enableBackgroundRecording: true,
        enableBackgroundPlayback: false,
      },
    ],
    'expo-asset',
    'expo-secure-store',
  ],
};

module.exports = config;
