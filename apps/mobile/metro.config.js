const path = require('path');

const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);
const defaultResolveRequest = config.resolver.resolveRequest;
const expoMessageSocketShim = path.join(__dirname, 'src/mobile-shims/expo-message-socket.ts');
const expoLogBoxDevServerEndpointsShim = path.join(
  __dirname,
  'src/mobile-shims/expo-logbox-dev-server-endpoints.ts'
);

const extraAssetExts = ['html', 'css', 'rendererjs', 'woff', 'woff2', 'ttf'];
for (const ext of extraAssetExts) {
  if (!config.resolver.assetExts.includes(ext)) {
    config.resolver.assetExts.push(ext);
  }
}

config.resolver.resolveRequest = (context, moduleName, platform) => {
  if (
    moduleName === './messageSocket' ||
    moduleName === './async-require/messageSocket' ||
    moduleName === 'expo/src/async-require/messageSocket' ||
    moduleName === 'expo/src/async-require/messageSocket.native' ||
    moduleName === 'expo/build/async-require/messageSocket' ||
    moduleName === 'expo/build/async-require/messageSocket.native' ||
    moduleName.endsWith('/async-require/messageSocket') ||
    moduleName.endsWith('/async-require/messageSocket.native')
  ) {
    return { filePath: expoMessageSocketShim, type: 'sourceFile' };
  }

  if (
    moduleName === './utils/devServerEndpoints' ||
    moduleName === '../utils/devServerEndpoints' ||
    moduleName === '@expo/log-box/src/utils/devServerEndpoints' ||
    moduleName === '@expo/log-box/build/utils/devServerEndpoints' ||
    moduleName.endsWith('/utils/devServerEndpoints')
  ) {
    return { filePath: expoLogBoxDevServerEndpointsShim, type: 'sourceFile' };
  }

  if (defaultResolveRequest) {
    return defaultResolveRequest(context, moduleName, platform);
  }

  return context.resolveRequest(context, moduleName, platform);
};

module.exports = config;
