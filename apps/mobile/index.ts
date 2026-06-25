import { AppRegistry } from 'react-native';

import App from './App';

// Do not use Expo's registerRootComponent here. In embedded/static iOS bundles
// it imports Expo.fx, whose dev async-require websocket throws when the bundle
// was not loaded directly from Metro.
AppRegistry.registerComponent('main', () => App);
