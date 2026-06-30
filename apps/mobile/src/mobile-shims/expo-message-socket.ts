// Expo's dev async-require websocket assumes the bundle was loaded directly
// from Metro. The standalone mobile WebView shell can run a dev bundle from an
// embedded/static source, where Expo's getDevServer().bundleLoadedFromServer is
// false and the default module throws during app startup.
//
// We do not use Expo async-require/RSC dev commands in this app, so this module
// intentionally replaces expo/src/async-require/messageSocket with a no-op.
export {};
