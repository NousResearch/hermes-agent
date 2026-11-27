import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.nousresearch.hermes.mobile',
  appName: 'Hermes',
  webDir: 'desktop-port/dist',
  backgroundColor: '#F8FAFF',
  ios: {
    // No rubber-band/page scrolling of the whole WebView (app feel instead of
    // website); inner overflow containers (chat, lists) keep scrolling.
    // zoomEnabled is already default false. contentInset explicitly 'never'.
    contentInset: 'never',
    scrollEnabled: false,
  },
  plugins: {
    CapacitorHttp: {
      enabled: true,
    },
  },
};

export default config;
