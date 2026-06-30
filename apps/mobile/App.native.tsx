import { useCallback, useRef, useState } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { WebView, type WebViewMessageEvent } from 'react-native-webview';

import { bundledRendererHtml } from './src/generated/bundled-renderer-html';

const defaultGatewayUrl = process.env.EXPO_PUBLIC_HERMES_GATEWAY_URL || '';
const defaultGatewayToken = process.env.EXPO_PUBLIC_HERMES_GATEWAY_TOKEN || '';
// When set (e.g. EXPO_PUBLIC_RENDERER_URL=http://127.0.0.1:5174) the WebView
// loads from Vite's dev server with HMR instead of the baked-in HTML — every
// renderer edit pushes in milliseconds, no `renderer:bundle` round-trip.
const rendererDevUrl = process.env.EXPO_PUBLIC_RENDERER_URL || '';

const mobileBridgeScript = `
(function () {
  try { document.documentElement.classList.add('hermes-mobile-standalone'); } catch (_) {}
  function stringify(value) {
    if (value instanceof Error) return value.stack || value.message;
    if (typeof value === 'string') return value;
    try { return JSON.stringify(value); } catch (_) { return String(value); }
  }
  function post(type, payload) {
    try { window.ReactNativeWebView && window.ReactNativeWebView.postMessage(JSON.stringify({ type: type, payload: payload })); } catch (_) {}
  }
  ['log','warn','error'].forEach(function (level) {
    var original = console[level] && console[level].bind(console);
    console[level] = function () {
      post('diagnostic', '[' + level + '] ' + Array.prototype.slice.call(arguments).map(stringify).join(' '));
      original && original.apply(console, arguments);
    };
  });
  window.addEventListener('error', function (event) {
    post('diagnostic', '[window-error] ' + [event.message, event.filename, event.lineno, event.colno, event.error && (event.error.stack || event.error.message)].map(stringify).join(' '));
  });
  window.addEventListener('unhandledrejection', function (event) {
    var reason = event.reason;
    post('diagnostic', '[unhandledrejection] ' + stringify(reason && (reason.stack || reason.message) || reason));
  });

  var defaultGatewayUrl = ${JSON.stringify(defaultGatewayUrl)}.replace(/\\/+$/, '');
  var defaultGatewayToken = ${JSON.stringify(defaultGatewayToken)};
  function cleanBaseUrl(value) {
    var trimmed = String(value || '').trim().replace(/\\/+$/, '');
    return trimmed || defaultGatewayUrl;
  }
  function toWsUrl(baseUrl, token) {
    var cleanUrl = cleanBaseUrl(baseUrl);
    if (!cleanUrl) return '';
    var url = new URL('/api/ws', cleanUrl);
    url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
    if (token) url.searchParams.set('token', token);
    return url.toString();
  }
  function readConnection() {
    var storedUrl = '';
    var storedToken = '';
    var storedProfile = '';
    try {
      storedUrl = localStorage.getItem('hermesDashboardUrl') || '';
      storedToken = localStorage.getItem('hermesDashboardToken') || '';
      storedProfile = localStorage.getItem('hermesDashboardProfile') || '';
    } catch (_) {}
    var baseUrl = cleanBaseUrl(storedUrl || defaultGatewayUrl);
    var token = String(storedToken || defaultGatewayToken || '').trim();
    var profile = String(storedProfile || 'default').trim() || 'default';
    return {
      baseUrl: baseUrl,
      isFullscreen: false,
      mode: 'remote',
      authMode: 'token',
      nativeOverlayWidth: 0,
      source: 'mobile-standalone',
      token: token,
      wsUrl: toWsUrl(baseUrl, token),
      logs: [],
      profile: profile,
      windowButtonPosition: null
    };
  }
  function saveConnection(next) {
    var baseUrl = cleanBaseUrl(next && (next.remoteUrl || next.url || next.baseUrl));
    var token = String(next && (next.remoteToken || next.token || next.sessionToken) || '').trim();
    var profile = String(next && next.profile || 'default').trim() || 'default';
    try {
      localStorage.setItem('hermesDashboardUrl', baseUrl);
      localStorage.setItem('hermesDashboardToken', token);
      localStorage.setItem('hermesDashboardProfile', profile);
    } catch (_) {}
    window.__HERMES_DASHBOARD_URL__ = baseUrl;
    window.__HERMES_DASHBOARD_TOKEN__ = token;
    return readConnection();
  }
  function clearConnection() {
    try {
      localStorage.removeItem('hermesDashboardUrl');
      localStorage.removeItem('hermesDashboardToken');
      localStorage.removeItem('hermesDashboardProfile');
    } catch (_) {}
    window.__HERMES_DASHBOARD_URL__ = defaultGatewayUrl;
    window.__HERMES_DASHBOARD_TOKEN__ = defaultGatewayToken;
  }
  function connectionConfig() {
    var conn = readConnection();
    return {
      envOverride: true,
      mode: 'remote',
      profile: conn.profile,
      remoteAuthMode: 'token',
      remoteOauthConnected: false,
      remoteTokenPreview: conn.token ? 'set' : null,
      remoteTokenSet: !!conn.token,
      remoteUrl: conn.baseUrl
    };
  }
  var noopUnsubscribe = function () { return function () {}; };
  var resolved = function (value) { return function () { return Promise.resolve(typeof value === 'function' ? value() : value); }; };

  window.__HERMES_DASHBOARD_URL__ = readConnection().baseUrl;
  window.__HERMES_DASHBOARD_TOKEN__ = readConnection().token;
  window.__HERMES_MOBILE_STANDALONE__ = true;

  window.hermesDesktop = {
    api: function (request) {
      return new Promise(function (resolve, reject) {
        var id = 'mobile-' + Date.now() + '-' + Math.random().toString(16).slice(2);
        var conn = readConnection();
        function onMessage(event) {
          var payload = event && event.data;
          if (typeof payload === 'string') {
            try { payload = JSON.parse(payload); } catch (_) {}
          }
          if (!payload || payload.type !== 'hermes:api:result' || payload.id !== id) return;
          window.removeEventListener('message', onMessage);
          if (payload.ok) resolve(payload.data);
          else reject(new Error(payload.error || 'Hermes API request failed'));
        }
        window.addEventListener('message', onMessage);
        post('hermes:api', {
          id: id,
          request: {
            baseUrl: conn.baseUrl,
            token: conn.token,
            method: request && request.method,
            path: request && request.path,
            body: request && request.body,
            timeoutMs: request && request.timeoutMs
          }
        });
      });
    },
    getConnection: resolved(readConnection),
    getConnectionConfig: resolved(connectionConfig),
    getBootProgress: resolved({ error: null, fakeMode: false, phase: 'done', message: 'Connected', progress: 100, running: false, timestamp: Date.now() }),
    getBootstrapState: resolved({ active: false, manifest: null, stages: {}, error: null, log: [], startedAt: null, completedAt: null, unsupportedPlatform: null }),
    getGatewayWsUrl: function () { return Promise.resolve(readConnection().wsUrl); },
    getRecentLogs: resolved({ path: '', lines: [] }),
    clearLogs: resolved(undefined),
    onBackendExit: noopUnsubscribe,
    onBackendOutput: noopUnsubscribe,
    onBootProgress: noopUnsubscribe,
    onBootstrapState: noopUnsubscribe,
    onConnectionChanged: noopUnsubscribe,
    onPowerResume: noopUnsubscribe,
    revalidateConnection: resolved({ ok: true, rebuilt: false }),
    touchBackend: resolved({ ok: true }),
    profile: { set: function (name) { var conn = readConnection(); saveConnection({ baseUrl: conn.baseUrl, token: conn.token, profile: name || 'default' }); return Promise.resolve({ name: name || 'default' }); } },
    settings: { getDefaultProjectDir: resolved({ defaultLabel: '~', dir: null, resolvedCwd: '~' }) },
    sanitizeWorkspaceCwd: function (cwd) { return Promise.resolve({ cwd: cwd || '~', sanitized: false }); },
    setTranslucency: resolved(undefined),
    notify: resolved(true),
    probeConnectionConfig: async function (remoteUrl) {
      // iOS WKWebView often reports generic Load failed for authenticated
      // Cloudflare/dashboard probes because Electron privileged networking is
      // not available here. Do not block first-run setup on a browser-side
      // probe; actual dashboard REST calls go through the React Native bridge.
      var baseUrl = cleanBaseUrl(remoteUrl);
      return { baseUrl: baseUrl, reachable: true, authMode: 'token', providers: [], version: null, error: null, mobileBypass: true };
    },
    testConnectionConfig: async function (config) {
      var probe = await window.hermesDesktop.probeConnectionConfig(config && (config.remoteUrl || config.url || config.baseUrl));
      return { baseUrl: probe.baseUrl, ok: true, connected: true, version: probe.version, mobileBypass: true };
    },
    saveConnectionConfig: function (config) { return Promise.resolve((saveConnection(config), connectionConfig())); },
    applyConnectionConfig: function (config) {
      if (config && config.mode === 'local') {
        clearConnection();
      } else {
        saveConnection(config || {});
      }
      window.location.reload();
      return Promise.resolve(connectionConfig());
    },
    resetMobileConnection: function () { clearConnection(); window.location.reload(); return Promise.resolve({ ok: true }); },
    resetBootstrap: function () { window.location.reload(); return Promise.resolve({ ok: true }); },
    repairBootstrap: resolved({ ok: false }),
    revealLogs: resolved({ ok: false, path: '', error: 'Logs are unavailable in mobile standalone' })
  };
  // Wrap the bridge so renderer code that calls a method we never stubbed
  // (e.g. fetchLinkTitle) gets a Promise back instead of undefined — which
  // would otherwise blow up at \`.then\` and tank the whole view.
  window.hermesDesktop = new Proxy(window.hermesDesktop, {
    get: function (target, prop) {
      if (prop in target) return target[prop];
      return function () { return Promise.resolve(undefined); };
    }
  });
})();
true;
`;

type BridgeMessage = {
  payload?: any;
  request?: {
    baseUrl?: string;
    body?: string;
    method?: string;
    path?: string;
    timeoutMs?: number;
    token?: string;
  };
  id?: string;
  type?: string;
};

function cleanBaseUrl(value: unknown) {
  return String(value || defaultGatewayUrl).trim().replace(/\/+$/, '');
}

function postResultScript(id: string, payload: { data?: unknown; error?: string; ok: boolean }) {
  return `window.dispatchEvent(new MessageEvent('message', { data: ${JSON.stringify(
    JSON.stringify({ type: 'hermes:api:result', id, ...payload })
  )} })); true;`;
}

export default function App() {
  const webViewRef = useRef<WebView>(null);
  const [status, setStatus] = useState('loading bundled Hermes');
  const [diagnostic, setDiagnostic] = useState('');

  const sendResult = useCallback((id: string, payload: { data?: unknown; error?: string; ok: boolean }) => {
    webViewRef.current?.injectJavaScript(postResultScript(id, payload));
  }, []);

  const handleMessage = useCallback(
    async (event: WebViewMessageEvent) => {
      let message: BridgeMessage | null = null;
      try {
        message = JSON.parse(event.nativeEvent.data) as BridgeMessage;
      } catch {
        setDiagnostic(event.nativeEvent.data.slice(0, 1600));
        return;
      }

      if (message.type === 'diagnostic') {
        // Console diagnostics from the WebView land here. Logging only — no UI:
        // the on-screen overlay is reserved for real WebView load failures.
        if (__DEV__) console.log('[hermes:webview]', String(message.payload || '').slice(0, 1800));
        return;
      }
      if (message.type !== 'hermes:api' || !message.payload?.id || !message.payload?.request?.path) return;

      const id = String(message.payload.id);
      const request = message.payload.request;
      const method = String(request.method || 'GET').toUpperCase();
      const baseUrl = cleanBaseUrl(request.baseUrl);
      const url = `${baseUrl}${request.path}`;
      const timeoutMs = Number(request.timeoutMs) > 0 ? Number(request.timeoutMs) : 30000;
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeoutMs);
      try {
        const headers: Record<string, string> = { accept: 'application/json' };
        if (request.body !== undefined) headers['content-type'] = 'application/json';
        if (request.token) {
          headers['X-Hermes-Session-Token'] = request.token;
          headers.authorization = `Bearer ${request.token}`;
        }
        let response: Response;
        try {
          response = await fetch(url, { body: request.body, headers, method, signal: controller.signal });
        } catch (fetchError: any) {
          const diag = {
            url,
            method,
            errName: fetchError?.name || 'unknown',
            errMsg: fetchError?.message || String(fetchError),
            errCode: fetchError?.code,
            errCause: fetchError?.cause ? String(fetchError.cause) : undefined,
            headers: Object.keys(headers),
            hasToken: Boolean(request.token),
            baseUrl,
            path: request.path,
          };
          setDiagnostic('FETCH FAILED:\n' + JSON.stringify(diag, null, 2));
          if (__DEV__) console.log('[hermes:native-fetch-fail]', diag);
          throw fetchError;
        }
        const text = await response.text();
        if (!response.ok) throw new Error(`Hermes API ${response.status} ${request.path}: ${text.slice(0, 240)}`);
        let data: unknown = text;
        try { data = text ? JSON.parse(text) : {}; } catch (_) {}
        sendResult(id, { ok: true, data });
      } catch (error) {
        const text = error instanceof Error ? error.message : String(error);
        sendResult(id, { ok: false, error: `${method} ${request.path} failed: ${text}` });
      } finally {
        clearTimeout(timer);
      }
    },
    [sendResult]
  );

  return (
    <View style={styles.root}>
      <WebView
        ref={webViewRef}
        source={rendererDevUrl ? { uri: rendererDevUrl } : { html: bundledRendererHtml, baseUrl: 'https://hermes.local/' }}
        style={styles.webview}
        containerStyle={styles.webview}
        originWhitelist={['*']}
        javaScriptEnabled
        domStorageEnabled
        injectedJavaScriptBeforeContentLoaded={mobileBridgeScript}
        scalesPageToFit={false}
        automaticallyAdjustContentInsets={false}
        contentInsetAdjustmentBehavior="never"
        hideKeyboardAccessoryView
        allowsBackForwardNavigationGestures
        setSupportMultipleWindows={false}
        onMessage={handleMessage}
        onLoadStart={() => setStatus('loading bundled Hermes')}
        onLoadEnd={() => setStatus('bundled WebView load ended')}
        onContentProcessDidTerminate={() => {
          setStatus('WebView content process terminated; reloading');
          webViewRef.current?.reload();
        }}
        onError={event => {
          setStatus(`WebView error: ${event.nativeEvent.description || 'unknown'}`);
          setDiagnostic(JSON.stringify(event.nativeEvent).slice(0, 1600));
        }}
      />
      {/* Diagnostic overlay removed from UI entirely — the JSON dump leaked
          into Release and there's no user-facing value in seeing fetch
          internals on screen. Diagnostic strings are still captured in the
          state ref for future logging hookup but not rendered. */}
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#f7f8fb' },
  webview: { flex: 1, backgroundColor: '#f7f8fb' },
  debugOverlay: {
    backgroundColor: 'rgba(0,0,0,0.76)',
    borderRadius: 12,
    bottom: 28,
    left: 12,
    maxHeight: 240,
    padding: 10,
    position: 'absolute',
    right: 12,
  },
  debugText: { color: '#fff', fontFamily: 'Menlo', fontSize: 11, lineHeight: 16 },
});
