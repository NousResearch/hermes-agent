// API-only plugin: the Mini App UI is a separate React root served at /miniapp
// (web/src/miniapp), so the desktop dashboard gets a hidden, empty component.
(function () {
  window.__HERMES_PLUGINS__.register("telegram-miniapp", function () {
    return null;
  });
})();
