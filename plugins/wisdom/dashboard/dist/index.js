// API-only plugin: the Desktop "Team Skills" page consumes /api/plugins/wisdom/*; the web
// dashboard registers a no-op component so the loader's contract is satisfied without a tab.
(function () {
  window.__HERMES_PLUGINS__.register("wisdom", function () { return null; });
})();
