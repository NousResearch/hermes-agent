(() => {
  const plugins = window.__HERMES_PLUGINS__;
  const sdk = window.__HERMES_PLUGIN_SDK__;
  if (!plugins || !sdk || !sdk.React) return;

  const { React } = sdk;

  function ExampleDashboardPlugin() {
    return React.createElement(
      "section",
      { className: "flex min-h-[16rem] flex-col gap-3 border border-current/20 bg-background-base/35 p-4 normal-case" },
      React.createElement("div", { className: "text-sm font-bold tracking-[0.08em] text-midground" }, "Example dashboard plugin"),
      React.createElement("p", { className: "max-w-2xl text-sm leading-relaxed text-midground/65" }, "This bundled plugin keeps the dashboard plugin API route testable without producing a missing JavaScript bundle in the browser."),
    );
  }

  plugins.register("example", ExampleDashboardPlugin);
})();
