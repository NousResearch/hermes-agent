/**
 * react-dom/client shim — pulls from the dashboard's plugin SDK.
 */
const SDK = window.__HERMES_PLUGIN_SDK__;
const ReactDOM = SDK.ReactDOM || SDK["react-dom"] || {};

export default ReactDOM;

export const createRoot = ReactDOM.createRoot || ((container) => ({
  render: (children) => {
    // Fallback: use legacy ReactDOM.render
    if (ReactDOM.render) {
      ReactDOM.render(children, container);
    }
  },
  unmount: () => {},
}));
